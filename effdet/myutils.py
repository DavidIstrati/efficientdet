import torch
import os
from datetime import datetime
import time
import random
import cv2
import numpy as np
import albumentations as A
from tqdm.auto import tqdm
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from glob import glob
from warmup_scheduler import GradualWarmupScheduler
from apex import amp


def get_train_transforms(img_size):
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(500, 500), height=512, width=512, p=0.5),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=0.2,
                    sat_shift_limit=0.2,
                    val_shift_limit=0.2,
                    p=0.9
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.9
                ),
            ], p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=img_size, width=img_size, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms(img_size):
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, data_path, img_size, transforms=None, test=False, cache=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.data_path = data_path
        self.transforms = transforms
        self.test = test
        self.imgs = [None] * len(image_ids)
        self.img_size = img_size
        if cache:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.image_ids)), desc='Caching images')
            for index in pbar:  # max 10k images
                image_id = self.image_ids[index]
                image = cv2.imread(f'{self.data_path}/{image_id}.jpg', cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.imgs[index] = image
                gb += image.nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        # image, boxes = self.load_mixup_image_and_boxes(index)
        rand_n = random.random()
        if self.test or rand_n >= 0.8:
            image, boxes = self.load_image_and_boxes(index)
            # print('default')
        elif 0.6 <= rand_n < 0.8:
            if rand_n < 0.7:
                image, boxes = self.load_mixup_image_and_boxes(index)
                # print('mixup')
            else:
                image, boxes = self.load_cutmix_image_and_boxes(index)
                # print('cutmix')
        elif 0.3 < rand_n < 0.6:
            image, boxes = self.load_mosaic_image_and_boxes(index)
            # print('mosaic')
        else:
            image, boxes = self.load_moaic_mixup_image_and_boxes(index)
            # print('mosaic_mixup')

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]),
                  'img_scale': torch.tensor([1.]), 'img_size': torch.tensor([(self.img_size, self.img_size)])}

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning
                    # target['labels'] = torch.stack(sample['labels'])
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image = self.imgs[index]
        image_id = self.image_ids[index]
        if image is None:  # not cache
            image = cv2.imread(f'{self.data_path}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_mixup_image_and_boxes(self, index, imsize=1024):
        img, labels = self.load_image_and_boxes(index)
        img2, labels2 = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        r = 0.5  # mixup 1:1
        img = (img * r + img2 * (1 - r))
        labels = np.concatenate((labels, labels2), 0)
        return img, labels

    def load_mosaic_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        image, boxes = self.load_image_and_boxes(index)
        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))

        mixup_image = image.copy()

        imsize = image.shape[0]
        x1, y1 = [int(random.uniform(imsize * 0.0, imsize * 0.45)) for _ in range(2)]
        x2, y2 = [int(random.uniform(imsize * 0.55, imsize * 1.0)) for _ in range(2)]

        mixup_boxes = r_boxes.copy()
        mixup_boxes[:, [0, 2]] = mixup_boxes[:, [0, 2]].clip(min=x1, max=x2)
        mixup_boxes[:, [1, 3]] = mixup_boxes[:, [1, 3]].clip(min=y1, max=y2)

        mixup_boxes = mixup_boxes.astype(np.int32)
        mixup_boxes = mixup_boxes[
            np.where((mixup_boxes[:, 2] - mixup_boxes[:, 0]) * (mixup_boxes[:, 3] - mixup_boxes[:, 1]) > 0)]

        # mix img
        mixup_image[y1:y2, x1:x2] = (mixup_image[y1:y2, x1:x2] + r_image[y1:y2, x1:x2]) / 2
        # mix boxes
        boxes = np.concatenate((boxes, mixup_boxes), 0)

        return mixup_image, boxes

    def load_moaic_mixup_image_and_boxes(self, index, imsize=1024):
        img, boxes = self.load_mosaic_image_and_boxes(index)
        img2, boxes2 = self.load_mosaic_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        r = 0.5  # mixup 1:1
        img = (img * r + img2 * (1 - r))
        boxes = np.concatenate((boxes, boxes2), 0)
        return img, boxes


class Fitter:

    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = config.OptimizerClass(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        if self.config.warmup:
            self.warmup_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=5,
                                                           after_scheduler=self.scheduler)

        self.log(f'Fitter prepared. Device is {self.device}')

        if self.config.apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

    def fit(self, train_loader, validation_loader):
        for epoch in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader, epoch)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        cls_loss = AverageMeter()
        box_loss = AverageMeter()
        t = time.time()
        nb = len(val_loader)
        pbar = tqdm(enumerate(val_loader), total=nb)
        for step, (images, targets, image_ids) in pbar:
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'cls_loss: {cls_loss.avg:.5f}, ' +
                        f'box_loss: {box_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                images = images.to(self.device).float()
                batch_size = images.shape[0]
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]
                img_scale = torch.tensor([target['img_scale'] for target in targets]).to(self.device).float()
                img_size = torch.tensor([(self.config.img_size, self.config.img_size) for target in targets]).to(
                    self.device).float()
                targets = {'bbox': boxes, 'cls': labels, 'img_scale': img_scale, 'img_size': img_size}

                output = self.model(images, targets)
                loss = output['loss']
                closs = output['class_loss']
                bloss = output['box_loss']

                summary_loss.update(loss.detach().item(), batch_size)
                cls_loss.update(closs.detach().item(), batch_size)
                box_loss.update(bloss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        if self.config.warmup:
            self.warmup_scheduler.step(epoch)
        summary_loss = AverageMeter()
        cls_loss = AverageMeter()
        box_loss = AverageMeter()
        t = time.time()
        nb = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total=nb)
        for step, (images, targets, image_ids) in pbar:
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    pbar.set_description(
                        f'Train Step {step}/{len(train_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'cls_loss: {cls_loss.avg:.5f}, ' +
                        f'box_loss: {box_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.5f}'
                    )

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            targets = {'bbox': boxes, 'cls': labels}
            # targets['image_id'] = target['image_id']

            output = self.model(images, targets)
            loss = output['loss']
            closs = output['class_loss']
            bloss = output['box_loss']

            if self.config.apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)
            cls_loss.update(closs.detach().item(), batch_size)
            box_loss.update(bloss.detach().item(), batch_size)

            if self.config.warmup:
                accumulate = min(self.config.accumulate,
                                 round(np.interp(epoch * nb + step, [1, epoch * nb * 5], [1, self.config.accumulate])))
            else:
                accumulate = self.config.accumulate

            if (epoch * nb + step) % accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        if self.config.step_scheduler:
            self.scheduler.step()

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
