# encoding: utf-8
from torch.utils import data

from .datasets.coco import CocoDataset
from .transforms import build_transforms
from torchvision.datasets import CocoCaptions


def build_dataset(transforms, is_train=True):
    coco_dataset = CocoCaptions(root='DATA/coco128/images/train2017',
                            annFile='DATA/coco128/labels/train2017',
                            transform=transforms)
    datasets = CocoDataset(coco_dataset,
                           transform=transforms)
    return datasets


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
