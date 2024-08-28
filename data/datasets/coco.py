# encoding: utf-8
from torchvision.datasets import Dataset


class CocoDataset(Dataset):
    def __init__(self, coco, preprocess):
        self.coco = coco
        self.preprocess = preprocess

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        image, captions = self.coco[idx]
        image = self.preprocess(image)
        return image, captions[0]
