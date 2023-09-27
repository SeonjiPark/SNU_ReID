# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from .random_erasing import RandomErasing

class ReidTransforms():

    def __init__(self, cfg):
        self.cfg = cfg

    def build_transforms(self, is_train=True):
        normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            transform = T.Compose([
                T.Resize([256,128]),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(int(self.cfg.input_padding)),
                T.RandomCrop([256,128]),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=0.5, mean = [0.485, 0.456, 0.406])
            ])
        else:
            transform = T.Compose([
                T.Resize([256, 128]),
                T.ToTensor(),
                normalize_transform
            ])

        return transform