
# encoding: utf-8
"""
Adapted and extended by:
@author: mikwieczorek
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import tensor
from tqdm import tqdm

from config import cfg
from modelling.bases import ModelBase
from utils.misc import run_main


class CTLModel(ModelBase):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.losses_names = [
            "query_xent",
            "query_triplet",
            "query_center",
            "centroid_triplet",
        ]
        self.losses_dict = {n: [] for n in self.losses_names}

    def forward(self, x):
                
        # Get backbone features
        _, features = self.backbone(x)

        return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLT Model Training")
    parser.add_argument(
        "--config_file", default="./configs/256_resnet50.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.TEST.ONLY_TEST = True

    run_main(cfg, CTLModel)
