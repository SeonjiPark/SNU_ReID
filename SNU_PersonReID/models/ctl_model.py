import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import tensor

from SNU_PersonReID.models.bases import ModelBase

class CTLModel(ModelBase):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.losses_names = [
            "query_xent",
            "query_triplet",
            "query_center",
            "centroid_triplet",
        ]
        self.losses_dict = {n: [] for n in self.losses_names}

    def forward(self, x):
        _, features = self.backbone(x)

        return features