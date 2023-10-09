# encoding: utf-8
"""
Partially based on work by:
@author:  sherlock
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

import glob
import json
import os.path as osp
import random
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler
from tqdm import tqdm

from .samplers import get_sampler
from .transforms import ReidTransforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ReidBaseDataModule(Dataset):
    """
    Base class for reid datasets
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.num_workers = kwargs.get("num_workers") if "num_workers" in kwargs else 6
        # 1 for PRW / 4 for market1501 (SJ No idea...)
        self.num_instances = (
            kwargs.get("num_instances") if "num_instances" in kwargs else 4 #changed from 4
        )

    def _get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, *_ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def _print_dataset_statistics(self, train, query=None, gallery=None):
        num_train_pids, num_train_imgs, num_train_cams = self._get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self._get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self._get_imagedata_info(
            gallery
        )

        # print("Dataset statistics:")
        # print("  ----------------------------------------")
        # print("  subset   | # ids | # images | # cameras")
        # print("  ----------------------------------------")
        # print(
        #     "  train    | {:5d} | {:8d} | {:9d}".format(
        #         num_train_pids, num_train_imgs, num_train_cams
        #     )
        # )
        # print(
        #     "  query    | {:5d} | {:8d} | {:9d}".format(
        #         num_query_pids, num_query_imgs, num_query_cams
        #     )
        # )
        # print(
        #     "  gallery  | {:5d} | {:8d} | {:9d}".format(
        #         num_gallery_pids, num_gallery_imgs, num_gallery_cams
        #     )
        # )
        # print("  ----------------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        # if not osp.exists(self.query_dir):
        #     raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def train_dataloader(
        self, cfg, sampler_name: str = "random_identity", **kwargs
    ):  
        sampler = get_sampler(
            sampler_name,
            data_source=self.train_dict,
            batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
            num_instances=self.num_instances,
            world_size = 1, #DEFINE LATER
            rank=0,
        )
        return DataLoader(
            self.train,
            self.cfg.SOLVER.IMS_PER_BATCH,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn_alternative,
            **kwargs,
        )

    def val_dataloader(self):
        sampler = SequentialSampler(
            self.val
        )  ## This get replaced with ddp mode by lightning
        return DataLoader(
            self.val,
            self.cfg.test_ims_per_batch,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
        )
    
    def gallery_dataloader(self):
        sampler = SequentialSampler(
            self.gallery_val
        )  ## This get replaced with ddp mode by lightning

        return DataLoader(
            self.gallery_val,
            self.cfg.test_ims_per_batch,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
        )

    def test_dataloader(self):
        sampler = SequentialSampler(
            self.train
        )  ## This get replaced with ddp mode by lightning
        return DataLoader(
            self.train,
            int(self.cfg.test_ims_per_batch),
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
        )

    @staticmethod
    def _load_json(path):
        with open(path, "r") as f:
            js = json.load(f)

        return js





class BaseDatasetLabelledPerPid(Dataset):
    def __init__(self, data, transform=None, num_instances=4, resample=False):
        self.samples = data
        self.transform = transform
        self.num_instances = num_instances
        self.resample = resample

    def __getitem__(self, pid):
        """
        Retrives self.num_instances per given pair_id
        Args:
            pid (int): Pair_id number actually

        Returns:
            num_instace of given pid
        """
        pid = int(pid)
        list_of_samples = self.samples[pid][
            :
        ]  # path, target, camid, idx <- in each inner tuple
        _len = len(list_of_samples)
        assert (
            _len > 1
        ), f"len of samples for pid: {pid} is <=1. len: {_len}, samples: {list_of_samples}"

        if _len < self.num_instances:
            choice_size = _len
            needPad = True
        else:
            choice_size = self.num_instances
            needPad = False

        # We shuffle self.samples[pid] as we extract instances from this dict directly
        random.shuffle(self.samples[pid])

        out = []
        for _ in range(choice_size):
            tup = self.samples[pid].pop(0)
            path, target, camid, idx = tup
            img = self.prepare_img(path)
            out.append(
                (img, target, camid, idx, True)
            )  ## True stand if the sample is real or mock

        if needPad:
            num_missing = self.num_instances - _len
            assert (
                num_missing != self.num_instances
            ), f"Number of missings sample in the batch is equal to num_instances. PID: {pid}"
            if self.resample:
                assert len(list_of_samples) > 0
                resampled = np.random.choice(
                    range(len(list_of_samples)), size=num_missing, replace=True
                )
                for idx in resampled:
                    path, target, camid, idx = list_of_samples[idx]
                    img = self.prepare_img(path)
                    out.append((img, target, camid, idx, True))
            else:
                img_mock = torch.zeros_like(img)
                for _ in range(num_missing):
                    out.append((img_mock, target, camid, idx, False))

        assert (
            len(out) == self.num_instances
        ), f"Number of returned tuples per id needs to be equal self.num_instance. It is: {len(out)}"

        return out

    def __len__(self):
        return len(self.samples) * self.num_instances

    def prepare_img(self, path):
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class BaseDatasetLabelled(Dataset):
    def __init__(self, data, transform=None, return_paths=False):
        self.samples = data
        self.transform = transform
        self.return_paths = return_paths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, camid, idx = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, target, camid, path
        else:
            return sample, target, camid, idx

    def __len__(self):
        return len(self.samples)


def collate_fn_alternative(batch):
    # imgs, pids, _, _, isReal = zip(*batch)
    imgs = [item[0] for sample in batch for item in sample]
    pids = [item[1] for sample in batch for item in sample]
    camids = [item[2] for sample in batch for item in sample]
    isReal = [item[4] for sample in batch for item in sample]

    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, torch.tensor(camids), torch.tensor(isReal)
