# encoding: utf-8
"""
@author: mikwieczorek
"""

import copy
import os
import random
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from tqdm import tqdm

from config import cfg
from losses.center_loss import CenterLoss
from losses.triplet_loss import CrossEntropyLabelSmooth, TripletLoss
from losses.octuplet_loss import OctupletLoss
from modelling.baseline import Baseline
from modelling.build import build_optimizer, build_scheduler
from utils.reid_metric import R1_mAP
from attributedict.collections import AttributeDict

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class ModelBase(nn.Module):
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        if cfg is None:
            hparams = {**kwargs}
        elif isinstance(cfg, dict):
            #print("hi")
            hparams = {**cfg, **kwargs}
            if cfg.TEST.ONLY_TEST:
                # To make sure that loaded hparams are overwritten by cfg we may have chnaged
                hparams = {**kwargs, **cfg}
        self.hparams = AttributeDict(hparams)

        #self.save_hyperparameters(self.hparams)
        #NEED TO SAVE HYPERPARAMETERS

        self.backbone = Baseline(self.hparams)

        #define losses
        self.contrastive_loss = TripletLoss(
            self.hparams.SOLVER.MARGIN, self.hparams.SOLVER.DISTANCE_FUNC
        )
        
        self.octuplet_loss = OctupletLoss(margin = 25)

        d_model = self.hparams.MODEL.BACKBONE_EMB_SIZE
        self.xent = CrossEntropyLabelSmooth(num_classes=self.hparams.num_classes)
        self.center_loss = CenterLoss(
            num_classes=self.hparams.num_classes, feat_dim=d_model
        )
        self.center_loss_weight = self.hparams.SOLVER.CENTER_LOSS_WEIGHT

        #define layers

        self.bn = torch.nn.BatchNorm1d(d_model)
        self.bn.bias.requires_grad_(False)

        self.fc_query = torch.nn.Linear(d_model, self.hparams.num_classes, bias=False)
        self.fc_query.apply(weights_init_classifier)


        self.losses_names = ["query_xent", "query_triplet", "query_center"]
        self.losses_dict = {n: [] for n in self.losses_names}


    @staticmethod
    def _calculate_centroids(vecs, dim=1):
        length = vecs.shape[dim]
        return torch.sum(vecs, dim) / length

    def configure_optimizers(self):
        optimizers_list = build_optimizer(self.named_parameters(), self.hparams)
        self.lr_scheduler = build_scheduler(optimizers_list[0], self.hparams)
        return optimizers_list, self.lr_scheduler
    
    def validation_create_centroids(
        self, embeddings, labels, camids, respect_camids=False
    ):
        num_query = self.hparams.num_query
        # Keep query data samples seperated
        embeddings_query = embeddings[:num_query].cpu()
        labels_query = labels[:num_query]

        # Process gallery samples further
        embeddings_gallery = embeddings[num_query:]
        labels_gallery = labels[num_query:]

        labels2idx = defaultdict(list)
        for idx, label in enumerate(labels_gallery):
            labels2idx[label].append(idx)

        labels2idx_q = defaultdict(list)
        for idx, label in enumerate(labels_query):
            labels2idx_q[label].append(idx)

        unique_labels = sorted(np.unique(list(labels2idx.keys())))

        centroids_embeddings = []
        centroids_labels = []

        if respect_camids:
            centroids_camids = []
            query_camid = camids[:num_query]

        # Create centroids for each pid seperately
        for label in unique_labels:
            cmaids_combinations = set()
            inds = labels2idx[label]
            inds_q = labels2idx_q[label]
            if respect_camids:
                selected_camids_g = camids[inds]

                selected_camids_q = camids[inds_q]
                unique_camids = sorted(np.unique(selected_camids_q))

                for current_camid in unique_camids:
                    # We want to select all gallery images that comes from DIFFERENT cameraId
                    camid_inds = np.where(selected_camids_g != current_camid)[0]
                    if camid_inds.shape[0] == 0:
                        continue
                    used_camids = sorted(
                        np.unique(
                            [cid for cid in selected_camids_g if cid != current_camid]
                        )
                    )
                    if tuple(used_camids) not in cmaids_combinations:
                        cmaids_combinations.add(tuple(used_camids))
                        centroids_emb = embeddings_gallery[inds][camid_inds]
                        centroids_emb = self._calculate_centroids(centroids_emb, dim=0)
                        centroids_embeddings.append(centroids_emb.detach().cpu())
                        centroids_camids.append(used_camids)
                        centroids_labels.append(label)

            else:
                centroids_labels.append(label)
                centroids_emb = embeddings_gallery[inds]
                centroids_emb = self._calculate_centroids(centroids_emb, dim=0)
                centroids_embeddings.append(centroids_emb.detach().cpu())

        # Make a single tensor from query and gallery data
        centroids_embeddings = torch.stack(centroids_embeddings).squeeze()
        centroids_embeddings = torch.cat(
            (embeddings_query, centroids_embeddings), dim=0
        )
        centroids_labels = np.hstack((labels_query, np.array(centroids_labels)))

        if respect_camids:
            query_camid = [[item] for item in query_camid]
            centroids_camids = query_camid + centroids_camids

        if not respect_camids:
            # Create dummy camids for query na gallery features
            # it is used in eval_reid script
            camids_query = np.zeros_like(labels_query)
            camids_gallery = np.ones_like(np.array(centroids_labels))
            centroids_camids = np.hstack((camids_query, np.array(camids_gallery)))

        return centroids_embeddings.cpu(), centroids_labels, centroids_camids

    def get_val_metrics(self, embeddings, labels, camids, val_dataloader):
        self.r1_map_func = R1_mAP(
            model=self,
            num_query=self.hparams.num_query,
            feat_norm=self.hparams.TEST.FEAT_NORM,
            val_dataloader = val_dataloader
        )
        respect_camids = (
            True
            if (
                self.hparams.MODEL.KEEP_CAMID_CENTROIDS
                and self.hparams.MODEL.USE_CENTROIDS
            )
            else False
        )
        cmc, mAP, all_topk = self.r1_map_func.compute(
            feats=embeddings.float(),
            pids=labels,
            camids=camids,
            respect_camids=respect_camids,
        )

        topks = {}
        for top_k, kk in zip(all_topk, [1, 5, 10, 20, 50]):
            print("top-k, Rank-{:<3}:{:.1%}".format(kk, top_k))
            topks[f"Top-{kk}"] = top_k
        print(f"mAP: {mAP}")

        log_data = {"mAP": mAP}

        # TODO This line below is hacky, but it works when grad_monitoring is active
        #self.trainer.logger_connector.callback_metrics.update(log_data)
        #log_data = {**log_data, **topks}
        #self.trainer.logger.log_metrics(log_data, step=self.trainer.current_epoch)
        
    @staticmethod
    def create_masks_train(class_labels):
        labels_dict = defaultdict(list)
        class_labels = class_labels.detach().cpu().numpy()
        for idx, pid in enumerate(class_labels):
            labels_dict[pid].append(idx)
        labels_list = [v for k, v in labels_dict.items()]
        labels_list_copy = copy.deepcopy(labels_list)
        lens_list = [len(item) for item in labels_list]
        lens_list_cs = np.cumsum(lens_list)

        max_gal_num = max(
            [len(item) for item in labels_dict.values()]
        )  ## TODO Should allow usage of all permuations

        masks = torch.ones((max_gal_num, len(class_labels)), dtype=bool)
        for _ in range(max_gal_num):
            for i, inner_list in enumerate(labels_list):
                if len(inner_list) > 0:
                    masks[_, inner_list.pop(0)] = 0
                else:
                    start_ind = lens_list_cs[i - 1]
                    end_ind = start_ind + lens_list[i]
                    masks[_, start_ind:end_ind] = 0

        return masks, labels_list_copy

