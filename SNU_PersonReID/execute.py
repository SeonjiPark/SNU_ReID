import csv
import re
import cv2, os, glob
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from SNU_PersonReID.inference.inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    calculate_centroids,
    create_pid_path_index,
    make_inference_data_loader,
    run_inference,
    run_inference_list,
)
from SNU_PersonReID.models.ctl_model import CTLModel
from SNU_PersonReID.utils.reid_metric import get_dist_func
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler
from datasets import init_dataset

### Functions used to extract pair_id
exctract_func = (
    lambda x: (x).rsplit(".", 1)[0].rsplit("_", 1)[0]
)  ## To extract pid from filename. Example: /path/to/dir/product001_04.jpg -> pid = product001

#changed to match
exctract_func = (
    lambda x: (x).rsplit(".", 1)[0].split('/')[-1].rsplit("_")[0]
)  ## To extract pid from filename. Example: /path/to/dir/product001_04.jpg -> pid = product001


def build_reid_model(args, device):
    # reid_network = CTLModel(args, device=device, dnn=False, data=args.data)
    reid_network = CTLModel(args, device=device, dnn=False)
    #print(reid_network)
    return reid_network



def do_reid(args, reid_network, pred_querys, gt_ids, gallery_dataloader):

    val_loader = make_inference_data_loader(args, args.gallery_path, ImageDataset)

    ### Inference
    embeddings_gallery, paths_gallery = run_inference(
        reid_network, val_loader, args, print_freq= 50, use_cuda=True
    )
    #print(embeddings_gallery.shape)
    #print(paths_gallery)

    ### Create centroids
    pid_path_index = create_pid_path_index(paths=paths_gallery, func=exctract_func) #returns a dictionary of pids, and the index of embedding
    embeddings_gallery, paths_gallery = calculate_centroids(embeddings_gallery, pid_path_index)
   

    #print(len(pred_querys_resized))
    ### Inference
    embeddings = run_inference_list(
        reid_network, pred_querys, args, use_cuda=True
    )

    #print(type(embeddings_gallery))
    if args.normalize_features:
        embeddings_gallery = torch.nn.functional.normalize(
            torch.from_numpy(embeddings_gallery), dim=1, p=2
        )
        embeddings = torch.nn.functional.normalize(
            torch.from_numpy(embeddings), dim=1, p=2
        )
    else:
        embeddings = torch.from_numpy(embeddings)



    ### Calculate similarity
    dist_func = get_dist_func("cosine")
    distmat = dist_func(x=embeddings, y=embeddings_gallery).cpu().numpy()
    indices = np.argsort(distmat, axis=1) #prints a matrix of sorted indices
    ### Constrain the results to only topk most similar ids
    indices = indices[:, : int(args.topk)] if args.topk else indices
    pred_ids = [
        int(paths_gallery[indices[q_num, :]][0].split('/')[-1].split('_')[0])
        for q_num, emb in enumerate(embeddings)
    ]
    

    thresholdlist = []
    if args.use_unknown == "True":
        distance = [
            distmat[q_num, indices[q_num, :]][0]
            for q_num, emb in enumerate(embeddings)
        ]
        for i in range(len(distance)):
            #print(distance[i])
            if distance[i] > float(args.reid_threshold):
                pred_ids[i] = 'unknown'

    return pred_ids, embeddings

def eval1(args, model, embeddings, labels):
    x = embeddings 
    x = torch.nn.functional.interpolate(x, scale_factor = 1/int(args.scale), mode = 'bicubic')
    x = torch.nn.functional.interpolate(x, scale_factor = int(args.scale), mode = 'bicubic')

    with torch.no_grad():
        _, emb = model.backbone(x)
        emb = model.bn(emb)
    output = {"emb": emb, "labels": labels.cuda()}
    return output


def eval2(model, batch, batch_idx):
    x, class_labels, camid, idx = batch
    x = x.cuda()
    with torch.no_grad():
        _, emb = model.backbone(x)
        emb = model.bn(emb)
    return {"emb": emb, "labels": class_labels.cuda()}


def _process_dir(dir_path, relabel=False):
    img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    dataset_dict = defaultdict(list)
    dataset = []

    for idx, img_path in enumerate(img_paths):
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        # assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel: pid = pid2label[pid]
        dataset.append((img_path, pid, camid, idx))
        dataset_dict[pid].append((img_path, pid, camid, idx))

    return dataset, dataset_dict

def load_gallery(args):
    dm = init_dataset(
        args.dataset_name, cfg=args, num_workers=8
    )
    dm.setup()

    gallery_loader = dm.gallery_dataloader()
    return gallery_loader

