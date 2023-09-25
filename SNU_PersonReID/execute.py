import csv
import cv2, os, glob
import torch
import numpy as np
from pathlib import Path
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

### Functions used to extract pair_id
exctract_func = (
    lambda x: (x).rsplit(".", 1)[0].rsplit("_", 1)[0]
)  ## To extract pid from filename. Example: /path/to/dir/product001_04.jpg -> pid = product001

#changed to match
exctract_func = (
    lambda x: (x).rsplit(".", 1)[0].split('/')[-1].rsplit("_")[0]
)  ## To extract pid from filename. Example: /path/to/dir/product001_04.jpg -> pid = product001


def build_reid_model(args, device):
    reid_network = CTLModel(args, device=device, dnn=False, data=args.data)
    #print(reid_network)
    return reid_network


# def preprocess_img(img, fp_flag, img_size, auto=True):
#     img = img.permute(0, 3, 1, 2)  # HWC to CHW

#     im = img.type(torch.half) if fp_flag else img.type(torch.float)  # uint8 to fp16/32
#     im /= 255  # 0 - 255 to 0.0 - 1.0
#     if len(im.shape) == 3:
#         im = im[None]  # expand for batch dim
#     return im


def do_reid(args, reid_network, pred_querys, gt_ids):

    val_loader = make_inference_data_loader(args, args.gallery_path, ImageDataset)

    ### Inference
    embeddings_gallery, paths_gallery = run_inference(
        reid_network, val_loader, args, print_freq= 50, use_cuda=True
    )


    ### Create centroids
    pid_path_index = create_pid_path_index(paths=paths_gallery, func=exctract_func) #returns a dictionary of pids, and the index of embedding
    embeddings_gallery, paths_gallery = calculate_centroids(embeddings_gallery, pid_path_index)
   
    #Preprocess
    #resize to 64x128
    pred_querys_resized = []
    for i in range(len(pred_querys)):
        pred_querys[i] = pred_querys[i].permute(2,1,0).unsqueeze(0).float()
        query = torch.nn.functional.interpolate(pred_querys[i], (128,256), mode = 'bicubic')
        pred_querys_resized.append(query)

    #print(len(pred_querys_resized))
    ### Inference
    embeddings = run_inference_list(
        reid_network, pred_querys_resized, args, use_cuda=True
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

def eval_reid(args, model, embeddings, labels):

    
    embeddings, labels, camids = model.validation_create_centroids(
        embeddings,
        labels,
        camids,
        respect_camids=True,
    )

    model.get_val_metrics(embeddings, labels, camids)
    del embeddings, labels, camids
