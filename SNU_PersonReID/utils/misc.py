# encoding: utf-8
"""
Adapted and extended by:
@author: mikwieczorek
"""

import multiprocessing
import os
import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from datasets import init_dataset
from SNU_PersonReID.models.backbones.resnet_ibn_a import resnet50_ibn_a
from SNU_PersonReID.models import build_optimizer, build_scheduler
from einops import rearrange, repeat
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_distributed_sampler(
    trainer, dataset, train, **kwargs
) -> torch.utils.data.sampler.Sampler:
    world_size = {
        "ddp": trainer.num_nodes * trainer.num_processes,
        "ddp_spawn": trainer.num_nodes * trainer.num_processes,
        "ddp2": trainer.num_nodes,
        "ddp_cpu": trainer.num_processes * trainer.num_nodes,
    }
    assert trainer.distributed_backend is not None
    kwargs = dict(
        num_replicas=world_size[trainer.distributed_backend], rank=trainer.global_rank
    )

    kwargs["shuffle"] = train and not trainer.overfit_batches
    sampler = DistributedSampler(dataset, **kwargs)
    return sampler

def get_backbone(name: str, **kwargs) -> torch.nn.Module:
    """
    Gets just the encoder portion of a torchvision model (replaces final layer with identity)
    :param name: (str) name of the model
    :param kwargs: kwargs to send to the model
    :return:
    """

    if name in torchvision.models.__dict__:
        model_creator = torchvision.models.__dict__.get(name)
    elif name == "resnet50_ibn_a":
        model = resnet50_ibn_a(last_stride=1, **kwargs)
        model_creator = True
    else:
        raise AttributeError(f"Unknown architecture {name}")

    assert model_creator is not None, f"no torchvision model named {name}"
    if name != "resnet50_ibn_a":
        model = model_creator(**kwargs)
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()
    else:
        raise NotImplementedError(f"Unknown class {model.__class__}")

    return model

def run_test(cfg, method, dm, load_path, scale = 1):
    checkpoint = torch.load(load_path)
    
    model = method(
                cfg,
                num_query=dm.num_query,
                num_classes=dm.num_classes,
            ).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    

    model.backbone.eval()
    model.bn.eval()
    outputs = []
    for batch in dm.val_dataloader():
        x, class_labels, camid, idx = batch
        x, class_labels, camid = x.cuda(), class_labels.cuda(), camid.cuda()

        x = torch.nn.functional.interpolate(x, scale_factor = 1/int(scale), mode = 'bicubic')
        x = torch.nn.functional.interpolate(x, scale_factor = int(scale), mode = 'bicubic')

        with torch.no_grad():
            _, emb = model.backbone(x)
            emb = model.bn(emb)
        outputs.append({"emb": emb, "labels": class_labels, "camid": camid, "idx": idx})

    embeddings = torch.cat([x.pop("emb") for x in outputs]).detach().cpu()
    labels = (
        torch.cat([x.pop("labels") for x in outputs]).detach().cpu().numpy()
    )
    camids = torch.cat([x.pop("camid") for x in outputs]).cpu().detach().numpy()
    del outputs

    if model.hparams.MODEL.USE_CENTROIDS:
        embeddings, labels, camids = model.validation_create_centroids(
            embeddings,
            labels,
            camids,
            respect_camids=model.hparams.MODEL.KEEP_CAMID_CENTROIDS,
        )

    model.get_val_metrics(embeddings, labels, camids,  dm.val_dataloader())
    del embeddings, labels, camids

def run_train(cfg, method, writer, dm, scale):

    if cfg.MODEL.RESUME_TRAINING:
        print("RESUME TRAINING")
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)
        cfg.OUTPUT_DIR = os.path.dirname(cfg.MODEL.PRETRAIN_PATH)
        model = method(
                    cfg,
                    num_query=dm.num_query,
                    num_classes=dm.num_classes,
                ).cuda()
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        scale = checkpoint['scale']
        
        optimizers_list, lr_scheduler = model.configure_optimizers()
        opt, opt_center = optimizers_list[0], optimizers_list[1]

        opt.load_state_dict(checkpoint['opt_state_dict'])
        opt_center.load_state_dict(checkpoint['opt_center_state_dict'])

        print("model loaded")
    else:
        model = method(
                    cfg,
                    num_query=dm.num_query,
                    num_classes=dm.num_classes,
                ).cuda()

        optimizers_list, lr_scheduler = model.configure_optimizers()
        opt, opt_center = optimizers_list[0], optimizers_list[1]
        epoch_start = 0
        print("NEW TRAINING")


    train_dataloader = dm.train_dataloader(cfg, sampler_name=cfg.DATALOADER.SAMPLER, drop_last=cfg.DATALOADER.DROP_LAST,)

    model.train()
    for epoch in range(epoch_start, cfg.SOLVER.MAX_EPOCHS):
        outputs = []
        with tqdm(train_dataloader, unit = "batch") as tepoch:
            for idx, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                x, class_labels, camid, isReal = batch
                
                unique_classes = len(np.unique(class_labels.detach().cpu()))

                x, class_labels, camid, isReal = x.cuda(), class_labels.cuda(), camid.cuda(), isReal.cuda()

                x = torch.nn.functional.interpolate(x, scale_factor = 1/int(scale), mode = 'bicubic')
                x = torch.nn.functional.interpolate(x, scale_factor = int(scale), mode = 'bicubic')

                features = model(x)

                opt_center.zero_grad()
                opt.zero_grad()

                #calc loss
                contrastive_loss_query, dist_ap, dist_an = model.contrastive_loss(
                features, class_labels, mask=isReal
                )
                contrastive_loss_query = (
                    contrastive_loss_query * model.hparams.SOLVER.QUERY_CONTRASTIVE_WEIGHT
                )

                center_loss = model.hparams.SOLVER.CENTER_LOSS_WEIGHT * model.center_loss(
                    features, class_labels
                )
                bn_features = model.bn(features)
                cls_score = model.fc_query(bn_features)
                xent_query = model.xent(cls_score, class_labels)
                xent_query = xent_query * model.hparams.SOLVER.QUERY_XENT_WEIGHT
                #print(model.hparams.MODEL.USE_CENTROIDS)
                if model.hparams.MODEL.USE_CENTROIDS:
                    # Prepare masks for uneven numbe of sample per pid in a batch
                    ir = isReal.view(unique_classes, -1)
                    t = repeat(ir, "c b -> c b s", s=model.hparams.DATALOADER.NUM_INSTANCE)
                    t_re = rearrange(t, "c b s -> b (c s)")
                    t_re = t_re & isReal

                    masks, labels_list = model.create_masks_train(class_labels)  ## True for gallery
                    masks = masks.to(features.device)
                    masks = masks & t_re

                    masks_float = masks.float().to(features.device)
                    padded = masks_float.unsqueeze(-1) * features.unsqueeze(0)  # For broadcasting

                    centroids_mask = rearrange(
                        masks, "i (ins s) -> i ins s", s=model.hparams.DATALOADER.NUM_INSTANCE
                    )
                    padded_tmp = rearrange(
                        padded,
                        "i (ins s) dim -> i ins s dim",
                        s=model.hparams.DATALOADER.NUM_INSTANCE,
                    )
                    valid_inst = centroids_mask.sum(-1)
                    valid_inst_bool = centroids_mask.sum(-1).bool()
                    centroids_emb = padded_tmp.sum(-2) / valid_inst.masked_fill(
                        valid_inst == 0, 1
                    ).unsqueeze(-1)

                    contrastive_loss_total = []
                    ap_total = []
                    an_total = []
                    l2_mean_norm_total = []
                    xent_centroids_total = []

                    for i in range(model.hparams.DATALOADER.NUM_INSTANCE):
                        if valid_inst_bool[i].sum() <= 1:
                            continue

                        current_mask = masks[i, :]
                        current_labels = class_labels[~current_mask & t_re[i]]
                        query_feat = features[~current_mask & t_re[i]]
                        current_centroids = centroids_emb[i]
                        current_centroids = current_centroids[
                            torch.abs(current_centroids).sum(1) > 1e-7
                        ]
                        embeddings_concat = torch.cat((query_feat, current_centroids))
                        labels_concat = torch.cat((current_labels, current_labels))

                        contrastive_loss, dist_ap, dist_an = model.contrastive_loss(
                            embeddings_concat, labels_concat
                        )

                        with torch.no_grad():
                            dist_ap = dist_ap.data.mean()
                            dist_an = dist_an.data.mean()
                        ap_total.append(dist_ap)
                        an_total.append(dist_an)

                        contrastive_loss_total.append(contrastive_loss)

                        # L2 norm of centroid vectors
                        l2_mean_norm = torch.norm(current_centroids, dim=1).mean()
                        l2_mean_norm_total.append(l2_mean_norm)

                    contrastive_loss_step = (
                        torch.mean(torch.stack(contrastive_loss_total))
                        * model.hparams.SOLVER.CENTROID_CONTRASTIVE_WEIGHT
                    )
                    dist_ap = torch.mean(torch.stack(ap_total))
                    dist_an = torch.mean(torch.stack(an_total))
                    l2_mean_norm_total = torch.mean(torch.stack(l2_mean_norm_total))
                else:
                    contrastive_loss_step = torch.tensor(0)
                
                total_loss = center_loss + xent_query + contrastive_loss_query + contrastive_loss_step

                #backprop
                total_loss.backward()
                opt.step()
                
                for param in model.center_loss.parameters():
                    param.grad.data *= 1.0 / model.hparams.SOLVER.CENTER_LOSS_WEIGHT

                opt_center.step()           

                losses = [xent_query, contrastive_loss_query, center_loss, contrastive_loss_step]
                losses = [item.detach() for item in losses]
                losses = list(map(float, losses))

                for name, loss_val in zip(model.losses_names, losses):
                    model.losses_dict[name].append(loss_val) 

                if model.hparams.MODEL.USE_CENTROIDS:
                    log_data = {
                        "step_dist_ap": float(dist_ap.mean()),
                        "step_dist_an": float(dist_an.mean()),
                        "l2_mean_centroid": float(l2_mean_norm_total),
                    }
                else:
                    log_data = {
                        "step_dist_ap": float(dist_ap.mean()),
                        "step_dist_an": float(dist_an.mean()),
                    }

                output = {"loss" : total_loss, "other": log_data} 
            
                if idx % 100 == 0: 
                    #print(output)              

                    n_iter = epoch * len(train_dataloader) + idx            
                    #logging
                    writer.add_scalar('Loss/total', total_loss, n_iter)
                    writer.add_scalar('Loss/xent_query', xent_query, n_iter)
                    writer.add_scalar('Loss/contrastive_loss_query', contrastive_loss_query, n_iter)
                    writer.add_scalar('Loss/center_loss', center_loss, n_iter)
                    writer.add_scalar('Loss/contrastive_loss_step', contrastive_loss_step, n_iter)

                outputs.append(output)

        #epoch end
        lr = model.lr_scheduler.get_last_lr()[0]
        loss = torch.stack([x.pop("loss") for x in outputs]).mean().cpu().detach()
        epoch_dist_ap = np.mean([x["other"].pop("step_dist_ap") for x in outputs])
        epoch_dist_an = np.mean([x["other"].pop("step_dist_an") for x in outputs])

        del outputs

        log_data = {
            "epoch_train_loss": float(loss),
            "epoch_dist_ap": epoch_dist_ap,
            "epoch_dist_an": epoch_dist_an,
            "lr": lr,
        }

        if hasattr(model, "losses_dict"):
            for name, loss_val in model.losses_dict.items():
                val_tmp = np.mean(loss_val)
                log_data.update({name: val_tmp})
                model.losses_dict[name] = []  ## Zeroing values after a completed epoch
        
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            print(log_data)
    
            savepath = cfg.OUTPUT_DIR
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'opt_center_state_dict': opt_center.state_dict(),
                'num_query': dm.num_query,
                'num_classes': dm.num_classes,
                'scale' : scale
                }, os.path.join(savepath, f"{epoch}.pth"))  
            
            load_path = os.path.join(savepath, f"{epoch}.pth")

            if model.hparams.MODEL.USE_CENTROIDS:
                cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
                print("Test baseline")
                run_test(cfg, method, dm, load_path)
                print("Test CTL")
                cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
                run_test(cfg, method, dm, load_path)
            else:
                print("Test baseline")
                run_test(cfg, method, dm, load_path)
    
   
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'opt_center_state_dict': opt_center.state_dict(),
        'num_query': dm.num_query,
        'num_classes': dm.num_classes,
        'scale' : scale
        }, os.path.join(savepath, f"{epoch}.pth"))  

    print("Final Evaluation and Save")
    load_path = os.path.join(savepath, f"{epoch}.pth")

    if model.hparams.MODEL.USE_CENTROIDS:
        cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
        print("Test baseline")
        run_test(cfg, method, dm, load_path)
        print("Test CTL")
        cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
        run_test(cfg, method, dm, load_path)
    else:
        print("Test baseline")
        run_test(cfg, method, dm, load_path)
    
def finetune_oct(cfg, method, writer, dm, scale):
    print("FINETUNING FROM CHECKPOINT")
    checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)
    cfg.OUTPUT_DIR = os.path.dirname(cfg.MODEL.PRETRAIN_PATH)
    model = method(
                cfg,
                num_query=dm.num_query,
                num_classes=dm.num_classes,
            ).cuda()
    epoch_start = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizers_list, lr_scheduler = model.configure_optimizers()
    opt, opt_center = optimizers_list[0], optimizers_list[1]

    opt.load_state_dict(checkpoint['opt_state_dict'])
    opt_center.load_state_dict(checkpoint['opt_center_state_dict'])

    print("model loaded")

    load_path = cfg.MODEL.PRETRAIN_PATH
    if model.hparams.MODEL.USE_CENTROIDS:
        cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
        print("Test baseline")
        run_test(cfg, method, dm, load_path, scale)
        print("Test CTL")
        cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
        run_test(cfg, method, dm, load_path, scale)
    else:
        print("Test baseline")
        run_test(cfg, method, dm, load_path, scale)


    #####################

    train_dataloader = dm.train_dataloader(cfg, sampler_name=cfg.DATALOADER.SAMPLER, drop_last=cfg.DATALOADER.DROP_LAST,)

    model.train()
    for epoch in range(epoch_start, epoch_start + 10):
        outputs = []
        with tqdm(train_dataloader, unit = "batch") as tepoch:
            for idx, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                model.hparams.SOLVER.BASE_LR = 0.001
                if epoch >= epoch_start+9:
                    model.hparams.SOLVER.BASE_LR = model.hparams.SOLVER.BASE_LR /8
                elif epoch >= epoch_start+7:
                    model.hparams.SOLVER.BASE_LR = model.hparams.SOLVER.BASE_LR /4
                elif epoch >= epoch_start+4:
                    model.hparams.SOLVER.BASE_LR = model.hparams.SOLVER.BASE_LR /2

                for pg in opt.param_groups:
                    pg["lr"] =  model.hparams.SOLVER.BASE_LR

                x, class_labels, camid, isReal = batch
                
                unique_classes = len(np.unique(class_labels.detach().cpu()))

                x, class_labels, camid, isReal = x.cuda(), class_labels.cuda(), camid.cuda(), isReal.cuda()

                #print(scale)

                x_lr = torch.clone(x)
                x_lr = torch.nn.functional.interpolate(x_lr, scale_factor = 1/int(scale), mode = 'bicubic')
                x_lr = torch.nn.functional.interpolate(x_lr, scale_factor = int(scale), mode = 'bicubic')

                features = model(x)
                features_lr = model(x_lr)

                features_cat = torch.cat((features, features_lr), dim = 0)

                octuplet_loss = model.octuplet_loss(features_cat, class_labels)


                opt_center.zero_grad()
                opt.zero_grad()

                #calc loss
                contrastive_loss_query, dist_ap, dist_an = model.contrastive_loss(
                features, class_labels, mask=isReal
                )
                contrastive_loss_query = (
                    contrastive_loss_query * model.hparams.SOLVER.QUERY_CONTRASTIVE_WEIGHT
                )

                center_loss = model.hparams.SOLVER.CENTER_LOSS_WEIGHT * model.center_loss(
                    features, class_labels
                )
                bn_features = model.bn(features)
                cls_score = model.fc_query(bn_features)
                xent_query = model.xent(cls_score, class_labels)
                xent_query = xent_query * model.hparams.SOLVER.QUERY_XENT_WEIGHT

                #print(model.hparams.MODEL.USE_CENTROIDS)
                if model.hparams.MODEL.USE_CENTROIDS:
                    # Prepare masks for uneven numbe of sample per pid in a batch
                    ir = isReal.view(unique_classes, -1)
                    t = repeat(ir, "c b -> c b s", s=model.hparams.DATALOADER.NUM_INSTANCE)
                    t_re = rearrange(t, "c b s -> b (c s)")
                    t_re = t_re & isReal

                    masks, labels_list = model.create_masks_train(class_labels)  ## True for gallery
                    masks = masks.to(features.device)
                    masks = masks & t_re

                    masks_float = masks.float().to(features.device)
                    padded = masks_float.unsqueeze(-1) * features.unsqueeze(0)  # For broadcasting

                    centroids_mask = rearrange(
                        masks, "i (ins s) -> i ins s", s=model.hparams.DATALOADER.NUM_INSTANCE
                    )
                    padded_tmp = rearrange(
                        padded,
                        "i (ins s) dim -> i ins s dim",
                        s=model.hparams.DATALOADER.NUM_INSTANCE,
                    )
                    valid_inst = centroids_mask.sum(-1)
                    valid_inst_bool = centroids_mask.sum(-1).bool()
                    centroids_emb = padded_tmp.sum(-2) / valid_inst.masked_fill(
                        valid_inst == 0, 1
                    ).unsqueeze(-1)

                    contrastive_loss_total = []
                    ap_total = []
                    an_total = []
                    l2_mean_norm_total = []
                    xent_centroids_total = []

                    for i in range(model.hparams.DATALOADER.NUM_INSTANCE):
                        if valid_inst_bool[i].sum() <= 1:
                            continue

                        current_mask = masks[i, :]
                        current_labels = class_labels[~current_mask & t_re[i]]
                        query_feat = features[~current_mask & t_re[i]]
                        current_centroids = centroids_emb[i]
                        current_centroids = current_centroids[
                            torch.abs(current_centroids).sum(1) > 1e-7
                        ]
                        embeddings_concat = torch.cat((query_feat, current_centroids))
                        labels_concat = torch.cat((current_labels, current_labels))

                        contrastive_loss, dist_ap, dist_an = model.contrastive_loss(
                            embeddings_concat, labels_concat
                        )

                        with torch.no_grad():
                            dist_ap = dist_ap.data.mean()
                            dist_an = dist_an.data.mean()
                        ap_total.append(dist_ap)
                        an_total.append(dist_an)

                        contrastive_loss_total.append(contrastive_loss)

                        # L2 norm of centroid vectors
                        l2_mean_norm = torch.norm(current_centroids, dim=1).mean()
                        l2_mean_norm_total.append(l2_mean_norm)

                    contrastive_loss_step = (
                        torch.mean(torch.stack(contrastive_loss_total))
                        * model.hparams.SOLVER.CENTROID_CONTRASTIVE_WEIGHT
                    )
                    dist_ap = torch.mean(torch.stack(ap_total))
                    dist_an = torch.mean(torch.stack(an_total))
                    l2_mean_norm_total = torch.mean(torch.stack(l2_mean_norm_total))
                else:
                    contrastive_loss_step = torch.tensor(0)
                
                total_loss = octuplet_loss

                #backprop
                total_loss.backward()
                opt.step()
                
                # for param in model.center_loss.parameters():
                #     param.grad.data *= 1.0 / model.hparams.SOLVER.CENTER_LOSS_WEIGHT

                # opt_center.step()           

                losses = [xent_query, contrastive_loss_query, center_loss, contrastive_loss_step]
                losses = [item.detach() for item in losses]
                losses = list(map(float, losses))

                for name, loss_val in zip(model.losses_names, losses):
                    model.losses_dict[name].append(loss_val) 

                if model.hparams.MODEL.USE_CENTROIDS:
                    log_data = {
                        "step_dist_ap": float(dist_ap.mean()),
                        "step_dist_an": float(dist_an.mean()),
                        "l2_mean_centroid": float(l2_mean_norm_total),
                    }
                else:
                    log_data = {
                        "step_dist_ap": float(dist_ap.mean()),
                        "step_dist_an": float(dist_an.mean()),
                    }

                output = {"loss" : total_loss, "other": log_data} 
            
                if idx % 100 == 0: 
                    #print(output)              

                    n_iter = epoch * len(train_dataloader) + idx            
                    #logging
                    writer.add_scalar('Loss/total', total_loss, n_iter)
                    writer.add_scalar('Loss/xent_query', xent_query, n_iter)
                    writer.add_scalar('Loss/contrastive_loss_query', contrastive_loss_query, n_iter)
                    writer.add_scalar('Loss/center_loss', center_loss, n_iter)
                    writer.add_scalar('Loss/contrastive_loss_step', contrastive_loss_step, n_iter)

                outputs.append(output)

        #epoch end
        lr = opt.param_groups[0]["lr"]
        loss = torch.stack([x.pop("loss") for x in outputs]).mean().cpu().detach()
        epoch_dist_ap = np.mean([x["other"].pop("step_dist_ap") for x in outputs])
        epoch_dist_an = np.mean([x["other"].pop("step_dist_an") for x in outputs])

        del outputs

        log_data = {
            "epoch_train_loss": float(loss),
            "epoch_dist_ap": epoch_dist_ap,
            "epoch_dist_an": epoch_dist_an,
            "lr": lr,
        }

        if hasattr(model, "losses_dict"):
            for name, loss_val in model.losses_dict.items():
                val_tmp = np.mean(loss_val)
                log_data.update({name: val_tmp})
                model.losses_dict[name] = []  ## Zeroing values after a completed epoch
        
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            print(log_data)
    
            savepath = cfg.OUTPUT_DIR
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'opt_center_state_dict': opt_center.state_dict(),
                'num_query': dm.num_query,
                'num_classes': dm.num_classes,
                'scale' : scale
                }, os.path.join(savepath, f"{epoch}_x{scale}.pth"))  
            
            load_path = os.path.join(savepath, f"{epoch}_x{scale}.pth")
            print(f"Test on x{scale}")
            if model.hparams.MODEL.USE_CENTROIDS:
                cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
                print("Test baseline")
                run_test(cfg, method, dm, load_path, scale)
                print("Test CTL")
                cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
                run_test(cfg, method, dm, load_path, scale)
            else:
                print("Test baseline")
                run_test(cfg, method, dm, load_path, scale)
    



    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'opt_center_state_dict': opt_center.state_dict(),
        'num_query': dm.num_query,
        'num_classes': dm.num_classes,
        'scale' : scale
        }, os.path.join(savepath, f"{epoch}_x{scale}.pth"))  

    print(f"Final Evaluation on x{scale} and Save")
    load_path = os.path.join(savepath, f"{epoch}_x{scale}.pth")

    if model.hparams.MODEL.USE_CENTROIDS:
        cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
        print("Test baseline")
        run_test(cfg, method, dm, load_path, scale)
        print("Test CTL")
        cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
        run_test(cfg, method, dm, load_path, scale)
    else:
        print("Test baseline")
        run_test(cfg, method, dm, load_path, scale)

def run_single(cfg, method, scale):
    print(f"GPU numbers: {torch.cuda.device_count()}")
    
    print("RUN SINGLE")
    dm = init_dataset(
        cfg.DATASETS.NAMES, cfg=cfg, num_workers=cfg.DATALOADER.NUM_WORKERS
    ) #dataset
    dm.setup()
    if cfg.TEST.ONLY_TEST:
        load_path = cfg.MODEL.PRETRAIN_PATH
        print("Test baseline")
        run_test(cfg, method, dm, load_path, scale)
        cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
        print("Test CTL")
        run_test(cfg, method, dm, load_path, scale)
        cfg.MODEL.USE_CENTROIDS = not cfg.MODEL.USE_CENTROIDS
    elif cfg.FINETUNE_OCT:
        print("Finetune octuplet loss")
        writer = SummaryWriter()
        print(cfg)
        finetune_oct(cfg, method, writer, dm, scale)
    else:
        print("Train")
        writer = SummaryWriter()
        run_train(cfg, method, writer, dm, scale)
        
def run_main(cfg, method, scale = 1):
    if cfg.MODEL.USE_CENTROIDS == True:
        cfg.DATALOADER.USE_RESAMPLING = False
    run_single(cfg, method, scale)