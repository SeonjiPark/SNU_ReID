python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501x4' \
DATASETS.ROOT_DIR '/data/jaep0805/datasets/PersonReID' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/256_resnet50' \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/data/jaep0805/PersonReID/centroids-reid/logs/market1501/256_resnet50/finetune_octuplet/version_18/checkpoints/epoch=129.ckpt"