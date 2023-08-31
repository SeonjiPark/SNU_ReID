python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'PRWx4' \
DATASETS.ROOT_DIR '../DATASET/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/PRW/256_resnet50' \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "./logs/PRW/256_resnet50/finetune_octuplet/version_2/auto_checkpoints/checkpoint_129.pth"