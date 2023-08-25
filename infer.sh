python inference/create_embeddings.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.ROOT_DIR '../DATASET/PRW_yolo/test/images' \
TEST.IMS_PER_BATCH 128 \
OUTPUT_DIR 'output' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "./checkpoints/market1501_resnet50_256_128_epoch_120.ckpt"
