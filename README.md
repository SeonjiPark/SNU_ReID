# SNU_ReID
## Person_Reidentification train, test용 코드

# Environments
Pytorch >= 1.7.0

Python >= 3.7.0

```
git clone -b ReID --single-branch https://github.com/SeonjiPark/SNU_ReID.git
cd SNU_ReID
conda create -n ENV_NAME python=3.7
conda activate ENV_NAME
pip install -r requirements.txt
```



# Directory 설명

```
|── datasets : dataset을 저장하는 폴더
|──SNU_ReID
    |── config : 실험에 필요한 모든 configuration 설정 파일
    |── configs : 모델 backbone에 해당하는 configuration을 저장해놓은 .yml 파일들
    |── datasets : dataset별 불러온느데 필요한 loader파일들
    |── inference : inference용 코드
    |── logs : 학습한 모델들의 웨이트가 저장되는 경로
    |── losses : 모델 학습에 필요한 loss 코드
    |── modelling : 모델 backbone을 구성하는 코드
    |── models : pre-trained된 모델 backbone 저장 위치
    |── output-dir : inference결과 저장 위치
    |── runs : tensorboard log 저장 위치
    |── utils : metric, evaluation, plot 등 실행과 계산에 필요한 util 모음
        |──train.py : train용 코드
        |──train_ctl.py : CTL를 활용한 train용 코드
        |──test.py : test용 코드
        |──finetune_octuplet.py : octuplet loss를 활용한 finetuning용 코드
```



# dataset 구성

#### dataset 폴더 구성 주의 사항: dataset 폴더는 python 코드보다 상위에 존재함

```
|── datasets : dataset을 저장하는 폴더
    |── {dataset 이름}
        |── bounding_box_test
        |── bounding_box_train
        |── gt_bbox
        |── gt_query
        |── query
    |── {dataset 이름}
    |── {dataset 이름}
    makelr.py


|──SNU_ReID
   .
   .
   .

```


# 코드 실행 설명
## === Train ===
### Baseline
```
python train.py --config_file "configs/256_resnet50.yml" --scale 4 DATASETS.NAMES 'market1501'
```

--config_file : backbone으로 사용할 네트워크 configuration설정, ./configs/ 폴더에서 적합한 .yml파일 골라서 사용

--scale : 학습할 downsampling 해상도 

DATASETS.NAMES : Train 진행할 데이터셋 이름 (PRW, market1501)

Train했을시에, ./logs폴더에 {DATASET.NAMES}/baseline/{scale}/exp0/안에 {epoch}.pth 이름으로 웨이트가 저장됨

### CTL
```
python train_ctl.py --config_file "configs/256_resnet50.yml" --scale 4 DATASETS.NAMES 'market1501'
```




## === Resume Training ===
### Baseline
```
python train.py --config_file "configs/256_resnet50.yml" DATASETS.NAMES 'market1501' MODEL.RESUME_TRAINING True MODEL.PRETRAIN_PATH "./logs/market1501/resnet50/baseline/exp0/x1/20.pth" 
```

--config_file : backbone으로 사용할 네트워크 configuration설정, ./configs/ 폴더에서 적합한 .yml파일 골라서 사용

DATASETS.NAMES : Resume할 데이터셋 이름 (기존 train된 데이터셋과 일치)

MODEL.RESUME_TRAINING : True로 설정해서 학습 이어나가기

MODEL.PRETRAIN_PATH : 이어서 학습할 모델의 웨이트 파일 경로

### CTL
```
python train_ctl.py --config_fil "configs/256_resnet50.yml" DATASETS.NAMES 'market1501' MODEL.RESUME_TRAINING True  MODEL.PRETRAIN_PATH "./logs/market1501/resnet50/ctl/exp0/x1/5.pth"
```

## === Finetune with Octuplet Loss ===

```
python finetune_octuplet.py --config_file "configs/256_resnet50.yml" --scale 4 DATASETS.NAMES 'market1501' MODEL.PRETRAIN_PATH "./logs/market1501/resnet50/base/exp0/x1/119_last.pth" 
```

--config_file : backbone으로 사용할 네트워크 configuration설정, ./configs/ 폴더에서 적합한 .yml파일 골라서 사용

--scale : Octuplet loss로 finetuning할 downsampling 해상도. 

DATASETS.NAMES : Finetune 진행할 데이터셋 이름 (PRW, market1501)

MODEL.PRETRAIN_PATH : 이어서 finetuning할 모델의 웨이트 파일 경로

Finetuning했을시에, 불러온 MODEL.PRETRAIN_PATH의 폴더안에 {epoch}x4.pth 이름으로 웨이트가 저장됨

### CTL
```
python finetune_octuplet_ctl.py --config_file "configs/256_resnet50.yml" --scale 4 DATASETS.NAMES 'market1501' MODEL.PRETRAIN_PATH "./logs/market1501/resnet50/ctl/exp0/x1/119_last.pth" 
```




## === Test ===
```
python test.py --config_file "configs/256_resnet50.yml" --scale 4 DATASETS.NAMES 'market1501' MODEL.PRETRAIN_PATH "./logs/market1501/resnet50/ctl/exp0/5.pth"
```

--config_file : backbone으로 사용할 네트워크 configuration설정, ./configs/ 폴더에서 적합한 .yml파일 골라서 사용 (학습한 모델과 동일 config 이용)

--scale : Test할 downsampling 해상도. 

DATASETS.NAMES : Test를 진행할 데이터셋 이름 (PRW, market1501)

MODEL.PRETRAIN_PATH : Test할 모델의 웨이트 파일 경로



위 코드 실행시 데이터셋의 query + gallery 이미지들에 대한 mAP 결과 및 top-k결과 출력


## === Inference ===

### Create Embeddings
```
python inference/create_embeddings.py --config_file "configs/256_resnet50.yml" DATASETS.ROOT_DIR '../datasets/market1501/bounding_box_test' MODEL.PRETRAIN_PATH "./logs/market1501/resnet50/baseline/exp0/25.pth"
```

--config_file: backbone으로 사용할 네트워크 configuration설정, ./configs/ 폴더에서 적합한 .yml파일 골라서 사용 (MODEL.PRETRAIN_PATH에서 불러와 사용하는 모델과 동일 config 이용)

DATASETS.ROOT_DIR: inference할 gallery 데이터셋 폴더 경로 지정 (데이터셋의 bounding_box_test의 경로)

MODEL.PRETRAIN_PATH : Inference를 진행할 모델의 웨이트 파일 경로



위 코드 실행시 ./output_dir 폴더가 생성되며 안에 embeddings.py, paths.npy 파일들이 생성됨

### Similarity Search

```
python inference/get_similar.py --config_file "configs/256_resnet50.yml" --gallery_data 'output_dir' DATASETS.ROOT_DIR '../datasets/market1501/query' MODEL.PRETRAIN_PATH "./logs/market1501/resnet50/baseline/exp0/25.pth"
```

--config_file: backbone으로 사용할 네트워크 configuration설정, ./configs/ 폴더에서 적합한 .yml파일 골라서 사용 (MODEL.PRETRAIN_PATH에서 불러와 사용하는 모델과 동일 config 이용)

DATASETS.ROOT_DIR: inference할 query 데이터셋 폴더 이름 지정 (데이터셋의 query 경로)

MODEL.PRETRAIN_PATH : Inference를 진행할 모델의 웨이트 파일 경로

### Inference : Create Embeddings + Similarity Search

Create Embeddings 와 Similarity Search를 진행.

```
python inference/infer.py --query_dir '../datasets/market1501/query' DATASETS.ROOT_DIR '../datasets/market1501/bounding_box_test' MODEL.PRETRAIN_PATH "./logs/market1501/resnet50/baseline/exp0/25.pth"
```

query_dir: inference할 query이미지가 담긴 데이터셋 폴더 이름 지정

DATASETS.ROOT_DIR: inference할 gallery 데이터셋 폴더 경로 지정

MODEL.PRETRAIN_PATH : Inference를 진행할 모델의 웨이트 파일 경로


위 코드 실행 시, query_dir안에 있는 모든 query이미지에 대해 cosine distance가 가장 가까운 gallery 데이터셋의 이미지를 계산해 query에대한 predicted class를 계산한다.
출력으로 output_dir 폴더에 result.csv가 저장되고 형태는 아래와 같다.
    query filename,       predicted class,  predicted path,           distance
예) 843_c1s4_054211.jpg,  843,              0843_c1s4_054211_00.jpg,  0.011188507

이에 더불어, query에대한 예측 결과의 accuracy를 출력한다.

## citation 

Wieczorek M., Rychalska B., Dąbrowski J. 
(2021) On the Unreasonable Effectiveness of Centroids in Image Retrieval.
In: Mantoro T., Lee M., Ayu M.A., Wong K.W., Hidayanto A.N. (eds) 
Neural Information Processing. ICONIP 2021.
Lecture Notes in Computer Science, vol 13111. Springer, Cham. https://doi.org/10.1007/978-3-030-92273-3_18

https://github.com/mikwieczorek/centroids-reid