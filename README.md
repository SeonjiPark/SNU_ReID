# SNU_ReID
## Detection & ReID Pipeline를 위한 inference 및 test 코드


# Environments
Pytorch >= 1.13.0

Python >= 3.7.0

```
git clone -b main_jae --single-branch https://github.com/SeonjiPark/SNU_ReID.git
cd SNU_ReID
conda create --name ENV_NAME --file requirements.txt
conda activate ENV_NAME
```


# Directory 설명

```
|──DATASET : market1501, PRW, MOT17 등 dataset을 저장하는 폴더
|──SNU_ReID
    |── results : inference결과가 저장되는 경로
    |── SNU_PersonDetection : Detection 모델 및 코드 경로
    |── SNU_PersonReID : ReID에 필요한 모델 및 코드 경로
    |── weights : inference에 필요한 Detection + ReID 모델들의 weight
    |── reid.cfg : 사용하는 모델들의 파라미터 (변경 x)
    |── config.py : infer, test시에 필요한 경로 및 파라미터 설정 
    |── infer.py : inference용 코드
    |── test.py : test용 코드

```

## DATASET 구성

#### dataset 폴더 구성 주의 사항: dataset 폴더는 python 코드보다 상위에 존재함

```
|── DATASET : dataset을 저장하는 폴더
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


# Pretrained Weights

Inference 및 Test에 필요한 웨이트는 detection weight와 reid weight가 있다.

[여기](https://drive.google.com/drive/folders/1Tc0NUviqcDMYbIYvT-fQE6dp92NnvSO8?usp=drive_link)에서 필요한 웨이트를 다운받아 weights폴더에 위치하면된다.

이후 config.py에서 

--detection_weight_file
--reid_weight_file

argument에 대한 경로값들을 수정해 원하는 웨이트를 사용한다.


# Configuration

Inference 및 Test에 사용할 경로 및 파라미터 설정은 config.py에서 한다.

### Directory parameter

--infer_data_dir : Detection + ReID pipeline을 돌릴 이미지들의 경로
--dataset_root_dir : 데이터셋 경로 (default : '../DATASET/')
--dataset_name : ReID에서 갤러리로 사용하고자 하는 데이터셋 이름 (예: market1501, PRW, MOT17)

--detection_weight_file : 사용할 detection weight 경로
--reid_weight_file : 사용할 ReID weight 경로





