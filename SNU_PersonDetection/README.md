# SNU_ReID
## Person_Detection train, test용 코드

# Environments
Pytorch >= 1.7.0

Python >= 3.7.0

```
git clone -b Detection --single-branch https://github.com/SeonjiPark/SNU_ReID.git
cd SNU_ReID
conda create -n ENV_NAME python=3.7
conda activate ENV_NAME
pip install -r requirements.txt
```

# Directory 설명

```
|── datasets : dataset을 저장하는 폴더
|──SNU_ReID
    |── data : dataset별 실행에 필요한 argument를 지정해주는 yaml
    |── models : yolov5s 모델을 구성하는 layer와 argument
    |── utils : loss, plot 등 실행과 계산에 필요한 util 모음
    |── weights : yolov5s 모델로 학습시킨 AD dataset의 pretrained weight
        |──detect.py: inference용 코드 (label 존재하지 않을 때)
        |──train.py : train용 코드
        |──val.py : test용 코드 (label 존재할 때)
        |──requirements.txt : 환경 설정 파일
```



# dataset 구성

#### dataset 폴더 구성 주의 사항: dataset 폴더는 python 코드보다 상위에 존재함
label 구성 : [{lcass_num} center_x, center_y, width, height], 모든 좌료값은 x, y 각각 0~1 normalized

Ex. 0 0.51 0.62 0.056 0.024


```
|── datasets : dataset을 저장하는 폴더
    |── {dataset 이름}
        |── train
            |── images
            |── labels
        |── valid
            |── images
            |── labels
  
|──SNU_ReID
   .
   .
   .

```

# 코드 실행 설명
## === Train ===
```
python train.py --data PRW.yaml --weights '' --cfg yolov5s.yaml --device {gpu index}
```
--data : ./data/PRW.yaml을 참고하여 학습시키고자 하는 dataset의 경로와 class의 갯수를 입력하여 사용

--weights : pretrained 파일이 없는 경우 ''로 argument를 주면 scratch부터 학습


## === Test ===
```
python val.py --weights ./weights/detection.pt --data PRW.yaml --device {gpu index}
```
--weights: 학습된 weight의 경로

--data: 학습시에 사용한 dataset의 yaml


위 코드 실행시 runs 폴더가 생성되며 validation 결과를 저장함


## === Inference ===
```
python detect.py --weights ./weights/detection.pt --source {inference 대상인 이미지 or 동영상 or 폴더의 경로} --data PRW.yaml --device {gpu index}
```
--weights: 학습된 weight의 경로

--data: 학습시에 사용한 dataset의 yaml

--source: inference를 진행할 이미지 / 동영상 / 폴더의 경로 Ex. --source ./test_image.png

위 코드 실행시 runs 폴더가 생성되며 inference 결과를 저장함



## citation 

YOLOv5 🚀 by Ultralytics, GPL-3.0 license

https://github.com/ultralytics/yolov5
