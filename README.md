# SNU_ReID


##Pretrained Weights

Inference에 필요한 웨이트는 detection weight와 reid weight가 있다.
[여기](https://drive.google.com/drive/folders/1Tc0NUviqcDMYbIYvT-fQE6dp92NnvSO8?usp=drive_link)에서 필요한 웨이트를 다운받아 weights폴더에 위치하면된다.
이후 config.py에서 
--detection_weight_file
--reid_weight_file
argument에 대한 값들을 수정하거나 입력으로 넣어주어 원하는 웨이트를 사용한다.





