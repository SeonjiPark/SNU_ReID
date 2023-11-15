import argparse
import configparser
import os, cv2
import sys
from pathlib import Path
#hi

import torch
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler
import ast
from config import parse_args

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    sys.path.append(os.path.join(str(ROOT), "SNU_PersonDetection")) 
    sys.path.append(os.path.join(str(ROOT), "SNU_PersonReID")) 

from SNU_PersonDetection.execute import *
from SNU_PersonReID.execute import *
from SNU_PersonReID.execute import _process_dir

import re

class ReIDPerson:

    def __init__(self):
        super().__init__()

        self.device = None
        self.args = None
        self.detection_network = None
        self.reid_network = None


    def initialize(self, cfg_dir):
        ################################################
        #   1. Read in parameters from config file     #
        ################################################

        self.args = parse_args()

        config = configparser.RawConfigParser()
        config.read(cfg_dir)

        detect_config = config["detect_config"]
        # add config values to self.args
        for k, v in detect_config.items():
            setattr(self.args, k, v)


        reid_config = config["reid_config"]
        # add config values to self.args
        for k, v in reid_config.items():
            setattr(self.args, k, v)
        
        ################################################
        #        1.2 for Detection                     #
        ################################################
        self.args.stride = int(self.args.stride)
        self.args.num_workers = int(self.args.num_workers)
        self.args.max_det = int(self.args.max_det)
        self.args.batch_size = int(self.args.batch_size)
        
        self.args.detect_imgsz = [int(self.args.imgsz), int(self.args.imgsz)]
        
        
        ################################################
        #        1.3 for ReID                          #
        ################################################
        
        self.args.gallery_path = os.path.join(self.args.dataset_root_dir, f'{self.args.dataset_name}_reid/bounding_box_test')

        # class override
        self.reid_checkpoint = torch.load(self.args.reid_weight_file)
        
        #JH 정리
        self.args.reid_batch_size = 128 #?
        self.args.use_unknown = True
        self.args.reid_threshold = 0.8
        self.args.topk = 1
        self.args.num_classes = self.reid_checkpoint['num_classes']  #697 for PRW or 751 for market1501 
        self.args.input_size_test = ast.literal_eval(self.args.input_size_test) # ast.literal_eval : to turn string of list to list
        self.args.input_pixel_mean = ast.literal_eval(self.args.input_pixel_mean)
        self.args.input_pixel_std = ast.literal_eval(self.args.input_pixel_std)
        self.gallery, self.gallery_dict = _process_dir(self.args.gallery_path, relabel=False, dataset_name = self.args.dataset_name)
        # self.query, self.query_dict = _process_dir(self.args.query_path, relabel=False) #len(query) = 3368
        # self.args.num_query = len(self.query)
        return (self.args)
    
    def init_device(self):
        ################################################
        #        1.1 Set GPU Device                    #
        ################################################
        
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu_num)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        return


    def init_network(self):
        ################################################
        #  3. Declare Detection / Recognition Network  #
        ################################################
        # Detection Network
        # SJ todo
        self.detection_network = build_detect_model(self.args, self.device)

        # Recognition Network
        # JH todo
        self.reid_network = build_reid_model(self.args, self.device)
        self.reid_network.to(self.device)

        ################################################
        #    4. Load Detection / Recognition Network   #
        ################################################
        #JH todo
        #reid_checkpoint = torch.load(self.args.reid_weight_file)
        self.reid_network.load_state_dict(self.reid_checkpoint['model_state_dict'], strict = False)

        with torch.no_grad():
            self.detection_network.eval()
            self.reid_network.eval()

        ################################################
        #      5. Make Result Saving Directories       #
        ################################################
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, "pred_txt"), exist_ok=True)
        if self.args.save_images: os.makedirs(os.path.join(self.args.output_dir, self.args.reid_save_dir), exist_ok=True)          
        if self.args.save_detection_images: os.makedirs(os.path.join(self.args.output_dir, self.args.detect_save_dir), exist_ok=True)
        return

    def read_data(self):
        ### Load Datas for eval
        # SJ to do
        # Collate = AlignCollate(IMGH, IMGW, PAD)
        
        if self.args.use_GT_IDs:
            test_dataset = LoadImagesandLabels(self.args.infer_data_dir, self.args.stride, self.args.detect_imgsz)
        else:
            test_dataset = LoadImages(self.args.infer_data_dir, self.args.stride, self.args.detect_imgsz)
            
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=int(self.args.batch_size),
            num_workers=int(self.args.num_workers),
            shuffle=False,
            drop_last=False,
            collate_fn=None
            )
        
        
        return test_dataloader

    def infer(self, dataloader):
        
        total_pred_class = []

        embeddings_gallery, paths_gallery = load_gallery(self.args, self.reid_network) #gallery loading

        for idx, data in enumerate(dataloader):
            path, original_img, img_preprocess, labels = data

            img_preprocess = img_preprocess.to(self.device)
            
            # detect_preds = [image1, image2, ...] of not normalized, HWC 
            # if not detected,  detect_preds = []
            
            # if dataset is not with GT or args.use_GT_IDs = False, GT_ids = None
            detect_preds, det, GT_ids = do_detect(self.args, self.detection_network, img_preprocess, original_img, labels)

            detect_preds_preprocessed = preprocess_reid(self.args, detect_preds)
            
            if len(detect_preds) != 0:
                pred_class, embedding = do_reid(self.args, self.reid_network, embeddings_gallery, paths_gallery, detect_preds_preprocessed)
            else:
                pred_class = []
            
            total_pred_class.append(pred_class)
            print("Predicted class:", pred_class)
            
            # SJ Todo
            if self.args.save_images:
                save_result(self.args, path[0], original_img[0], det, pred_class, self.detection_network.names, GT_ids)
            if self.args.save_detection_images:
                save_result(self.args, path[0], original_img[0], det, pred_class, self.detection_network.names, GT_ids)
            save_txt(self.args, path[0], det, pred_class)

        return total_pred_class

if __name__ == "__main__":
        
    reidperson = ReIDPerson()
    print("Start Init")
    reidperson.initialize('reid.cfg')
    reidperson.init_device()
    reidperson.init_network()
    
    print("Start Load Network")
    dataloader = reidperson.read_data()
    
    print("Start Re-Identification")
    recog_result = reidperson.infer(dataloader)
    
    print(recog_result)
    