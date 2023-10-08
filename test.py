import argparse
import configparser
import os, cv2
import sys
import ast
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler

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
        
        #JH 정리
        self.args.reid_batch_size = 128 #?
        self.args.use_unknown = True
        self.args.reid_threshold = 0.8
        self.args.topk = 1
        self.args.num_classes = 697  #697 for PRW or 751 for market1501
        self.args.input_size_test = ast.literal_eval(self.args.input_size_test) # ast.literal_eval : to turn string of list to list
        self.args.input_pixel_mean = ast.literal_eval(self.args.input_pixel_mean)
        self.args.input_pixel_std = ast.literal_eval(self.args.input_pixel_std)
        self.gallery, self.gallery_dict = _process_dir(self.args.gallery_path, relabel=False, dataset_name = self.args.dataset_name)
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
        reid_checkpoint = torch.load(self.args.reid_weight_file)
        self.reid_network.load_state_dict(reid_checkpoint['model_state_dict'], strict = False)

        with torch.no_grad():
            self.detection_network.eval()
            self.reid_network.eval()

        ################################################
        #      5. Make Result Saving Directories       #
        ################################################
        os.makedirs(self.args.output_dir, exist_ok=True)
        return

    def read_data(self):
        ### Load Datas for eval
        # SJ to do
        # Collate = AlignCollate(IMGH, IMGW, PAD)
        
        test_dataset = LoadImagesandLabels(self.args.infer_data_dir, self.args.stride, self.args.detect_imgsz)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=int(self.args.batch_size),
            num_workers=int(self.args.num_workers),
            shuffle=False,
            drop_last=False,
            collate_fn=None
            )
        
        
        return test_dataloader

    def test(self, dataloader):
        
        total_embedding = []
        total_gt = []
        total_pred_class = []
        outputs = []

        for idx, data in enumerate(dataloader):
            path, original_img, img_preprocess, labels = data

            img_preprocess = img_preprocess.to(self.device)
            labels = labels.to(self.device)
            
            # detect_preds = [image1, image2, ...] of not normalized, HWC 
            # GT_ids = [id1, id2, ...] of int
            # if not detected, both are []
            detect_preds, det, GT_ids = do_detect(self.args, self.detection_network, img_preprocess, original_img, labels[0])
            
            detect_preds_preprocessed = preprocess_reid(self.args, detect_preds)

            if len(detect_preds) != 0:
                pred_class, embedding = do_reid(self.args, self.reid_network, detect_preds_preprocessed, GT_ids)
            else:
                pred_class = []

            gt_list = GT_ids.tolist()
            gt_list_int = [int(x) for x in gt_list]
            
            total_pred_class.append(pred_class)
            print("Predicted class:", pred_class)
            print("GT class:", gt_list_int)
            # SJ Todo
            # save_result
            
            """

            #eval
            embs = []

            if len(detect_preds) != 0:
                detect_preds_resized_batched = detect_preds_resized[0]
                for i in range(1, len(detect_preds_resized)):
                    detect_preds_resized_batched = torch.cat((detect_preds_resized_batched, detect_preds_resized[i]))
                

                
                output = eval1(self.args, self.reid_network, detect_preds_resized_batched.cuda(), GT_ids)
                outputs.append(output)
                    
            if idx > 2:
                break        

        gallery_loader = load_gallery(self.args)
        
        #add gallery embeddings to outputs
        for i, batch in enumerate(gallery_loader):
            x, class_labels, camid, idx = batch
            output = eval1(self.args, self.reid_network, x.cuda(), class_labels.cuda())
            outputs.append(output)
        
        embeddings = torch.cat([x.pop("emb") for x in outputs]).detach().cpu()
        labels = (
            torch.cat([x.pop("labels") for x in outputs]).detach().cpu().numpy()
        )
        del outputs

        embeddings, labels, camids = self.reid_network.validation_create_centroids(
            embeddings,
            labels,
        )

        self.reid_network.get_val_metrics(embeddings, labels, dataloader)
        """
        

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
    recog_result = reidperson.test(dataloader)
    
    print(recog_result)
    