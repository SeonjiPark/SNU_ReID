import argparse
import configparser
import os, cv2
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    sys.path.append(os.path.join(str(ROOT), "SNU_PersonDetection")) 
    sys.path.append(os.path.join(str(ROOT), "SNU_PersonReID")) 

from SNU_PersonDetection.execute import *


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
        parser = argparse.ArgumentParser()

        self.args = parser.parse_args()

        config = configparser.RawConfigParser()
        config.read(cfg_dir)

        person_config = config["person_config"]
        
        
        # add config values to self.args
        for k, v in person_config.items():
            setattr(self.args, k, v)
        
        
        ################################################
        #        1.2 for Detection                     #
        ################################################
        self.args.stride = 32
        self.args.num_workers = 4
        self.args.max_det = 1000
        
        
        self.args.conf_thres = float(self.args.conf_thres)
        self.args.iou_thres = float(self.args.iou_thres)
        self.args.detect_imgsz = [int(self.args.detect_imgsz), int(self.args.detect_imgsz)]
        
        
        
        ################################################
        #        1.3 for ReID                          #
        ################################################
        
        

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
        # self.reid_network = build_reid_model(self.args, self.device)
        # self.reid_network.to(self.device)

        ################################################
        #    4. Load Detection / Recognition Network   #
        ################################################
        #JH todo
        # recognition_checkpoint = torch.load(self.args.reid_weight_file)
        # self.recognition_network.load_state_dict(recognition_checkpoint['network'])
        with torch.no_grad():
            self.detection_network.eval()
            # self.recognition_network.eval()

        ################################################
        #      5. Make Result Saving Directories       #
        ################################################
        os.makedirs(self.args.output_dir, exist_ok=True)
        return

    def read_data(self):
        ### Load Datas for eval
        # SJ to do
        # Collate = AlignCollate(IMGH, IMGW, PAD)
        
        test_dataset = LoadImagesandLabels(self.args.data_dir, self.args.stride, self.args.detect_imgsz)
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
        
        for data in dataloader:
            
            path, original_img, img_preprocess, labels = data
            img_preprocess = img_preprocess.to(self.device)
            
            # detect_preds = [image1, image2, ...] of not normalized, HWC 
            # GT_ids = [id1, id2, ...] of int
            # if not detected, both are []
            detect_preds, GT_ids = do_detect(self.args, self.detection_network, img_preprocess, original_img)

            # JH todo 
            
            ### if GT_ids == None: just run and save extracted features (infer)
            ### else (GT_ids !=None): run and eval ReID score(test)
            ### if test and infer architecture differs a lot, just define two functions seperately. 
            
            # reid_result = do_recognition()
            # reid_result = post_process(reid_result)
            reid_result = None
            
            
            # SJ Todo
            # save_result
            
        # JH Todo
        # Print(ctl eval result)

        return reid_result
        

    """
    
    def save_result_snu(args, images, recog_preds):
        INPUT_IMAGE_FILE_ONLYNM = os.path.basename(str(args.source))
        file_ext = os.path.splitext(INPUT_IMAGE_FILE_ONLYNM)[-1]
        file_nm = os.path.splitext(INPUT_IMAGE_FILE_ONLYNM)[0]
        OUTPUT_IMAGE_FILE = os.path.join(args.output_dir, file_nm + "_recognized" + file_ext)

        # In case input is folder (not image or video)
        if file_ext == "":
            file_nm = file_nm.split("/")[-1]

        OUTPUT_BASE_DIR = os.path.join(args.output_dir, file_nm)
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        args.output_base_dir = OUTPUT_BASE_DIR

        if args.result_savefile:
            RECOG_SAVE_DIR = os.path.join(args.output_dir, file_nm, "recognition")
            DETECT_SAVE_DIR = os.path.join(args.output_dir, file_nm, "detection")
            LABEL_SAVE_DIR = os.path.join(args.output_dir, file_nm, "labels")

            os.makedirs(RECOG_SAVE_DIR, exist_ok=True)
            os.makedirs(DETECT_SAVE_DIR, exist_ok=True)
            os.makedirs(LABEL_SAVE_DIR, exist_ok=True)
        else:
            RECOG_SAVE_DIR = None
            DETECT_SAVE_DIR = None
            LABEL_SAVE_DIR = None

        args.recog_save_dir = RECOG_SAVE_DIR
        args.detect_save_dir = DETECT_SAVE_DIR
        args.label_save_dir = LABEL_SAVE_DIR

        if file_ext in VIDEO_FORMATS:
            args.isvideo = True
        else:
            args.isvideo = False

        print("{0}\n{1}\n{2}\n".format(INPUT_IMAGE_FILE_ONLYNM, file_ext, file_nm))
        args.input_onlynm = INPUT_IMAGE_FILE_ONLYNM
        args.input_ext = file_ext
        args.input_filenm = file_nm

        if args.result_savefile:
            for idx, img in enumerate(images):
                ## save detect result
                img = img.permute(1,2,0) * 255
                img = img[:, :, [2, 1, 0]]
                img = img.cpu().numpy().astype(np.uint8)

                if args.save_detect_result:
                    annotator_detect = Annotator(img, line_width=3, pil=True, example=str(args.names))
                if args.save_recog_result:
                    annotator_recog = Annotator(img, line_width=3, font='gulim.ttc', font_size=40, pil=True, example=str(args.names))
                gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                savefile = args.input_filenm + "_detection_" + str(idx + 1).zfill(10)

                for *xyxy, conf, cls, lp_characters in reversed(recog_preds[idx]):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format

                    # Save label
                    TXT_PATH = os.path.join(LABEL_SAVE_DIR, savefile + ".txt")
                    with open(TXT_PATH, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    c = int(cls)  # integer class

                    # Save Detection Result
                    if args.save_detect_result:
                        if args.hide_labels and args.hide_conf:
                            label = None
                        elif args.hide_labels and not args.hide_conf:
                            label = f'{conf:.2f}'
                        elif not args.hide_labels and args.hide_conf:
                            label = args.names[c]
                        else:
                            label = f'{args.names[c]} {conf:.2f}'

                        annotator_detect.box_label(xyxy, label, color=colors(c, True))

                    # Save Recognition Result
                    if args.save_recog_result:
                        label = f'{lp_characters}'
                        annotator_recog.box_label(xyxy, label, color=colors(c,True))

                # Stream results
                if args.save_detect_result:
                    im_detect = annotator_detect.result()
                    outname = os.path.join(DETECT_SAVE_DIR, savefile + '.jpg')
                    cv2.imwrite(outname, im_detect)

                if args.save_recog_result:
                    im_recog = annotator_recog.result()
                    outname = os.path.join(RECOG_SAVE_DIR, savefile + '.jpg')
                    cv2.imwrite(outname, im_recog)

    """


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
    