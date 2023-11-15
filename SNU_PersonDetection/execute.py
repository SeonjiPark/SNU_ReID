import cv2, os, glob
import torch
import numpy as np
from pathlib import Path

from SNU_PersonDetection.utils.general import non_max_suppression, scale_coords, check_img_size, xywhn2xyxy
from SNU_PersonDetection.utils.augmentations import letterbox
from SNU_PersonDetection.utils.metrics import box_iou
from SNU_PersonDetection.utils.plots import Annotator, colors

from SNU_PersonDetection.models.common import DetectMultiBackend

def build_detect_model(args, device):
    detection_network = DetectMultiBackend(args.detection_weight_file, device=device, dnn=False, data=args.yolo_data)
    imgsz = check_img_size(args.detect_imgsz, s=args.stride)  # check image size
    detection_network.warmup(imgsz=(1, 3, *imgsz))  # warmup
    
    return detection_network


def preprocess_img(img, fp_flag, img_size, auto=True):
    img = img.permute(0, 3, 1, 2)  # HWC to CHW

    im = img.type(torch.half) if fp_flag else img.type(torch.float)  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im

def do_detect(args, detection_network, img, original_img, labels=None):
    # Do Detection Inference
    im_resize = preprocess_img(img, detection_network.fp16, args.detect_imgsz, args.stride)
    pred = detection_network(im_resize, augment=False, visualize=False)

    # Detection NMS
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, None, False, max_det=args.max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per predictions
        # Rescale boxes from img_size to img size
        det[:, :4] = scale_coords(im_resize.shape[2:], det[:, :4], original_img[0].shape).round()
    
    # xywh2xyxy
    h, w = im_resize.shape[2:]
    H, W = original_img.shape[1:3]
    gain = min(h / H, w / W)  # gain  = old / new
    pad = (w - W * gain) / 2, (h - H * gain) / 2  # wh padding
    pred_images = post_preds_images(det, original_img)
    if labels != []:
        #print(labels)
        #print(labels.shape)
        labels = labels.unsqueeze(0)
        labels= xywhn2xyxy(labels[0], w=W, h=H, padw=pad[0], padh=pad[1])
        ids = find_gt_ids(det, labels.to(detection_network.device))
        return pred_images, det, ids
    else:
        return pred_images, det, None
    
    

def save_detected_boxes(args, pred_images, GT_ids, path):
    path = path[0]
    name = path.split("/")[-1].split(".")[0]
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    im_save_folder = os.path.join(args.output_dir, args.detect_save_dir)
    if not os.path.exists(im_save_folder):
        os.mkdir(im_save_folder)
    
    
    for idx, gtid in enumerate(GT_ids):
        if int(gtid.item()) > 0:
            im_save_path = os.path.join(im_save_folder, f"{str(int(gtid.item())).zfill(4)}_{name}_{str(idx).zfill(2)}.jpg")
        else: 
            im_save_path = os.path.join(im_save_folder, f"{int(gtid.item())}_{name}_{str(idx).zfill(2)}.jpg")
        cv2.imwrite(im_save_path, pred_images[idx].numpy())
        

def save_detection_boxes(args, pred_images, GT_ids, path):
    path = path[0]
    name = path.split("/")[-1].split(".")[0]
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    im_save_folder = os.path.join(args.output_dir, args.detect_save_dir)
    if not os.path.exists(im_save_folder):
        os.mkdir(im_save_folder)
    
    
    for idx, gtid in enumerate(GT_ids):
        if int(gtid.item()) > 0:
            im_save_path = os.path.join(im_save_folder, f"{str(int(gtid.item())).zfill(4)}_{name}_{str(idx).zfill(2)}.jpg")
        else: 
            im_save_path = os.path.join(im_save_folder, f"{int(gtid.item())}_{name}_{str(idx).zfill(2)}.jpg")
        cv2.imwrite(im_save_path, pred_images[idx].numpy())
        
        
    
def save_result(args, path, original_img, det, pred_class, names, GT_ids):
    annotator = Annotator(original_img.numpy(), line_width=3, pil=True, example=str(names))
    # import ipdb; ipdb.set_trace()
    
    im_name = path.split("/")[-1]
    
    if args.use_GT_IDs:
        for box, pred, gt in zip(det, pred_class, GT_ids):
            label = f'{pred}__{int(gt)}'
            annotator.box_label(box, label, color=colors(0, True))

    else:
        for box, pred in zip(det, pred_class):
            label = f'{pred}'
            annotator.box_label(box[:4].cpu().detach().numpy(), label, color=colors(0, True))
        
    
    im = annotator.result()
    cv2.imwrite(os.path.join(os.path.join(args.output_dir, args.reid_save_dir), im_name), im)
    return None

def save_txt(args, path, det, pred_class):
    name = path.split("/")[-1]
    f = open(os.path.join(os.path.join(args.output_dir, "pred_txt"), name.split(".")[0] + ".txt"), "w")
    for box, pred in zip(det, pred_class):
        f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {pred}\n")
    
    return None

    
def post_preds_images(det, original_img):
    pred_images = []
    if det == []: return pred_images
    for d in det:
        x1 = int(d[0])
        y1 = int(d[1])
        x2 = int(d[2])
        y2 = int(d[3])
        
        pred_images.append(original_img[0, y1:y2, x1:x2, :])
    
    return pred_images

def find_gt_ids(det, labels):
    # det [N, 6], labels [N, 5] 
    # both x1,y1,x2,y2
    gt_ids = []
    if det == []:
        return gt_ids
    else:
        iou = box_iou(labels[:, :4], det[:, :4]) # [N_lab, N_pred]
        max_value, max_index = torch.max(iou, dim=0)
        gt_ids = labels[max_index.cpu(), -1]
            
    return gt_ids

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct
    
    
class LoadImagesandLabels:
    def __init__(self, path, stride, img_size):
        self.stride = stride
        self.img_size = img_size
        
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

        self.imgfiles = images
        
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels_id' + os.sep  # /images/, /labels/ substrings
        self.labfiles = sorted([sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in self.imgfiles])
        #print(self.labfiles)
        assert len(self.labfiles) == len(self.imgfiles), f'Number of images and labels are not same.'
        
        self.labels = []
        for labpath in self.labfiles:
            lab = open(labpath, "r")
            datas = lab.readlines()
            
            label = []
            for d in datas:
                if len(d.split()) == 6:
                    cls, cx, cy, w, h, id = d.split()
                else:
                    cls, cx, cy, w, h = d.split()
                    id = -1
                label.append([float(cx), float(cy), float(w), float(h), int(id)])
            label = np.asarray(label)
            
            self.labels.append(label)


    def __getitem__(self, index):
        impath = self.imgfiles[index]
        
        img = cv2.imread(impath)  # BGR
        assert img is not None, f'Image Not Found {impath}'
        img_preprocess = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]
        
        label = self.labels[index]
        
        return impath, img, img_preprocess, label

    def __len__(self):
        return len(self.imgfiles)  # number of files



class LoadImages:
    def __init__(self, path, stride, img_size):
        self.stride = stride
        self.img_size = img_size
        
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

        self.imgfiles = images
        
        sa = os.sep + 'images' + os.sep # /images/ substrings


    def __getitem__(self, index):
        impath = self.imgfiles[index]
        
        img = cv2.imread(impath)  # BGR
        assert img is not None, f'Image Not Found {impath}'
        img_preprocess = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]
        
        return impath, img, img_preprocess, []

    def __len__(self):
        return len(self.imgfiles)  # number of files