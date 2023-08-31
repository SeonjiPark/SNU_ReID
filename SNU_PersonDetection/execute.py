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
    detection_network = DetectMultiBackend(args.detection_weight_file, device=device, dnn=False, data=args.data)
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

def do_detect(args, detection_network, img, original_img, labels):
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
    gain = min(im_resize.shape[0] / original_img.shape[0], im_resize.shape[1] / original_img.shape[1])  # gain  = old / new
    pad = (im_resize.shape[1] - original_img.shape[1] * gain) / 2, (im_resize.shape[0] - original_img.shape[0] * gain) / 2  # wh padding
    labels= xywhn2xyxy(labels, w=original_img[0].shape[0], h=original_img[0].shape[1], padw=pad[0], padh=pad[1])
    
    
    # SJ todo 
    # import ipdb; ipdb.set_trace()
    
    pred_images = post_preds_images(det, original_img)
    ids = find_gt_ids(det, labels)
    
    
    return pred_images, ids

def save_detection_result(args, det, names, p, mode, frame, imc, save_dir):
    p = Path(p)  # to Path
    save_path = os.path.join(save_dir, p.name)  # im.jpg
    txt_path = os.path.join(save_dir, 'labels', p.stem) + ('' if mode == 'image' else f'_{frame}')  # im.txt
    gn = torch.tensor(imc.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(imc, line_width=3, example=str(names))


    for *xyxy, conf, cls in reversed(det):
        if args.save_bbox:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

        if args.save_detect_img:  # Add bbox to image
            c = int(cls)  # integer class
            label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))

    # Stream results
    im0 = annotator.result()

    # Save results (image with detections)
    if args.save_detect_img:
        if mode == 'image':
            cv2.imwrite(save_path, cv2.cvtColor(im0, cv2.COLOR_RGB2BGR))
        elif mode == 'video':
            im_save_path = save_path[:-4] + "_" + str(frame) + ".jpg"
            cv2.imwrite(im_save_path, cv2.cvtColor(im0, cv2.COLOR_RGB2BGR))

    # Print detection results
    # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if args.save_bbox else ''
    
    
    
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
    
    gt_ids = []
    # import ipdb; ipdb.set_trace()
    # for d in det:
    # iou = box_iou(labels[:, :4].to("cuda"), det[:, :4])
    
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


class LoadDatas:
    def __init__(self, path):
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
        VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        # self.img_size = img_size
        # self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        # self.auto = auto
        self.fps = 24.0
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            self.fps = 24.0
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'

        return path, img0

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
    
    
    
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
        
        assert len(self.labfiles) == len(self.imgfiles), f'Number of images and labels are not same.'
        
        self.labels = []
        for labpath in self.labfiles:
            lab = open(labpath, "r")
            datas = lab.readlines()
            
            label = []
            for d in datas:
                cls, cx, cy, w, h, id = d.split()
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

