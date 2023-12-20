import argparse

def parse_training_args(parser):
    
    # Directory parameters
    
    #MOT17
    # parser.add_argument('--infer_data_dir', type=str, default='../DATASET/MOT17/train/MOT17-04-FRCNN_query/images', help='Only for inference')
    # parser.add_argument('--dataset_root_dir', type=str, default='../DATASET/', help='root data dir for gallery, gallery must be placed like "{root_dir}/{dataset_name}_reid/gallery" ')
    # parser.add_argument('--dataset_name', type=str, default='MOT17', help='for setting gallery path')
    
    # parser.add_argument('--detection_weight_file', type=str, default='./weights/detection_MOT17.pt')
    # parser.add_argument('--reid_weight_file', type=str, default='./weights/reid_market1501_pretrained.pth')
    
    # #PRW_yolo
    parser.add_argument('--infer_data_dir', type=str, default='./watosys', help='Only for inference')
    parser.add_argument('--dataset_root_dir', type=str, default='../DATASET/', help='root data dir for gallery, gallery must be placed like "{root_dir}/{dataset_name}_reid/gallery" ')
    parser.add_argument('--dataset_name', type=str, default='PRW', help='for setting gallery path')
    
    parser.add_argument('--detection_weight_file', type=str, default='./weights/detection_MOT17.pt')
    parser.add_argument('--reid_weight_file', type=str, default='./weights/reid_PRW_pretrained.pth')



    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--detect_save_dir', type=str, default='detection')
    parser.add_argument('--reid_save_dir', type=str, default='reid')
    
    
    # Session Parameters
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--save_detection_images', type=str2bool, default=False)
    parser.add_argument('--save_images', type=str2bool, default=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf_thres', type=float, default=0.3, help='conf_threshold for detection')
    parser.add_argument('--iou_thres', type=str, default=0.3, help='conf_threshold for detection')
    
    parser.add_argument('--scale', type=int, default=1, help='for infer low-resolution reid') #test 때만 필요
    parser.add_argument('--use_GT_IDs', action='store_true', help='if dataset has GT IDs, visualize with GT') 

    
    # detection parameter
    parser.add_argument('--yolo_data', type=str, default='./SNU_PersonDetection/data/PRW.yaml')
    

    # reid parameter

    

def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser(description='UNet')
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))