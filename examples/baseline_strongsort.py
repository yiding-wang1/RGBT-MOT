import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from boxmot.trackers import DeepOCSORT, StrongSORT
from ultralytics import YOLO
from boxmot.utils import ROOT, WEIGHTS
import os
import csv

from functools import partial

from examples.detectors import get_yolo_inferer


## 尚未实现：写mot格式输出、相关滤波插补
## 不完善：  跳过插补

@torch.no_grad()
def single_set_run(args, img_path, set_name, det_path):
    # Load a model
    # yolo = YOLO(args.yolo_model)
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    # yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args

    tracker = StrongSORT(
        model_weights=Path('./weights/osnet_x0_25_msmt17.pt'),  # which ReID model to use
        device='cpu',
        fp16=False,
    )

    # 读取图片
    mot_imgpath = "../data/MOT16-13/img1"
    img_list = os.listdir(img_path)
    color = (0, 0, 255)  # BGR
    thickness = 2
    fontscale = 0.5
    visualize_tracking = False

    # 显示图像的参数
    frame_number = 0
    xyxys = None
    object_num_list = []
    save_mot = False
    save_path = "./output/baseline/mot17/train/"+set_name.replace('16', '17') + "_FRCNN.txt"
    total_frames = len(img_list)
    half_val = True  # 仅跟踪后半段目标

    dec_inner = False  # 使用自带检测模型，否则使用外部文件作为检测结果

    # yolox_mot17_ = np.load("../data/MOT17-13-FRCNN.npy")
    # with open(det_path, 'r') as file: # 读取txt格式detect
    #     reader = csv.reader(file)
    #     det_data = np.array(list(reader), dtype=float)
    if half_val:  # 仅跟踪后半段目标
        det_data = np.load(det_path)
        frame_number = int(total_frames/2)
        img_list = img_list[frame_number:]

    for im1_name in img_list:
        im1 = cv2.imread(img_path + '/' + im1_name)
        if dec_inner:
            dets = yolo.predict(source=im1, save=True, imgsz=args.imgsz, classes=args.classes, conf=args.conf)
            for det in dets:
                boxes = det.boxes.xyxy
                confs = det.boxes.conf
                cls = det.boxes.cls
                # 将PyTorch张量转换为NumPy数组
                boxes_np = boxes.cpu().numpy()
                confs_np = confs.cpu().numpy()
                cls_np = cls.cpu().numpy()
                print(boxes_np)

                # 将boxes、confs和cls堆叠成一个数组
                detection_results = np.column_stack((boxes_np, confs_np, cls_np))
        else:
            dets = det_data[det_data[:, 0] == frame_number + 1]  # 筛选相应帧的检测
            boxes_np = np.column_stack((dets[:,2:4], dets[:,2:4]+dets[:,4:6]))
            confs_np = dets[:, 6]
            cls_np = np.zeros(shape=confs_np.shape)
            detection_results = np.column_stack((boxes_np, confs_np, cls_np))


        frame_number = frame_number + 1
        print(f"当前帧号: {frame_number}/{total_frames}")

        # print(f"det_[1]:{detection_results.shape[1]},det_{detection_results}")
        tracks, flag_rapid_pivot = tracker.update(detection_results,
                                                  im1)  # --> (x, y, x, y, id, conf, cls, ind)
        if tracks.shape[0] != 0:
            xyxys = tracks[:, 0:4].astype('int')  # float64 to int
            ids = tracks[:, 4].astype('int')  # float64 to int
            confs = tracks[:, 5]
            clss = tracks[:, 6].astype('int')  # float64 to int

            if save_mot and frame_number > 1: # 写跟踪内容
                fi = open(save_path, 'a+')
                for wid, wxyxy, wc in zip(ids, xyxys, confs):
                    fi.write(
                        f"{frame_number - 1},{wid},{wxyxy[0]},{wxyxy[1]},{(wxyxy[2] - wxyxy[0])},{(wxyxy[3] - wxyxy[1])},1,{str(args.classes + 1)},1" + '\n')
                fi.close()
            elif frame_number == 1:  # 清空跟踪结果文件之前的结果
                fi = open(save_path, 'w')
                fi.close()

        object_num_list.append(detection_results.shape[0])
        print(f"object_num_{detection_results.shape[0]}")
        if visualize_tracking: # 可视化跟踪过程
            if tracks.shape[0] != 0:
                for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                    im1 = cv2.rectangle(
                        im1,
                        (xyxy[0], xyxy[1]),
                        (xyxy[2], xyxy[3]),
                        color,
                        thickness
                    )
                    cv2.putText(
                        im1,
                        f'{id} ',  # f'id: {id}, conf: {conf}, c: {cls}',
                        (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        color,
                        thickness
                    )

            # show image with bboxes, ids, classes and confidences
            # origin_size = (img_w, img_h)
            im1 = cv2.resize(im1, (1920, 1080))
            cv2.imshow('frame', im1)
            cv2.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def run(args):
    root_set_path = "../../../MOT16/train"
    # root_det_path = "../data/MOT17-FRCNN"  # 使用MOT17数据集提供的全部检测结果
    root_det_path = "../data/MOT17-YOLOX-Half"  # 使用StrongSort提供的后半段检测结果
    set_list = os.listdir(root_set_path)
    det_list = os.listdir(root_det_path)
    for set, det in zip(set_list,det_list):
        # single_set_run(args, root_set_path+'/'+set+'/img1', set, root_det_path+'/'+det+'/det/det.txt') # 使用MOT17数据集提供的全部检测结果
        single_set_run(args, root_set_path + '/' + set + '/img1', set, root_det_path + '/' + det) # 使用StrongSort提供的后半段检测结果

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolox_x.pt',  # yolov8n
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='strongsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.6,  # 0.5
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.8,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=0,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',  # labels    store_false
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_true',  # conf    store_false
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_false',  # store_true,
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',  # id-crops    store_true
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',  # 保存mot txt文件，与输入视频同目录。
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=1, type=int,  # default=None
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=1, type=int,
                        help='video frame-rate stride')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
