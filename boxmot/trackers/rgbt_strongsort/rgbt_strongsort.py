# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import sys

import numpy as np
from torch import device
from pathlib import Path
import torchvision.transforms as transforms
import onnxruntime as ort
import copy

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.rgbt_strongsort.sort.detection import Detection
from boxmot.trackers.rgbt_strongsort.sort.rgbt_tracker import Tracker
from boxmot.trackers.rgbt_strongsort.sort.sep_tracker import SeperateTracker

from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh
from boxmot.trackers.basetracker import BaseTracker

import os
import glob
import csv
import cv2
from PIL import Image
import pickle
import time
import json


class RGBT_StrongSort(object):
    """
    StrongSORT Tracker: A tracking algorithm that utilizes a combination of appearance and motion-based tracking.

    Args:
        model_weights (str): Path to the model weights for ReID (Re-Identification).
        device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').
        fp16 (bool): Whether to use half-precision (fp16) for faster inference on compatible devices.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        max_dist (float, optional): Maximum cosine distance for ReID feature matching in Nearest Neighbor Distance Metric.
        max_iou_dist (float, optional): Maximum Intersection over Union (IoU) distance for data association. Controls the maximum allowed distance between tracklets and detections for a match.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        n_init (int, optional): Number of consecutive frames required to confirm a track.
        nn_budget (int, optional): Maximum size of the feature library for Nearest Neighbor Distance Metric. If the library size exceeds this value, the oldest features are removed.
        mc_lambda (float, optional): Weight for motion consistency in the track state estimation. Higher values give more weight to motion information.
        ema_alpha (float, optional): Alpha value for exponential moving average (EMA) update of appearance features. Controls the contribution of new and old embeddings in the ReID model.
    """

    def __init__(
            self,
            reid_weights: Path,
            device: device,
            half: bool,
            per_class: bool = False,
            max_cos_dist=0.2,  #0.2
            max_iou_dist=0.7,
            max_age=30,
            n_init=3,
            nn_budget=100,
            mc_lambda=0.98,
            ema_alpha=0.9,
            track_id=0,
            track_all=True
    ):
        # stack = inspect.stack()
        self.per_class = per_class
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model

        max_cos_dist = 0.2
        max_age = 30
        max_iou_dist = 0.9  # ！！！！！！！！！！！！！！！！！
        self.exp_id = '58_r'

        # whether track different model seperately
        self.seperate_track = False
        if self.seperate_track:
            self.tracker = SeperateTracker(
                metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
                max_iou_dist=max_iou_dist,
                max_age=max_age,
                n_init=n_init,
                mc_lambda=mc_lambda,
                ema_alpha=ema_alpha,
            )
        else:
            self.tracker = Tracker(
                metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
                max_iou_dist=max_iou_dist,
                max_age=max_age,
                n_init=n_init,
                mc_lambda=mc_lambda,
                ema_alpha=ema_alpha,
                exp_id=self.exp_id
            )
        self.cmc_vi = get_cmc_method('ecc')()
        self.cmc_ir = get_cmc_method('ecc')()

        self.frame_num = 0
        self.total_time = 0
        self.subset = 'the2ndboyunderbasket'

        if track_all:
            self.img_path = track_id
            self.subset = os.path.basename(track_id)
        else:
            self.img_path = 'E:/lasher/LasHeR_Unalined_960_0615/seleted/'+self.subset
            #blackboy' #2ndboyfarintheforest2right'
            # leftunderbasket,midof3girls,righthunchblack,manbikecoming

        self.dataset_path = 'E:/lasher/LasHeR_Unalined_960_0615/seleted'
        self.visible_img_list = get_img_names(self.img_path, 'visible')
        self.infrared_img_list = get_img_names(self.img_path, 'infrared')
        self.visible_img_size = cv2.imread(os.path.join(self.img_path, 'visible', self.visible_img_list[0])).shape
        self.infrared_img_size = cv2.imread(os.path.join(self.img_path, 'infrared', self.infrared_img_list[0])).shape
        self.frame_len = len(self.infrared_img_list)
        self.visible_dets_list = get_img_dets(self.img_path, 'visible/det', '424_Two_Iwo_', thres=0.4)
        self.infrared_dets_list = get_img_dets(self.img_path, 'infrared/det', '424_Two_Iwo_', thres=0.4)

        self.visualize = not track_all
        self.save_output = track_all
        self.track_id = track_id
        self.track_all = track_all

        self.output_path = "../output_tracks/"+self.exp_id+"/data/" + os.path.basename(self.img_path)
        self.pose_only = True

        # self.output_path = "../../../TrackEval-master/TrackEval-master/data/tracks/228/data/" +os.path.basename(self.img_path)

        # "../../../TrackEval-master/TrackEval-master/data/tracks/228/data/"
        # os.environ["CUDA VISIBLE DEVICES"] = "0"

    @BaseTracker.per_class_decorator
    def update(self, visible_dets: np.ndarray, visible_img: np.ndarray,
               ) -> np.ndarray:
        assert isinstance(
            visible_dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(visible_dets)}', valid format is np.ndarray"
        assert isinstance(
            visible_img, np.ndarray
        ), f"Unsupported 'img' input format '{type(visible_img)}', valid format is np.ndarray"
        assert (
                len(visible_dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
                visible_dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        # read img_pairs,
        # shared_space image pairs
        # and det_results,
        visible_img = self.visible_img_list[self.frame_num]
        infrared_img = self.infrared_img_list[self.frame_num]
        visible_img = cv2.imread(os.path.join(self.img_path, 'visible', visible_img))
        infrared_img = cv2.imread(os.path.join(self.img_path, 'infrared', infrared_img))
        infrared_img = infrared_preprocess(infrared_img)
        visible_img = infrared_preprocess(visible_img)

        visible_dets = self.visible_dets_list[self.frame_num] if self.frame_num in self.visible_dets_list.keys() \
            else np.zeros(shape=(0, 8))
        infrared_dets = self.infrared_dets_list[self.frame_num] if self.frame_num in self.infrared_dets_list.keys() \
            else np.zeros(shape=(0, 8))

        self.frame_num = self.frame_num + 1
        print(f'round {self.frame_num}')

        visible_dets = np.array(visible_dets, dtype=float)
        visible_xyxyn = visible_dets[:, 1:5].astype(float)
        visible_xyxy = xyxyn2xyxy(visible_xyxyn, self.visible_img_size)
        visible_confs = visible_dets[:, 5]
        visible_clss = visible_dets[:, 6].astype(int)
        visible_det_ind = visible_dets[:, 7].astype(int)

        infrared_dets = np.array(infrared_dets, dtype=float)
        infrared_xyxyn = infrared_dets[:, 1:5].astype(float)
        infrared_xyxy = xyxyn2xyxy(infrared_xyxyn, self.infrared_img_size)
        infrared_confs = infrared_dets[:, 5]
        infrared_clss = infrared_dets[:, 6].astype(int)
        infrared_det_ind = infrared_dets[:, 7].astype(int)

        # print(f"infrared_bbox={infrared_xyxy}")
        # extract appearance information for each detection -- visible

        # 实验：图像熵
        vi_entropies = self.detections_entropy(visible_xyxy, visible_img)
        ir_entropies = self.detections_entropy(infrared_xyxy, infrared_img)

        visible_features = self.model.get_features(visible_xyxy, visible_img)
        if self.pose_only:
            share_visible_features = visible_features
        else:
            share_visible_features = self.get_modality_features_deen_vi(visible_xyxy, visible_img)
        visible_tlwh = xyxy2tlwh(visible_xyxy)
        visible_detections = [
            Detection(box, conf, cls, det_ind, feat, share_feat, vi_entropy) for
            box, conf, cls, det_ind, feat, share_feat, vi_entropy in
            zip(visible_tlwh, visible_confs, visible_clss, visible_det_ind, visible_features
                , share_visible_features, vi_entropies)
        ]

        start_time1 = time.time()  # cmc time

        if len(self.tracker.visible_tracks) >= 1:
            warp_matrix_vi, conf_cmc_vi = self.cmc_vi.apply(visible_img, visible_xyxy)
            for track in self.tracker.visible_tracks:
                track.camera_update(warp_matrix_vi)
            if len(self.tracker.infrared_tracks) >= 1:
                for track in self.tracker.infrared_tracks:
                    track.camera_update(warp_matrix_vi)

        # if len(self.tracker.visible_tracks) >= 1 and len(self.tracker.infrared_tracks) >= 1:
        #     warp_matrix_vi, conf_cmc_vi = self.cmc_vi.apply(visible_img, visible_xyxy)
        #     warp_matrix_ir, conf_cmc_ir = self.cmc_ir.apply(infrared_img, infrared_xyxy)
        #     print(conf_cmc_vi, conf_cmc_ir)
        #     if conf_cmc_vi == 0.:
        #         warp_matrix_b = warp_matrix_ir
        #     else:
        #         warp_matrix_b = warp_matrix_vi
        #     for track in self.tracker.visible_tracks:
        #         track.camera_update(warp_matrix_b)
        #     for track in self.tracker.infrared_tracks:
        #         track.camera_update(warp_matrix_b)
        #
        # elif len(self.tracker.visible_tracks) >= 1:
        #     warp_matrix_vi, conf_cmc_vi = self.cmc_vi.apply(visible_img, visible_xyxy)
        #     for track in self.tracker.visible_tracks:
        #         track.camera_update(warp_matrix_vi)
        #     for track in self.tracker.infrared_tracks:
        #         track.camera_update(warp_matrix_vi)

        time_cmc = time.time() - start_time1

        # extract appearance information for each detection -- infrared
        infrared_features = self.model.get_features(infrared_xyxy, infrared_img)
        if self.pose_only:
            share_infrared_features = infrared_features
        else:
            share_infrared_features = self.get_modality_features_deen_ir(infrared_xyxy, infrared_img)
        infrared_tlwh = xyxy2tlwh(infrared_xyxy)
        infrared_detections = [
            Detection(box, conf, cls, det_ind, feat, share_feat, ir_entropy) for
            box, conf, cls, det_ind, feat, share_feat, ir_entropy in
            zip(infrared_tlwh, infrared_confs, infrared_clss, infrared_det_ind,
                infrared_features, share_infrared_features, ir_entropies)
        ]

        start_time2 = time.time()  # tracking algorithm time

        # update tracker with dual modality detections, within-modality features and cross-modality features
        self.tracker.predict()
        self.tracker.update(visible_detections, infrared_detections, self.frame_num)

        time_all = time.time() - start_time2 + time_cmc
        self.total_time += time_all

        # output bbox identities in both modality
        visible_outputs = []
        for track in self.tracker.visible_tracks:
            if not track.is_confirmed() or track.time_since_update >= 1: # 尚未确认、暂未更新
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            visible_outputs.append(
                np.concatenate(([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])).reshape(1, -1)
            )
        if len(visible_outputs) > 0:
            visible_outputs = np.concatenate(visible_outputs)
            print(f"visible_outputs:{visible_outputs[:, 4]}")
        else:
            visible_outputs = np.array([])

        infrared_outputs = []
        for track in self.tracker.infrared_tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue
            x1, y1, x2, y2 = track.to_tlbr()
            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind
            infrared_outputs.append(
                np.concatenate(([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])).reshape(1, -1)
            )
        if len(infrared_outputs) > 0:
            infrared_outputs = np.concatenate(infrared_outputs)
            print(f"infrared_outputs:{infrared_outputs[:, 4]}")
        else:
            infrared_outputs = np.array([])

        paired_tracks = self.tracker.paired_crossmodel_ids if not self.seperate_track else []

        # print(f"visible_outputs:{visible_outputs[0,:,4]}",f"infrared_outputs:{infrared_outputs[0,:,4]}")
        # print(f"infrared_outputs:{infrared_outputs}")

        if self.visualize:
            pass
            # show_both_result(visible_outputs, infrared_outputs, copy.deepcopy(visible_img), copy.deepcopy(infrared_img),self.frame_num, self.subset)
            # save_both_results(self.frame_num, save_path=self.output_path,
            #                   visible_outputs=visible_outputs, infrared_outputs=infrared_outputs,
            #                   paired_tracks=paired_tracks, separate_tracking=self.seperate_track)
            # show_both_det(visible_xyxy, infrared_xyxy,
            #               copy.deepcopy(visible_img), copy.deepcopy(infrared_img),
            #               self.frame_num, self.subset, visible_confs, infrared_confs)
        if self.save_output:
            save_both_results(self.frame_num, save_path=self.output_path,
                              visible_outputs=visible_outputs, infrared_outputs=infrared_outputs,
                              paired_tracks=paired_tracks, separate_tracking=self.seperate_track, time_=self.total_time)
            # show_both_result(visible_outputs, infrared_outputs,
            #                  copy.deepcopy(visible_img), copy.deepcopy(infrared_img),
            #                  self.frame_num, self.subset)
            # show_both_det(visible_xyxy, infrared_xyxy,
            #               copy.deepcopy(visible_img), copy.deepcopy(infrared_img),
            #               self.frame_num, self.subset, visible_confs, infrared_confs)
        return np.array([])

    def get_modality_features_deen_vi(self, modality_xyxys, modality_img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 144)),
            transforms.ToTensor(),
            normalize,
        ])
        outputs = []
        providers = [("CUDAExecutionProvider", {"device_id": 0})]
        # if 'TensorrtExecutionProvider' in providers:
        #     providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        # else:
        #     providers = ['CPUExecutionProvider']
        for xyxy in modality_xyxys:
            x1, y1, x2, y2 = xyxy.astype(int)
            ort_session = ort.InferenceSession("./weights/deen_reid_vis.onnx", providers=providers)
            input_data = modality_img[y1:y2, x1:x2]
            # 预处理图像
            input_tensor = transform_test(input_data)
            input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
            input_data = input_tensor.numpy()

            # 获取输入名称
            input_names = [input.name for input in ort_session.get_inputs()]
            # 运行推理
            output1, output2 = ort_session.run(None, {input_names[0]: input_data})
            output = output1 + output2
            output = output[0] + output[1] + output[2]
            outputs.append(output)
        return np.array(outputs)

    def get_modality_features_deen_ir(self, modality_xyxys, modality_img):
        normalize = transforms.Normalize(mean=[0.496, 0.496, 0.496], std=[0.195, 0.195, 0.195])  # from ImageNet
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 144)),
            transforms.ToTensor(),
            normalize,
        ])
        outputs = []
        providers = [("CUDAExecutionProvider", {"device_id": 0})]
        # if 'TensorrtExecutionProvider' in providers:
        #     providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        # else:
        #     providers = ['CPUExecutionProvider']
        for xyxy in modality_xyxys:
            x1, y1, x2, y2 = xyxy.astype(int)
            ort_session = ort.InferenceSession("./weights/deen_reid_ir.onnx", providers=providers)
            input_data = modality_img[y1:y2, x1:x2]

            # 预处理图像
            input_tensor = transform_test(input_data)
            input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
            input_data = input_tensor.numpy()

            # 获取输入名称
            input_names = [input.name for input in ort_session.get_inputs()]
            # 运行推理
            output1, output2 = ort_session.run(None, {input_names[0]: input_data})
            output = output1+output2
            output = output[0]+output[1]+output[2]
            outputs.append(output)
        return np.array(outputs)

    def get_modality_features_jsia(self, modality_xyxys, modality_img):
        transform_test = transforms.Compose([
            transforms.Resize([256, 128], interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            # we normalize it in reid model with imagenet mean and std
        ])
        outputs = []
        for xyxy in modality_xyxys:
            x1, y1, x2, y2 = xyxy.astype(int)
            # providers = ort.get_available_providers()
            # if 'CUDAExecutionProvider' in providers:
            #     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # else:
            #     providers = ['CPUExecutionProvider']
            ort_session = ort.InferenceSession("./weights/jsia_reid.onnx")
            input_data = Image.fromarray(modality_img[y1:y2, x1:x2], mode='RGB')
            # 预处理图像
            input_tensor = transform_test(input_data)
            input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
            input_data = input_tensor.numpy()

            # 获取输入名称
            input_names = [input.name for input in ort_session.get_inputs()]
            # 运行推理
            _, output = ort_session.run(None, {input_names[0]: input_data})
            outputs.append(output)
        return np.array(outputs)

    def plot_results(self, orig_img, show_trajectories):
        pass

    def detections_entropy(self, xyxys, img):
        def _calculate_single_channel_entropy(channel):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = 0.0
            for p in hist:
                if p > 0:
                    entropy -= p * np.log2(p)
            return entropy

        entrophys = []
        h, w = img.shape[:2]
        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            entrophys.append(_calculate_single_channel_entropy(crop))
        return entrophys

def get_img_names(root_dir, sub_dir):
    # 拼接指定子目录的完整路径
    target_dir = os.path.join(root_dir, sub_dir)
    # 使用 glob 模块查找指定子目录下所有的 .jpg 文件
    jpg_files = glob.glob(os.path.join(target_dir, '*.jpg'))
    # 从完整文件路径中提取文件名
    file_names = [os.path.basename(file) for file in jpg_files]
    return file_names


def get_img_dets(root_dir, sub_dir, version, thres):
    # 拼接指定子目录的完整路径
    target_dir = os.path.join(root_dir, sub_dir, version+'det.csv')
    # 从完整文件路径中提取文件名
    result = {}
    with open(target_dir, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            float_row = []
            for value in row:
                float_value = float(value)
                float_row.append(float_value)
            cls = int(float_row[6])
            score = float(float_row[5])
            if cls!=0 or score<thres:
                continue
            try:
                index = int(float_row[0])
                if index not in result:
                    result[index] = []
                # row.append(clk)
                # clk+=1
                result[index].append(row)
            except (ValueError, IndexError):
                print(f"处理行 {float_row} 时出错，索引{index}可能不是有效的整数或者行为空。")

    for key, value in result.items():
        # 获取当前二维列表的行数
        rows = len(value)
        for i in range(rows):
            if len(value[i]) > 0:
                # 如果二维列表的子列表不为空，则设置最后一列的值
                if len(value[i]) == 1:
                    # 如果子列表只有一个元素，则直接赋值
                    value[i] = [i]
                else:
                    # 否则，最后一个元素
                    value[i].append(i)
    return result


def show_both_result(visible_outputs, infrared_outputs, visible_img, infrared_img, frame_num, subset):
    color = (0, 0, 255)  # BGR
    thickness = 2
    fontscale = 0.5
    if len(visible_outputs) != 0:
        for x in visible_outputs:
            x1, y1, x2, y2, id, conf, cls, ind = x
            cv2.rectangle(
                visible_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness
            )
            cv2.putText(
                visible_img,
                f'id:{int(id)}',#,conf:{conf:.2f}',  # f'id: {id}, conf: {conf}, c: {cls}',
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )
    if len(infrared_outputs) != 0:
        for x in infrared_outputs:
            x1, y1, x2, y2, id, conf, cls, ind = x

            cv2.rectangle(
                infrared_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness
            )
            cv2.putText(
                infrared_img,
                f'id:{int(id)}',#,conf:{conf:.2f}',  # f'id: {id}, conf: {conf}, c: {cls}',
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )
    combined_img = cv2.hconcat([visible_img, infrared_img])
    save_path = f"./output_imgs_428_i/{subset}/{frame_num}.jpg"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, combined_img)
    # combined_img = cv2.resize(combined_img, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow(f'frame {frame_num}', combined_img)
    # cv2.waitKey()

    return

def show_both_det(visible_xyxy, infrared_xyxy, visible_img, infrared_img, frame_num, subset, visible_confs, infrared_confs):
    color = (0, 0, 255)  # BGR
    thickness = 2
    fontscale = 0.5
    for x, conf in zip(visible_xyxy, visible_confs):
        x1,y1,x2,y2=x
        cv2.rectangle(
            visible_img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness
        )
        cv2.putText(
            visible_img,
            f'{conf:.2f} ',  # f'id: {id}, conf: {conf}, c: {cls}',
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            color,
            thickness
        )
    for x, conf in zip(infrared_xyxy, infrared_confs):
        x1, y1, x2, y2 = x
        cv2.rectangle(
            infrared_img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness
        )
        cv2.putText(
            infrared_img,
            f'{conf:.2f} ',  # f'id: {id}, conf: {conf}, c: {cls}',
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            color,
            thickness
        )
    combined_img = cv2.hconcat([visible_img, infrared_img])
    save_path = f"./output_imgs_det_en/{subset}_det/{frame_num}.jpg"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, combined_img)
    print('save successfully')
    # combined_img = cv2.resize(combined_img, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow('frame', combined_img)
    # cv2.waitKey()
    return


def save_both_results(frame_number, save_path, visible_outputs, infrared_outputs, paired_tracks, time_, separate_tracking=True):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    visible_path = save_path + '_visible.txt'
    infrared_path = save_path + '_infrared.txt'
    paired_id_path = save_path + '_paired_id.pickle'
    fps_path = os.path.dirname(save_path)+'/fps.json'
    if frame_number > 1:
        v_fi = open(visible_path, 'a+')
        for x in visible_outputs:
            x1, y1, x2, y2, id, conf, cls, ind = x
            v_fi.write(f"{frame_number - 1},"
                     f"{id},"
                     f"{x1},"
                     f"{y1},"
                     f"{x2-x1},"
                     f"{y2-y1},1,"
                     f"{cls},1"
                     + '\n')
        v_fi.close()

        i_fi = open(infrared_path, 'a+')
        for x in infrared_outputs:
            x1, y1, x2, y2, id, conf, cls, ind = x
            i_fi.write(f"{frame_number - 1},"
                     f"{id},"
                     f"{x1},"
                     f"{y1},"
                     f"{x2-x1},"
                     f"{y2-y1},1,"
                     f"{cls},1"
                     + '\n')
        i_fi.close()

        if not separate_tracking:
            try:
                with open(paired_id_path, 'rb') as p_fi:
                    data_list = pickle.load(p_fi)
                    p_fi.close()
            except (FileNotFoundError, EOFError):
                data_list = []
            data_list.append(paired_tracks)
            with open(paired_id_path, 'wb') as p_fi:
                pickle.dump(data_list, p_fi)
                p_fi.close()

        try:
            with open(fps_path, 'r+') as t_fi:
                history_t = json.load(t_fi)
                history_t = [history_t[0] + time_, history_t[1] + 1]
                t_fi.seek(0)
                json.dump(history_t, t_fi)
                t_fi.truncate()
        except FileNotFoundError:
            with open(fps_path, 'w') as t_fi:
                json.dump([time_, 1.], t_fi)

        ##  !!!!!!!!!!!标签

    elif frame_number == 1:  # refresh history records
        v_fi = open(visible_path, 'w')
        v_fi.close()
        i_fi = open(infrared_path, 'w')
        i_fi.close()
        p_fi = open(paired_id_path, 'w')
        p_fi.close()


def xyxyn2xyxy(xyxyn, img_shape):
    xyxy = np.zeros(shape=xyxyn.shape)
    xyxy[:, 0] = xyxyn[:, 0] * img_shape[1]
    xyxy[:, 1] = xyxyn[:, 1] * img_shape[0]
    xyxy[:, 2] = xyxyn[:, 2] * img_shape[1]
    xyxy[:, 3] = xyxyn[:, 3] * img_shape[0]
    return xyxy


def infrared_preprocess(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 分离图像的三个通道
    b, g, r = cv2.split(image)
    # 对每个通道应用 CLAHE 算法
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)

    # 合并处理后的通道
    image_clahe = cv2.merge((b_clahe, g_clahe, r_clahe))
    return image_clahe