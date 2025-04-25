# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import cv2
import os
import csv
import glob
from torch import device
from pathlib import Path

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh
from boxmot.trackers.basetracker import BaseTracker


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

def get_img_names(root_dir, sub_dir):
    # 拼接指定子目录的完整路径
    target_dir = os.path.join(root_dir, sub_dir)
    # 使用 glob 模块查找指定子目录下所有的 .jpg 文件
    jpg_files = glob.glob(os.path.join(target_dir, '*.jpg'))
    # 从完整文件路径中提取文件名
    file_names = [os.path.basename(file) for file in jpg_files]
    return file_names

def xyxyn2xyxy(xyxyn, img_shape):
    xyxy = np.zeros(shape=xyxyn.shape)
    xyxy[:, 0] = xyxyn[:, 0] * img_shape[1]
    xyxy[:, 1] = xyxyn[:, 1] * img_shape[0]
    xyxy[:, 2] = xyxyn[:, 2] * img_shape[1]
    xyxy[:, 3] = xyxyn[:, 3] * img_shape[0]
    return xyxy


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


def save_results(frame_number, save_path, outputs, modality):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    visible_path = save_path + '_'+modality+'.txt'
    if frame_number > 1:
        v_fi = open(visible_path, 'a+')
        for x in outputs:
            x1, y1, x2, y2, id, conf, cls, ind = x[0]
            v_fi.write(f"{frame_number - 1},"
                     f"{id},"
                     f"{x1},"
                     f"{y1},"
                     f"{x2-x1},"
                     f"{y2-y1},1,"
                     f"{cls},1"
                     + '\n')
        v_fi.close()

    elif frame_number == 1:  # refresh history records
        v_fi = open(visible_path, 'w')
        v_fi.close()


class StrongSort(object):
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
        max_cos_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.98,
        ema_alpha=0.9,
        track_id=0,
        track_all=True
    ):

        self.per_class = per_class
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model
        self.visualize = False
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        self.cmc = get_cmc_method('ecc')()

        self.frame_count = 0
        self.modality = 'visible'
        self.img_path = track_id
        self.visible_img_list = get_img_names(self.img_path, self.modality)
        self.visible_img_size = cv2.imread(os.path.join(self.img_path, self.modality, self.visible_img_list[0])).shape
        self.track_all = track_all
        self.subset = os.path.basename(track_id)
        self.detects = get_img_dets(self.img_path, self.modality + '/det', '424_Ten_Ien_', 0.4)
        self.output_path = "../output_tracks/strongsort_424_Ten_Ien/data/" + os.path.basename(self.img_path)

    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        visible_dets = self.detects[self.frame_count] if self.frame_count in self.detects.keys() \
            else np.zeros(shape=(0, 8))
        dets = np.array(visible_dets, dtype=float)[:, 1:]
        xyxyn = dets[:, :4].astype(float)
        xyxy = xyxyn2xyxy(xyxyn, self.visible_img_size)
        dets[:, :4] = xyxy
        img = self.visible_img_list[self.frame_count]
        img = cv2.imread(os.path.join(self.img_path, self.modality, img))
        img = infrared_preprocess(img)

        self.frame_count += 1

        # dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        # xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        if len(self.tracker.tracks) >= 1:
            warp_matrix, _ = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # extract appearance information for each detection
        if embs is not None:
            features = embs
        else:
            features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat) for
            box, conf, cls, det_ind, feat in
            zip(tlwh, confs, clss, det_ind, features)
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])).reshape(1, -1)
            )
        # print(outputs)
        if self.visualize:
            color = (0, 0, 255)  # BGR
            thickness = 2
            fontscale = 0.5
            if len(outputs) != 0:
                for x in outputs:
                    x1, y1, x2, y2, id, conf, cls, ind = x[0]
                    cv2.rectangle(
                        img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        thickness
                    )
                    cv2.putText(
                        img,
                        f'{id} ',  # f'id: {id}, conf: {conf}, c: {cls}',
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        color,
                        thickness
                    )

                # show image with bboxes, ids, classes and confidences
                # origin_size = (img_w, img_h)
            # im1 = cv2.resize(img, (1920, 1080))
            cv2.imshow('frame', img)
            cv2.waitKey()

        # save_results(self.frame_count, self.output_path, outputs, self.modality)
        show_both_result(img, outputs, self.frame_count, self.subset, self.modality)
        if len(outputs) > 0:
            return np.concatenate(outputs)

        return np.array([])

    def plot_results(img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5):
        pass


def show_both_result(visible_img, visible_outputs, frame_num, subset, modality):
    color = (0, 0, 255)  # BGR
    thickness = 2
    fontscale = 0.5
    if len(visible_outputs) != 0:
        for x in visible_outputs:
            x1, y1, x2, y2, id, conf, cls, ind = x[0]
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
    save_path = f"./output_imgs_strongsort/{subset}_{modality}/{frame_num}.jpg"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, visible_img)
    # combined_img = cv2.resize(combined_img, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow(f'frame {frame_num}', combined_img)
    # cv2.waitKey()

    return