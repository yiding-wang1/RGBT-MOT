# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
import numpy as np
from collections import deque
import os
import csv
import cv2
import glob
import time
import json

from boxmot.motion.kalman_filters.xysr_kf import KalmanFilterXYSR
from boxmot.utils.association import associate, linear_assignment
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.ops import xyxy2xysr



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


def get_img_dets(root_dir, sub_dir, thres):
    # 拼接指定子目录的完整路径
    target_dir = os.path.join(root_dir, sub_dir, 'det.csv')
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


def save_results(frame_number, save_path, outputs, modality, time_):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    visible_path = save_path + '_'+modality+'.txt'
    fps_path = os.path.dirname(save_path)+f'/{modality}_fps.json'

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

    elif frame_number == 1:  # refresh history records
        v_fi = open(visible_path, 'w')
        v_fi.close()


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, cls, det_ind, delta_t=3, max_obs=50, Q_xy_scaling = 0.01, Q_s_scaling = 0.0001):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        self.det_ind = det_ind

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0

        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_s_scaling

        self.kf.x[:4] = xyxy2xysr(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[-1]
        self.cls = cls
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox, cls, det_ind):
        """
        Updates the state vector with observed bbox.
        """
        self.det_ind = det_ind
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(xyxy2xysr(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class RGBT_OcSort(BaseTracker):
    """
    OCSort Tracker: A tracking algorithm that utilizes motion-based tracking.

    Args:
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        det_thresh (float, optional): Detection confidence threshold. Detections below this threshold are ignored in the first association step.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        min_hits (int, optional): Minimum number of hits required to confirm a track.
        asso_threshold (float, optional): Threshold for the association step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
        delta_t (int, optional): Time delta for velocity estimation in Kalman Filter.
        asso_func (str, optional): Association function to use for data association. Options include "iou" for IoU-based association.
        inertia (float, optional): Weight for inertia in motion modeling. Higher values make tracks less responsive to changes.
        use_byte (bool, optional): Whether to use BYTE association in the second association step.
        Q_xy_scaling (float, optional): Scaling factor for the process noise covariance in the Kalman Filter for position coordinates.
        Q_s_scaling (float, optional): Scaling factor for the process noise covariance in the Kalman Filter for scale coordinates.
    """
    def __init__(
        self,
        per_class: bool = False,
        det_thresh: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
        asso_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
        track_id=0,
        track_all=True,
    ):
        super().__init__(max_age=max_age, per_class=per_class, asso_func=asso_func)
        """
        Sets key parameters for SORT
        """
        self.per_class = per_class
        self.max_age = max_age
        self.min_hits = min_hits
        self.asso_threshold = asso_threshold
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling
        KalmanBoxTracker.count = 0

        self.modality = 'visible'
        self.img_path = track_id
        self.visible_img_list = get_img_names(self.img_path, self.modality)
        self.visible_img_size = cv2.imread(os.path.join(self.img_path, self.modality, self.visible_img_list[0])).shape
        self.track_all = track_all
        self.subset = os.path.basename(track_id)
        self.detects = get_img_dets(self.img_path, self.modality + '/det', thres=0.4)
        self.output_path = "../output_tracks/ocsort_Two_Iwo/data/" + os.path.basename(self.img_path)
        self.total_time = 0

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        visible_dets = self.detects[self.frame_count] if self.frame_count in self.detects.keys() \
            else np.zeros(shape=(0, 7))
        dets = np.array(visible_dets, dtype=float)[:, 1:]
        xyxyn = dets[:, :4].astype(float)
        xyxy = xyxyn2xyxy(xyxyn, self.visible_img_size)
        dets[:, :4] = xyxy
        dets = dets[:, :6]
        img = self.visible_img_list[self.frame_count]
        img = cv2.imread(os.path.join(self.img_path, self.modality, img))
        img = infrared_preprocess(img)
        self.check_inputs(dets, img)

        self.frame_count += 1
        h, w = img.shape[0:2]

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]

        start_time1 = time.time()  # cmc time

        inds_low = confs > 0.1
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(
            inds_low, inds_high
        )  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = confs > self.det_thresh
        dets = dets[remain_inds]

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.active_tracks), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.active_tracks
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.active_tracks
            ]
        )

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5], trks, self.asso_func, self.asso_threshold, velocities, k_observations, self.inertia, w, h
        )
        for m in matched:
            self.active_tracks[m[1]].update(dets[m[0], :5], dets[m[0], 5], dets[m[0], 6])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(
                dets_second, u_trks
            )  # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.asso_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(
                        dets_second[det_ind, :5], dets_second[det_ind, 5], dets_second[det_ind, 6]
                    )
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.asso_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(dets[det_ind, :5], dets[det_ind, 5], dets[det_ind, 6])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], dets[i, 6], delta_t=self.delta_t, Q_xy_scaling=self.Q_xy_scaling, Q_s_scaling=self.Q_s_scaling, max_obs=self.max_obs)
            self.active_tracks.append(trk)
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(
                    np.concatenate((d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])).reshape(
                        1, -1
                    )
                )
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)
        time_all = time.time() - start_time1
        self.total_time += time_all
        save_results(self.frame_count, self.output_path, ret, self.modality, time_all)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])