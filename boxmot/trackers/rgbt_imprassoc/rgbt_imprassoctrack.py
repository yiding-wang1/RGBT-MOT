# Raif Olson

import numpy as np
from collections import deque
from pathlib import Path
from torch import device

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc.sof import SOF
from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.trackers.imprassoc.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment,
                                   d_iou_distance)
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh
from boxmot.trackers.basetracker import BaseTracker
import os
import csv
import cv2
import glob


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
            x1, y1, x2, y2, id, conf, cls, ind = x
            v_fi.write(f"{frame_number - 1},"
                     f"{id},"
                     f"{x1},"
                     f"{y1},"
                     f"{x2-x1},"
                     f"{y2-y1},1,1,1"
                     + '\n')
        v_fi.close()

    elif frame_number == 1:  # refresh history records
        v_fi = open(visible_path, 'w')
        v_fi.close()



class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, det, feat=None, feat_history=15, max_obs=15):
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs=max_obs
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.conf)
        self.history_observations = deque([], maxlen=self.max_obs)

        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, conf):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += conf
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, conf])
                self.cls = cls
        else:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_count):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # from OAI track, no unconfirmed tracks.
        self.is_activated = True
        self.frame_count = frame_count
        self.start_frame = frame_count

    def re_activate(self, new_track, frame_count, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh, self.conf
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_count = frame_count
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

        self.update_cls(new_track.cls, new_track.conf)

    def update(self, new_track, frame_count):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_count: int
        :type update_feature: bool
        :return:
        """
        self.frame_count = frame_count
        self.tracklet_len += 1

        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh, self.conf
        )

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret


class rgbt_ImprAssocTrack(BaseTracker):
    """
    ImprAssocTrack Tracker: A tracking algorithm that utilizes a combination of appearance and motion-based tracking.

    Args:
        model_weights (str): Path to the model weights for ReID (Re-Identification).
        device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').
        fp16 (bool): Whether to use half-precision (fp16) for faster inference on compatible devices.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        track_high_thresh (float, optional): High threshold for detection confidence. Detections above this threshold are used in the first association round.
        track_low_thresh (float, optional): Low threshold for detection confidence. Detections below this threshold are ignored.
        new_track_thresh (float, optional): Threshold for creating a new track. Detections above this threshold will be considered as potential new tracks.
        match_thresh (float, optional): Threshold for the matching step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
        second_match_thresh (float, optional): Threshold for the second round of matching, used to associate low confidence detections.
        overlap_thresh (float, optional): Threshold for discarding overlapping detections after association.
        lambda_ (float, optional): Weighting factor for combining different association costs (e.g., IoU and ReID distance).
        track_buffer (int, optional): Number of frames to keep a track alive after it was last detected. A longer buffer allows for more robust tracking but may increase identity switches.
        proximity_thresh (float, optional): Threshold for IoU (Intersection over Union) distance in first-round association.
        appearance_thresh (float, optional): Threshold for appearance embedding distance in the ReID module.
        cmc_method (str, optional): Method for correcting camera motion. Options include "sparseOptFlow" (Sparse Optical Flow).
        frame_rate (int, optional): Frame rate of the video being processed. Used to scale the track buffer size.
        with_reid (bool, optional): Whether to use ReID (Re-Identification) features for association.
    """
    def __init__(
        self,
        reid_weights: Path,
        device: device,
        half: bool,
        per_class: bool = False,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        match_thresh: float = 0.65, # bigger?
        second_match_thresh: float = 0.19,
        overlap_thresh: float = 0.55,
        lambda_: float = 0.2,
        track_buffer: int = 35,
        proximity_thresh: float = 0.1,
        appearance_thresh: float = 0.25,
        cmc_method: str = "sparseOptFlow",
        frame_rate=30,
        with_reid: bool = True,
        track_id=0,
        track_all=True
    ):
        super().__init__(per_class=per_class)
        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.second_match_thresh = second_match_thresh
        self.overlap_thresh = overlap_thresh
        self.lambda_ = lambda_

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid
        if self.with_reid:
            rab = ReidAutoBackend(
                weights=reid_weights, device=device, half=half
            )
            self.model = rab.get_backend()

        self.cmc = SOF()

        self.modality = 'infrared'
        self.img_path = track_id
        self.visible_img_list = get_img_names(self.img_path, self.modality)
        self.visible_img_size = cv2.imread(os.path.join(self.img_path, self.modality, self.visible_img_list[0])).shape
        self.track_all = track_all
        self.subset = os.path.basename(track_id)
        self.detects = get_img_dets(self.img_path, self.modality + '/det', '424_Two_Iwo_', 0.4)
        self.output_path = "../output_tracks/imprassoc_Two_Iwo/data/" + os.path.basename(self.img_path)

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:

        visible_dets = self.detects[self.frame_count] if self.frame_count in self.detects.keys() \
            else np.zeros(shape=(0, 7))
        dets = np.array(visible_dets, dtype=float)[:, 1:]
        if dets.shape[0] != 0:
            xyxyn = dets[:, :4].astype(float)
            xyxy = xyxyn2xyxy(xyxyn, self.visible_img_size)
            dets[:, :4] = xyxy
            dets = np.delete(dets, 5, axis=1)

        img = self.visible_img_list[self.frame_count]
        img = cv2.imread(os.path.join(self.img_path, self.modality, img))
        img = infrared_preprocess(img)

        self.check_inputs(dets, img)

        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        # Remove bad detections
        confs = dets[:, 4]

        # find second round association detections
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]

        # find first round association detections
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]

        """Extract embeddings """
        # appearance descriptor extraction
        if self.with_reid:
            if embs is not None:
                features_high = embs
            else:
                # (Ndets x X) [512, 1024, 2048]
                features_high = self.model.get_features(dets_first[:, 0:4], img)

        if len(dets) > 0:
            """Detections"""
            if self.with_reid:
                detections = [STrack(det, f, max_obs=self.max_obs) for (det, f) in zip(dets_first, features_high)]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for (det) in np.array(dets_first)]
        else:
            detections = []

        """ Add newly detected tracklets to active_tracks"""
        unconfirmed = []
        low_tent = []
        high_tent = []
        active_tracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)

        '''Improved Association: First they calc the cost matrix of the high
        detections(func_1 -> cost_h), then the calc the cost matrix of the low
        detections (func_2 -> cost_l) and get the max values of both. Then
        B = det_h_max / det_l_max.
        Finally they calc cost = concat(cost_h, B*cost_l) for the matching
        '''

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # Fix camera motion
        warp = self.cmc.apply(img, dets_first)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Associate with high score detection boxes
        d_ious_dists = d_iou_distance(strack_pool, detections)
        ious = 1 - iou_distance(strack_pool, detections)
        ious_dists_mask = (ious < self.proximity_thresh) # o_min in ImprAssoc paper

        num_high_detections = len(detections)

        if self.with_reid:
            # ConfTrack version
            # emb_dists = embedding_distance(strack_pool, detections) / 2.0
            # raw_emb_dists = emb_dists.copy()
            # emb_dists[emb_dists > self.appearance_thresh] = 1.0
            # emb_dists[ious_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0

            # Improved Association Version (CD)
            emb_dists = embedding_distance(strack_pool, detections) # high dets
            dists = self.lambda_*d_ious_dists + (1-self.lambda_)*emb_dists
            dists[ious_dists_mask] = self.match_thresh + 0.00001
        else:
            dists = d_ious_dists
            dists[ious_dists_mask] = self.match_thresh + 0.00001

        # Add in the low score detection boxes

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(det, max_obs=self.max_obs) for
                                 (det) in np.array(dets_second)]
        else:
            detections_second = []
        dists_second = iou_distance(strack_pool, detections_second)
        dists_second_mask = (dists_second > self.second_match_thresh) # this is what the paper used
        dists_second[dists_second_mask] = self.second_match_thresh + 0.00001


        B = self.match_thresh/self.second_match_thresh

        combined_dists = np.concatenate((dists, B*dists_second), axis=1)

        matches, track_conf_remain, det_remain = linear_assignment(combined_dists, thresh=self.match_thresh)

        # concat detections so that it all works
        detections = np.concatenate((detections, detections_second), axis=0)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        '''Deal with lost tracks'''

        # left over confirmed tracks get lost
        for it in track_conf_remain:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''now do OAI from Improved Association paper'''
        # calc the iou between every unmatched det and all tracks if the max iou
        # for a det D is above overlap_thresh, discard it.
        sdet_remain = [detections[i] for i in det_remain]

        if self.with_reid:
            # if we don't need to recompute features
            if (self.new_track_thresh >= self.track_high_thresh) and features_high is not None:
                features = [features_high[i] for i in det_remain if i < num_high_detections]
            else:
                bboxes = [track.xyxy for track in sdet_remain]
                bboxes = np.array(bboxes)
                # (Ndets x X) [512, 1024, 2048]
                features = self.model.get_features(bboxes, img)

        unmatched_overlap = 1 - iou_distance(strack_pool, sdet_remain)

        for det_ind in range(unmatched_overlap.shape[1]): # loop over the rows
            if len(unmatched_overlap[:, det_ind]) != 0:
                if np.max(unmatched_overlap[:, det_ind]) < self.overlap_thresh:
                    # now initialize it
                    track = sdet_remain[det_ind]
                    if track.conf > self.new_track_thresh:
                        track.activate(self.kalman_filter, self.frame_count)
                        if self.with_reid:
                            track.update_features(features[det_ind])
                        activated_starcks.append(track)
            else:
                # if no curr tracks, then init one
                track = sdet_remain[det_ind]
                if track.conf > self.new_track_thresh:
                    track.activate(self.kalman_filter, self.frame_count)
                    if self.with_reid:
                        track.update_features(features[det_ind])
                    activated_starcks.append(track)


        """ Step 6: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        output_stracks = [track for track in self.active_tracks]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)

        outputs = np.asarray(outputs)

        save_results(self.frame_count, self.output_path, outputs, self.modality)

        return outputs


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_count - stracksa[p].start_frame
        timeq = stracksb[q].frame_count - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
