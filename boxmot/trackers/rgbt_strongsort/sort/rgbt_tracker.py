# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import absolute_import

import copy

import numpy as np
import random
import pyswarms as ps
import torch

from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.rgbt_strongsort.sort import iou_matching, linear_assignment
from boxmot.trackers.rgbt_strongsort.sort.track import Track
from boxmot.utils.matching import chi2inv95, _nn_cosine_distance, _nn_euclidean_distance

from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

from boxmot.motion.kalman_filters.xyah_fkf import FederatedKalmanFilterXYAH


class FKFMode:
    miss2 = 0
    miss_vi = 1
    miss_ir = 2
    both = 3


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """


    GATING_THRESHOLD = np.sqrt(chi2inv95[4])
    def __init__(
            self,
            metric,
            max_iou_dist=0.9,
            max_age=30,
            n_init=3,
            _lambda=0,
            ema_alpha=0.9,
            conf_ema_alpha=0.1,
            bias_ema_alpha=1,
            mc_lambda=0.995,
            deep_track_dist=0.45,
            pos_track_dist=0.4,
            pair_delete_pos_thres=0.5,
            pair_delete_time_thres_max=30,
            pair_delete_time_thres_min=10,
            pair_delete_deep_thres=0.55,
            soft_nms_thres=0.8,
            exp_id='',
            adaptive_pose_thres=True,  # [1, 0.1, 1, 0.2]
    ):
        if adaptive_pose_thres:
            adaptive_pose_thres = [1, 0.1, 1, 0.3]
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.conf_ema_alpha = conf_ema_alpha
        self.mc_lambda = mc_lambda
        self.deep_track_dist = deep_track_dist
        self.pos_track_dist = pos_track_dist
        self.pair_delete_pos_thres = pair_delete_pos_thres
        self.pair_delete_time_thres_max = pair_delete_time_thres_max
        self.pair_delete_time_thres_min = pair_delete_time_thres_min

        self.pair_delete_deep_thres = pair_delete_deep_thres
        self.soft_nms_thres = soft_nms_thres
        self.bias_ema_alpha = bias_ema_alpha
        self.exp_id = exp_id

        self.visible_tracks = []
        self.infrared_tracks = []

        self.single_visible_ids = []
        self.single_infrared_ids = []
        self.paired_crossmodel_ids = []

        self.paired_bias_set = []
        self.frame_num = 0
        self._next_id = 1
        self.cmc = get_cmc_method('ecc')()

        # self.fkf = FederatedKalmanFilterXYAH()
        self.dt = 1.
        self.pos_track_rate = 1.
        self.deep_track_rate = 1.

        self.pose_only = True
        self.adaptive_pose_thres = adaptive_pose_thres

        self.save_args()

    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.visible_tracks:
            track.predict()
        for track in self.infrared_tracks:
            track.predict()

    def increment_ages(self):
        for track in self.visible_tracks:
            track.increment_age()
            track.mark_missed()
        for track in self.infrared_tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, visible_detections, infrared_detections, frame_num):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        self.frame_num = frame_num

        # Run matching cascade.  # 对可见光、红外分别进行轨迹与检测目标的级联匹配
        visible_matches, visible_unmatched_tracks, visible_unmatched_detections = self._match_v1(visible_detections,
                                                                                              'visible')
        infrared_matches, infrared_unmatched_tracks, infrared_unmatched_detections = self._match_v1(infrared_detections,
                                                                                                 'infrared')

        # Update track set.
        # #1.1、匹配更新视觉、位置特征
        # 更新det-traj匹配成功的轨迹集合：运行update_feat，更新一个位置特例、更新视觉特征，不更新实际bbox
        for track_idx, detection_idx in visible_matches:
            self.visible_tracks[track_idx].update_feat(visible_detections[detection_idx])
        for track_idx, detection_idx in infrared_matches:
            self.infrared_tracks[track_idx].update_feat(infrared_detections[detection_idx])

        # #1.2、未匹配目标生成新生轨迹
        for visible_detection_idx in visible_unmatched_detections:
            self._initiate_track(visible_detections[visible_detection_idx], modality='visible')
            self.single_visible_ids.append(self._next_id - 1)
        for infrared_detection_idx in infrared_unmatched_detections:
            self._initiate_track(infrared_detections[infrared_detection_idx], modality='infrared')
            self.single_infrared_ids.append(self._next_id - 1)

        # #2、轨迹间匹配:输入：两模态各自的单模态轨迹；过程：删除匹配成功的单模态轨迹，增加匹配的跨模态轨迹，输出：类成员中的仨列表
        self.crossmodality_match()

        # #3、未匹配轨迹管理
        # 3.1：单模态未匹配轨迹，标记失踪
        for track_idx in visible_unmatched_tracks:
            if self.visible_tracks[track_idx].id in self.single_visible_ids:
                self.visible_tracks[track_idx].mark_missed()
        for track_idx in infrared_unmatched_tracks:
            if self.infrared_tracks[track_idx].id in self.single_infrared_ids:
                self.infrared_tracks[track_idx].mark_missed()

        # 3.2：跨模态轨迹对-未匹配轨迹：一个未检出，忽略;
        for visible_idx, infrared_idx in self.paired_crossmodel_ids:
            pair_vi_idx = [i for i, t in enumerate(self.visible_tracks) if t.id == visible_idx][0]
            pair_ir_idx = [i for i, t in enumerate(self.infrared_tracks) if t.id == infrared_idx][0]
            if (pair_vi_idx in visible_unmatched_tracks) and (pair_ir_idx in infrared_unmatched_tracks):
                self.visible_tracks[pair_vi_idx].mark_missed()
                self.infrared_tracks[pair_ir_idx].mark_missed()

        # #4、对单模态、跨模态轨迹滤波更新，并且更新距离度量？
        # 模块输入：单模态轨迹ids *2 跨模态轨迹对ids*1
        # 4.1
        for track_idx, detection_idx in visible_matches:  # 更新匹配成功的轨迹集合：对单模态轨迹ids *2分别运行 update+partial_fit
            if self.visible_tracks[track_idx].id in self.single_visible_ids:
                self.visible_tracks[track_idx].update_pos_and_state(visible_detections[detection_idx])
        for track_idx, detection_idx in infrared_matches:
            if self.infrared_tracks[track_idx].id in self.single_infrared_ids:
                self.infrared_tracks[track_idx].update_pos_and_state(infrared_detections[detection_idx])

        # 4.2 更新匹配成功的轨迹集合：跨模态轨迹对ids*1 运行update+partial_fit
        for visible_track_idx, infrared_track_idx in self.paired_crossmodel_ids:  # 轨迹id，列表位置id
            # 无检测
            visible_track_ = self.find_visible_track(visible_track_idx)
            infrared_track_ = self.find_infrared_track(infrared_track_idx)
            visible_unmatched_tracks_idx = [t.id for i, t in enumerate(self.visible_tracks) if
                                            i in visible_unmatched_tracks]
            infrared_unmatched_tracks_idx = [t.id for i, t in enumerate(self.infrared_tracks) if
                                             i in infrared_unmatched_tracks]

            if (visible_track_idx in visible_unmatched_tracks_idx) and (
                    infrared_track_idx in infrared_unmatched_tracks_idx):
                self.update_fkf_miss2(visible_track_, infrared_track_)

            # 单模态有检测
            elif (visible_track_idx in visible_unmatched_tracks_idx) and (
                    infrared_track_idx not in infrared_unmatched_tracks_idx):
                for t in infrared_matches:
                    if self.infrared_tracks[t[0]].id == infrared_track_idx:
                        infrared_det_ = infrared_detections[t[1]]
                        self.update_fkf_miss_visible(visible_track_, infrared_track_, infrared_det_)

            elif (visible_track_idx not in visible_unmatched_tracks_idx) and (
                    infrared_track_idx in infrared_unmatched_tracks_idx):
                for t in visible_matches:
                    if self.visible_tracks[t[0]].id == visible_track_idx:
                        visible_det_ = visible_detections[t[1]]
                        self.update_fkf_miss_infrared(visible_track_, infrared_track_, visible_det_)

            # 双模态均有检测
            else:
                for t in visible_matches:  # t[0]:轨迹列表中的索引，非轨迹id
                    # print("visible_pairs:", self.visible_tracks[t[0]].id, visible_track_idx)
                    if self.visible_tracks[t[0]].id == visible_track_idx:
                        # visible_track_ = self.visible_tracks[t[0]]
                        visible_det_ = visible_detections[t[1]]
                        break

                for t in infrared_matches:
                    # print(self.infrared_tracks[t[0]].id, infrared_track_idx)
                    if self.infrared_tracks[t[0]].id == infrared_track_idx:
                        # infrared_track_ = self.infrared_tracks[t[0]]
                        infrared_det_ = infrared_detections[t[1]]
                        break

                self.update_fkf(
                    visible_track_,
                    visible_det_,
                    infrared_track_,
                    infrared_det_
                )
                # visible_track_.update_pos_and_state(visible_det_)
                # infrared_track_.update_pos_and_state(infrared_det_)
                visible_track_.update_pos_and_state([])
                infrared_track_.update_pos_and_state([])

        # self.delete_time_adjust()
        if self.pose_only:
            self.delete_bad_track_pairs()
        else:
            self.delete_bad_track_pairs('pos_deep_time')
        self.soft_nms()
        # 5.Update distance metric. necessary! since original procedure should be maintained

        active_visible_targets = [t.id for t in self.visible_tracks if t.is_confirmed()]
        active_infrared_targets = [t.id for t in self.infrared_tracks if t.is_confirmed()]
        active_targets = active_visible_targets + active_infrared_targets
        features, targets = [], []
        for track in self.visible_tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        for track in self.infrared_tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]

        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )
        print(
            f"visible_trackers:{active_visible_targets} infrared_trackers:{active_infrared_targets} \npairs:{self.paired_crossmodel_ids}")
        print(f'single_vi&ir:{self.single_visible_ids}', self.single_infrared_ids)
        print(
            f'v-time-since-update{[f.time_since_update for f in self.visible_tracks if f.id in active_visible_targets]}')
        print(
            f'v-pair-time-since-update{[f.pair_time_since_update for f in self.visible_tracks if f.id in active_visible_targets]}')

        print(
            f'i-time-since-update{[f.time_since_update for f in self.infrared_tracks if f.id in active_infrared_targets]}')
        print(
            f'i-pair-time-since-update{[f.pair_time_since_update for f in self.infrared_tracks if f.id in active_infrared_targets]}')

    def _match(self, detections, modality):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])  # feat:det.feat
            targets = np.array([tracks[i].id for i in track_indices])  # targets:track.id
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )

            return cost_matrix

        if modality == 'visible':
            # Split track set into confirmed and unconfirmed tracks.
            confirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if not t.is_confirmed()]

            confirmed_paired_tracks = [i for i, t in enumerate(self.visible_tracks) if t.is_confirmed()and t.id not in self.single_visible_ids]
            confirmed_single_tracks = [i for i, t in enumerate(self.visible_tracks) if t.is_confirmed()and t.id in self.single_visible_ids]
            #
            # # Associate confirmed tracks using appearance features.  第一级匹配，使用外观特征
            # matches_a_, unmatched_tracks_a_, unmatched_detections_ = linear_assignment.matching_cascade(
            #     gated_metric,
            #     self.metric.matching_threshold,
            #     self.max_age,
            #     self.visible_tracks,
            #     detections,
            #     confirmed_paired_tracks,
            # )
            # #
            # unmatched_tracks_a = unmatched_tracks_a_ + confirmed_single_tracks

            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.visible_tracks,
                detections,
                confirmed_tracks,
                # unmatched_tracks_a,
                # unmatched_detections_
            )

            # Associate remaining tracks together with unconfirmed tracks using IOU.  第二级匹配，使用交并比
            # 备选tracks：unconfirm（检测过少，未形成）+第一轮unmatch中上一frame更新仅一轮的轨迹
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.visible_tracks[k].time_since_update == 1
            ]  # 确定无缘的轨迹：第一轮unmatch且之前frame已经unmatch的轨迹
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if self.visible_tracks[k].time_since_update != 1
            ]

            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.visible_tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )

            matches = matches_a + matches_b# + matches_a_
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            return matches, unmatched_tracks, unmatched_detections

        elif modality == 'infrared':
            # Split track set into confirmed and unconfirmed tracks.
            confirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if not t.is_confirmed()]

            # Associate confirmed tracks using appearance features.  第一级匹配，使用外观特征
            # confirmed_paired_tracks = [i for i, t in enumerate(self.infrared_tracks) if
            #                            t.is_confirmed() and t.id not in self.single_infrared_ids]
            # confirmed_single_tracks = [i for i, t in enumerate(self.infrared_tracks) if
            #                            t.is_confirmed() and t.id in self.single_infrared_ids]
            #
            # # Associate confirmed tracks using appearance features.  第一级匹配，使用外观特征
            # matches_a_, unmatched_tracks_a_, unmatched_detections_ = linear_assignment.matching_cascade(
            #     gated_metric,
            #     self.metric.matching_threshold,
            #     self.max_age,
            #     self.infrared_tracks,
            #     detections,
            #     confirmed_paired_tracks,
            # )
            # unmatched_tracks_a = unmatched_tracks_a_ + confirmed_single_tracks
            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.infrared_tracks,
                detections,
                confirmed_tracks,
                # unmatched_tracks_a,
                # unmatched_detections_
            )
            # Associate remaining tracks together with unconfirmed tracks using IOU.  第二级匹配，使用交并比
            # 备选tracks：unconfirm（检测过少，未形成）+第一轮unmatch中上一frame更新仅一轮的轨迹
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.infrared_tracks[k].time_since_update == 1
            ]  # 确定无缘的轨迹：第一轮unmatch且之前frame已经unmatch的轨迹
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if self.infrared_tracks[k].time_since_update != 1
            ]

            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.infrared_tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )

            matches = matches_a + matches_b# + matches_a_
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            return matches, unmatched_tracks, unmatched_detections

    def _match_v1(self, detections, modality):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])  # feat:det.feat
            targets = np.array([tracks[i].id for i in track_indices])  # targets:track.id
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )

            return cost_matrix

        if modality == 'visible':
            # Split track set into confirmed and unconfirmed tracks.
            confirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if not t.is_confirmed()]

            confirmed_paired_tracks = [i for i, t in enumerate(self.visible_tracks) if
                                       t.is_confirmed()
                                       and t.id not in self.single_visible_ids]
                                       # and t.pair_time_since_update <= 2]
            confirmed_single_tracks = [i for i, t in enumerate(self.visible_tracks) if
                                       t.is_confirmed() and
                                       t.id in self.single_visible_ids]
                                       # or t.pair_time_since_update > 2)]
            #
            # # Associate confirmed tracks using appearance features.  第一级匹配，使用外观特征
            matches_a_, unmatched_tracks_a_, unmatched_detections_ = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold+0.02,
                self.max_age,
                self.visible_tracks,
                detections,
                confirmed_paired_tracks,
            )
            #
            unmatched_tracks_a = unmatched_tracks_a_ + confirmed_single_tracks

            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.visible_tracks,
                detections,
                unmatched_tracks_a,
                unmatched_detections_
            )

            # Associate remaining tracks together with unconfirmed tracks using IOU.  第二级匹配，使用交并比
            # 备选tracks：unconfirm（检测过少，未形成）+第一轮unmatch中上一frame更新仅一轮的轨迹
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.visible_tracks[k].time_since_update == 1
            ]  # 确定无缘的轨迹：第一轮unmatch且之前frame已经unmatch的轨迹
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if self.visible_tracks[k].time_since_update != 1
            ]

            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.visible_tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )

            matches = matches_a + matches_b + matches_a_
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            return matches, unmatched_tracks, unmatched_detections

        elif modality == 'infrared':
            # Split track set into confirmed and unconfirmed tracks.
            confirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if not t.is_confirmed()]

            # Associate confirmed tracks using appearance features.  第一级匹配，使用外观特征
            confirmed_paired_tracks = [i for i, t in enumerate(self.infrared_tracks) if
                                       t.is_confirmed()
                                       and t.id not in self.single_infrared_ids]
                                       # and t.pair_time_since_update <= 2]
            confirmed_single_tracks = [i for i, t in enumerate(self.infrared_tracks) if
                                       t.is_confirmed() and
                                       t.id in self.single_infrared_ids]
                                        # or t.pair_time_since_update > 2)]

            # Associate confirmed tracks using appearance features.  第一级匹配，使用外观特征
            matches_a_, unmatched_tracks_a_, unmatched_detections_ = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold+0.02,
                self.max_age,
                self.infrared_tracks,
                detections,
                confirmed_paired_tracks,
            )
            unmatched_tracks_a = unmatched_tracks_a_ + confirmed_single_tracks
            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.infrared_tracks,
                detections,
                unmatched_tracks_a,
                unmatched_detections_
            )
            # Associate remaining tracks together with unconfirmed tracks using IOU.  第二级匹配，使用交并比
            # 备选tracks：unconfirm（检测过少，未形成）+第一轮unmatch中上一frame更新仅一轮的轨迹
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.infrared_tracks[k].time_since_update == 1
            ]  # 确定无缘的轨迹：第一轮unmatch且之前frame已经unmatch的轨迹
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if self.infrared_tracks[k].time_since_update != 1
            ]

            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.infrared_tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )

            matches = matches_a + matches_b + matches_a_
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, modality):
        if modality == 'visible':
            self.visible_tracks.append(
                Track(
                    detection,
                    self._next_id,
                    modality,
                    self.n_init,
                    self.max_age,
                    self.ema_alpha,
                    self.conf_ema_alpha
                )
            )
            self._next_id += 1
        elif modality == 'infrared':
            self.infrared_tracks.append(
                Track(
                    detection,
                    self._next_id,
                    modality,
                    self.n_init,
                    self.max_age,
                    self.ema_alpha,
                    self.conf_ema_alpha
                )
            )
            self._next_id += 1

    def crossmodality_match(self):
        # 提取可见光与红外的单模态轨迹
        confirmed_visible_tracks = [t.id for t in self.visible_tracks if
                                    t.is_confirmed() and t.id in self.single_visible_ids]
        confirmed_infrared_tracks = [t.id for t in self.infrared_tracks if
                                     t.is_confirmed() and t.id in self.single_infrared_ids]
        unconfirmed_visible_tracks = [t.id for t in self.visible_tracks if t.is_confirmed()]
        unconfirmed_infrared_tracks = [t.id for t in self.infrared_tracks if t.is_confirmed()]
        all_visible_tracks = [t.id for t in self.visible_tracks if t.is_confirmed() and t.time_since_update <= 2]
        all_infrared_tracks = [t.id for t in self.infrared_tracks if t.is_confirmed() and t.time_since_update <= 2]

        # 输出管理1：删除不合理的已匹配轨迹对！！！！！！！

        # 第一层：视觉匹配 ：
        # 筛选需要匹配的轨迹集合，计算相似度矩阵，计算匈牙利匹配，整理输出轨迹集合
        # if not self.pose_only:
        #     matched_track_pairs_a, unmatched_visible_tracks_a, unmatched_infrared_tracks_a \
        #         = self._crossmodality_match(
        #         confirmed_visible_tracks,
        #         confirmed_infrared_tracks,
        #         _nn_cosine_distance,
        #         self.deep_track_dist,
        #         feat="deep"
        #     )

        matched_track_pairs_b, _, _ = self._crossmodality_match(
            all_visible_tracks,
            all_infrared_tracks,
            _nn_iou_distance,
            self.pos_track_dist,
            all_visible_tracks,
            all_infrared_tracks,
            "pos"
        )
        matches = matched_track_pairs_b

        # deep feat update
        if not self.pose_only:
            for m in matches:
                f_vi = self.find_visible_track(m[0]).share_modality_features[-1]
                f_ir = self.find_infrared_track(m[1]).share_modality_features[-1]
                # d = _nn_cosine_distance(f_vi, f_ir)
                d = 1.0 - np.dot(f_vi, f_ir.T)
                if d > self.deep_track_dist:
                    matches.remove(m)

        # 输出管理2：增加匹配轨迹对，删除已匹配轨迹,！！！！！！！！！！！！！！！！可优化
        vi_in_matches = [t[0] for t in matches]
        ir_in_matches = [t[1] for t in matches]
        p_ids = copy.deepcopy(self.paired_crossmodel_ids)
        remove_pairs = []
        # for m in matches:
        #     if m[0] in [p[0] for p in self.paired_crossmodel_ids]:
        #         remove_pairs.append(p_ids[[p[0] for p in self.paired_crossmodel_ids].index(m[0])])
        #         continue
        #     elif m[1] in [p[1] for p in self.paired_crossmodel_ids]:
        #         remove_pairs.append(p_ids[[p[1] for p in self.paired_crossmodel_ids].index(m[1])])

        # for m in remove_pairs:
            # self.paired_crossmodel_ids.remove(m)

        for m in p_ids:
            if m[0] in vi_in_matches:
                remove_pairs.append(matches[vi_in_matches.index(m[0])])
                continue
            elif m[1] in ir_in_matches:
                remove_pairs.append(matches[ir_in_matches.index(m[1])])

        remove_pairs = list(set(remove_pairs))
        # deleted = []
        for m in remove_pairs:
            matches.remove(m)
            # if m in deleted:
            #     continue
            # if (m[0] in [i[0] for i in p_ids]) and (m[1] not in [i[1] for i in p_ids]):
            #     # ir发生切换
            #     id1 = m[1]
            #     id2_ = [i[0] for i in p_ids].index(m[0])
            #     id2 = p_ids[id2_][1]
            #     self.replace_infrared_tracks(id1, id2)
            # elif (m[0] not in [i[0] for i in p_ids]) and (m[1] in [i[1] for i in p_ids]):
            #     # vi发生切换
            #     id1 = m[0]
            #     id2_ = [i[1] for i in p_ids].index(m[1])
            #     id2 = p_ids[id2_][0]
            #     self.replace_visible_tracks(id1, id2)
            #
            # elif (m[0] in [i[0] for i in p_ids]) and (m[1] in [i[1] for i in p_ids]):
            #     # 发生轨迹互换：信置信度更大的模态：
            #     # 置信度衡量：time since update？
            #     v_id1=m[0]
            #     v_id2_ = [i[1] for i in p_ids].index(m[1])
            #     v_id2 = p_ids[v_id2_][0]
            #     i_id1 = m[1]
            #     i_id2_ = [i[0] for i in p_ids].index(m[0])
            #     i_id2 = p_ids[i_id2_][1]
            #
            #     v_t1, v_t2 = self.find_visible_track(v_id1), self.find_visible_track(v_id2)
            #     i_t1, i_t2 = self.find_infrared_track(i_id1), self.find_infrared_track(i_id2)
            #     if v_t1.time_since_update +v_t2.time_since_update>i_t1.time_since_update+i_t1.time_since_update:
            #         self.replace_visible_tracks(v_id1, v_id2)
            #     else:
            #         self.replace_infrared_tracks(i_id1, i_id2)
            #
            #     if (v_id2, i_id2) in remove_pairs:
            #         deleted.append((v_id2, i_id2))

            # self.single_visible_ids.append(m[0])
            # self.single_visible_ids = list(set(self.single_visible_ids))
            # self.single_infrared_ids.append(m[1])
            # self.single_infrared_ids = list(set(self.single_infrared_ids))
            # self.paired_crossmodel_ids.remove(m)

        #     if (m_old[0] in vi_in_matches) and (m_old[1] not in ir_in_matches):
        #         # new track id换成 old id， old track 删除，
        #         m_new = matches[vi_in_matches==m_old[0]]
        #         matches.remove(m_new)
        #         # self.find_infrared_track(m_old[1]).state=TrackState.Deleted
        #         # self.find_infrared_track(m_new[1]).id = m_old[1]
        #     elif (m_old[0] not in vi_in_matches) and (m_old[1] in ir_in_matches):
        #         m_new = matches[ir_in_matches == m_old[1]]
        #         matches.remove(m_new)

        # self.find_visible_track(m_old[0]).state = TrackState.Deleted
        # self.find_visible_track(m_new[0]).id = m_old[0]

            # if m not in self.paired_crossmodel_ids:
            #     if m[0] in vi_in_matches:
            #         self.find_infrared_track(m[1]).time_since_update += 1
            #     elif m[1] in ir_in_matches:
            #         self.find_visible_track(m[0]).time_since_update += 1

        self.paired_crossmodel_ids = list(set(matches + self.paired_crossmodel_ids))
        # print('after new match', self.paired_crossmodel_ids)

        for i in self.paired_crossmodel_ids:
            if i[0] in self.single_visible_ids:
                self.single_visible_ids.remove(i[0])
            if i[1] in self.single_infrared_ids:
                self.single_infrared_ids.remove(i[1])
        return

    def _crossmodality_match(self, visible_id, infrared_id, metric_function, distance_thres, all_visible_id=None,
                             all_infrared_id=None, feat="deep"):
        # 第一层：视觉相似度匹配：距离计算，遍历形成相似度矩阵；匈牙利匹配
        if feat == "deep":
            visible_features = [t.share_modality_features[0] for t in self.visible_tracks if
                                t.is_confirmed() and (t.id in visible_id)]
            infrared_features = [t.share_modality_features[0] for t in self.infrared_tracks if
                                 t.is_confirmed() and (t.id in infrared_id)]
            distance_thres = distance_thres * self.deep_track_rate
        elif feat == "pos":
            visible_features = [t.to_xywh() for t in self.visible_tracks if
                                t.is_confirmed() and (t.id in visible_id)]  # m*4,xywh
            infrared_features = [t.to_xywh() for t in self.infrared_tracks if
                                 t.is_confirmed() and (t.id in infrared_id)]  # n*4,xywh
            all_visible_features = [t.to_xywh() for t in self.visible_tracks if
                                    t.is_confirmed() and (t.id in all_visible_id)]  # m*4,xywh
            all_infrared_features = [t.to_xywh() for t in self.infrared_tracks if
                                     t.is_confirmed() and (t.id in all_infrared_id)]
            all_visible_features_ = copy.deepcopy(np.array(all_visible_features))
            all_infrared_features = np.array(all_infrared_features)

        if len(visible_features) == 0 or len(infrared_features) == 0:
            return [], visible_id, infrared_id  # Nothing to match.

        visible_features_ = copy.deepcopy(np.array(visible_features))
        infrared_features = np.array(infrared_features)

        if feat == 'pos':  # adjust pos based on global bias/icp algorithm
            if np.mod(self.frame_num, 10) == 0 or self.paired_bias_set == []:
                self.bias_score_ema()
                pose, score = self.ps_bbox_translation(all_visible_features_, all_infrared_features)
                if self.adaptive_pose_thres:
                    self.pos_track_dist = max(score * self.adaptive_pose_thres[0] + self.adaptive_pose_thres[1],0.4)
                    self.pair_delete_pos_thres = max(score * self.adaptive_pose_thres[2] + self.adaptive_pose_thres[3], 0.5)
            else:
                pose = self.best_bias()

            visible_features_ = self.bias_adjust(visible_features_, pose)

        cost_matrix = self.track_feature_distance(visible_features_, infrared_features, metric_function)
        cost_matrix[cost_matrix > distance_thres] = distance_thres + 1e-5
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        pairs, unpaired_visible, unpaired_infrared = [], [], []
        for col, visible_idx in enumerate(visible_id):
            if col not in col_indices:
                unpaired_visible.append(visible_idx)
        for row, infrared_idx in enumerate(infrared_id):
            if row not in row_indices:
                unpaired_infrared.append(infrared_idx)
        for row, col in zip(row_indices, col_indices):
            infrared_idx = infrared_id[col]
            visible_idx = visible_id[row]
            if cost_matrix[row, col] > distance_thres:  # 待定阈值！！！距离比待定阈值还远，直接不匹配
                unpaired_visible.append(visible_idx)
                unpaired_infrared.append(infrared_idx)
            else:
                pairs.append((visible_idx, infrared_idx))

        return pairs, unpaired_visible, unpaired_infrared

    def delete_time_adjust(self):
        def iou(bbox1, bbox2):
            source_xyxy = np.hstack((bbox1[:2] - bbox1[2:] / 2, bbox1[:2] + bbox1[2:] / 2))
            target_xyxy = np.hstack((bbox2[:2] - bbox2[2:] / 2, bbox2[:2] + bbox2[2:] / 2))
            boxes1 = torch.tensor([source_xyxy], dtype=torch.float)
            boxes2 = torch.tensor([target_xyxy], dtype=torch.float)
            return box_iou(boxes1, boxes2).numpy()

        all_visible_tracks = [t for t in self.visible_tracks if t.is_confirmed() and t.time_since_update < 1]
        all_infrared_tracks = [t for t in self.infrared_tracks if t.is_confirmed() and t.time_since_update < 1]
        all_iou = []
        for i, t1 in enumerate(all_visible_tracks):
            for j, t2 in enumerate(all_visible_tracks[i + 1:]):
                all_iou.append(iou(t1.to_xywh(), t2.to_xywh())[0][0])

        for i, t1 in enumerate(all_infrared_tracks):
            for j, t2 in enumerate(all_infrared_tracks[i + 1:]):
                all_iou.append(iou(t1.to_xywh(), t2.to_xywh())[0][0])
        self.pair_delete_time_thres = max(10 - np.sum(all_iou) * 50, 6)
        print("delete time", self.pair_delete_time_thres, np.mean(all_iou))

    def delete_time_adjust_iou(self, pairs):
        vi_bbox = self.find_visible_track(pairs[0]).to_xywh()
        ir_bbox = self.find_infrared_track(pairs[1]).to_xywh()

        all_vi_bboxes = [t.to_xywh() for t in self.visible_tracks if t.is_confirmed() and t.time_since_update < 1]
        all_ir_bboxes = [t.to_xywh() for t in self.infrared_tracks if t.is_confirmed() and t.time_since_update < 1]
        all_iou = []
        vi_iou = 1 - _nn_iou_distance(np.array([vi_bbox]), np.array(all_vi_bboxes))
        vi_iou[vi_iou == 1.0] = -1
        ir_iou = 1 - _nn_iou_distance(np.array([ir_bbox]), np.array(all_ir_bboxes))
        ir_iou[ir_iou == 1.0] = -1
        if vi_iou != [[]]:
            all_iou.append(max(vi_iou))
        else:
            all_iou.append(-1)
        if ir_iou != [[]]:
            all_iou.append(max(ir_iou))
        else:
            all_iou.append(-1)
        return max(max(all_iou), 0)

    def delete_bad_track_pairs(self, feat='pos-time'):
        # 遍历所有已匹配轨迹。满足一定标准，删除轨迹匹配关系
        # 该函数仅基于视觉特征实现。
        paired_crossmodel_ids = copy.deepcopy(self.paired_crossmodel_ids)
        for t_pairs_ in paired_crossmodel_ids:
            t_pairs = copy.deepcopy(t_pairs_)
            vi_feat = copy.deepcopy(self.find_visible_track(t_pairs[0]).to_xywh())
            ir_feat = self.find_infrared_track(t_pairs[1]).to_xywh()
            bias = self.best_bias()
            vi_feat[0] = vi_feat[0] * bias[2] + bias[0]
            vi_feat[1] = vi_feat[1] * bias[3] + bias[1]
            vi_feat[2] = vi_feat[2] * bias[2]
            vi_feat[3] = vi_feat[3] * bias[3]
            source_xyxy = np.hstack((vi_feat[:2] - vi_feat[2:] / 2, vi_feat[:2] + vi_feat[2:] / 2))
            target_xyxy = np.hstack((ir_feat[:2] - ir_feat[2:] / 2, ir_feat[:2] + ir_feat[2:] / 2))
            boxes1 = torch.tensor([source_xyxy], dtype=torch.float)
            boxes2 = torch.tensor([target_xyxy], dtype=torch.float)
            distances = 1 - box_iou(boxes1, boxes2).numpy()  # smaller better
            fix = self.delete_time_adjust_iou(t_pairs)
            if feat == 'pos':
                if distances > self.pair_delete_pos_thres:  # self.pos_track_dist+0.2:
                    # print(t_pairs, distances, "delete pairs!")
                    self.paired_crossmodel_ids.remove(t_pairs)
                    self.single_visible_ids.append(t_pairs[0])
                    self.single_infrared_ids.append(t_pairs[1])
            elif feat == 'pos-time':
                vi_time_since_update = self.find_visible_track(t_pairs[0]).pair_time_since_update
                ir_time_since_update = self.find_infrared_track(t_pairs[1]).pair_time_since_update
                fixed_time_thres = max(self.pair_delete_time_thres_max - fix * 30, self.pair_delete_time_thres_min)
                print('fixed_thres:', fixed_time_thres)
                if vi_time_since_update + ir_time_since_update > fixed_time_thres:
                    self.find_visible_track(t_pairs[0]).time_since_update += 1
                    self.find_infrared_track(t_pairs[1]).time_since_update += 1

                if distances > self.pair_delete_pos_thres \
                        or ir_time_since_update + vi_time_since_update > fixed_time_thres:
                    self.paired_crossmodel_ids.remove(t_pairs)
                    self.single_visible_ids.append(t_pairs[0])
                    self.single_infrared_ids.append(t_pairs[1])
                # if self.frame_num >= 177:
                #     # print('pair', t_pairs[1], ir_time_since_update, self.find_infrared_track(t_pairs[1]).time_since_update)
                #     print('7', self.infrared_tracks[1].pair_time_since_update, self.infrared_tracks[1].time_since_update)

            elif feat == 'pos_deep_time':
                vi_time_since_update = self.find_visible_track(t_pairs[0]).pair_time_since_update
                ir_time_since_update = self.find_infrared_track(t_pairs[1]).pair_time_since_update
                f_vi = self.find_visible_track(t_pairs[0]).share_modality_features[-1]
                f_ir = self.find_infrared_track(t_pairs[1]).share_modality_features[-1]
                deep_dist = 1.0 - np.dot(f_vi, f_ir.T)

                if vi_time_since_update > self.pair_delete_time_thres:
                    self.find_visible_track(t_pairs[0]).time_since_update += 1
                if ir_time_since_update > self.pair_delete_time_thres:
                    self.find_infrared_track(t_pairs[1]).time_since_update += 1

                if distances > self.pair_delete_pos_thres \
                        or ir_time_since_update > self.pair_delete_time_thres \
                        or vi_time_since_update > self.pair_delete_time_thres \
                        or deep_dist > self.pair_delete_deep_thres:
                    # print(t_pairs, distances, "delete pairs!")
                    self.paired_crossmodel_ids.remove(t_pairs)
                    self.single_visible_ids.append(t_pairs[0])
                    self.single_infrared_ids.append(t_pairs[1])
        # print('finish pairs delete', self.paired_crossmodel_ids)

    def soft_nms(self):
        def iou(bbox1, bbox2):
            source_xyxy = np.hstack((bbox1[:2] - bbox1[2:] / 2, bbox1[:2] + bbox1[2:] / 2))
            target_xyxy = np.hstack((bbox2[:2] - bbox2[2:] / 2, bbox2[:2] + bbox2[2:] / 2))
            boxes1 = torch.tensor([source_xyxy], dtype=torch.float)
            boxes2 = torch.tensor([target_xyxy], dtype=torch.float)
            return box_iou(boxes1, boxes2).numpy()

        def nms(t1, t2, paired_ids):
            # step2: paired time: newer updated. Believe track that fresher.
            if t1.pair_time_since_update > t2.pair_time_since_update:
                t1.time_since_update += 1
                return
            elif t1.pair_time_since_update < t2.pair_time_since_update:
                t2.time_since_update += 1
                return
            # # step3: hit more
            if t1.hits > t2.hits:
                t2.time_since_update += 1
                return
            elif t2.hits > t1.hits:
                t1.time_since_update += 1
                return
            # step4: score:remain higher
            if t1.conf_ema > t2.conf_ema:
                t1.time_since_update += 1
                return
            else:
                t2.time_since_update += 1
                return

        all_visible_tracks = [t for t in self.visible_tracks if t.is_confirmed() and t.time_since_update < 1]
        all_infrared_tracks = [t for t in self.infrared_tracks if t.is_confirmed() and t.time_since_update < 1]
        paired_visible_ids = [t[0] for t in self.paired_crossmodel_ids]
        paired_infrared_ids = [t[1] for t in self.paired_crossmodel_ids]

        for i, t1 in enumerate(all_visible_tracks):
            for j, t2 in enumerate(all_visible_tracks[i + 1:]):
                t_iou = iou(t1.to_xywh(), t2.to_xywh())
                if t_iou > self.soft_nms_thres:
                    nms(t1, t2, paired_visible_ids)

        for i, t1 in enumerate(all_infrared_tracks):
            for j, t2 in enumerate(all_infrared_tracks[i + 1:]):
                t_iou = iou(t1.to_xywh(), t2.to_xywh())
                if t_iou > self.soft_nms_thres:  # execute nms:
                    nms(t1, t2, paired_infrared_ids)
        return

    def track_feature_distance(self, visible_feats, infrared_feats, metric_funcion):
        cost_matrix = np.zeros((len(visible_feats), len(infrared_feats)))
        for i, visible_feat in enumerate(visible_feats):
            cost_matrix[i, :] = metric_funcion([visible_feat], infrared_feats)
        return cost_matrix

    def update_fkf(self, visible_track_, visible_det, infrared_track_, infrared_det):
        # visible_track_ = self.visible_tracks[visible_track_idx]
        # visible_track_.bbox = visible_det.to_xyah()
        # visible_track_.conf = visible_det.conf
        # visible_track_.cls = visible_det.cls
        # visible_track_.det_ind = visible_det.det_ind

        # infrared_track_ = self.infrared_tracks[infrared_track_idx]
        # infrared_track_.bbox = infrared_det.to_xyah()
        # infrared_track_.conf = infrared_det.conf?????????????????????
        # infrared_track_.cls = infrared_det.cls
        # infrared_track_.det_ind = infrared_det.det_ind

        # 执行fkf更新:
        # 输入：主滤波器（self），子滤波器(均值、协方差，估计置信度?（用于分配总协方差）)
        # 流程：子滤波器估计结果，
        # 1、取vel，加权平均得到融合速度；
        # 2、利用速度更新子滤波器状态；
        # 3、计算、分配子滤波器协方差；
        # 输出：主滤波器（self），融合后子滤波器均值、协方差.
        # 在完成融合计算后，直接将融合滤波器结果更新子滤波器均值与方差

        self._update_fkf(visible_track_, infrared_track_, FKFMode.both)

        # 更新指数移动平均：视觉特征、其他轨迹信息
        visible_track_.time_since_update = 0
        visible_track_.pair_time_since_update = 0
        if visible_track_.state == TrackState.Tentative and visible_track_.hits >= visible_track_._n_init:
            visible_track_.state = TrackState.Confirmed

        # infrared features smooth update
        infrared_track_.time_since_update = 0
        infrared_track_.pair_time_since_update = 0
        if infrared_track_.state == TrackState.Tentative and infrared_track_.hits >= infrared_track_._n_init:
            infrared_track_.state = TrackState.Confirmed

    def update_fkf_miss2(self, visible_track_, infrared_track_):
        visible_track_.pair_time_since_update += 1
        infrared_track_.pair_time_since_update += 1
        self._update_fkf(visible_track_, infrared_track_, FKFMode.miss2)

    def update_fkf_miss_visible(self, visible_track_, infrared_track_, infrared_det):
        # infrared_track_.conf = infrared_det.conf
        self._update_fkf(visible_track_, infrared_track_, FKFMode.miss_vi)
        # 更新指数移动平均：视觉特征、其他轨迹信息
        visible_track_.time_since_update = 0
        visible_track_.pair_time_since_update += 1
        if visible_track_.state == TrackState.Tentative and visible_track_.hits >= visible_track_._n_init:
            visible_track_.state = TrackState.Confirmed

        # infrared features smooth update
        infrared_track_.time_since_update = 0
        infrared_track_.pair_time_since_update = 0
        if infrared_track_.state == TrackState.Tentative and infrared_track_.hits >= infrared_track_._n_init:
            infrared_track_.state = TrackState.Confirmed

    def update_fkf_miss_infrared(self, visible_track_, infrared_track_, visible_det):
        # visible_track_.conf = visible_det.conf
        self._update_fkf(visible_track_, infrared_track_, FKFMode.miss_ir)
        # 更新指数移动平均：视觉特征、其他轨迹信息
        visible_track_.time_since_update = 0
        visible_track_.pair_time_since_update = 0
        if visible_track_.state == TrackState.Tentative and visible_track_.hits >= visible_track_._n_init:
            visible_track_.state = TrackState.Confirmed

        # infrared features smooth update
        infrared_track_.time_since_update = 0
        infrared_track_.pair_time_since_update += 1
        if infrared_track_.state == TrackState.Tentative and infrared_track_.hits >= infrared_track_._n_init:
            infrared_track_.state = TrackState.Confirmed

    def _update_fkf(self, visible_track_, infrared_track_, mode=FKFMode.both):
        # 流程：子滤波器估计结果，
        # 1、取vel，加权平均得到融合速度；
        # 2、利用速度更新子滤波器状态；
        # 3、计算、分配子滤波器协方差；
        v_conf = copy.deepcopy(visible_track_.conf_ema)
        v_match_mean, v_match_covariance = copy.deepcopy(visible_track_.match_mean), copy.deepcopy(
            visible_track_.match_covariance)
        v_vel_mean = v_match_mean[4:8]
        # v_vel_cov = v_match_covariance[4:, 4:]

        i_conf = copy.deepcopy(infrared_track_.conf_ema)
        i_match_mean, i_match_covariance = copy.deepcopy(infrared_track_.match_mean), copy.deepcopy(
            infrared_track_.match_covariance)
        i_vel_mean = i_match_mean[4:]
        # i_vel_cov = i_match_covariance[4:, 4:]
        # v_conf_, i_conf_ = v_conf ** 1, i_conf ** 1

        if mode == FKFMode.both:
            v_conf_, i_conf_ = v_conf ** 10, i_conf ** 10
        elif mode == FKFMode.miss_vi:
            v_conf_, i_conf_ = 0, i_conf
        elif mode == FKFMode.miss_ir:
            v_conf_, i_conf_ = v_conf, 0
        elif mode == FKFMode.miss2:
            v_conf_, i_conf_ = v_conf, i_conf

        # 按照匹配度，融合子滤波器均值并更新。匹配度构成：级联匹配处的特征相似度
        fkf_vel = (i_conf_ * i_vel_mean + v_conf_ * v_vel_mean) / (i_conf_ + v_conf_)
        visible_track_.mean[0:4] = visible_track_.mean[0:4] + fkf_vel * self.dt
        visible_track_.mean[4:] = fkf_vel
        infrared_track_.mean[0:4] = infrared_track_.mean[0:4] + fkf_vel * self.dt
        infrared_track_.mean[4:] = fkf_vel

        # 联邦滤波器协方差融合、子滤波器分配。此处协方差矩阵为对角阵，求逆步骤可以直接跳过，
        # fkf_vel_cov = np.linalg.inv(v_vel_cov) + np.linalg.inv(i_vel_cov)
        # fkf_vel_cov = np.linalg.inv(fkf_vel_cov)
        # fkf_vel_cov = v_vel_cov + i_vel_cov
        # v_vel_cov = fkf_vel_cov * (v_conf / (i_conf + v_conf))
        # i_vel_cov = fkf_vel_cov * (i_conf / (i_conf + v_conf))
        # visible_track_.match_covariance[4:, 4:] = v_vel_cov
        # infrared_track_.match_covariance[4:, 4:] = i_vel_cov

        # 联邦滤波器协方差融合、子滤波器分配。
        fkf_cov = v_match_covariance + i_match_covariance
        v_vel_cov = fkf_cov * (i_conf / (i_conf + v_conf))
        i_vel_cov = fkf_cov * (v_conf / (i_conf + v_conf))
        # visible_track_.match_covariance = v_vel_cov
        # infrared_track_.match_covariance = i_vel_cov

        # fkf_cov = (i_conf + v_conf) * (v_match_covariance @ i_match_covariance) \
        #           @ np.linalg.inv(v_conf * i_match_covariance + i_conf*v_match_covariance)
        # fkf_cov = (v_conf * np.linalg.inv(v_match_covariance) + i_conf * np.linalg.inv(i_match_covariance)) / (i_conf + v_conf)
        # fkf_cov = np.linalg.inv(fkf_cov)
        # v_vel_cov = fkf_cov * (i_conf + v_conf) / v_conf
        # i_vel_cov = fkf_cov * (i_conf + v_conf) / i_conf
        visible_track_.match_covariance = v_vel_cov
        infrared_track_.match_covariance = i_vel_cov
        return

    def bias_adjust(self, adjust_features, pose):
        # bias = self.ransac_bias()
        # # a = np.array(adjust_features[:,0:2]) + np.array([bias[0:2]])
        adjust_features_ = np.zeros(shape=adjust_features.shape)
        adjust_features_[:, 0] = adjust_features[:, 0] * pose[2] + pose[0]
        adjust_features_[:, 1] = adjust_features[:, 1] * pose[3] + pose[1]
        adjust_features_[:, 2] = adjust_features[:, 2] * pose[2]
        adjust_features_[:, 3] = adjust_features[:, 3] * pose[3]
        return adjust_features_

    def sample_from_gaussians(self, num_samples, bounds, means, stds=np.array([10, 10, 0.1, 0.1])):
        all_samples = []
        # 遍历四个高斯分布的均值和标准差

        for mean, std in zip(means, stds):
            # 从当前的一维高斯分布中进行采样
            samples = np.random.normal(mean, std, num_samples)
            all_samples.append(samples)
        return np.clip(np.array(all_samples).T, bounds[0], bounds[1])

    def ransac_bias(self, num_iterations=10, distance_threshold=0.1, min_points=3):
        best_inliers = []
        best_center = None
        points = np.array([t[0] for t in self.paired_bias_set])
        # num_paired = np.array([t[1] for t in self.paired_bias_set])

        if len(points) <= min_points:
            return np.array([0, 0, 1, 1])
        num_iterations = num_iterations * len(points)

        # normalization
        points[0:2] /= 20

        for _ in range(num_iterations):
            # 随机选择一个点作为中心点的初始猜测
            sample = random.choice(points)
            # 计算所有点到该中心点的距离
            distances = np.linalg.norm(points - sample, axis=1)
            # 找出内点（距离小于阈值的点）
            inliers = points[distances < distance_threshold]
            # inlier_confidences = num_paired[distances < distance_threshold]

            # 如果当前内点数量多于之前的最佳内点数量，则更新最佳模型
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_center = np.mean(inliers, axis=0)

        best_center[0:2] *= 20
        return np.array(best_center)

    def best_bias(self):
        if not self.paired_bias_set:
            return [0, 0, 1, 1]
        else:
            points = np.array([t[0] for t in self.paired_bias_set])
            score = np.array([t[1] for t in self.paired_bias_set])
            return points[np.argmax(score)]

    def bias_score_ema(self):
        for t in self.paired_bias_set:
            t[1] = t[1] * self.bias_ema_alpha

    def find_visible_track(self, id):
        for t in self.visible_tracks:
            if t.id == id:
                return t

    def find_infrared_track(self, id):
        for t in self.infrared_tracks:
            if t.id == id:
                return t

    def replace_visible_tracks(self, id1, id2):
        t1 = self.find_visible_track(id1)
        t2 = self.find_visible_track(id2)
        t1_ = copy.deepcopy(t1)
        t2_ = copy.deepcopy(t2)
        t1 = t2_
        t1.id, t1.time_since_update, t1.pair_time_since_update = id1, t1_.time_since_update, t1_.pair_time_since_update
        t2 = t1_
        t2.id, t2.time_since_update, t2.pair_time_since_update = id2, t2_.time_since_update, t2_.pair_time_since_update


    def replace_infrared_tracks(self, id1, id2):
        t1 = self.find_infrared_track(id1)
        t2 = self.find_infrared_track(id2)
        t1_ = copy.deepcopy(t1)
        t2_ = copy.deepcopy(t2)
        t1 = t2_
        t1.id, t1.time_since_update, t1.pair_time_since_update = id1, t1_.time_since_update, t1_.pair_time_since_update
        t2 = t1_
        t2.id, t2.time_since_update, t2.pair_time_since_update = id2, t2_.time_since_update, t2_.pair_time_since_update

    def icp_2d_translation(self, source_, target_, max_iterations=100, tolerance=1e-6):

        def ransac_point_selection(points1, points2, num_iterations=100, distance_threshold=80):
            """
            使用 RANSAC 算法根据两组对应点之间的欧氏距离筛选点。

            参数:
            points1 (np.ndarray): 第一组点坐标，形状为 (N, 2) 或 (N, 3)
            points2 (np.ndarray): 第二组点坐标，形状为 (N, 2) 或 (N, 3)，与 points1 中的点一一对应
            num_iterations (int): RANSAC 算法的迭代次数
            distance_threshold (float): 欧氏距离阈值，用于判断点是否为内点

            返回:
            np.ndarray: 保留的点的索引数组
            """
            num_points = points1.shape[0]
            best_inliers = np.array([])
            point_bias = points1 - points2

            for _ in range(num_iterations):
                # 随机选择一个点
                random_index = np.random.randint(0, num_points)

                # 计算所有点到该随机点对应点的欧氏距离
                distances = np.linalg.norm(point_bias - point_bias[random_index], axis=1)

                # 根据距离阈值确定内点
                inliers = np.where(distances < distance_threshold)[0]

                # 如果当前内点数量多于之前的最佳内点数量，则更新最佳内点
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers

            return best_inliers

        scale_x = 1.0
        scale_y = 1.0
        translation = np.zeros(2)

        source = source_[:, :2]
        target = target_[:, :2]
        pre_mse = np.inf
        ori_source = copy.deepcopy(source)
        # ori_target = copy.deepcopy(target)

        if len(source) <= 2 or len(target) <= 2:
            return np.eye(3), 0.1

        for _ in range(max_iterations):
            # 最近点匹配
            # 问题：缩放幅度过大，最后所有点都落到重心上
            distances = np.linalg.norm(source[:, np.newaxis] - target, axis=2)
            closest_indices = np.argmin(distances, axis=1)
            closest_points = target[closest_indices]

            idx = ransac_point_selection(source, closest_points)

            # 计算质心
            source_centroid = np.mean(source[idx], axis=0)
            target_centroid = np.mean(closest_points[idx], axis=0)

            # 分别计算 x 和 y 方向的缩放因子
            numerator_x = np.sum((source[idx, 0] - source_centroid[0]) * (closest_points[idx, 0] - target_centroid[0]))
            denominator_x = np.sum((source[idx, 0] - source_centroid[0]) ** 2)
            new_scale_x = numerator_x / denominator_x

            numerator_y = np.sum((source[idx, 1] - source_centroid[1]) * (closest_points[idx, 1] - target_centroid[1]))
            denominator_y = np.sum((source[idx, 1] - source_centroid[1]) ** 2)
            new_scale_y = numerator_y / denominator_y

            # 计算平移向量
            new_translation = target_centroid - np.array(
                [new_scale_x * source_centroid[0], new_scale_y * source_centroid[1]])

            # 更新变换参数
            scale_x *= new_scale_x
            scale_y *= new_scale_y
            translation += new_translation

            # 应用变换
            source[:, 0] = scale_x * ori_source[:, 0] + translation[0]
            source[:, 1] = scale_y * ori_source[:, 1] + translation[1]

            # 评估误差
            mse = np.mean(np.linalg.norm(source - closest_points, axis=1) ** 2)

            # 判断是否收敛
            if abs(pre_mse - mse) < tolerance:
                score = _softmax_1000(mse)
                break
            pre_mse = mse

        # 构建齐次变换矩阵
        transformation_matrix = np.array([[scale_x, 0, translation[0]],
                                          [0, scale_y, translation[1]],
                                          [0, 0, 1]])

        return transformation_matrix, score

        # 确定配对点、未匹配点
        paired_indices = []
        source_matched = np.zeros(len(source), dtype=bool)
        target_matched = np.zeros(len(target), dtype=bool)

        for i in range(len(source)):
            j = closest_indices[i]
            paired_indices.append([i, j])
            source_matched[i] = True
            target_matched[j] = True

        paired_indices = np.array(paired_indices)
        source_unmatched = np.where(~source_matched)[0]
        target_unmatched = np.where(~target_matched)[0]

        # return paired_indices, source_unmatched, target_unmatched, translation

        return translation, score

    def icp_bbox_translation(self, source_, target_, max_iterations=100, tolerance=1e-6):
        # bbox:xyxy
        def ransac_point_selection(points1, points2, num_iterations=100, distance_threshold=80):
            """
            使用 RANSAC 算法根据两组对应点之间的欧氏距离筛选点。

            参数:
            points1 (np.ndarray): 第一组点坐标，形状为 (N, 2) 或 (N, 3)
            points2 (np.ndarray): 第二组点坐标，形状为 (N, 2) 或 (N, 3)，与 points1 中的点一一对应
            num_iterations (int): RANSAC 算法的迭代次数
            distance_threshold (float): 欧氏距离阈值，用于判断点是否为内点

            返回:
            np.ndarray: 保留的点的索引数组
            """
            num_points = points1.shape[0]
            best_inliers = np.array([])
            point_bias = points1 - points2

            for _ in range(num_iterations):
                # 随机选择一个点
                random_index = np.random.randint(0, num_points)

                # 计算所有点到该随机点对应点的欧氏距离
                distances = np.linalg.norm(point_bias - point_bias[random_index], axis=1)

                # 根据距离阈值确定内点
                inliers = np.where(distances < distance_threshold)[0]

                # 如果当前内点数量多于之前的最佳内点数量，则更新最佳内点
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers

            return best_inliers

        scale_x = 1.0
        scale_y = 1.0
        translation = np.zeros(2)

        source_pos = source_[:, :2]
        target_pos = target_[:, :2]
        source_scale = source_[:, 2:]
        target_scale = target_[:, 2:]

        source_xyxy = np.hstack((source_pos - source_scale / 2, source_pos - source_scale / 2))
        target_xyxy = np.hstack((target_pos - target_scale / 2, target_pos - target_scale / 2))

        pre_mse = np.inf
        ori_source_pos = copy.deepcopy(source_pos)
        ori_source_scale = copy.deepcopy(source_scale)
        # ori_target = copy.deepcopy(target)

        if len(source_) <= 2 or len(target_) <= 2:
            return np.eye(3), 0.1

        for _ in range(max_iterations):
            # 最近bbox匹配
            # 问题：缩放幅度过大，最后所有点都落到重心上
            boxes1 = torch.Tensor(source_xyxy, dtype=torch.float)
            boxes2 = torch.Tensor(target_xyxy, dtype=torch.float)
            distances = box_iou(boxes1, boxes2).numpy()
            closest_indices = np.argmin(distances, axis=1)
            closest_boxes = target_xyxy[closest_indices]
            closest_pos = target_pos[closest_indices]
            closest_scale = target_scale[closest_indices]

            # idx = ransac_point_selection(source, closest_points)

            # 计算质心
            source_centroid = np.mean(source_pos, axis=0)
            target_centroid = np.mean(target_pos, axis=0)

            # 分别计算 x 和 y 方向的缩放因子
            numerator_x = np.sum((source_pos[0] - source_centroid[0]) * (closest_pos[0] - target_centroid[0]))
            denominator_x = np.sum((source_pos[0] - source_centroid[0]) ** 2)
            new_scale_x = numerator_x / denominator_x

            numerator_y = np.sum((source_pos[1] - source_centroid[1]) * (closest_pos[1] - target_centroid[1]))
            denominator_y = np.sum((source_pos[1] - source_centroid[1]) ** 2)
            new_scale_y = numerator_y / denominator_y

            # 计算平移向量
            new_translation = target_centroid - np.array(
                [new_scale_x * source_centroid[0], new_scale_y * source_centroid[1]])

            # 更新变换参数
            scale_x *= new_scale_x
            scale_y *= new_scale_y
            translation += new_translation

            # 应用变换
            source_pos[:, 0] = scale_x * ori_source_pos[:, 0] + translation[0]
            source_pos[:, 1] = scale_y * ori_source_pos[:, 1] + translation[1]
            source_scale[:, 0] = scale_x * ori_source_pos[:, 0]
            source_scale[:, 1] = scale_y * ori_source_pos[:, 1]

            # 评估误差
            mse = np.mean(np.linalg.norm(distances, axis=1) ** 2)

            # 判断是否收敛
            if abs(pre_mse - mse) < tolerance:
                score = _softmax_1000(mse)
                break
            pre_mse = mse

        # 构建齐次变换矩阵
        transformation_matrix = np.array([[scale_x, 0, translation[0]],
                                          [0, scale_y, translation[1]],
                                          [0, 0, 1]])

        return pos, cost

        # 确定配对点、未匹配点
        paired_indices = []
        source_matched = np.zeros(len(source), dtype=bool)
        target_matched = np.zeros(len(target), dtype=bool)

        for i in range(len(source)):
            j = closest_indices[i]
            paired_indices.append([i, j])
            source_matched[i] = True
            target_matched[j] = True

        paired_indices = np.array(paired_indices)
        source_unmatched = np.where(~source_matched)[0]
        target_unmatched = np.where(~target_matched)[0]

        # return paired_indices, source_unmatched, target_unmatched, translation

        return pos, cost

    def ps_bbox_translation(self, source_, target_, max_iterations=150, partical_num=40):
        """
        Rosenbrock 函数的实现
        :param x: 输入的变量，形状为 (n_particles, dimensions),n*[dx,dy,rx,ry]
        :return: 每个粒子对应的函数值，形状为 (n_particles,)
        """
        from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
        import matplotlib.pyplot as plt

        def paired_num(x_, source, target):
            s_ = copy.deepcopy(source)
            s_[:, 0] = x_[2] * source[:, 0] + x_[0]
            s_[:, 1] = x_[3] * source[:, 1] + x_[1]
            s_[:, 2] = x_[2] * source[:, 2]
            s_[:, 3] = x_[3] * source[:, 3]

            source_xyxy = np.hstack((s_[:, :2] - s_[:, 2:] / 2, s_[:, :2] + s_[:, 2:] / 2))
            target_xyxy = np.hstack((target[:, :2] - target[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2))
            boxes1 = torch.tensor(source_xyxy, dtype=torch.float)
            boxes2 = torch.tensor(target_xyxy, dtype=torch.float)
            distances = 1 - box_iou(boxes1, boxes2).numpy()  # smaller better
            row_indices, col_indices = linear_sum_assignment(distances)
            num_paired = np.sum(distances[row_indices, col_indices] < self.pos_track_dist)
            return np.mean(distances[row_indices, col_indices]), num_paired

        def rosenbrock(x, source, target):
            x = np.asarray(x)
            score = []
            # ori_source = copy.deepcopy(source)
            s_ = copy.deepcopy(source)
            for x_ in x:
                # 计算source的变换，xywh格式
                s_[:, 0] = x_[2] * source[:, 0] + x_[0]
                s_[:, 1] = x_[3] * source[:, 1] + x_[1]
                s_[:, 2] = x_[2] * source[:, 2]
                s_[:, 3] = x_[3] * source[:, 3]

                source_xyxy = np.hstack((s_[:, :2] - s_[:, 2:] / 2, s_[:, :2] + s_[:, 2:] / 2))
                target_xyxy = np.hstack((target[:, :2] - target[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2))
                boxes1 = torch.tensor(source_xyxy, dtype=torch.float)
                boxes2 = torch.tensor(target_xyxy, dtype=torch.float)
                distances = 1 - box_iou(boxes1, boxes2).numpy()  # smaller better
                row_indices, col_indices = linear_sum_assignment(distances)
                iou_dist = np.mean(distances[row_indices, col_indices])

                score.append(iou_dist)

            return np.array(score)

        dimensions = 4
        bounds = (np.array([-200, -10, 0.7, 0.7]), np.array([200, 10, 1.3, 1.3]))

        init_mean = self.best_bias()  # ransac_bias()

        init_points = self.sample_from_gaussians(partical_num, bounds, init_mean)

        if len(source_) < 3 or len(target_) < 3:
            return init_mean, 0.4

        source = copy.deepcopy(source_)

        options = {'c1': 3, 'c2': 0.5, 'w': 0.9}
        optimized_rosenbrock = lambda x: rosenbrock(x, source, target_)

        # 创建全局最优 PSO 优化器
        optimizer = ps.single.GlobalBestPSO(n_particles=partical_num, dimensions=dimensions, options=options,
                                            bounds=bounds, init_pos=init_points)
        cost, pos = optimizer.optimize(optimized_rosenbrock, iters=max_iterations)

        # plot_cost_history(cost_history=optimizer.cost_history)
        # # pos_history = np.array(optimizer.pos_history)[:,:,:2]
        # # plot_contour(pos_history)
        # plt.show()

        # dist, paired_num_ = paired_num(pos, source, target_)  # 修改
        self.paired_bias_set.append([pos, 1 - cost])  # 修改

        return pos, cost

    def save_args(self):
        args_ = {
            'max_iou_dist': self.max_iou_dist,
            'max_age': self.max_age,
            'n_init': self.n_init,
            '_lambda': self._lambda,
            'ema_alpha': self.ema_alpha,
            'conf_ema_alpha': self.conf_ema_alpha,
            'bias_ema_alpha': self.bias_ema_alpha,
            'mc_lambda': self.mc_lambda,
            'deep_track_dist': self.deep_track_dist,
            'pos_track_dist': self.pos_track_dist,
            'pair_delete_pos_thres': self.pair_delete_pos_thres,
            'pair_delete_time_thres_max': self.pair_delete_time_thres_max,
            'pair_delete_time_thres_min': self.pair_delete_time_thres_min,
            'pair_delete_deep_thres': self.pair_delete_deep_thres,
            'soft_nms_thres': self.soft_nms_thres
        }
        import csv
        import os
        keys = args_.keys()  # 获取字典中的键

        filename = '../output_tracks/' + self.exp_id + '/args_info.csv'
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as file:
                print(args_)
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()  # 写入标题行
                writer.writerows([args_])  # 写入数据行


def _softmax_1000(x):
    a = 0.8
    b = 0.4
    k = 0.01
    c = 1000
    return a + b * (np.exp(k * (x - c)) / (1 + np.exp(k * (x - c))))


def _nn_iou_distance(bbox, bboxes):
    """
    计算一个边界框与多个边界框之间的 IoU。

    :param bbox: 单个边界框，形状为 (1, 4)，格式为 (center_x, center_y, w, h)
    :param bboxes: 多个边界框组成的数组，形状为 (n, 4)，格式为 (center_x, center_y, w, h)
    :return: 包含 IoU 值的数组，形状为 (n,)
    """

    # 将中心坐标和宽高转换为左上角和右下角坐标
    def center_to_corners(bbox):
        center_x, center_y, w, h = bbox
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        return x1, y1, x2, y2

    # 处理单个边界框
    x1_1, y1_1, x2_1, y2_1 = center_to_corners(bbox[0])

    # 处理多个边界框
    n = bboxes.shape[0]
    ious = np.zeros(n)

    for i in range(n):
        x1_2, y1_2, x2_2, y2_2 = center_to_corners(bboxes[i])

        # 计算交集区域的边界
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        # 计算交集的宽度和高度
        width_inter = max(0, x2_inter - x1_inter)
        height_inter = max(0, y2_inter - y1_inter)

        # 计算交集的面积
        area_inter = width_inter * height_inter

        # 计算两个边界框的面积
        area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # 计算并集的面积
        area_union = area_box1 + area_box2 - area_inter

        # 避免除零错误
        if area_union == 0:
            ious[i] = 1
        else:
            # 计算 IoU
            # 计算尺度相似性
            scale = min((x2_1 - x1_1), (x2_2 - x1_2)) / max((x2_1 - x1_1), (x2_2 - x1_2)) * \
                    min((y2_1 - y1_1), (y2_2 - y1_2)) / max((y2_1 - y1_1), (y2_2 - y1_2))

            ious[i] = 1 - area_inter / area_union * scale

    return ious
