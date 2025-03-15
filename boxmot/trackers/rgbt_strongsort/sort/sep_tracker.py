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


class SeperateTracker:
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
            mc_lambda=0.995,
            deep_track_dist=0.5,
            pos_track_dist=0.7,
            conf_ema_alpha=0.5
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        self.deep_track_dist = deep_track_dist
        self.pos_track_dist = pos_track_dist
        self.conf_ema_alpha = conf_ema_alpha

        self.visible_tracks = []
        self.infrared_tracks = []

        self._next_id = 1
        self.cmc = get_cmc_method('ecc')()

        # self.fkf = FederatedKalmanFilterXYAH()
        self.dt = 1.
        self.pos_track_rate = 1.
        self.deep_track_rate = 1.

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

    def update(self, visible_detections, infrared_detections, image_num):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.  # 对可见光、红外分别进行轨迹与检测目标的级联匹配
        visible_matches, visible_unmatched_tracks, visible_unmatched_detections = self._match(visible_detections,
                                                                                              'visible')
        infrared_matches, infrared_unmatched_tracks, infrared_unmatched_detections = self._match(infrared_detections,
                                                                                                 'infrared')

        # Update track set.
        # #1.1、匹配更新视觉、位置特征
        for track_idx, detection_idx in visible_matches:
            self.visible_tracks[track_idx].update(visible_detections[detection_idx])
        for track_idx, detection_idx in infrared_matches:
            self.infrared_tracks[track_idx].update(infrared_detections[detection_idx])

        # #1.2、未匹配目标生成新生轨迹
        for visible_detection_idx in visible_unmatched_detections:
            self._initiate_track(visible_detections[visible_detection_idx], modality='visible')
        for infrared_detection_idx in infrared_unmatched_detections:
            self._initiate_track(infrared_detections[infrared_detection_idx], modality='infrared')

        # #3、未匹配轨迹管理
        # 3.1：单模态未匹配轨迹，标记失踪
        for track_idx in visible_unmatched_tracks:
            self.visible_tracks[track_idx].mark_missed()
        for track_idx in infrared_unmatched_tracks:
            self.infrared_tracks[track_idx].mark_missed()

        # #4、对单模态、跨模态轨迹滤波更新，并且更新距离度量
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

            # Associate confirmed tracks using appearance features.  第一级匹配，使用外观特征
            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.visible_tracks,
                detections,
                confirmed_tracks,
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

            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            return matches, unmatched_tracks, unmatched_detections

        elif modality == 'infrared':
            # Split track set into confirmed and unconfirmed tracks.
            confirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if not t.is_confirmed()]

            # Associate confirmed tracks using appearance features.  第一级匹配，使用外观特征
            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.infrared_tracks,
                detections,
                confirmed_tracks,
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

            matches = matches_a + matches_b
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

    def find_visible_track(self, id):
        for t in self.visible_tracks:
            if t.id == id:
                return t

    def find_infrared_track(self, id):
        for t in self.infrared_tracks:
            if t.id == id:
                return t
