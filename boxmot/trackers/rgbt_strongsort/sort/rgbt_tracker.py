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
from sko.GA import GA

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
            pair_delete_time_thres_max=60,
            pair_delete_time_thres_min=10,
            pair_delete_deep_thres=0.45,
            soft_nms_thres=0.85,
            vi_entropy_thres=6.6,
            ir_entropy_thres=6.,
            exp_id='',
            adaptive_pose_thres=True,
            input_fusion=False
    ):
        if adaptive_pose_thres:
            adaptive_pose_thres = [1, 0.1, 1.7, 0.15]
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
        self.vi_entropy_thres = vi_entropy_thres
        self.ir_entropy_thres = ir_entropy_thres

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
        self.input_fusion = input_fusion
        self.adaptive_pose_thres = adaptive_pose_thres

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

        visible_entropy = [d.entrophy for d in visible_detections]
        infrared_entrophy = [d.entrophy for d in infrared_detections]

        # Run matching cascade.  # 对可见光、红外分别进行轨迹与检测目标的级联匹配
        if not visible_entropy:
            visible_matches, visible_unmatched_tracks, visible_unmatched_detections = self._match(visible_detections,
                                                                                                     'visible')
        elif min(visible_entropy) < self.vi_entropy_thres:
            visible_matches, visible_unmatched_tracks, visible_unmatched_detections = self._match_v1(visible_detections,
                                                                                              'visible')
        else:
            visible_matches, visible_unmatched_tracks, visible_unmatched_detections = self._match(visible_detections,
                                                                                                  'visible')

        if not infrared_entrophy:
            infrared_matches, infrared_unmatched_tracks, infrared_unmatched_detections = self._match(
                infrared_detections, 'infrared')
        elif min(infrared_entrophy) < self.ir_entropy_thres:
            infrared_matches, infrared_unmatched_tracks, infrared_unmatched_detections = self._match_v1(infrared_detections,
                                                                                                 'infrared')
        else:
            infrared_matches, infrared_unmatched_tracks, infrared_unmatched_detections = self._match(
                infrared_detections, 'infrared')

        # Update track set.
        for track_idx, detection_idx in visible_matches:
            self.visible_tracks[track_idx].update_feat(visible_detections[detection_idx])
        for track_idx, detection_idx in infrared_matches:
            self.infrared_tracks[track_idx].update_feat(infrared_detections[detection_idx])

        for visible_detection_idx in visible_unmatched_detections:
            self._initiate_track(visible_detections[visible_detection_idx], modality='visible')
            self.single_visible_ids.append(self._next_id - 1)
        for infrared_detection_idx in infrared_unmatched_detections:
            self._initiate_track(infrared_detections[infrared_detection_idx], modality='infrared')
            self.single_infrared_ids.append(self._next_id - 1)

        self.crossmodality_match()

        for track_idx in visible_unmatched_tracks:
            if self.visible_tracks[track_idx].id in self.single_visible_ids:
                self.visible_tracks[track_idx].mark_missed()
        for track_idx in infrared_unmatched_tracks:
            if self.infrared_tracks[track_idx].id in self.single_infrared_ids:
                self.infrared_tracks[track_idx].mark_missed()

        for visible_idx, infrared_idx in self.paired_crossmodel_ids:
            pair_vi_idx = [i for i, t in enumerate(self.visible_tracks) if t.id == visible_idx][0]
            pair_ir_idx = [i for i, t in enumerate(self.infrared_tracks) if t.id == infrared_idx][0]
            if (pair_vi_idx in visible_unmatched_tracks) and (pair_ir_idx in infrared_unmatched_tracks):
                self.visible_tracks[pair_vi_idx].mark_missed()
                self.infrared_tracks[pair_ir_idx].mark_missed()

        for track_idx, detection_idx in visible_matches:
            if self.visible_tracks[track_idx].id in self.single_visible_ids:
                self.visible_tracks[track_idx].update_pos_and_state(visible_detections[detection_idx])
        for track_idx, detection_idx in infrared_matches:
            if self.infrared_tracks[track_idx].id in self.single_infrared_ids:
                self.infrared_tracks[track_idx].update_pos_and_state(infrared_detections[detection_idx])

        visible_unmatched_tracks_idx = [t.id for i, t in enumerate(self.visible_tracks) if
                                        i in visible_unmatched_tracks]
        infrared_unmatched_tracks_idx = [t.id for i, t in enumerate(self.infrared_tracks) if
                                         i in infrared_unmatched_tracks]
        for visible_track_idx, infrared_track_idx in self.paired_crossmodel_ids:
            visible_track_ = self.find_visible_track(visible_track_idx)
            infrared_track_ = self.find_infrared_track(infrared_track_idx)

            if (visible_track_idx in visible_unmatched_tracks_idx) and (
                    infrared_track_idx in infrared_unmatched_tracks_idx):
                self.update_fkf_miss2(visible_track_, infrared_track_)

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

            else:
                for t in visible_matches:
                    if self.visible_tracks[t[0]].id == visible_track_idx:
                        visible_det_ = visible_detections[t[1]]
                        break

                for t in infrared_matches:
                    if self.infrared_tracks[t[0]].id == infrared_track_idx:
                        infrared_det_ = infrared_detections[t[1]]
                        break

                self.update_fkf(
                    visible_track_,
                    visible_det_,
                    infrared_track_,
                    infrared_det_
                )
            visible_track_.update_pos_and_state([])
            infrared_track_.update_pos_and_state([])


        self.delete_bad_track_pairs()
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

    def _match(self, detections, modality):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
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
            confirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if not t.is_confirmed()]

            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.visible_tracks,
                detections,
                confirmed_tracks,
            )

            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.visible_tracks[k].time_since_update == 1
            ]
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
            confirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if not t.is_confirmed()]

            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.infrared_tracks,
                detections,
                confirmed_tracks,
            )
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.infrared_tracks[k].time_since_update == 1
            ]
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


    def _match_v1(self, detections, modality):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
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
            unconfirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if not t.is_confirmed()]

            confirmed_paired_tracks = [i for i, t in enumerate(self.visible_tracks) if
                                       t.is_confirmed()
                                       and t.id not in self.single_visible_ids]
            confirmed_single_tracks = [i for i, t in enumerate(self.visible_tracks) if
                                       t.is_confirmed() and
                                       t.id in self.single_visible_ids]

            matches_a_, unmatched_tracks_a_, unmatched_detections_ = linear_assignment.matching_cascade(
                iou_matching.iou_cost,
                self.max_iou_dist-0.4,
                self.max_age,
                self.visible_tracks,
                detections,
                confirmed_paired_tracks,
            )

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

            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.visible_tracks[k].time_since_update == 1
            ]
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
            unconfirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if not t.is_confirmed()]

            confirmed_paired_tracks = [i for i, t in enumerate(self.infrared_tracks) if
                                       t.is_confirmed()
                                       and t.id not in self.single_infrared_ids]
            confirmed_single_tracks = [i for i, t in enumerate(self.infrared_tracks) if
                                       t.is_confirmed() and
                                       t.id in self.single_infrared_ids]

            matches_a_, unmatched_tracks_a_, unmatched_detections_ = linear_assignment.matching_cascade(
                iou_matching.iou_cost,
                self.max_iou_dist-0.3,
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
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if self.infrared_tracks[k].time_since_update == 1
            ]
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

    def _match_v3(self, detections, modality):   # pos match only
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
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
            confirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.visible_tracks) if not t.is_confirmed()]

            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.visible_tracks,
                detections,
                confirmed_tracks+unconfirmed_tracks,
            )

            matches = matches_b
            unmatched_tracks = unmatched_tracks_b
            return matches, unmatched_tracks, unmatched_detections

        elif modality == 'infrared':
            confirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if t.is_confirmed()]
            unconfirmed_tracks = [i for i, t in enumerate(self.infrared_tracks) if not t.is_confirmed()]

            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.infrared_tracks,
                detections,
                confirmed_tracks + unconfirmed_tracks,
            )

            matches = matches_b
            unmatched_tracks = unmatched_tracks_b
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
        all_visible_tracks = [t.id for t in self.visible_tracks if t.is_confirmed() and t.time_since_update <= 2]
        all_infrared_tracks = [t.id for t in self.infrared_tracks if t.is_confirmed() and t.time_since_update <= 2]

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
                d = 1.0 - np.dot(f_vi, f_ir.T)
                if d > self.deep_track_dist:
                    matches.remove(m)

        # TODO ablation for match update
        self.paired_crossmodel_ids = self.update_p_t(self.paired_crossmodel_ids, matches)

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
            if np.mod(self.frame_num, 10) == 0 or len(self.paired_bias_set) <= 5:
                self.bias_score_ema()
                # TODO:ablation for pso
                _, _ = self.ps_bbox_translation(all_visible_features_, all_infrared_features)
                pose, score = self.best_bias()
                score = 1-score
                if self.adaptive_pose_thres:
                    self.pos_track_dist = max(score * self.adaptive_pose_thres[0] + self.adaptive_pose_thres[1], 0.6)
                    self.pair_delete_pos_thres = \
                        max(score * self.adaptive_pose_thres[2] + self.adaptive_pose_thres[3], 0.8)
                    # print('dists:', score, self.pos_track_dist, self.pair_delete_pos_thres)
            else:
                pose, _ = self.best_bias()

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

    def update_p_t(self, P_t_prev, Q_t):
        # 策略1：保留符合条件的旧匹配对
        prev_M = {m for (m, n) in P_t_prev}
        prev_N = {n for (m, n) in P_t_prev}
        Q_M = {m for (m, n) in Q_t}
        Q_N = {n for (m, n) in Q_t}

        P_retained = set()
        for pair in P_t_prev:
            m_p, n_p = pair
            if pair in Q_t or (m_p not in Q_M and n_p not in Q_N):
                P_retained.add(pair)
        P_t = set(P_retained)

        for q_pair in Q_t:
            m, n = q_pair
            if q_pair in P_retained:
                continue
            m_in_prev_M = m in prev_M
            n_in_prev_N = n in prev_N
            if not m_in_prev_M and not n_in_prev_N:
                P_t.add(q_pair)
            else:
                if (m_in_prev_M and not n_in_prev_N) or (not m_in_prev_M and n_in_prev_N):
                    to_remove = set()
                    if m_in_prev_M:
                        pair_m = next((p for p in P_t_prev if p[0] == m), None)
                        if pair_m in P_retained:
                            to_remove.add(pair_m)
                    if n_in_prev_N:
                        pair_n = next((p for p in P_t_prev if p[1] == n), None)
                        if pair_n in P_retained:
                            to_remove.add(pair_n)
                    P_t -= to_remove
                    P_t.add(q_pair)
                elif m_in_prev_M and n_in_prev_N:
                    pair_m = next((p for p in P_t_prev if p[0] == m), None)
                    pair_n = next((p for p in P_t_prev if p[1] == n), None)
                    if pair_m != pair_n and pair_m is not None and pair_n is not None:
                        to_remove = set()
                        if pair_m in P_retained:
                            to_remove.add(pair_m)
                        if pair_n in P_retained:
                            to_remove.add(pair_n)
                        P_t -= to_remove
                        P_t.add(q_pair)
        return P_t

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
        paired_crossmodel_ids = copy.deepcopy(self.paired_crossmodel_ids)
        for t_pairs_ in paired_crossmodel_ids:
            t_pairs = copy.deepcopy(t_pairs_)
            vi_feat = copy.deepcopy(self.find_visible_track(t_pairs[0]).to_xywh())
            ir_feat = self.find_infrared_track(t_pairs[1]).to_xywh()
            bias, _ = self.best_bias()
            vi_feat[0] = vi_feat[0] * bias[2] + bias[0]
            vi_feat[1] = vi_feat[1] * bias[3] + bias[1]
            vi_feat[2] = vi_feat[2] * bias[2]
            vi_feat[3] = vi_feat[3] * bias[3]
            source_xyxy = np.hstack((vi_feat[:2] - vi_feat[2:] / 2, vi_feat[:2] + vi_feat[2:] / 2))
            target_xyxy = np.hstack((ir_feat[:2] - ir_feat[2:] / 2, ir_feat[:2] + ir_feat[2:] / 2))

            distances = 1 - self.np_iou(source_xyxy, target_xyxy)
            fix = self.delete_time_adjust_iou(t_pairs)
            if feat == 'pos':
                if distances > self.pair_delete_pos_thres:
                    self.paired_crossmodel_ids.remove(t_pairs)
                    self.single_visible_ids.append(t_pairs[0])
                    self.single_infrared_ids.append(t_pairs[1])
            elif feat == 'pos-time':
                vi_time_since_update = self.find_visible_track(t_pairs[0]).pair_time_since_update
                ir_time_since_update = self.find_infrared_track(t_pairs[1]).pair_time_since_update

                fixed_time_thres = max(self.pair_delete_time_thres_max - fix * 30, self.pair_delete_time_thres_min)
                if vi_time_since_update + ir_time_since_update > fixed_time_thres:
                    self.find_visible_track(t_pairs[0]).time_since_update += 1
                    self.find_infrared_track(t_pairs[1]).time_since_update += 1

                if distances > self.pair_delete_pos_thres \
                        or ir_time_since_update + vi_time_since_update > fixed_time_thres:
                    self.paired_crossmodel_ids.remove(t_pairs)
                    self.single_visible_ids.append(t_pairs[0])
                    self.single_infrared_ids.append(t_pairs[1])

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

    def soft_nms(self):
        def iou(bbox1, bbox2):
            source_xyxy = np.hstack((bbox1[:2] - bbox1[2:] / 2, bbox1[:2] + bbox1[2:] / 2))
            target_xyxy = np.hstack((bbox2[:2] - bbox2[2:] / 2, bbox2[:2] + bbox2[2:] / 2))
            return self.np_iou(source_xyxy, target_xyxy)

        def nms(t1, t2):
            if t1.pair_time_since_update > t2.pair_time_since_update:
                t1.time_since_update += 1
                return
            elif t1.pair_time_since_update < t2.pair_time_since_update:
                t2.time_since_update += 1
                return
            if t1.hits > t2.hits:
                t2.time_since_update += 1
                return
            elif t2.hits > t1.hits:
                t1.time_since_update += 1
                return
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
        if self.input_fusion:
            self.measurements_fusion(visible_track_, infrared_track_, FKFMode.both)
        else:
            self._update_fkf(visible_track_, infrared_track_, FKFMode.both)

        visible_track_.time_since_update = 0
        visible_track_.pair_time_since_update = 0
        if visible_track_.state == TrackState.Tentative and visible_track_.hits >= visible_track_._n_init:
            visible_track_.state = TrackState.Confirmed

        infrared_track_.time_since_update = 0
        infrared_track_.pair_time_since_update = 0
        if infrared_track_.state == TrackState.Tentative and infrared_track_.hits >= infrared_track_._n_init:
            infrared_track_.state = TrackState.Confirmed

    def update_fkf_miss2(self, visible_track_, infrared_track_):
        visible_track_.pair_time_since_update += 1
        infrared_track_.pair_time_since_update += 1
        if self.input_fusion:
            self.measurements_fusion(visible_track_, infrared_track_, FKFMode.miss2)
        else:
            self._update_fkf(visible_track_, infrared_track_, FKFMode.miss2)

    def update_fkf_miss_visible(self, visible_track_, infrared_track_, infrared_det):
        # infrared_track_.conf = infrared_det.conf
        if self.input_fusion:
            self.measurements_fusion(visible_track_, infrared_track_, FKFMode.miss_vi)
        else:
            self._update_fkf(visible_track_, infrared_track_, FKFMode.miss_vi)
        visible_track_.time_since_update = 0
        visible_track_.pair_time_since_update += 1
        if visible_track_.state == TrackState.Tentative and visible_track_.hits >= visible_track_._n_init:
            visible_track_.state = TrackState.Confirmed

        infrared_track_.time_since_update = 0
        infrared_track_.pair_time_since_update = 0
        if infrared_track_.state == TrackState.Tentative and infrared_track_.hits >= infrared_track_._n_init:
            infrared_track_.state = TrackState.Confirmed

    def update_fkf_miss_infrared(self, visible_track_, infrared_track_, visible_det):
        if self.input_fusion:
            self.measurements_fusion(visible_track_, infrared_track_, FKFMode.miss_ir)
        else:
            self._update_fkf(visible_track_, infrared_track_, FKFMode.miss_ir)
        visible_track_.time_since_update = 0
        visible_track_.pair_time_since_update = 0
        if visible_track_.state == TrackState.Tentative and visible_track_.hits >= visible_track_._n_init:
            visible_track_.state = TrackState.Confirmed

        infrared_track_.time_since_update = 0
        infrared_track_.pair_time_since_update += 1
        if infrared_track_.state == TrackState.Tentative and infrared_track_.hits >= infrared_track_._n_init:
            infrared_track_.state = TrackState.Confirmed

    def _update_fkf(self, visible_track_, infrared_track_, mode=FKFMode.both):
        best_bias,_ = self.best_bias()

        v_conf = copy.deepcopy(visible_track_.conf_ema)
        v_mean, v_covariance = copy.deepcopy(visible_track_.mean), copy.deepcopy(
            visible_track_.covariance)
        v_vel_mean = v_mean[4:8]

        i_conf = copy.deepcopy(infrared_track_.conf_ema)
        i_mean, i_covariance = copy.deepcopy(infrared_track_.mean), copy.deepcopy(
            infrared_track_.covariance)
        i_vel_mean = i_mean[4:]

        if mode == FKFMode.both:
            v_conf_, i_conf_ = v_conf**10, i_conf ** 10
        elif mode == FKFMode.miss_vi:
            v_conf_, i_conf_ = 0, i_conf
        elif mode == FKFMode.miss_ir:
            v_conf_, i_conf_ = v_conf, 0
        elif mode == FKFMode.miss2:
            v_conf_, i_conf_ = v_conf, i_conf

        if mode == FKFMode.both:
            s_vi = visible_track_.bbox[2] * (visible_track_.bbox[3]**2) * best_bias[2] *best_bias[3]
            s_ir = infrared_track_.bbox[2] * (infrared_track_.bbox[3]**2)

            if s_ir > 1.5*s_vi:
                fkf_vel = np.divide(
                    i_vel_mean, np.array([best_bias[2], best_bias[3], best_bias[2] / best_bias[3], best_bias[3]]))
                visible_track_.mean[0:4] = visible_track_.match_mean[0:4] + fkf_vel * self.dt
                visible_track_.mean[4:] = fkf_vel
            elif s_vi > 1.5*s_ir:
                fkf_vel = np.multiply(
                    v_vel_mean, np.array([best_bias[2], best_bias[3], best_bias[2] / best_bias[3], best_bias[3]]))
                infrared_track_.mean[0:4] = infrared_track_.match_mean[0:4] + fkf_vel * self.dt
                infrared_track_.mean[4:] = fkf_vel

        if mode == FKFMode.miss2:
            fkf_vel = (i_conf_ * i_vel_mean + v_conf_ * v_vel_mean) / (i_conf_ + v_conf_)
            visible_track_.mean[0:4] = visible_track_.match_mean[0:4] + fkf_vel * self.dt
            visible_track_.mean[4:] = fkf_vel
            infrared_track_.mean[0:4] = infrared_track_.match_mean[0:4] + fkf_vel * self.dt
            infrared_track_.mean[4:] = fkf_vel

        elif mode == FKFMode.miss_ir:
            fkf_vel = copy.deepcopy(v_vel_mean)
            fkf_vel = np.multiply(
                fkf_vel, np.array([best_bias[2], best_bias[3], best_bias[2]/best_bias[3], best_bias[3]]))
            infrared_track_.mean[0:4] = infrared_track_.match_mean[0:4] + fkf_vel * self.dt
            infrared_track_.mean[4:] = fkf_vel

        elif mode == FKFMode.miss_vi:
            fkf_vel = copy.deepcopy(i_vel_mean)
            fkf_vel = np.divide(
                fkf_vel, np.array([best_bias[2], best_bias[3], best_bias[2] / best_bias[3], best_bias[3]]))
            visible_track_.mean[0:4] = visible_track_.match_mean[0:4] + fkf_vel * self.dt
            visible_track_.mean[4:] = fkf_vel

        fkf_cov = v_covariance + i_covariance
        v_vel_cov = fkf_cov * (min(i_conf, v_conf) / (i_conf + v_conf))
        i_vel_cov = fkf_cov * (min(i_conf, v_conf) / (i_conf + v_conf))

        visible_track_.match_covariance = v_vel_cov
        infrared_track_.match_covariance = i_vel_cov
        return

    def measurements_fusion(self, t_vi, t_ir, mode):
        def c_solve(L, b):
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x
        best_bias,_ = self.best_bias()
        if mode == FKFMode.both:
            s_vi = t_vi.bbox[2] * (t_vi.bbox[3] ** 2) * best_bias[2] * best_bias[3]
            s_ir = t_ir.bbox[2] * (t_ir.bbox[3] ** 2)

            if s_ir > 1.5 * s_vi:
                mode == FKFMode.miss_vi
            elif s_vi > 1.5 * s_ir:
                mode == FKFMode.miss_ir
            mean_vi, mean_ir = t_vi.to_xywh(), t_ir.to_xywh()
            mean_vi[2] /= mean_vi[3]
            mean_ir[2] /= mean_ir[3]
            z_vi, z_ir = copy.deepcopy(t_vi.bbox), copy.deepcopy(t_ir.bbox)
            R_vi = t_vi.kf._get_measurement_noise_std(mean_vi, t_vi.conf)
            R_vi = [(1-t_vi.conf) * x for x in R_vi]
            R_vi = np.diag(R_vi)
            R_ir = t_ir.kf._get_measurement_noise_std(mean_ir, t_ir.conf)
            R_ir = [(1 - t_ir.conf) * x for x in R_ir]
            R_ir = np.diag(R_ir)
            best_bias,_ = self.best_bias()
            #
            delta_z_vi, delta_z_ir = z_vi-mean_vi, z_ir-mean_ir
            h_delta_z_vi = np.multiply(
                        delta_z_vi, np.array([best_bias[2], best_bias[3], best_bias[2] / best_bias[3], best_bias[3]]))

            L_vi, L_ir = np.linalg.cholesky(R_vi), np.linalg.cholesky(R_ir)
            Rz_vi_inv, Rz_ir_inv = c_solve(L_vi, h_delta_z_vi), c_solve(L_ir, delta_z_ir)
            R_vi_inv, R_ir_inv = c_solve(L_vi, np.eye(4)), c_solve(L_ir, np.eye(4))
            R_inv_sum = R_vi_inv + R_ir_inv
            L_sum = np.linalg.cholesky(R_inv_sum)
            R_bar = c_solve(L_sum, np.eye(4))
            z_bar = R_bar@(Rz_vi_inv + Rz_ir_inv)

            z_bar_vi = np.divide(
                z_bar, np.array([best_bias[2], best_bias[3], best_bias[2] / best_bias[3], best_bias[3]]))
            r_bar = 1-(1-t_vi.conf)*(1-t_ir.conf)/((1-t_vi.conf)+(1-t_ir.conf))
            t_vi.bbox, t_vi.conf = mean_vi + z_bar_vi, r_bar
            t_ir.bbox, t_ir.conf = mean_ir + z_bar, r_bar

            t_vi.mean, t_vi.covariance = t_vi.kf.update(t_vi.mean, t_vi.covariance, t_vi.bbox, t_vi.conf)
            t_ir.mean, t_ir.covariance = t_ir.kf.update(t_ir.mean, t_ir.covariance, t_ir.bbox, t_ir.conf)


        if mode == FKFMode.miss_vi:
            mean_vi, mean_ir = copy.deepcopy(t_vi.match_mean[:4]), copy.deepcopy(t_ir.match_mean[:4])
            z_ir = copy.deepcopy(t_ir.bbox)
            delta_z_ir = z_ir - mean_ir

            best_bias, _ = self.best_bias()
            h_delta_z_ir = copy.deepcopy(delta_z_ir)
            h_delta_z_ir[0] = delta_z_ir[0] / best_bias[2]
            h_delta_z_ir[1] = delta_z_ir[1] / best_bias[3]
            h_delta_z_ir[2] = delta_z_ir[2] / best_bias[2] * best_bias[3]
            h_delta_z_ir[3] = delta_z_ir[3] / best_bias[3]

            t_vi.bbox, t_vi.conf = mean_vi + h_delta_z_ir, t_ir.conf
            t_vi.mean, t_vi.covariance = t_vi.kf.update(t_vi.mean, t_vi.covariance, t_vi.bbox, t_vi.conf)

        elif mode == FKFMode.miss_ir:
            mean_vi, mean_ir = copy.deepcopy(t_vi.match_mean[:4]), copy.deepcopy(t_ir.match_mean[:4])
            z_vi = copy.deepcopy(t_vi.bbox)
            delta_z_vi = z_vi - mean_vi

            best_bias, _ = self.best_bias()
            h_delta_z_vi = copy.deepcopy(delta_z_vi)
            h_delta_z_vi[0] = delta_z_vi[0] * best_bias[2]
            h_delta_z_vi[1] = delta_z_vi[1] * best_bias[3]
            h_delta_z_vi[2] = delta_z_vi[2] * best_bias[2] / best_bias[3]
            h_delta_z_vi[3] = delta_z_vi[3] * best_bias[3]

            t_ir.bbox, t_ir.conf = mean_ir + h_delta_z_vi, t_vi.conf
            t_ir.mean, t_ir.covariance = t_ir.kf.update(t_ir.mean, t_ir.covariance, t_ir.bbox, t_ir.conf)

        elif mode == FKFMode.miss2:
            pass
        return

    def bias_adjust(self, adjust_features, pose):
        adjust_features_ = np.zeros(shape=adjust_features.shape)
        adjust_features_[:, 0] = adjust_features[:, 0] * pose[2] + pose[0]
        adjust_features_[:, 1] = adjust_features[:, 1] * pose[3] + pose[1]
        adjust_features_[:, 2] = adjust_features[:, 2] * pose[2]
        adjust_features_[:, 3] = adjust_features[:, 3] * pose[3]
        return adjust_features_

    def sample_from_gaussians(self, num_samples, bounds, means, stds=np.array([10, 10, 0.1, 0.1])):
        all_samples = []
        for mean, std in zip(means, stds):
            samples = np.random.normal(mean, std, num_samples)
            all_samples.append(samples)
        return np.clip(np.array(all_samples).T, bounds[0], bounds[1])

    def best_bias(self):
        if not self.paired_bias_set:# or max([s[1] for s in self.paired_bias_set])<0.5:
            return [0, 0, 1, 1], 0.
        else:
            points = np.array([t[0] for t in self.paired_bias_set[-10:]])
            score = np.array([t[1] for t in self.paired_bias_set[-10:]])
            return points[np.argmax(score)], max(score)

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

    def ps_bbox_translation(self, source_, target_, max_iterations=150, partical_num=40):  # 150,40 ,60x
        def paired_num(x_, source, target):
            s_ = copy.deepcopy(source)
            s_[:, 0] = x_[2] * source[:, 0] + x_[0]
            s_[:, 1] = x_[3] * source[:, 1] + x_[1]
            s_[:, 2] = x_[2] * source[:, 2]
            s_[:, 3] = x_[3] * source[:, 3]

            source_xyxy = np.hstack((s_[:, :2] - s_[:, 2:] / 2, s_[:, :2] + s_[:, 2:] / 2))
            target_xyxy = np.hstack((target[:, :2] - target[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2))
            distances = 1 - self.np_iou_distance_matrix(source_xyxy, target_xyxy)

            row_indices, col_indices = linear_sum_assignment(distances)
            num_paired = np.sum(distances[row_indices, col_indices] < self.pos_track_dist)
            return np.mean(distances[row_indices, col_indices]), num_paired

        def rosenbrock(x, source, target):
            x = np.asarray(x)
            score = []
            s_ = copy.deepcopy(source)
            for x_ in x:
                s_[:, 0] = x_[2] * source[:, 0] + x_[0]
                s_[:, 1] = x_[3] * source[:, 1] + x_[1]
                s_[:, 2] = x_[2] * source[:, 2]
                s_[:, 3] = x_[3] * source[:, 3]

                source_xyxy = np.hstack((s_[:, :2] - s_[:, 2:] / 2, s_[:, :2] + s_[:, 2:] / 2))
                target_xyxy = np.hstack((target[:, :2] - target[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2))
                distances = 1 - self.np_iou_distance_matrix(source_xyxy, target_xyxy)

                row_indices, col_indices = linear_sum_assignment(distances)
                iou_dist = np.mean(distances[row_indices, col_indices])
                score.append(iou_dist)

            return np.array(score)

        dimensions = 4
        bounds = (np.array([-250, -40, 0.67, 0.67]), np.array([250, 40, 1.5, 1.5]))

        init_mean, _ = self.best_bias()
        init_points = self.sample_from_gaussians(partical_num, bounds, init_mean)
        if len(source_) < 2 or len(target_) < 2:
            return init_mean, 0.8
        source = copy.deepcopy(source_)
        if len(self.paired_bias_set) < 10:
            options = {'c1': 3.2, 'c2': 0.4, 'w': 0.9}
        else:
            options = {'c1': 3, 'c2': 0.4, 'w': 0.8}
        optimized_rosenbrock = lambda x: rosenbrock(x, source, target_)
        optimizer = ps.single.GlobalBestPSO(n_particles=partical_num, dimensions=dimensions, options=options,
                                            bounds=bounds, init_pos=init_points)
        cost, pos = optimizer.optimize(optimized_rosenbrock, iters=max_iterations)
        dist, paired_num_ = paired_num(pos, source, target_)
        self.paired_bias_set.append([pos, (1 - cost/np.log10(paired_num_))])
        return pos, cost

    def np_iou(self, bbox1, bbox2):
        """
        Compute IoU between two bounding boxes.
        Bounding boxes are formatted as [x1, y1, x2, y2], where (x1, y1) is the top-left corner
        and (x2, y2) is the bottom-right corner.
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

        x1_intersect = np.maximum(x1_1, x1_2)
        y1_intersect = np.maximum(y1_1, y1_2)
        x2_intersect = np.minimum(x2_1, x2_2)
        y2_intersect = np.minimum(y2_1, y2_2)

        intersection_width = np.maximum(0, x2_intersect - x1_intersect)
        intersection_height = np.maximum(0, y2_intersect - y1_intersect)
        intersection_area = intersection_width * intersection_height

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = area1 + area2 - intersection_area

        iou = intersection_area / (union_area + 1e-10)
        return iou

    def np_iou_distance_matrix(self, boxes1, boxes2):
        boxes1 = boxes1[:, np.newaxis, :]
        boxes2 = boxes2[np.newaxis, :, :]

        left = np.maximum(boxes1[..., 0], boxes2[..., 0])
        right = np.minimum(boxes1[..., 2], boxes2[..., 2])
        top = np.maximum(boxes1[..., 1], boxes2[..., 1])
        bottom = np.minimum(boxes1[..., 3], boxes2[..., 3])

        inter_width = np.clip(right - left, a_min=0, a_max=None)
        inter_height = np.clip(bottom - top, a_min=0, a_max=None)
        inter_area = inter_width * inter_height

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union_area = area1 + area2 - inter_area

        iou = np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=(union_area != 0))
        return iou


def _softmax_1000(x):
    a = 0.8
    b = 0.4
    k = 0.01
    c = 1000
    return a + b * (np.exp(k * (x - c)) / (1 + np.exp(k * (x - c))))


def _nn_iou_distance(bbox, bboxes):
    def center_to_corners(bbox):
        center_x, center_y, w, h = bbox
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = center_to_corners(bbox[0])
    n = bboxes.shape[0]
    ious = np.zeros(n)

    for i in range(n):
        x1_2, y1_2, x2_2, y2_2 = center_to_corners(bboxes[i])

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        width_inter = max(0, x2_inter - x1_inter)
        height_inter = max(0, y2_inter - y1_inter)

        area_inter = width_inter * height_inter

        area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        area_union = area_box1 + area_box2 - area_inter

        if area_union == 0:
            ious[i] = 1
        else:
            scale = min((x2_1 - x1_1), (x2_2 - x1_2)) / max((x2_1 - x1_1), (x2_2 - x1_2)) * \
                    min((y2_1 - y1_1), (y2_2 - y1_2)) / max((y2_1 - y1_1), (y2_2 - y1_2))

            ious[i] = 1 - area_inter / area_union * scale
    return ious
