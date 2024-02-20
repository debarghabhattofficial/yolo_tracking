# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from collections import deque

import numpy as np

from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.motion.cmc.sof import SparseOptFlow
from boxmot.motion.kalman_filters.botsort_rgbd_kf import KalmanFilter
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (
    embedding_distance_with_depth, 
    fuse_score_with_depth, 
    iou_distance_with_depth, 
    linear_assignment_with_depth
)
from boxmot.utils.ops import (
    xywh2xyxy_with_depth, 
    xyxy2xywh_with_depth
)


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, det, feat=None, feat_history=50):
        # wait activate
        # self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)  # ORIGINAL
        # NOTE: Replace every instance of self.xywh with
        # self.xywh (later self.sydwh) in the code.
        self.xywh_with_depth = xyxy2xywh_with_depth(det[0:5])  # (x1, y1, x2, y2, depth) --> (xc, yc, depth, w, h)  # DEB
        print(f"xyxy2xywh: \n{det[0:5]}")  # DEB
        print("-" * 75)  # DEB
        print(f"self.xywh_with_depth: \n{self.xywh_with_depth}")  # DEB
        print("-" * 75)  # DEB
        self.score = det[5]
        print(f"self.score: {self.score}")  # DEB
        print("-" * 75)
        self.cls = det[6]
        print(f"self.cls: {self.cls}")  # DEB
        print("-" * 75)
        self.det_ind = det[7]
        print(f"self.det_ind: {self.det_ind}")  # DEB
        print("-" * 75)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.score)

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

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[8] = 0
            mean_state[9] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            print("BEFORE")
            print("-" * 75)
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            print(f"multi_mean shape: {multi_mean.shape}")  # DEB
            print(f"multi_mean: \n{multi_mean}")  # DEB
            print("-" * 75)  # DEB
            multi_covariance = np.asarray([st.covariance for st in stracks])
            print(f"multi_covariance shape: {multi_covariance}")  # DEB
            print(f"multi_covariance: \n{multi_covariance}")  # DEB
            print("-" * 75)  # DEB
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][8] = 0
                    multi_mean[i][9] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                mean=multi_mean, 
                covariance=multi_covariance
            )
            print("AFTER")
            print("-" * 75)
            print(f"multi_mean shape: {multi_mean.shape}")  # DEB
            print(f"multi_mean: \n{multi_mean}")  # DEB
            print("-" * 75)  # DEB
            print(f"multi_covariance shape: {multi_covariance}")  # DEB
            print(f"multi_covariance: \n{multi_covariance}")  # DEB
            print("-" * 75)  # DEB
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
            print(f"INSIDE multi_predict()")
            print(f"stracks: \n{stracks}")
            print("-" * 75)

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        # NOTE: Make changes to this method to account for
        # camera motion compensation while updating the
        # state of the tracklets.
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

    # NOTE: Replace every call to self.multi_gmc() method with
    # self.multi_gmc_with_depth() method.
    @staticmethod
    def multi_gmc_with_depth(stracks, H=np.eye(2, 3)):
        # NOTE: Make changes to this method to account for
        # camera motion compensation while updating the
        # state of the tracklets.
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            # Changed the identity matrix in Knornocker product
            # from 4x4 to 5x5 since out measurement vector has
            # 5 measurement varibles (x, y, depth, w, h).
            R10x10 = np.kron(np.eye(5, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R10x10.dot(mean)
                mean[:2] += t
                cov = R10x10.dot(cov).dot(R10x10.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(
            measurement=self.xywh_with_depth
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            mean=self.mean, 
            covariance=self.covariance, 
            measurement=new_track.xywh_with_depth
        )
        if new_track.curr_feat is not None:
            self.update_features(feat=new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

        self.update_cls(
            cls=new_track.cls, 
            score=new_track.score
        )

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            mean=self.mean, 
            covariance=self.covariance, 
            measurement=new_track.xywh_with_depth
        )

        if new_track.curr_feat is not None:
            self.update_features(feat=new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(
            cls=new_track.cls, 
            score=new_track.score
        )

    # NOTE: Following code was in the original BoT-SORT algorithm, 
    # but we have replaced it with self.xyxy_with_depth() 
    # in the BoT-SORT + RGBD algorithm.
    # =========================================================
    # @property
    # def xyxy(self):
    #     """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    #     `(top left, bottom right)`.
    #     """
    #     if self.mean is None:
    #         ret = self.xywh_with_depth.copy()  # (xc, yc, depth, w, h)
    #     else:
    #         ret = self.mean[:5].copy()  # kf (xc, yc, depth, w, h)
    #     ret = xywh2xyxy_with_depth(ret)  # (xc, yc, depth, w, h) --> (x1, y1, x2, y2, depth)
    #     return ret
    # =========================================================
    
    # NOTE: Replace every call to self.xyxy() method with
    # self.xyxy_with_depth() method.
    @property
    def xyxy_with_depth(self):
        """
        Convert bounding box to format 
        `(min x, min y, max x, max y, depth)`, i.e.,
        `(top left coord, bottom right coord, centre depth)`.
        """
        print(f"INSIDE WITH xyxy_with_depth()")  # DEB
        print("-" * 75)  # DEB
        print(f"self.mean: \n{self.mean}")  # DEB
        print("-" * 75)  # DEB
        if self.mean is None:
            ret = self.xywh_with_depth.copy()  # (xc, yc, depth, w, h)
        else:
            ret = self.mean[:5].copy()  # kf (xc, yc, depth, w, h)
        print(f"ret: \n{ret}")  # DEB
        print("-" * 75)  # DEB
        ret = xywh2xyxy_with_depth(ret)  # (xc, yc, depth, w, h) --> (x1, y1, x2, y2, depth)
        print(f"ret: \n{ret}")  # DEB
        print("-" * 75)  # DEB
        return ret


class BoTSORTRGBD(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "sparseOptFlow",
        frame_rate=30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
    ):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid
        if self.with_reid:
            self.model = ReIDDetectMultiBackend(
                weights=model_weights, 
                device=device, 
                fp16=fp16
            )

        self.cmc = SparseOptFlow()
        self.fuse_first_associate = fuse_first_associate

    def update(self, dets, img):
        err_msg = f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray."
        assert isinstance(
            dets, np.ndarray
        ), err_msg

        err_msg = f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), err_msg
        
        err_msg = f"Unsupported 'dets' dimensions '{len(dets.shape)}', valid number of dimensions is two."
        assert (
            len(dets.shape) == 2
        ), err_msg
        
        err_msg = f"Unsupported 'dets' 2nd dimension length '{dets.shape[1]}', valid lenghts is 7."
        assert (
            dets.shape[1] == 7
        ), err_msg

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Appends a new column to the dets array, where the new 
        # column contains integers representing the indices of 
        # the rows in the original dets array.
        dets = np.hstack([
            dets, 
            np.arange(len(dets)).reshape(-1, 1)
        ])
        print(f"dets: {dets.shape}")  # DEB
        print(f"dets: \n{dets}")  # DEB
        print("-" * 75)  # DEB

        # Remove bad detections
        # NOTE: Depth of bbox centre inserted in index 4. 
        # Therefore, confidence values are now available 
        # at index 5.
        confs = dets[:, 5]
        print(f"confs shape: {confs.shape}")  # DEB
        print(f"confs: \n{confs}")  # DEB
        print("-" * 75)  # DEB

        # find second round association detections
        second_mask = np.logical_and(
            confs > self.track_low_thresh, 
            confs < self.track_high_thresh
        )
        dets_second = dets[second_mask]
        print(f"second_mask: \n{second_mask}")  # DEB
        print("-" * 75)  # DEB
        print(f"dets_second: \n{dets_second}")
        print("-" * 75)

        # find first round association detections
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        print(f"first_mask: \n{first_mask}")
        print("-" * 75)
        print(f"dets_first: \n{dets_first}")
        print("-" * 75)

        """Extract embeddings """
        if self.with_reid:
            print(f"ReID step is active.")  # DEB
            print("-" * 75)  # DEB
            features_high = self.model.get_features(
                xyxys=dets_first[:, 0:4], 
                img=img
            )
            print(f"features_high: \n{features_high}")  # DEB
            print("-" * 75)  # DEB

        if len(dets) > 0:
            """Detections"""
            print(f"len(dets) > 0 is True.")  # DEB
            print("-" * 75)  # DEB
            if self.with_reid:
                detections = [
                    STrack(det, f) 
                    for (det, f) in zip(dets_first, features_high)
                ]
            else:
                detections = [
                    STrack(det) 
                    for (det) in np.array(dets_first)
                ]
        else:
            print(f"len(dets) > 0 is False.")  # DEB
            print("-" * 75)  # DEB
            detections = []
        print(f"detections (inside BoTSORT+RGBD): \n{detections}")  # DEB
        print("-" * 75)  # DEB

        """
        Add newly detected tracklets to tracked_stracks.
        """
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        print(f"unconfirmed: \n{unconfirmed}")  # DEB
        print("-" * 75)  # DEB
        print(f"tracked_stracks: \n{tracked_stracks}")  # DEB
        print("-" * 75)  # DEB

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(
            tlista=tracked_stracks, 
            tlistb=self.lost_stracks
        )
        print(f"strack_pool: \n{strack_pool}")  # DEB
        print("-" * 75)  # DEB

        # Predict the current location with KF.
        STrack.multi_predict(strack_pool)

        # Fix camera motion.
        # NOTE: Here, we apply CMC to update the state of 
        # tracklets in the strack_pool. Understand it thoroughly,
        # and if possible, adopt the same in he OC-SORT + RGBD
        # algorithm.
        # ORIGINAL
        # =========================================================
        # warp = self.cmc.apply(img, dets_first)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)
        # =========================================================
        # DEB
        # =========================================================
        warp = self.cmc.apply(img, dets_first[:, :4])  # DEB
        STrack.multi_gmc_with_depth(strack_pool, warp)
        STrack.multi_gmc_with_depth(unconfirmed, warp)
        # =========================================================

        # Associate with high score detection boxes
        ious_dists = iou_distance_with_depth(
            atracks=strack_pool, 
            btracks=detections
        )
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
          ious_dists = fuse_score_with_depth(
              cost_matrix=ious_dists, 
              detections=detections
            )

        if self.with_reid:
            emb_dists = embedding_distance_with_depth(
                tracks=strack_pool, 
                detections=detections
            ) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment_with_depth(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(dets_second) 
                for dets_second in dets_second
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance_with_depth(
            atracks=r_tracked_stracks, 
            btracks=detections_second
        )
        matches, u_track, \
            u_detection_second = linear_assignment_with_depth(
                cost_matrix=dists, 
                thresh=0.5
            )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance_with_depth(
            atracks=unconfirmed, 
            btracks=detections
        )
        ious_dists_mask = ious_dists > self.proximity_thresh

        ious_dists = fuse_score_with_depth(
            cost_matrix=ious_dists, 
            detections=detections
        )
        
        if self.with_reid:
            emb_dists = embedding_distance_with_depth(
                tracks=unconfirmed, 
                detections=detections
            ) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, \
            u_detection = linear_assignment_with_depth(
                cost_matrix=dists, 
                thresh=0.7
            )
        for itracked, idet in matches:
            unconfirmed[itracked].update(
                new_track=detections[idet], 
                frame_id=self.frame_id
            )
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(
                kalman_filter=self.kalman_filter, 
                frame_id=self.frame_id
            )
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [
            t 
            for t in self.tracked_stracks 
            if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(
            tlista=self.tracked_stracks, 
            tlistb=activated_starcks
        )
        self.tracked_stracks = joint_stracks(
            tlista=self.tracked_stracks, 
            tlistb=refind_stracks
        )
        self.lost_stracks = sub_stracks(
            tlista=self.lost_stracks, 
            tlistb=self.tracked_stracks
        )
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            tlista=self.lost_stracks, 
            tlistb=self.removed_stracks
        )
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            stracksa=self.tracked_stracks, 
            stracksb=self.lost_stracks
        )

        output_stracks = [
            track 
            for track in self.tracked_stracks 
            if track.is_activated
        ]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy_with_depth)  # (x1, y1, x2, y2, depth)
            output.append(t.id)  # track ID
            output.append(t.score)  # confidence score
            output.append(t.cls)  # class ID
            output.append(t.det_ind)  # detection index
            outputs.append(output)

        outputs = np.asarray(outputs)
        print(f"outputs shape: {outputs.shape}")  # DEB
        print(f"outputs: \n{outputs}")  # DEB
        print("-" * 75)  # DEB
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
    pdist = iou_distance_with_depth(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
