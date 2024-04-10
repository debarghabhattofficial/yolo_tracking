# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from collections import deque

import numpy as np

from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.motion.cmc.sof import SparseOptFlow
from boxmot.motion.kalman_filters.botsort_dtc_kf import KalmanFilter
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (
    fuse_score_dtc, 
    iou_distance_dtc, 
    linear_assignment_dtc
)

# Following package and related code was
# added by DEB to print the CMC-related
# before and after KF state mean bboxes
# for debugging purposes; and to print 
# average IoU and average centroid 
# distances for debugging purposes.
# =======================================================
import cv2
from boxmot.utils.iou import iou_batch, centroid_batch
# =======================================================


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, det, feat=None, feat_history=50):
        # wait activate
        self.xyxyd = det[0:5].copy()  # (x1, y1, x2, y2, depth)
        self.score = det[5]
        self.cls = det[6]
        self.det_ind = det[7]
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
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                mean=multi_mean, 
                covariance=multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    # NOTE: Replace every call to self.multi_gmc() method with
    # self.multi_gmc_dtc() method.
    @staticmethod
    def multi_gmc_dtc(stracks, H=np.eye(2, 3), img=None):
        # NOTE: Make changes to this method to account for
        # camera motion compensation while updating the
        # state of the tracklets.
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            # Using the same identity matrix in Knornocker product
            # of size 4x4 as in original BoTSORT since out 
            # we exclude depth-related dimensions from the affine 
            # transformations.
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                # Display bbox for state mean vector before
                # applying the affine transformation.
                # =================================================
                b_text = f"B: {i + 1}"
                (b_text_w, b_text_h), _ = cv2.getTextSize(
                    text=b_text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1
                )
                b_tl, b_br = mean[:2], mean[2:4]
                b_tl = (int(b_tl[0]), int(b_tl[1]))
                b_br = (int(b_br[0]), int(b_br[1]))
                cv2.rectangle(
                    img=img, 
                    pt1=b_tl, 
                    pt2=b_br, 
                    color=(255, 255, 255),  # (0, 255, 0)
                    thickness=2
                )
                cv2.rectangle(
                    img=img, 
                    pt1=b_tl, 
                    pt2=(b_tl[0] + b_text_w - 1, b_tl[1] + b_text_h - 1), 
                    color=(255, 255, 255),  # (0, 255, 0)
                    thickness=-1
                )
                cv2.putText(
                    img=img,
                    text=b_text,
                    org=(b_tl[0], b_tl[1] + b_text_h - 1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 0),  # (0, 255, 0)
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
                # =================================================
                
                # Get transformed state mean vector.
                sub_mean = np.concatenate([
                    mean[:4],  # 4 x 1 
                    mean[5:9]  # 4 x 1
                ])
                sub_mean = R8x8.dot(sub_mean)
                sub_mean[:2] += t
                sub_mean[2:4] += t
                mean[:4] = sub_mean[:4]
                mean[5:9] = sub_mean[4:]

                # Display bbox for state mean vector after
                # applying the affine transformation.
                # =================================================
                a_text = f"A: {i + 1}"
                (a_text_w, a_text_h), _ = cv2.getTextSize(
                    text=a_text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1
                )
                a_tl, a_br = mean[:2], mean[2:4]
                a_tl = (int(a_tl[0]), int(a_tl[1]))
                a_br = (int(a_br[0]), int(a_br[1]))
                cv2.rectangle(
                    img=img, 
                    pt1=a_tl, 
                    pt2=a_br, 
                    color=(0, 0, 0),  # (0, 255, 0)
                    thickness=2
                )
                cv2.rectangle(
                    img=img, 
                    pt1=a_tl, 
                    pt2=(a_tl[0] + a_text_w - 1, a_tl[1] + a_text_h - 1), 
                    color=(0, 0, 0),  # (0, 255, 0)
                    thickness=-1
                )
                cv2.putText(
                    img=img,
                    text=a_text,
                    org=(a_tl[0], a_tl[1] + a_text_h - 1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),  # (0, 255, 0)
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
                # =================================================

                # Get transformed state covariance matrix.
                sub_cov = np.block([
                    [cov[:4, :4], cov[:4, 5:9]],
                    [cov[5:9, :4], cov[5:9, 5:9]]
                ])
                sub_cov = R8x8.dot(sub_cov).dot(R8x8.transpose())
                cov[:4, :4] = sub_cov[:4, :4]
                cov[:4, 5:9] = sub_cov[:4, 4:]
                cov[5:9, :4] = sub_cov[4:, :4]
                cov[5:9, 5:9] = sub_cov[4:, 4:]

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(
            measurement=self.xyxyd
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
            measurement=new_track.xyxyd
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
            measurement=new_track.xyxyd
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
    
    # NOTE: Replace every call to self.xyxy() method with
    # self.xyxy_dtc() method.
    @property
    def xyxy_dtc(self):
        """
        Convert bounding box to format 
        `(min x, min y, max x, max y, depth)`, i.e.,
        `(top left coord, bottom right coord, centre depth)`.
        """
        if self.mean is None:
            ret = self.xyxyd.copy()  # (u1, v1, u2, v2 d)
        else:
            ret = self.mean[:5].copy()  # kf (u1, v1, u2, v2, d)
        return ret


class BoTSORT_DTC(object):
    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate=30,
        fuse_first_associate: bool = False,
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
        h, w = img.shape[0:2]

        # Appends a new column to the dets array, where the new 
        # column contains integers representing the indices of 
        # the rows in the original dets array.
        dets = np.hstack([
            dets, 
            np.arange(len(dets)).reshape(-1, 1)
        ])

        # Remove bad detections
        # NOTE: Depth of bbox centre inserted in index 4. 
        # Therefore, confidence values are now available 
        # at index 5.
        confs = dets[:, 5]

        # find second round association detections
        second_mask = np.logical_and(
            confs > self.track_low_thresh, 
            confs < self.track_high_thresh
        )
        dets_second = dets[second_mask]

        # find first round association detections
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(det) 
                for (det) in np.array(dets_first)
            ]
        else:
            detections = []

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

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(
            tlista=tracked_stracks, 
            tlistb=self.lost_stracks
        )

        # Predict the current location with KF.
        STrack.multi_predict(strack_pool)

        # Fix camera motion.
        warp = self.cmc.apply(img, dets_first[:, :4])
        STrack.multi_gmc_dtc(strack_pool, warp, img)
        STrack.multi_gmc_dtc(unconfirmed, warp, img)

        # Associate with high score detection boxes.
        ious_dists = iou_distance_dtc(
            atracks=strack_pool, 
            btracks=detections
        )
        if self.fuse_first_associate:
            ious_dists = fuse_score_dtc(
              cost_matrix=ious_dists, 
              detections=detections
            )

        dists = ious_dists

        matches, u_track, u_detection = linear_assignment_dtc(
            dists, thresh=self.match_thresh
        )
        if matches.shape[0] > 0:
            avg_iou = np.around(
                np.mean(np.diag(iou_batch(
                    bboxes1=np.array([
                        strack_pool[matches[m_i, 0]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    bboxes2=np.array([
                        detections[matches[m_i, 1]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ])
                ))), 4
            )  # DEB
            avg_centroid_dist = np.around(
                np.mean(np.diag(centroid_batch(
                    bboxes1=np.array([
                        strack_pool[matches[m_i, 0]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    bboxes2=np.array([
                        detections[matches[m_i, 1]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    w=w, h=h
                ))), 4
            )  # DEB
            print("ASSOCIATION ROUND 1")  # DEB
            print(f"AVG IOU: {avg_iou}")  # DEB
            print(f"AVG CENTROID DIST: {avg_centroid_dist}")  # DEB
            print("-" * 75)  # DEB

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
        dists = iou_distance_dtc(
            atracks=r_tracked_stracks, 
            btracks=detections_second
        )
        matches, u_track, \
            u_detection_second = linear_assignment_dtc(
                cost_matrix=dists, 
                thresh=0.5
            )
        if matches.shape[0] > 0:
            avg_iou = np.around(
                np.mean(np.diag(iou_batch(
                    bboxes1=np.array([
                        r_tracked_stracks[matches[m_i, 0]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    bboxes2=np.array([
                        detections_second[matches[m_i, 1]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ])
                ))), 4
            )  # DEB
            avg_centroid_dist = np.around(
                np.mean(np.diag(centroid_batch(
                    bboxes1=np.array([
                        r_tracked_stracks[matches[m_i, 0]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    bboxes2=np.array([
                        detections_second[matches[m_i, 1]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    w=w, h=h
                ))), 4
            )  # DEB
            print("ASSOCIATION ROUND 2")  # DEB
            print(f"AVG IOU: {avg_iou}")  # DEB
            print(f"AVG CENTROID DIST: {avg_centroid_dist}")  # DEB
            print("-" * 75)  # DEB
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
        ious_dists = iou_distance_dtc(
            atracks=unconfirmed, 
            btracks=detections
        )

        ious_dists = fuse_score_dtc(
            cost_matrix=ious_dists, 
            detections=detections
        )
        
        dists = ious_dists

        matches, u_unconfirmed, \
            u_detection = linear_assignment_dtc(
                cost_matrix=dists, 
                thresh=0.7
            )
        if matches.shape[0] > 0:
            avg_iou = np.around(
                np.mean(np.diag(iou_batch(
                    bboxes1=np.array([
                        unconfirmed[matches[m_i, 0]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    bboxes2=np.array([
                        detections[matches[m_i, 1]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ])
                ))), 4
            )  # DEB
            avg_centroid_dist = np.around(
                np.mean(np.diag(centroid_batch(
                    bboxes1=np.array([
                        unconfirmed[matches[m_i, 0]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    bboxes2=np.array([
                        detections[matches[m_i, 1]].xyxy_dtc
                        for m_i in range(matches.shape[0])
                    ]),
                    w=w, h=h
                ))), 4
            )  # DEB
            print("UNCONFIRMED TRACKS")
            print(f"AVG IOU: {avg_iou}")  # DEB
            print(f"AVG CENTROID DIST: {avg_centroid_dist}")  # DEB
            print("-" * 75)  # DEB
        
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
            output.extend(t.xyxy_dtc)  # (x1, y1, x2, y2, depth)
            output.append(t.id)  # track ID
            output.append(t.score)  # confidence score
            output.append(t.cls)  # class ID
            output.append(t.det_ind)  # detection index
            outputs.append(output)

        outputs = np.asarray(outputs)
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
    pdist = iou_distance_dtc(stracksa, stracksb)
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
