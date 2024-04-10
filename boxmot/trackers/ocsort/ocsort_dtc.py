# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

# NOTE: Here, we are fusing depth of the centre of
# bounding box with the state variables of the Kalman
# filters. This is done by modifying the structure of
# detected bounding boxes to include depth information.
# Specifically, the bounding box is represented as
# [x1, y1, x2, y2, depth, score, class]. The depth 
# corresponds to the depth of the centre of the bounding
# box.

"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
import logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

import numpy as np

from boxmot.motion.kalman_filters.ocsort_dtc_kf import KalmanFilter
from boxmot.utils.association import associate_dtc, linear_assignment
from boxmot.utils.iou import get_asso_func
from boxmot.utils.iou import run_asso_func
from boxmot.motion.cmc.sof import SparseOptFlow

# Following package and related code was
# added by DEB to print the CMC-related
# before and after KF state mean bboxes
# for debugging purposes.
# =======================================================
import cv2
from boxmot.utils.iou import iou_batch, centroid_batch
# =======================================================


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


# @nb.njit(fastmath=True, cache=True)
def convert_bbox_to_z_dtc(bbox):
    """
    Takes a bounding box in the form 
    [u1, v1, u2, v2, d, conf] 
    and  returns z in the form [u1, v1, u2, v2, d] 
    where (u1, v1) is the bbox top left, (u2, v2) is the
    bbox top right, and d is the bbox centre depth.
    """
    u1 = bbox[0]
    v1 = bbox[1]
    u2 = bbox[2]
    v2 = bbox[3]
    d = bbox[4]
    z = np.array([u1, v1, u2, v2, d]).reshape((5, 1))
    return z


# @nb.njit(fastmath=True, cache=True)
def convert_x_to_bbox_dtc(x, score=None):
    """
    Takes a bounding box in the tlbr form 
    [u1, v1, u2, v2, d] 
    and returns it in the form [u1, v1, u2, v2, d] where 
    (u1, v1) is the top left and (u2, v2) is the bottom 
    right.
    """
    if score is None:
        return np.array([
            x[0], 
            x[1], 
            x[2], 
            x[3],
            x[4]
        ]).reshape((1, 5))
    else:
        return np.array([
            x[0], 
            x[1], 
            x[2], 
            x[3], 
            x[4],
            score
        ]).reshape((1, 6))


# NOTE: Replace every call to speed_direction() method
# with speed_direction_dtc() method.
# @nb.njit(fastmath=True, cache=True)
def speed_direction_dtc(bbox1, bbox2):
    # Coordinates for bbox1 centre
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cz1 = bbox1[4]

    # Coordinates for bbox2 centre
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    cz2 = bbox2[4]

    speed = np.array([cx2 - cx1, cy2 - cy1, cz2 - cz1])
    norm = np.sqrt(
        ((cx2 - cx1) ** 2) + ((cy2 - cy1) ** 2) + ((cz2 - cz1) ** 2)
    ) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual 
    tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, cls, det_ind, delta_t=3):
        """
        Initialises a tracker using initial bounding box.

        """
        # Define constant velocity model.
        # No. of state variables (x) = 10
        # Here, we have:
        #   - 4 tlbr coordinates ((u1, v1) & (u2, v2))
        #   - 1 centre depth (d)
        #   - 4 tlbr velocity ((du1, dv1) & (du2, dv2))
        #   - 1 depth velocity (dd)
        # No. of measuement varaible (z) = 5
        # Here, we have:
        #   - 4 tlbr coordinates ((u1, v1) & (u2, v2))
        #   - 1 centre depth (d)
        self.det_ind = det_ind
        self.kf = KalmanFilter(dim_x=10, dim_z=5)

        # State transition matrix F
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # tl u1
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # tl v1
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # br u2
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # br v2
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # centre depth d
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # velocity tl du1
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # velocity tl dv1
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # velocity br du2
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # velocity br dv2
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # velocity centre depth dd
            ]
        )

        # Measurement matrix H
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # tl u1
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # tl v1
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # br u2
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # br v2
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # centre depth d
            ]
        )

        # # Measurement noise covariance R
        # self.kf.R[3:, 3:] *= 10.0  # Increase the measurement noise for scale and aspect ratio.
        
        # State covariance P
        self.kf.P[5:, 5:] *= 1000.0  # Increase the uncertainty for the unobservable initial velocities.
        self.kf.P *= 10.0

        # Process noise covariance Q
        # self.kf.Q[-1, -1] *= 0.01  # Lower the uncertainty in process's scale velocity.
        self.kf.Q[5:, 5:] *= 0.01  # Lower the uncertainty all velocities of the process function.

        # Initial state [u1, v1, u2, v2, d, du1, dv1, du2, dv2, dd]
        self.kf.x[:5] =  convert_bbox_to_z_dtc(bbox)  # Convert bounding box to state variables.
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[-1]
        self.cls = cls
        """
        NOTE: [-1, -1, -1, -1, -1, -1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.

        Also, please keep in the mind that the original algorithm 
        had a 5 element array for the last observation (since the bbox
        for the trackers were represented by [u1, v1, u2, v2, score]).
        In our modified algorithm, where we make use of the bbox 
        centre depth, we added another element to the placeholder. In
        other words, we have a placholder of size 6 as follows:
        [u1, v1, u2, v2, d, score] where d is depth.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox, cls, det_ind):
        """
        Updates the state vector with observed bbox.
        INPUT:
            bbox: List, [u1, v1, u2, v2, d, conf]
            cls: int
            det_ind: int
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
                # Estimate the track speed direction with 
                # observations \Delta t steps away.
                self.velocity = speed_direction_dtc(
                    bbox1=previous_box, 
                    bbox2=bbox
                )

            # Insert new observations. This is a ugly way 
            # to maintain both self.observations and 
            # self.history_observations. Bear it for the 
            # moment.
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(z=convert_bbox_to_z_dtc(bbox)) 
        else:
            self.kf.update(z=bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted 
        bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox_dtc(x=self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox_dtc(x=self.kf.x)
    
    @staticmethod
    def multi_gmc_dtc(stracks, H=np.eye(2, 3), img=None):
        # NOTE: Make changes to this method to account for
        # camera motion compensation while updating the
        # state of the tracklets.
        if len(stracks) > 0:
            multi_mean = np.asarray([st.kf.x.copy() for st in stracks])
            multi_covariance = np.asarray([st.kf.P for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2].reshape(-1, 1)

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

                stracks[i].kf.x = mean
                stracks[i].kf.P = cov

    @staticmethod
    def multi_gmc_lo_dtc(stracks, H=np.eye(2, 3), img=None):
        # NOTE: Make changes to this method to account for
        # camera motion compensation while updating the
        # state of the tracklets.
        if len(stracks) > 0:
            multi_lo = np.asarray([st.last_observation for st in stracks])

            R = H[:2, :2]
            R4x4 = np.kron(np.eye(2, dtype=float), R)
            t = H[:2, 2]

            for i, lo in enumerate(multi_lo):
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
                b_tl, b_br = lo[:2], lo[2:4]
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

                # Get transformed state last observation vector.
                sub_lo = lo[:4]
                sub_lo = R4x4.dot(sub_lo)
                sub_lo[:2] += t
                sub_lo[2:4] += t
                lo[:4] = sub_lo[:4]

                # Display bbox for state last observation vector after
                # applying the affine transformation.
                # =================================================
                a_text = f"A: {i + 1}"
                (a_text_w, a_text_h), _ = cv2.getTextSize(
                    text=a_text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1
                )
                a_tl, a_br = lo[:2], lo[2:4]
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

                stracks[i].last_observation = lo


class OCSORT_DTC(object):
    def __init__(self,
                 det_thresh=0.2,
                 max_age=30,
                 min_hits=3,
                 asso_threshold=0.2,
                 delta_t=3,
                 asso_func="iou",
                 inertia=0.2,
                 use_byte=False):
        """
        Sets key parameters for SORT.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.asso_threshold = asso_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0
        self.cmc = SparseOptFlow()

    def update(self, dets, img):
        """
        Params:
          dets - a numpy array of detections in the format 
            [
                [u1, v1, u2, v2, depth, score, class],
                [u1, v1, u2, v2, depth, score, class],
                ...
            ]
        Requires: this method must be called once for each 
        frame even with empty detections (use np.empty((0, 6)) 
        for frames without detections).
        Returns the a similar array, where the last column 
        is the object ID.
        NOTE: The number of objects returned may differ 
        from the number of detections provided.
        """

        err_msg = f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray."
        assert isinstance(
            dets, np.ndarray
        ), err_msg

        err_msg = f"Unsupported 'dets' dimensions '{len(dets.shape)}', valid number of dimensions is two."
        assert (
            len(dets.shape) == 2
        ), err_msg
        
        err_msg = f"Unsupported 'dets' 2nd dimension length '{dets.shape[1]}', valid lenghts is 7. It is '{dets}'"
        assert (
            dets.shape[1] == 7
        ), err_msg

        self.frame_count += 1
        h, w = img.shape[0:2]

        # Appends a new column to the dets array, where the new 
        # column contains integers representing the indices of 
        # the rows in the original dets array.
        dets = np.hstack([
            dets, 
            np.arange(len(dets)).reshape(-1, 1)
        ])
        # Depth of bbox centre inserted in index 4.
        # Confidence values are now available at index 5.
        confs = dets[:, 5]

        inds_low = confs > 0.1
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(
            inds_low, inds_high
        )  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = confs > self.det_thresh
        dets = dets[remain_inds]

        # Get predicted locations from existing trackers.
        # Depth of bbox centre inserted in index 4.
        # Therefore, size along axis=1 increases by 1 to 6.
        trks = np.zeros((len(self.trackers), 6))  
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Fix camera motion.
        warp = self.cmc.apply(img, dets[:, :4])
        KalmanBoxTracker.multi_gmc_dtc(self.trackers, warp, img)
        KalmanBoxTracker.multi_gmc_lo_dtc(self.trackers, warp, img)

        # NOTE: In the else condition, np.array((0, 0)) changed 
        # to np.array((0, 0, 0)) to account for depth.
        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0, 0))
                for trk in self.trackers
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.trackers
            ]
        )

        """
        First round of association
        """
        # NOTE: Original OC-SORT + RGBD code called associate_dtc() 
        # method instead of associate_dtc() method.
        matched, unmatched_dets, unmatched_trks = associate_dtc(
            detections=dets[:, 0:6], 
            trackers=trks, 
            asso_func=self.asso_func, 
            iou_threshold=self.asso_threshold, 
            velocities=velocities, 
            previous_obs=k_observations, 
            vdc_weight=self.inertia, 
            w=w, 
            h=h
        )
        avg_iou = np.around(
            np.mean(np.diag(iou_batch(
                bboxes1=dets[matched[:, 0], :4],
                bboxes2=trks[matched[:, 1], :4]
            ))), 4
        )  # DEB
        avg_centroid_dist = np.around(
            np.mean(np.diag(centroid_batch(
                bboxes1=dets[matched[:, 0], :4],
                bboxes2=trks[matched[:, 1], :4],
                w=w, h=h
            ))), 4
        )  # DEB
        print("ASSOCIATION ROUND 1")  # DEB
        print(f"AVG IOU: {avg_iou}")  # DEB
        print(f"AVG CENTROID DIST: {avg_centroid_dist}")  # DEB
        print("-" * 75)  # DEB
        for m in matched:
            self.trackers[m[1]].update(
                bbox=dets[m[0], :6], 
                cls=dets[m[0], 6], 
                det_ind=dets[m[0], 7]
            )

        """
        Second round of associaton by OCR 
        (Observation-Centric Re-update)
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
                avg_iou = np.around(
                    np.mean(np.diag(iou_batch(
                        bboxes1=dets_second[matched_indices[:, 0], :4],
                        bboxes2=u_trks[matched_indices[:, 1], :4]
                    ))), 4
                )  # DEB
                avg_centroid_dist = np.around(
                    np.mean(np.diag(centroid_batch(
                        bboxes1=dets_second[matched_indices[:, 0], :4],
                        bboxes2=u_trks[matched_indices[:, 1], :4],
                        w=w, h=h
                    ))), 4
                )  # DEB
                print("ASSOCIATION ROUND 2")  # DEB
                print(f"AVG IOU: {avg_iou}")  # DEB
                print(f"AVG CENTROID DIST: {avg_centroid_dist}")  # DEB
                print("-" * 75)  # DEB
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.trackers[trk_ind].update(
                        bbox=dets_second[det_ind, :6], 
                        cls=dets_second[det_ind, 6], 
                        det_ind=dets_second[det_ind, 7]
                    )
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = run_asso_func(self.asso_func, left_dets, left_trks, w, h)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.asso_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                avg_iou = np.around(
                    np.mean(np.diag(iou_batch(
                        bboxes1=dets[unmatched_dets[rematched_indices[:, 0]], :4],
                        bboxes2=last_boxes[unmatched_trks[rematched_indices[:, 1]], :4]
                    ))), 4
                )  # DEB
                avg_centroid_dist = np.around(
                    np.mean(np.diag(centroid_batch(
                        bboxes1=dets[unmatched_dets[rematched_indices[:, 0]], :4],
                        bboxes2=last_boxes[unmatched_trks[rematched_indices[:, 1]], :4],
                        w=w, h=h
                    ))), 4
                )  # DEB
                print("UNMATCHED DETECTIONS")  # DEB
                print(f"AVG IOU: {avg_iou}")  # DEB
                print(f"AVG CENTROID DIST: {avg_centroid_dist}")  # DEB
                print("-" * 75)  # DEB
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.trackers[trk_ind].update(
                        bbox=dets[det_ind, :6], 
                        cls=dets[det_ind, 6], 
                        det_ind=dets[det_ind, 7]
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, 
                    np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, 
                    np.array(to_remove_trk_indices)
                )

        for m in unmatched_trks:
            self.trackers[m].update(
                bbox=None, 
                cls=None, 
                det_ind=None
            )

        # Create and initialise new trackers for unmatched 
        # detections.
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                bbox=dets[i, :6], 
                cls=dets[i, 6], 
                det_ind=dets[i, 7], 
                delta_t=self.delta_t
            )
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                This is optional to use the recent observation 
                or the kalman filter prediction. 
                We didn't notice significant difference here.
                """
                d = trk.last_observation[:5]
            if (trk.time_since_update < 1) and (
                (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)
            ):
                # +1 as MOT benchmark requires positive
                ret.append(
                    np.concatenate((
                        d, 
                        [trk.id + 1], 
                        [trk.conf], 
                        [trk.cls], 
                        [trk.det_ind]
                    )).reshape(1, -1)
                )
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])
