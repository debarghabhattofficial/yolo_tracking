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
import numpy as np

from boxmot.motion.kalman_filters.ocsort_kf import KalmanFilter
from boxmot.utils.association import associate_with_depth, linear_assignment
from boxmot.utils.iou import get_asso_func
from boxmot.utils.iou import run_asso_func


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and 
    returns z in the form [x, y, s, r] where (x, y) is the 
    centre of the box and s is the scale/area and r is the 
    aspect ratio
    """
    w = bbox[2] - bbox[0]  # width
    h = bbox[3] - bbox[1]  # height
    x = bbox[0] + w / 2.0  # x coordinate of the centre
    y = bbox[1] + h / 2.0  # y coordinate of the centre
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)  # aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))


# NOTE: Replace every call to convert_bbox_to_z() method
# with convert_bbox_to_z_with_depth() method.
def convert_bbox_to_z_with_depth(bbox, depth_mat=None):
    """
    This method takes a bounding box in the form 
    [u1, v1, u2, v2, w] (where (u1, v1) is top left corner,  
    (u2, v2) is the bottom right corner, and w is the depth),
    and returns z in the form [u, v, w, s, r] where (u, v) 
    is the centre of the box, w is the depth of the center, 
    and s is the scale/area and r is the aspect ratio.
    """ 
    # z will be of the form [u, v, s, r]
    # where (u, v) is the center of the box.
    centre_depth = bbox[4]
    bbox = np.delete(bbox, 4)
    z = convert_bbox_to_z(bbox)
    z = np.insert(z, 2, centre_depth)
    return z


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x, y, s, r] 
    and returns it in the form [x1, y1, x2, y2] where 
    (x1, y1) is the top left and (x2, y2) is the bottom 
    right.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([
            x[0] - w / 2.0, 
            x[1] - h / 2.0, 
            x[0] + w / 2.0, 
            x[1] + h / 2.0
        ]).reshape((1, 4))
    else:
        return np.array([
            x[0] - w / 2.0, 
            x[1] - h / 2.0, 
            x[0] + w / 2.0, 
            x[1] + h / 2.0, 
            score
        ]).reshape((1, 5))


# NOTE: Replace every call to convert_x_to_bbox() method
# with convert_x_to_bbox_with_depth() method.
def convert_x_to_bbox_with_depth(x, score=None):
    """
    This method takes a bounding box in the centre form
    [u, v, w, s, r] and returns it in the form 
    [u1, v1, u2, v2, w] where (u1, v1) is the top left,  
    (u2, v2) is the bottom right, and w is the centre.
    """
    centre_depth = x[2]
    x = np.delete(x, 2)
    bbox = convert_x_to_bbox(x, score=None)
    bbox = np.insert(bbox, 2, centre_depth)
    return bbox


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


# NOTE: Replace every call to speed_direction() method
# with speed_direction_with_depth() method.
def speed_direction_with_depth(bbox1, bbox2):
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
        # No. of state variables (x) = 9
        # Here, we have:
        #   - 3 center coordinates (u, v, w)
        #   - 1 scale/area (s)
        #   - 1 aspect ratio (r)
        #   - 4 change in velocity (du, dv, dw, ds)
        # No. of measuement varaible (z) = 5
        # Here, we have:
        #   - 3 center coordinates (u, v, w)
        #   - 1 scale/area (s)
        #   - 1 aspect ratio (r)
        self.det_ind = det_ind
        self.kf = KalmanFilter(dim_x=9, dim_z=5)

        # State transition matrix F
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 0],  # Center u
                [0, 1, 0, 0, 0, 0, 1, 0, 0],  # Center v
                [0, 0, 1, 0, 0, 0, 0, 1, 0],  # Center w
                [0, 0, 0, 1, 0, 0, 0, 0, 1],  # Scale s
                [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Aspect ratio r
                [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Velocity for u
                [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Velocity for v
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Velocity for w
                [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Velocity for scale s
            ]
        )

        # Measurement matrix H
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Center u
                [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Center v
                [0, 0, 1, 0, 0, 0, 0, 0, 0],  # Center w
                [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Scale s
                [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Aspect ratio r
            ]
        )

        # Measurement noise covariance R
        self.kf.R[3:, 3:] *= 10.0  # Increase the measurement noise for scale and aspect ratio.
        
        # State covariance P
        self.kf.P[5:, 5:] *= 1000.0  # Increase the uncertainty for the unobservable initial velocities.
        self.kf.P *= 10.0

        # Process noise covariance Q
        self.kf.Q[-1, -1] *= 0.01  # Lower the uncertainty in process's scale velocity.
        self.kf.Q[5:, 5:] *= 0.01  # Lower the uncertainty all velocities of the process function.

        # Initial state [u, v, w, s, r, du, dv, dw, ds]
        self.kf.x[:5] = convert_bbox_to_z_with_depth(bbox)  # Convert bounding box to state variables.
        
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
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
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
                # Estimate the track speed direction with 
                # observations \Delta t steps away.
                # self.velocity = speed_direction(previous_box, bbox)  # ORIGINAL
                self.velocity = speed_direction_with_depth(previous_box, bbox)  # DEB

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
            # self.kf.update(convert_bbox_to_z(bbox))  # ORIGINAL
            self.kf.update(convert_bbox_to_z_with_depth(bbox))  # DEB
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted 
        bounding box estimate.
        """
        # Scale and scale velocity is represented by the 
        # state variables at index 3 and 8 respectively.
        if (self.kf.x[8] + self.kf.x[3]) <= 0:
            self.kf.x[8] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        # self.history.append(convert_x_to_bbox(self.kf.x))  # ORIGINAL
        self.history.append(convert_x_to_bbox_with_depth(self.kf.x))  # DEB
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        # return convert_x_to_bbox(self.kf.x)  # ORIGINAL
        return convert_x_to_bbox_with_depth(self.kf.x)  # DEB


class OCSortRGBD(object):
    def __init__(
        self,
        per_class=True,
        det_thresh=0.2,
        max_age=30,
        min_hits=3,
        asso_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
    ):
        """
        Sets key parameters for SORT
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

    def update(self, dets, img):
        """
        Params:
          dets - a numpy array of detections in the format 
            [
                [x1, y1, x2, y2, depth, score, class],
                [x1, y1, x2, y2, depth, score, class],
                ...
            ]
        Requires: this method must be called once for each 
        frame even with empty detections (use np.empty((0, 5)) 
        for frames without detections).
        Returns the a similar array, where the last column 
        is the object ID.
        NOTE: The number of objects returned may differ 
        from the number of detections provided.
        """

        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray."
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two."
        assert (
            dets.shape[1] == 7
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 7."

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

        # get predicted locations from existing trackers.
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
        # NOTE: Original code called associate() method instead of
        # associate_with_depth() method.
        matched, unmatched_dets, unmatched_trks = associate_with_depth(
            detections=dets[:, 0:6], 
            trackers=trks, 
            assoc_func=self.asso_func, 
            iou_threshold=self.asso_threshold, 
            velocities=velocities, 
            previous_obs=k_observations, 
            vdc_weights=self.inertia, 
            w=w, 
            h=h
        )
        for m in matched:
            self.trackers[m[1]].update(
                bbox=dets[m[0], :6], 
                cls=dets[m[0], 6], 
                det_ind=dets[m[0], 7]
            )

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
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.trackers[trk_ind].update(
                        bbox=dets[det_ind, :6], 
                        cls=ets[det_ind, 6], 
                        det_ind=dets[det_ind, 7]
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
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
