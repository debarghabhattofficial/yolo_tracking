# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
from copy import deepcopy


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in RGBD image 
    space.

    The 8-dimensional state space

        u1, v1, u2, v2, d, du1, dv1, du2, dv2, dd

    contains the bounding box top left (u1, v1),
    bottom right (u2, v2), centre depth d, and their 
    respective velocities.

    Object motion follows a constant velocity model. The bounding box 
    location (u1, v1, u2, v2, d) and is taken as direct observation 
    of the state space (linear observation model).

    """

    def __init__(self):
        # ndim, dt = 4, 1.  # ORIGINAL
        ndim, dt = 5, 1.

        self.mean = np.zeros(10)  # x = [u1, v1, u2, v2, d, du1, dv1, du2, dv2, dd]
        self.covariance = np.eye(10)

        # Create Kalman filter model matrices.
        
        # State transition matrix F
        self._motion_mat = np.array(
            [
                [1,  0,  0,  0,  0, dt,  0,  0,  0,  0],  # u1 + du1
                [0,  1,  0,  0,  0,  0, dt,  0,  0,  0],  # v1 + dv1
                [0,  0,  1,  0,  0,  0,  0, dt,  0,  0],  # u2 + du2
                [0,  0,  0,  1,  0,  0,  0,  0, dt,  0],  # v2 + dv2
                [0,  0,  0,  0,  1,  0,  0,  0,  0, dt],  # d + dd
                [0,  0,  0,  0,  0,  1,  0,  0,  0,  0],  # du1
                [0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # dv1
                [0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # du2
                [0,  0,  0,  0,  0,  0,  0,  0,  1,  0],  # dv2
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # dd

            ]
        )
        
        # Measurement matrix H
        self._update_mat = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # u1
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # v1
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # u2
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # v2
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # d
            ]
        )

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        
        # Uncertainty in depth measurement. This is a hack which
        # makes the depth measurement uncertainty constant irrespective
        # of the current measurement.
        self._std_centre_depth = 0.01 ** (1/2)  # 0.01 ** (1/2)  # Here, we assume variance is 1.0.

        # Keep all observations. 
        self.history_obs = []

        self.attr_saved = None
        self.observed = False 

    def _init_mat(self, dt):
        # acceleration-based process noise
        acc_cov = np.diag([0.25 * dt**4] * 4 + [dt**2] * 4)
        acc_cov[4:, :4] = np.eye(4) * (0.5 * dt**3)
        acc_cov[:4, 4:] = np.eye(4) * (0.5 * dt**3)

        update_mat = np.eye(5, 10)
        motion_mat = np.eye(10)
        for i in range(4):
            motion_mat[i, i + 4] = self.vel_coupling * dt
            motion_mat [i, (i + 2) % 4 + 4] = (1. - self.vel_coupling) * dt
            motion_mat [i + 4, i + 4] = 0.5**(dt / self.vel_half_life)
        return acc_cov, update_mat, motion_mat 

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bbox top left (u1, v1), bottom right (u2, v2),,
            and centre depth d.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and 
            covariance matrix (10x10 dimensional) of the 
            new track. Unobserved velocities are 
            initialized to 0 mean.

        """
        # Initial state, x = [u, v, depth, w, h, du, dv, ddepth, dw, dh]
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        w = measurement[2] - measurement[0]
        h = measurement[3] - measurement[1]

        # State covariance P
        std = [
             2 * self._std_weight_position * w,  # u1
             2 * self._std_weight_position * h,  # v1
             2 * self._std_weight_position * w,  # u2
             2 * self._std_weight_position * h,  # v2
             2 * self._std_centre_depth,  # d
            10 * self._std_weight_velocity * w,  # du1
            10 * self._std_weight_velocity * h,  # dv1
            10 * self._std_weight_velocity * w,  # du2
            10 * self._std_weight_velocity * h,  # dv2
            10 * self._std_centre_depth,  # dd
        ]
        covariance = np.diag(np.square(std))

        self.mean = mean  # DEB
        self.covariance = covariance  # DEB

        # return mean, covariance  # ORIGINAL
        return

    def predict(self):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step: [u1, v1, u2, v2, d, du1, dv1, du2, dv2, dd].
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        w = self.mean[2] - self.mean[0]
        h = self.mean[3] - self.mean[1]

        std_pos = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_centre_depth
        ]
        std_vel = [
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
            self._std_centre_depth
        ]
        motion_cov = np.diag(
            np.square(np.r_[std_pos, std_vel])
        )
        # Predicted state covariance P = FPF' + Q
        covariance = np.linalg.multi_dot(
            (self._motion_mat, self.covariance, self._motion_mat.T)
        ) + motion_cov

        # Calculate prediction (or prior).
        # Predicted state vector x = [u, v, depth, w, h, du, dv, ddepth, dw, dh]
        mean = np.dot(self.mean, self._motion_mat.T)

        self.mean = mean  # DEB
        self.covariance = covariance  # DEB

        # return mean, covariance  # ORIGINAL
        return  # DEB

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.
        In other words, it:
            - projects the prior state mean vector to measuement space 
              using the Hx form.
            - projects the prior state covariance matrix to measurement 
              space using the HPH' + R form.
        Here, x is the prior state mean vector, and P is the prior state
        covariance matrix.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the 
            given state estimate.

        """
        # Calculate new covariance.
        w = mean[2] - mean[0]
        h = mean[3] - mean[1]
        std = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_centre_depth
        ]
        innovation_cov = np.diag(np.square(std))
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        innovation_cov = covariance + innovation_cov
        # Calculate new mean.
        mean = np.dot(self._update_mat, mean)

        return mean, innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        ws = mean[:, 2] - mean[:, 0]
        hs = mean[:, 3] - mean[:, 1]
        std_pos = [
            self._std_weight_position * ws,  # u1
            self._std_weight_position * hs,  # v1
            self._std_weight_position * ws,  # u2
            self._std_weight_position * hs,  # v2
            np.repeat(self._std_centre_depth, mean.shape[0]),  # d
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],  # du1
            self._std_weight_velocity * mean[:, 4],  # dv1
            self._std_weight_velocity * mean[:, 3],  # du2
            self._std_weight_velocity * mean[:, 4],  # dv2
            np.repeat(self._std_centre_depth, mean.shape[0]),  # dd
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        # Calculate prediction (or prior).
        # Predicted state vector x = [u, v, depth, w, h, du, dv, ddepth, dw, dh]
        mean = np.dot(mean, self._motion_mat.T)
        # Predicted state covariance P = FPF' + Q
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance
    
    def freeze(self):
        """
        Save the parameters before non-observation forward.
        """
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self):
        if self.attr_saved is not None:
            new_history = deepcopy(self.history_obs)  # ORIGINAL
            # new_history = np.copy(self.history_obs)  # DEB
            self.__dict__ = self.attr_saved
            # self.history_obs = new_history 
            self.history_obs = self.history_obs[:-1]
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur)==0)[0]
            index_1 = indices[-2]
            index_2 = indices[-1]
            box_1 = new_history[index_1]
            u1_1, v1_1, u2_1, v2_1, depth_1 = box_1
            box_2 = new_history[index_2]
            u1_2, v1_2, u2_2, v2_2, depth_2 = box_2
            time_gap = index_2 - index_1
            du1 = (u1_2 - u1_1) / time_gap
            dv1 = (v1_2 - v1_1) / time_gap
            du2 = (u2_2 - u2_1) / time_gap
            dv2 = (v2_2 - v2_1) / time_gap
            ddepth = (depth_2 - depth_1) / time_gap
            for i in range(index_2 - index_1):
                """
                The default virtual trajectory generation is by linear
                motion (constant speed hypothesis), you could modify this 
                part to implement your own. 
                """
                u1_n = u1_1 + (i+1) * du1
                v1_n = v1_1 + (i+1) * dv1
                u2_n = u2_1 + (i+1) * du2
                v2_n = v2_1 + (i+1) * dv2
                depth_n = depth_1 + (i+1) * ddepth
                new_box = np.array([u1_n, v1_n, u2_n, v2_n, depth_n]).reshape((5, 1))
                """
                I still use predict-update loop here to refresh the 
                parameters, but this can be faster by directly 
                modifying the internal parameters as suggested in 
                the paper. I keep this naive but slow way for 
                easy read and understanding
                """
                self.update(measurement=new_box)
                if not i == (index_2 - index_1 - 1):
                    self.predict()
    
    def update(self, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (10 dimensional).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        measurement : ndarray
            The 5 dimensional measurement vector (x, y, depth, w, h), 
            where 
                (x, y) is the center position, 
                depth is the dentre depth, 
                w the width, and 
                h the height 
            of the bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # The following code block inside `===` is written by DEB.
        # It is adapted from the original KF code of OC-SORT.
        # ======================================================
        # Append the observation.
        self.history_obs.append(measurement)

        if measurement is None:
            if self.observed:
                """
                Got no observation so freeze the current parameters 
                for future potential online smoothing.
                """
                self.freeze()
            self.observed = False
            return

        # self.observed = True
        if not self.observed:
            """
                Get observation, use online smoothing to re-update parameters
            """
            self.unfreeze()
        self.observed = True
        # ======================================================

        projected_mean, projected_cov = self.project(
            mean=self.mean, covariance=self.covariance
        )

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, 
            lower=True, 
            check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), 
            np.dot(self.covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_covariance = self.covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        new_mean = self.mean + np.dot(innovation, kalman_gain.T)

        self.mean == new_mean  # DEB
        self.covariance = new_covariance  # DEB

        # return new_mean, new_covariance  # ORIGINAL
        return  # DEB

    def gating_distance(self, 
                        mean, 
                        covariance, 
                        measurements,
                        only_position=False, 
                        metric="maha"):
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. 
        If `only_position` is False, the chi-square distribution has 
        4 degrees of freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (10 dimensional).
        covariance : ndarray
            Covariance of the state distribution (10x10 dimensional).
        measurements : ndarray
            An Nx5 dimensional matrix of N measurements, each in
            format (u1, v1, u2, v2, d) where (u1, v1) is bbox top
            left, (u2, v2) is the bottom right, and d is the centre 
            depth.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the 
            bounding box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element 
            contains the squared Mahalanobis distance between 
            (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:5], covariance[:5, :5]
            measurements = measurements[:, :5]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                a=cholesky_factor, 
                b=d.T, 
                lower=True, 
                check_finite=False,
                overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("Invalid distance metric.")