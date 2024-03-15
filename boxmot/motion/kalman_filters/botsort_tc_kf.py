# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


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

        u1, v1, u2, v2, du1, dv1, du2, dv2

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

        # Create Kalman filter model matrices.
        
        # State transition matrix F
        self._motion_mat = np.array(
            [
                [1,  0,  0,  0, dt,  0,  0,  0],  # u1 + du1
                [0,  1,  0,  0,  0, dt,  0,  0],  # v1 + dv1
                [0,  0,  1,  0,  0,  0, dt,  0],  # u2 + du2
                [0,  0,  0,  1,  0,  0,  0, dt],  # v2 + dv2
                [0,  0,  0,  0,  1,  0,  0,  0],  # du1
                [0,  0,  0,  0,  0,  1,  0,  0],  # dv1
                [0,  0,  0,  0,  0,  0,  1,  0],  # du2
                [0,  0,  0,  0,  0,  0,  0,  1],  # dv2
            ]
        )
        
        # Measurement matrix H
        self._update_mat = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],  # u1
                [0, 1, 0, 0, 0, 0, 0, 0],  # v1
                [0, 0, 1, 0, 0, 0, 0, 0],  # u2
                [0, 0, 0, 1, 0, 0, 0, 0],  # v2
            ]
        )

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bbox top left (u1, v1), and 
            bottom right (u2, v2).

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
            10 * self._std_weight_velocity * w,  # du1
            10 * self._std_weight_velocity * h,  # dv1
            10 * self._std_weight_velocity * w,  # du2
            10 * self._std_weight_velocity * h,  # dv2
        ]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step: [u1, v1, u2, v2, du1, dv1, du2, dv2].
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        w = mean[2] - mean[0]
        h = mean[3] - mean[1]

        std_pos = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h
        ]
        std_vel = [
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
            self._std_weight_velocity * w,
            self._std_weight_velocity * h
        ]
        motion_cov = np.diag(
            np.square(np.r_[std_pos, std_vel])
        )

        # Calculate prediction (or prior).
        # Predicted state vector x = [u, v, depth, w, h, du, dv, ddepth, dw, dh]
        mean = np.dot(mean, self._motion_mat.T)
        # Predicted state covariance P = FPF' + Q
        covariance = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)
        ) + motion_cov

        return mean, covariance

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
        w = mean[2] - mean[0]
        h = mean[3] - mean[1]
        std = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        innovation_cov = covariance + innovation_cov
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
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],  # du1
            self._std_weight_velocity * mean[:, 4],  # dv1
            self._std_weight_velocity * mean[:, 3],  # du2
            self._std_weight_velocity * mean[:, 4],  # dv2
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        # Calculate prediction (or prior).
        # Predicted state vector x = [u1, v1, u2, v2, du1, dv1, du2, dv2]
        mean = np.dot(mean, self._motion_mat.T)
        # Predicted state covariance P = FPF' + Q
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (10 dimensional).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        measurement : ndarray
            The 5 dimensional measurement vector (u1, v1, u2, v2), 
            where 
                (u1, v1) is the top left position, and
                (u2, v2) is the bottom right position
            of the bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, 
            lower=True, 
            check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), 
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

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
            format (u1, v1, u2, v2) where (u1, v1) is bbox top
            left, and (u2, v2) is the bottom right position.
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
            mean, covariance = mean[:4], covariance[:4, :4]
            measurements = measurements[:, :4]

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