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

        x, y, depth, w, h, vx, vy, vdepth, vw, vh

    contains the bounding box center position (x, y), width w, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box 
    location (x, y, depth, w, h) and is taken as direct observation 
    of the state space (linear observation model).

    """

    def __init__(self):
        # ndim, dt = 4, 1.  # ORIGINAL
        ndim, dt = 5, 1.

        # Create Kalman filter model matrices.
        
        # State transition matrix F
        self._motion_mat = np.array(
            [
                [1,  0,  0,  0,  0, dt,  0,  0,  0,  0],  # u + du
                [0,  1,  0,  0,  0,  0, dt,  0,  0,  0],  # v + dv
                [0,  0,  1,  0,  0,  0,  0, dt,  0,  0],  # depth + ddepth
                [0,  0,  0,  1,  0,  0,  0,  0, dt,  0],  # w + dw
                [0,  0,  0,  0,  1,  0,  0,  0,  0, dt],  # h + dh
                [0,  0,  0,  0,  0,  1,  0,  0,  0,  0],  # du
                [0,  0,  0,  0,  0,  0,  1,  0,  0,  0],  # dv
                [0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # ddepth
                [0,  0,  0,  0,  0,  0,  0,  0,  1,  0],  # dw
                [0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # dh

            ]
        )
        
        # Measurement matrix H
        self._update_mat = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # u
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # v
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # depth
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # w
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # h
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

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, depth, w, h) with center position (x, y),
            centre depth d, width w, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and covariance matrix (10x10
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        # Initial state, x = [u, v, depth, w, h, du, dv, ddepth, dw, dh]
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # State covariance P
        std = [
             2 * self._std_weight_position * measurement[3],  # u
             2 * self._std_weight_position * measurement[4],  # v
             2 * self._std_centre_depth,  # depth
             2 * self._std_weight_position * measurement[3],  # w
             2 * self._std_weight_position * measurement[4],  # h
            10 * self._std_weight_velocity * measurement[3],  # du
            10 * self._std_weight_velocity * measurement[4],  # dv
            10 * self._std_centre_depth,  # ddepth
            10 * self._std_weight_velocity * measurement[3],  # dw
            10 * self._std_weight_velocity * measurement[4]  #  dh
        ]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step: [u, v, depth, w, h, du, dv, ddepth, dw, dh].
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[4],
            self._std_centre_depth,
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[4]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[4],
            self._std_centre_depth,
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[4]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

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
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[4],
            self._std_centre_depth,
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[4]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

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
        std_pos = [
            self._std_weight_position * mean[:, 3],  # u
            self._std_weight_position * mean[:, 4],  # v
            np.repeat(self._std_centre_depth, mean.shape[0]),  # depth
            self._std_weight_position * mean[:, 3],  # width w
            self._std_weight_position * mean[:, 4]]  # height h
        std_vel = [
            self._std_weight_velocity * mean[:, 3],  # du
            self._std_weight_velocity * mean[:, 4],  # dv
            np.repeat(self._std_centre_depth, mean.shape[0]),  # ddepth
            self._std_weight_velocity * mean[:, 3],  # dw
            self._std_weight_velocity * mean[:, 4]]  # dh
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

    def update(self, mean, covariance, measurement):
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
            format (x, y, depth, a, h) where (x, y) is the bounding 
            box center position, depth is the centre depth, a the 
            aspect ratio, and h the height.
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
            mean, covariance = mean[:3], covariance[:3, :3]
            measurements = measurements[:, :3]

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