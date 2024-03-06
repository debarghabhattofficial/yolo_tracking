from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

# Define the number of objects to track
num_objects = 2  # Update this based on the number of objects you want to track

# Define the number of state variables for each object
num_state_variables = 9  # 3 (center coordinates) + 1 (scale) + 1 (aspect ratio) + 4 (change in velocity)

# Initialize Kalman filter for each object
kalman_filters = [
    KalmanFilter(dim_x=num_state_variables, dim_z=5) 
    for _ in range(num_objects)
]

for kalman in kalman_filters:
    # State transition matrix F (constant velocity model)
    kalman.F = np.array(
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
    # np.eye(num_state_variables)
    # kalman.F[:3, 3:6] = np.eye(3)  # Change in velocity affects center coordinates
    # kalman.F[3:6, 6:9] = np.eye(3)  # Change in velocity affects scale
    # kalman.F[6:9, 6:9] = np.eye(3)  # Change in velocity affects aspect ratio

    # Measurement matrix H
    kalman.H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Center u
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Center v
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  # Center w
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Scale s
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Aspect ratio r
        ]
    )
    # kalman.H = np.concatenate([np.eye(4), np.zeros((4, num_state_variables - 4))], axis=1)

    # Measurement noise covariance R
    kalman.R[3:, 3:] *= 10.0  # Increase the measurement noise for scale and aspect ratio.
    # kalman.R = np.eye(4) * 1e-2

    # State covariance P
    kalman.P[5:, 5:] *= 1000.0  # Increase the uncertainty for the unobservable initial velocities.
    kalman.P *= 10.0 # Increase the uncertainty in all state variables.

    # Process noise covariance Q
    kalman.Q[-1, -1] *= 0.01  # Lower the uncertainty in process's scale velocity.
    kalman.Q[5:, 5:] *= 0.01  # Lower the uncertainty all velocities of the process function.
    # kalman.Q = np.eye(num_state_variables) * 1e-4

    # Initial state [u, v, w, s, r, du, dv, dw, ds]
    kalman.x[:5] = convert_bbox_to_z(bbox)  # Convert bounding box to state variables.
    # kalman.x = np.zeros(num_state_variables)

# Generate simulated data (replace this with actual measurements)
num_steps = 100
true_states = [np.zeros((num_steps, num_state_variables), dtype=np.float32) for _ in range(num_objects)]
measurements = [np.zeros((num_steps, 4), dtype=np.float32) for _ in range(num_objects)]

for i in range(num_steps):
    for obj in range(num_objects):
        # Simulate true state (constant velocity)
        true_states[obj][i, :3] = true_states[obj][i-1, :3] + true_states[obj][i-1, 6:9]  # Center coordinates
        true_states[obj][i, 3] = true_states[obj][i-1, 3] + true_states[obj][i-1, 8]  # Scale
        true_states[obj][i, 4] = true_states[obj][i-1, 4]  # Aspect ratio
        true_states[obj][i, 5:9] = true_states[obj][i-1, 5:9]  # Change in velocity

        # Simulate noisy measurements (add measurement noise)
        measurements[obj][i] = true_states[obj][i, :4] + np.random.normal(0, 0.1, 4)

        # Predict step
        kalman_filters[obj].predict()

        # Update step
        kalman_filters[obj].update(measurements[obj][i])

# Plot the results
plt.figure(figsize=(15, 15))

for obj in range(num_objects):
    plt.subplot(num_objects, 1, obj + 1)
    plt.plot(true_states[obj][:, 0], label='True Center U', marker='o')
    plt.plot(true_states[obj][:, 1], label='True Center V', marker='o')
    plt.plot(true_states[obj][:, 2], label='True Center W', marker='o')
    plt.plot(true_states[obj][:, 3], label='True Scale (Area)', marker='o')
    plt.plot(true_states[obj][:, 4], label='True Aspect Ratio', marker='o')

    plt.plot(measurements[obj][:, 0], label='Measured Center U', marker='x')
    plt.plot(measurements[obj][:, 1], label='Measured Center V', marker='x')
    plt.plot(measurements[obj][:, 2], label='Measured Center W', marker='x')
    plt.plot(measurements[obj][:, 3], label='Measured Scale (Area)', marker='x')

    plt.legend()
    plt.xlabel('Time Steps')
    plt.title(f'Object {obj + 1} Tracking (Constant Velocity)')

plt.tight_layout()
plt.show()