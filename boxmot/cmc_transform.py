import numpy as np

def apply_affine_transform_with_covariance(state_vector, 
                                           covariance_matrix, 
                                           affine_matrix):
    # Extract position and velocity components from the state vector
    # Adjusted to take only the first two elements for 2D case
    position = state_vector[:2]  # 1x2 vector
    velocity = state_vector[5:7]  # 1x2 velocity vector

    # Get transformed position vector (rotation/scaling + translation).
    """
    NOTE: Technique 1
    """
    # # Append 1 to the position vector for homogeneous coordinates
    # position_homogeneous = np.append(position, 1)  # 1x3 position vector
    # # Apply affine transformation to position vector
    # transformed_position_homogeneous = np.dot(
    #     affine_matrix,  # 2x3 affine transformation matrix
    #     position_homogeneous  # 1x3 position vector
    # )
    # # Extract the transformed position and remove the homogeneous coordinate
    # # Adjusted to take only the first two elements for 2D case
    # transformed_position = transformed_position_homogeneous  # 1x2 position vector
    """
    NOTE: Technique 2
    """
    # Result is a 1x2 vector
    transformed_position = np.dot(
        affine_matrix[:, :2],  # 2x2 matrix
        position  # 1x2 vector
    ) + affine_matrix[:, 2]  # 1x2 vector
    print(f"position shape: {position.shape}")
    print(f"position: \n{position}")
    print("-" * 75)
    print(f"transformed_position shape: {transformed_position.shape}")
    print(f"transformed_position: \n{transformed_position}")
    print("-" * 75)

    # Get transformed velocity vector (translation).
    # Adjusted to take only the first two columns for 2D case
    # Result is a 1x2 vector.
    transformed_velocity = np.dot(
        affine_matrix[:, :2],  # 2x2 vector
        velocity  # 1x2 velocity vector
    )
    print(f"velocity shape: {velocity.shape}")
    print(f"velocity: \n{velocity}")
    print("-" * 75)
    print(f"transformed_velocity shape: {transformed_velocity.shape}")
    print(f"transformed_velocity: \n{transformed_velocity}")
    print("-" * 75)

    # Get updated state vector by combining transformed 
    # position and velocity with unchanged depth, area and 
    # aspect ratio.
    updated_state_vector = np.concatenate([
        transformed_position,  # 1x2 vector (position)
        state_vector[2:5],  # 1x3 vector (depth, area and aspect ratio)
        transformed_velocity,  # 1x2 vector (velocity)
        state_vector[7:]  # 1x1 vector (orientation)
    ])
    print(f"state_vector shape: {state_vector.shape}")
    print(f"state_vector: \n{state_vector}")
    print("-" * 75)
    print(f"updated_state_vector shape: {updated_state_vector.shape}")
    print(f"updated_state_vector: \n{updated_state_vector}")
    print("-" * 75)

    # Get transformed covariance matrix.
    R4x4 = np.kron(np.eye(2, dtype=float), affine_matrix[:, :2])
    cov_sub_mat = np.block([
        [covariance_matrix[:2, :2], covariance_matrix[:2, 5:7]],
        [covariance_matrix[5:7, :2], covariance_matrix[5:7, 5:7]]
    ])
    updated_cov_sub_mat = R4x4.dot(cov_sub_mat).dot(R4x4.T)
    updated_covariance_matrix = np.copy(covariance_matrix)
    updated_covariance_matrix[:2, :2] = updated_cov_sub_mat[:2, :2]
    updated_covariance_matrix[:2, 5:7] = updated_cov_sub_mat[:2, 2:]
    updated_covariance_matrix[5:7, :2] = updated_cov_sub_mat[2:, :2]
    updated_covariance_matrix[5:7, 5:7] = updated_cov_sub_mat[2:, 2:]
    print(f"covariance_matrix shape: {covariance_matrix.shape}")
    print(f"covariance_matrix: \n{covariance_matrix}")
    print("-" * 75)
    print(f"updated_covariance_matrix shape: {updated_covariance_matrix.shape}")
    print(f"updated_covariance_matrix: \n{updated_covariance_matrix}")
    print("-" * 75)

    # jacobian_matrix = np.block([
    #     [
    #         affine_matrix[:, :2],  # 2x2 matrix
    #         np.zeros((2, 1)),  # 2x1 matrix
    #         np.zeros((2, 3))  # 2x3 matrix
    #     ],
    #     [
    #         np.zeros((3, 2)),  # 3x2 matrix
    #         np.eye(3)  # 3x3 matrix
    #     ]
    # ])

    # # Apply the covariance matrix transformation
    # updated_covariance_matrix = np.dot(
    #     jacobian_matrix, 
    #     np.dot(covariance_matrix, jacobian_matrix.T)
    # )

    return updated_state_vector, updated_covariance_matrix


# Example usage:
# Assume state_vector is your original 9-dimensional state vector
state_vector = np.array([10, 20, 30, 40, 1.5, 2, 3, 4, 0.1])

# Assume covariance_matrix is your original 9x9 state covariance matrix
covariance_matrix = np.array([[1,    0,    0,    0,    0,    0,    0,    0,    0],
                              [0,    2,    0,    0,    0,    0,    0,    0,    0],
                              [0,    0,    3,    0,    0,    0,    0,    0,    0],
                              [0,    0,    0,    4,    0,    0,    0,    0,    0],
                              [0,    0,    0,    0,  0.1,    0,    0,    0,    0],
                              [0,    0,    0,    0,    0,  0.2,    0,    0,    0],
                              [0,    0,    0,    0,    0,    0,  0.3,    0,    0],
                              [0,    0,    0,    0,    0,    0,    0,  0.4,    0],
                              [0,    0,    0,    0,    0,    0,    0,    0, 0.01]])

# Example affine transformation matrix (replace this with your actual matrix)
affine_matrix = np.array([
    [ 1.2, 0.3, 10],
    [-0.1, 0.8,  5]
])
print(f"affine_matrix shape: {affine_matrix.shape}")
print(f"affine_matrix: \n{affine_matrix}")
print("-" * 75)

# Apply affine transformation to the state vector and covariance matrix
updated_state_vector, updated_covariance_matrix = apply_affine_transform_with_covariance(state_vector, covariance_matrix, affine_matrix)

# # Display the original and updated state vector and covariance matrix
# print("Original State Vector:")
# print(state_vector)
# print("\nUpdated State Vector:")
# print(updated_state_vector)
# print("\nOriginal Covariance Matrix:")
# print(covariance_matrix)
# print("\nUpdated Covariance Matrix:")
# print(updated_covariance_matrix)
