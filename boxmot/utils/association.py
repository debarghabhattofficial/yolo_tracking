# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np

from boxmot.utils.iou import iou_batch, centroid_batch, run_asso_func


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


# NOTE: Replace every call to speed_direction_batch() by
# speed_direction_batch_d() in the 
# associate_with_depth() method definition.
def speed_direction_batch_d(dets, tracks):
    tracks = tracks[..., np.newaxis]
    # Center coordinates of bbox1
    CX1 = (dets[:, 0] + dets[:, 2]) / 2.0
    CY1 = (dets[:, 1] + dets[:, 3]) / 2.0
    CZ1 = dets[:, 4]
    # Center coordinates of bbox2
    CX2 = (tracks[:, 0] + tracks[:, 2]) / 2.0
    CY2 = (tracks[:, 1] + tracks[:, 3]) / 2.0
    CZ2 = tracks[:, 4]
    # Compute speed and direction.
    dx = CX1 - CX2
    dy = CY1 - CY2
    dz = CZ1 - CZ2
    norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    dz = dz / norm
    return dx, dy, dz  # size: num_track x num_det


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array([list(zip(x, y))])


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def compute_aw_max_metric(emb_cost, w_association_emb, bottom=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)

    for idx in range(emb_cost.shape[0]):
        inds = np.argsort(-emb_cost[idx])
        # If there's less than two matches, just keep original weight
        if len(inds) < 2:
            continue
        if emb_cost[idx, inds[0]] == 0:
            row_weight = 0
        else:
            row_weight = 1 - max(
                (emb_cost[idx, inds[1]] / emb_cost[idx, inds[0]]) - bottom, 0
            ) / (1 - bottom)
        w_emb[idx] *= row_weight

    for idj in range(emb_cost.shape[1]):
        inds = np.argsort(-emb_cost[:, idj])
        # If there's less than two matches, just keep original weight
        if len(inds) < 2:
            continue
        if emb_cost[inds[0], idj] == 0:
            col_weight = 0
        else:
            col_weight = 1 - max(
                (emb_cost[inds[1], idj] / emb_cost[inds[0], idj]) - bottom, 0
            ) / (1 - bottom)
        w_emb[:, idj] *= col_weight

    return w_emb * emb_cost


def associate(
    detections,
    trackers,
    asso_func,
    iou_threshold,
    velocities,
    previous_obs,
    vdc_weight,
    w,
    h,
    emb_cost=None,
    w_assoc_emb=None,
    aw_off=None,
    aw_param=None,
    
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    iou_matrix = run_asso_func(asso_func, detections, trackers, w, h)
    #iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape):
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost = emb_cost
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
                else:
                    emb_cost *= w_assoc_emb

            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            matched_indices = linear_assignment(final_cost)
            if matched_indices.size == 0:
                matched_indices = np.empty(shape=(0, 2))

    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# NOTE: This function is not used in the current implementation.
# Replace every call to associate() by associate_with_depth() in the
# the tracking algorithms (currently working with OC-SORT).
def associate_d(detections,
                trackers,
                asso_func,
                iou_threshold,
                velocities,
                previous_obs,
                vdc_weight,
                w,
                h,
                emb_cost=None,
                w_assoc_emb=None,
                aw_off=None,
                aw_param=None):
    if len(trackers) == 0:
        # NOTE: np.empty((0, 6)) changed to np.empty((0, 5)) 
        # to include bbox centre depth at index 4.
        # Therefore, current structure of detections 
        # is [u1, v1, u2, v2, depth, score]
        #
        # If there are no trackers (len(trackers) == 0), 
        # the function returns empty arrays for matches, 
        # unmatched detections, and unmatched trackers.
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 6), dtype=int),
        )

    # compute the speed and direction of the detections 
    # based on their previous observations.
    X, Y, Z = speed_direction_batch_d(detections, previous_obs)
    
    # Extract the X, Y and Z components of velocities and 
    # repeat them to match the shape of the computed 
    # directions (X, Y and Z).
    inertia_X = velocities[:, 0]
    inertia_Y = velocities[:, 1]
    inertia_Z = velocities[:, 2]
    inertia_X = np.repeat(
        a=inertia_X[:, np.newaxis], 
        repeats=X.shape[1], 
        axis=1
    )
    inertia_Y = np.repeat(
        a=inertia_Y[:, np.newaxis], 
        repeats=Y.shape[1], 
        axis=1
    )
    inertia_Z = np.repeat(
        a=inertia_Z[:, np.newaxis], 
        repeats=Z.shape[1], 
        axis=1
    )
    
    # Compute the cosine of the angle difference between 
    # velocities and directions
    diff_angle_cos = (inertia_X * X) + (inertia_Y * Y) + (inertia_Z * Z)
    # Clip it to the range [-1, 1]. 
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    # Compute the angle in radians.
    diff_angle = np.arccos(diff_angle_cos)
    # Normalize the angle difference to a value between 
    # 0 and 1.
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    # np.where condition changed from 
    # (previous_obs[:, 4] < 0) to (previous_obs[:, 5] < 0)
    # to account for depth values at index 4, which shifts
    # the confidence score to index 5.
    valid_mask[np.where(previous_obs[:, 5] < 0)] = 0

    iou_matrix = run_asso_func(asso_func, detections, trackers, w, h)
    #iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(
        a=detections[:, -1][:, np.newaxis], 
        repeats=trackers.shape[0], 
        axis=1
    )
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(
        a=valid_mask[:, np.newaxis], 
        repeats=X.shape[1], 
        axis=1
    )

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape):
        # Create a binary matrix a indicating whether the
        # IOU is above the threshold.
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # If each detection matches at most one tracker 
            # and each tracker matches at most one detection, 
            # directly stack the indices of matching pairs.
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Handle the case when embedding cost is 
            # provided. If aw_off is False, apply 
            # additional processing using the 
            # compute_aw_max_metric function.
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost = emb_cost
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost = compute_aw_max_metric(
                        emb_cost=emb_cost, 
                        w_association_emb=w_assoc_emb, 
                        bottom=aw_param
                    )
                else:
                    emb_cost *= w_assoc_emb
            # Compute the final cost by combining IOU, angle 
            # difference, and embedding cost. 
            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            # Use the Hungarian algorithm to find the optimal 
            # assignment of detections to trackers based on the 
            # total cost.
            matched_indices = linear_assignment(final_cost)
            if matched_indices.size == 0:
                matched_indices = np.empty(shape=(0, 2))

    else:
        matched_indices = np.empty(shape=(0, 2))

    # Identify unmatched detections based on the 
    # matched indices.
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # Identify unmatched trackers based on the 
    # matched indices.
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IOU and update 
    # lists of unmatched detections and trackers 
    # accordingly.
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        # If there are no matches, set matches to an 
        # empty array.
        matches = np.empty((0, 2), dtype=int)
    else:
        # Otherwise, concatenate the matches.
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# NOTE: This function is not used in the current implementation.
# Replace every call to associate() by associate_with_depth() in the
# the tracking algorithms (currently working with OC-SORT).
def associate_dt(detections,
                 trackers,
                 asso_func,
                 iou_threshold,
                 velocities,
                 previous_obs,
                 vdc_weight,
                 w,
                 h,
                 emb_cost=None,
                 w_assoc_emb=None,
                 aw_off=None,
                 aw_param=None):
    if len(trackers) == 0:
        # NOTE: np.empty((0, 6)) changed to np.empty((0, 5)) 
        # to include bbox centre depth at index 4.
        # Therefore, current structure of detections 
        # is [u1, v1, u2, v2, depth, score]
        #
        # If there are no trackers (len(trackers) == 0), 
        # the function returns empty arrays for matches, 
        # unmatched detections, and unmatched trackers.
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 6), dtype=int),
        )

    # compute the speed and direction of the detections 
    # based on their previous observations.
    X, Y, Z = speed_direction_batch_d(detections, previous_obs)
    
    # Extract the X, Y and Z components of velocities and 
    # repeat them to match the shape of the computed 
    # directions (X, Y and Z).
    inertia_X = velocities[:, 0]
    inertia_Y = velocities[:, 1]
    inertia_Z = velocities[:, 2]
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_Z = np.repeat(inertia_Z[:, np.newaxis], Z.shape[1], axis=1)
    
    # Compute the cosine of the angle difference between 
    # velocities and directions
    diff_angle_cos = inertia_X * X + inertia_Y * Y + inertia_Z * Z
    # Clip it to the range [-1, 1]. 
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    # Compute the angle in radians.
    diff_angle = np.arccos(diff_angle_cos)
    # Normalize the angle difference to a value between 
    # 0 and 1.
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    # np.where condition changed from 
    # (previous_obs[:, 4] < 0) to (previous_obs[:, 5] < 0)
    # to account for depth values at index 4, which shifts
    # the confidence score to index 5.
    valid_mask[np.where(previous_obs[:, 5] < 0)] = 0

    iou_matrix = run_asso_func(asso_func, detections, trackers, w, h)
    #iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape):
        # Create a binary matrix a indicating whether the
        # IOU is above the threshold.
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # If each detection matches at most one tracker 
            # and each tracker matches at most one detection, 
            # directly stack the indices of matching pairs.
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Handle the case when embedding cost is 
            # provided. If aw_off is False, apply 
            # additional processing using the 
            # compute_aw_max_metric function.
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost = emb_cost
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost = compute_aw_max_metric(
                        emb_cost=emb_cost, 
                        w_association_emb=w_assoc_emb, 
                        bottom=aw_param
                    )
                else:
                    emb_cost *= w_assoc_emb
            # Compute the final cost by combining IOU, angle 
            # difference, and embedding cost. 
            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            # Use the Hungarian algorithm to find the optimal 
            # assignment of detections to trackers based on the 
            # total cost.
            matched_indices = linear_assignment(final_cost)
            if matched_indices.size == 0:
                matched_indices = np.empty(shape=(0, 2))

    else:
        matched_indices = np.empty(shape=(0, 2))

    # Identify unmatched detections based on the 
    # matched indices.
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # Identify unmatched trackers based on the 
    # matched indices.
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IOU and update 
    # lists of unmatched detections and trackers 
    # accordingly.
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        # If there are no matches, set matches to an 
        # empty array.
        matches = np.empty((0, 2), dtype=int)
    else:
        # Otherwise, concatenate the matches.
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# NOTE: This function is not used in the current implementation.
# Replace every call to associate() by associate_with_depth() in the
# the tracking algorithms (currently working with OC-SORT).
def associate_dtc(detections,
                  trackers,
                  asso_func,
                  iou_threshold,
                  velocities,
                  previous_obs,
                  vdc_weight,
                  w,
                  h,
                  emb_cost=None,
                  w_assoc_emb=None,
                  aw_off=None,
                  aw_param=None):
    if len(trackers) == 0:
        # NOTE: np.empty((0, 6)) changed to np.empty((0, 5)) 
        # to include bbox centre depth at index 4.
        # Therefore, current structure of detections 
        # is [u1, v1, u2, v2, depth, score]
        #
        # If there are no trackers (len(trackers) == 0), 
        # the function returns empty arrays for matches, 
        # unmatched detections, and unmatched trackers.
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 6), dtype=int),
        )

    # compute the speed and direction of the detections 
    # based on their previous observations.
    X, Y, Z = speed_direction_batch_d(detections, previous_obs)
    
    # Extract the X, Y and Z components of velocities and 
    # repeat them to match the shape of the computed 
    # directions (X, Y and Z).
    inertia_X = velocities[:, 0]
    inertia_Y = velocities[:, 1]
    inertia_Z = velocities[:, 2]
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_Z = np.repeat(inertia_Z[:, np.newaxis], Z.shape[1], axis=1)
    
    # Compute the cosine of the angle difference between 
    # velocities and directions
    diff_angle_cos = inertia_X * X + inertia_Y * Y + inertia_Z * Z
    # Clip it to the range [-1, 1]. 
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    # Compute the angle in radians.
    diff_angle = np.arccos(diff_angle_cos)
    # Normalize the angle difference to a value between 
    # 0 and 1.
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    # np.where condition changed from 
    # (previous_obs[:, 4] < 0) to (previous_obs[:, 5] < 0)
    # to account for depth values at index 4, which shifts
    # the confidence score to index 5.
    valid_mask[np.where(previous_obs[:, 5] < 0)] = 0

    iou_matrix = run_asso_func(asso_func, detections, trackers, w, h)
    #iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape):
        # Create a binary matrix a indicating whether the
        # IOU is above the threshold.
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # If each detection matches at most one tracker 
            # and each tracker matches at most one detection, 
            # directly stack the indices of matching pairs.
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Handle the case when embedding cost is 
            # provided. If aw_off is False, apply 
            # additional processing using the 
            # compute_aw_max_metric function.
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost = emb_cost
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost = compute_aw_max_metric(
                        emb_cost=emb_cost, 
                        w_association_emb=w_assoc_emb, 
                        bottom=aw_param
                    )
                else:
                    emb_cost *= w_assoc_emb
            # Compute the final cost by combining IOU, angle 
            # difference, and embedding cost. 
            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            # Use the Hungarian algorithm to find the optimal 
            # assignment of detections to trackers based on the 
            # total cost.
            matched_indices = linear_assignment(final_cost)
            if matched_indices.size == 0:
                matched_indices = np.empty(shape=(0, 2))

    else:
        matched_indices = np.empty(shape=(0, 2))

    # Identify unmatched detections based on the 
    # matched indices.
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # Identify unmatched trackers based on the 
    # matched indices.
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IOU and update 
    # lists of unmatched detections and trackers 
    # accordingly.
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        # If there are no matches, set matches to an 
        # empty array.
        matches = np.empty((0, 2), dtype=int)
    else:
        # Otherwise, concatenate the matches.
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_dtc_b(detections,
                    trackers,
                    asso_func,
                    iou_threshold,
                    velocities,
                    previous_obs,
                    vdc_weight,
                    w,
                    h,
                    emb_cost=None,
                    w_assoc_emb=None,
                    aw_off=None,
                    aw_param=None):
    if len(trackers) == 0:
        # NOTE: np.empty((0, 6)) changed to np.empty((0, 5)) 
        # to include bbox centre depth at index 4.
        # Therefore, current structure of detections 
        # is [u1, v1, u2, v2, depth, score]
        #
        # If there are no trackers (len(trackers) == 0), 
        # the function returns empty arrays for matches, 
        # unmatched detections, and unmatched trackers.
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 6), dtype=int),
        )

    # compute the speed and direction of the detections 
    # based on their previous observations.
    X, Y, Z = speed_direction_batch_d(detections, previous_obs)
    
    # Extract the X, Y and Z components of velocities and 
    # repeat them to match the shape of the computed 
    # directions (X, Y and Z).
    inertia_X = velocities[:, 0]
    inertia_Y = velocities[:, 1]
    inertia_Z = velocities[:, 2]
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_Z = np.repeat(inertia_Z[:, np.newaxis], Z.shape[1], axis=1)
    
    # Compute the cosine of the angle difference between 
    # velocities and directions
    diff_angle_cos = inertia_X * X + inertia_Y * Y + inertia_Z * Z
    # Clip it to the range [-1, 1]. 
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    # Compute the angle in radians.
    diff_angle = np.arccos(diff_angle_cos)
    # Normalize the angle difference to a value between 
    # 0 and 1.
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    # np.where condition changed from 
    # (previous_obs[:, 4] < 0) to (previous_obs[:, 5] < 0)
    # to account for depth values at index 4, which shifts
    # the confidence score to index 5.
    valid_mask[np.where(previous_obs[:, 5] < 0)] = 0

    iou_matrix = run_asso_func(asso_func, detections, trackers, w, h)
    #iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(
        a=detections[:, -1][:, np.newaxis], 
        repeats=trackers.shape[0], 
        axis=1
    )
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(
        a=valid_mask[:, np.newaxis], 
        repeats=X.shape[1], 
        axis=1
    )

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape):
        # Create a binary matrix a indicating whether the
        # IOU is above the threshold.
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # If each detection matches at most one tracker 
            # and each tracker matches at most one detection, 
            # directly stack the indices of matching pairs.
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Handle the case when embedding cost is 
            # provided. If aw_off is False, apply 
            # additional processing using the 
            # compute_aw_max_metric function.
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost = emb_cost
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost = compute_aw_max_metric(
                        emb_cost=emb_cost, 
                        w_association_emb=w_assoc_emb, 
                        bottom=aw_param
                    )
                else:
                    emb_cost *= w_assoc_emb
            # Compute the final cost by combining IOU, angle 
            # difference, and embedding cost. 
            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            # Use the Hungarian algorithm to find the optimal 
            # assignment of detections to trackers based on the 
            # total cost.
            matched_indices = linear_assignment(final_cost)
            if matched_indices.size == 0:
                matched_indices = np.empty(shape=(0, 2))

    else:
        matched_indices = np.empty(shape=(0, 2))

    # Identify unmatched detections based on the 
    # matched indices.
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # Identify unmatched trackers based on the 
    # matched indices.
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IOU and update 
    # lists of unmatched detections and trackers 
    # accordingly.
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        # If there are no matches, set matches to an 
        # empty array.
        matches = np.empty((0, 2), dtype=int)
    else:
        # Otherwise, concatenate the matches.
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_kitti(
    detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    """
        Cost from the velocity direction consistency
    """
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    """
        Cost from IoU
    """
    iou_matrix = iou_batch(detections, trackers)

    """
        With multiple categories, generate the cost for catgory mismatch
    """
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
        for j in range(num_trk):
            if det_cates[i] != trackers[j, 4]:
                cate_matrix[i][j] = -1e6

    cost_matrix = -iou_matrix - angle_diff_cost - cate_matrix

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
