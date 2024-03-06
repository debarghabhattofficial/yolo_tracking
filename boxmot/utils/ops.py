# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


# NOTE: Replace every call to xyxy2xywh() method in 
# BoT-SORT + RGBD code with xyxy2xywh_with_depth() 
# method.
def xyxy2xywh_with_depth(x):
    """
    Convert bounding box coordinates from 
    (x1, y1, x2, y2, depth) format to 
    (x, y, depth, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): 
        The input bounding box coordinates in 
        (x1, y1, x2, y2, depth) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): 
       The bounding box coordinates in 
       (x, y, depth, width, height) format.
    """
    if isinstance(x, torch.Tensor):
        y = x.clone()
        # Remove the depth tensor from the input tensor.
        y = torch.cat((
            y[..., :4], 
            y[..., 4+1:]
        ), dim=-1)
        # Convert the bounding box coordinates from
        # (x1, y1, x2, y2) format to (x, y, width, height) format.
        y = xyxy2xywh(x=y)
        # Insert the depth tensor back to the tensor.
        y = torch.cat((
            y[..., :2], 
            x[..., 4].unsqueeze(dim=-1), 
            y[..., 2:]
        ), dim=-1)
    else:
        y = np.copy(x)
        # Remove the depth array from the input array.
        y = np.delete(arr=y, obj=4, axis=-1)
        # Convert the bounding box coordinates from
        # (x1, y1, x2, y2) format to (x, y, width, height) format.
        y = xyxy2xywh(x=y)
        # Insert the depth array back to the array.
        y = np.insert(arr=y, obj=2, values=x[..., 4], axis=-1)

    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x_c, y_c, width, height) format to
    (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2)
    is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., ]
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


# NOTE: Replace every call to xywh2xyxy() method in 
# BoT-SORT + RGBD code with xywh2xyxy_with_depth() 
# method.
def xywh2xyxy_with_depth(x):
    """
    Convert bounding box coordinates from 
    (x_c, y_c, depth, width, height) format to
    (x1, y1, x2, y2, depth) format where 
        (x1, y1) is the top-left corner,
        (x2, y2) is the bottom-right corner, and
        depth is the bounding box centre depth.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding 
        box coordinates in (x, y, depth, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box 
        coordinates in (x1, y1, x2, y2, depth) format.
    """
    if isinstance(x, torch.Tensor):
        y = x.clone()
        # Remove the depth tensor from the input tensor.
        y = torch.cat((
            y[..., :2], 
            y[..., 2+1:]
        ), dim=-1)
        # Convert the bounding box coordinates from
        # (x, y, w, h) format to (x1, y1, x2, y2) format.
        y = xywh2xyxy(x=y)
        # Insert the depth tensor back to the tensor.
        y = torch.cat((
            y[..., :4], 
            x[..., 2].unsqueeze(dim=-1), 
            y[..., 4:]
        ), dim=-1)
    else:
        y = np.copy(x)
        # Remove the depth array from the input array.
        y = np.delete(arr=y, obj=2, axis=-1)
        # Convert the bounding box coordinates from
        # (x1, y1, x2, y2) format to (x, y, width, height) format.
        y = xywh2xyxy(x=y)
        # Insert the depth array back to the array.
        y = np.insert(arr=y, obj=4, values=x[..., 2], axis=-1)
        
    return y


def xywh2tlwh(x):
    """
    Convert bounding box coordinates from (x c, y c, w, h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.cl1one() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0  # xc --> t
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0  # yc --> l
    y[..., 2] = x[..., 2]                    # width
    y[..., 3] = x[..., 3]                    # height
    return y


def tlwh2xyxy(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y


def xyxy2tlwh(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def tlwh2xyah(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h)
    to (center x, center y, aspect ratio, height)`, where the aspect ratio is `width / height`.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + (x[..., 2] / 2)
    y[..., 1] = x[..., 1] + (x[..., 3] / 2)
    y[..., 2] = x[..., 2] / x[..., 3]
    y[..., 3] = x[..., 3]
    return y
