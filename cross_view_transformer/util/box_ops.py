# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import numpy as np
from pyquaternion import Quaternion

def box_cxcywh_to_xyxy(x, transform=False):
    if isinstance(x, np.ndarray):
        cx = x[..., 0]
        cy = x[..., 1]
        w = x[..., 2]
        h = x[..., 3]
        if transform:
            return np.stack([
                cx + 0.5*w, cy + 0.5*h,
                cx - 0.5*w, cy - 0.5*h,
            ],1)
        else:
            return np.stack([
            cx - 0.5*w, cy - 0.5*h,
            cx + 0.5*w, cy + 0.5*h,
        ],1)
    
    x_c, y_c, w, h = x.unbind(-1)
    if transform:
        b = [(x_c + 0.5 * w), (y_c + 0.5 * h),
            (x_c - 0.5 * w), (y_c - 0.5 * h)]
    else:
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x, transform=False):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def lidar_to_bev(x, view):

    x0, y0, x1, y1 = x.unbind(-1)
    p1, p2 = torch.stack([x0,y0], dim=-1), torch.stack([x1,y1], dim=-1)
    p1, p2 = torch.nn.functional.pad(p1,(0,1), value=1).permute(0,2,1), torch.nn.functional.pad(p2,(0,1), value=1).permute(0,2,1)
    p1 = torch.einsum('b i j, b j n -> b i n', view, p1).permute(0,2,1)[...,:2]
    p2 = torch.einsum('b i j, b j n -> b i n', view, p2).permute(0,2,1)[...,:2]
    x = torch.cat([p1,p2],dim=-1)
    return x

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def sincos2quaternion(sin, cos):
    rotation = [
        [cos, sin, 0.0],
        [-sin, cos, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return Quaternion(matrix=np.array(rotation))