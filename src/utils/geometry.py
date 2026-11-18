"""Hàm hình học dùng chung."""
from __future__ import annotations

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU giữa mọi cặp (a_i, b_j). Box dạng [x1, y1, x2, y2]. Trả về ma trận (N, M)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def bottom_center(box) -> tuple[float, float]:
    """Điểm giữa cạnh dưới của bbox ~ điểm xe chạm mặt đường."""
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y2))
