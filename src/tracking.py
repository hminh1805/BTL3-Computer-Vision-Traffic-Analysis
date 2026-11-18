"""Module 5 - Theo dõi đối tượng (đề bài mục 3.3: tracking).

Bộ tracker kiểu SORT tự viết (không phụ thuộc thư viện ngoài):
  * Mỗi track có bộ lọc Kalman (cv2.KalmanFilter) với trạng thái [cx, cy, w, h, vcx, vcy, vw, vh].
  * Mỗi frame: dự đoán vị trí mới -> ghép với detection bằng IoU (tham lam, IoU lớn nhất trước)
    -> cập nhật track khớp, tạo track mới cho detection chưa khớp, xoá track mất dấu quá max_age.
  * Track chỉ được "xác nhận" (trả ra ngoài) khi đã khớp >= min_hits lần để loại phát hiện nhiễu.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import cv2
import numpy as np

from .config import TrackingConfig
from .detection import Detection
from .utils.geometry import bottom_center, iou_matrix


class Track:
    _next_id = 1

    def __init__(self, det: Detection, frame_idx: int, trace_length: int):
        self.id = Track._next_id
        Track._next_id += 1
        self.label = det.label
        self.score = det.score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.trace: Deque[Tuple[float, float]] = deque(maxlen=trace_length)  # điểm chạm đất trên ảnh
        self._init_kalman(det.bbox)
        self.trace.append(self.foot)

    # ------------------------------------------------------------------ Kalman
    def _init_kalman(self, bbox: np.ndarray) -> None:
        kf = cv2.KalmanFilter(8, 4)
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = 1.0
        kf.transitionMatrix = F
        H = np.zeros((4, 8), dtype=np.float32)
        H[:4, :4] = np.eye(4)
        kf.measurementMatrix = H
        kf.processNoiseCov = np.diag([1, 1, 1, 1, 0.5, 0.5, 0.01, 0.01]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([4, 4, 8, 8]).astype(np.float32)
        kf.errorCovPost = np.diag([10, 10, 10, 10, 1000, 1000, 100, 100]).astype(np.float32)
        z = self._bbox_to_z(bbox)
        kf.statePost = np.array([[z[0]], [z[1]], [z[2]], [z[3]], [0], [0], [0], [0]], dtype=np.float32)
        self.kf = kf

    @staticmethod
    def _bbox_to_z(b) -> np.ndarray:
        x1, y1, x2, y2 = b
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32)

    @property
    def bbox(self) -> np.ndarray:
        cx, cy, w, h = self.kf.statePost[:4, 0]
        w, h = max(float(w), 1.0), max(float(h), 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    @property
    def foot(self) -> Tuple[float, float]:
        return bottom_center(self.bbox)

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, det: Detection) -> None:
        self.kf.correct(self._bbox_to_z(det.bbox).reshape(4, 1))
        self.hits += 1
        self.time_since_update = 0
        self.score = det.score
        self.label = det.label
        self.trace.append(self.foot)


class Tracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.tracks: List[Track] = []
        Track._next_id = 1

    def update(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        predicted = np.array([t.predict() for t in self.tracks], dtype=np.float32).reshape(-1, 4)
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32).reshape(-1, 4)

        matched_t, matched_d = set(), set()
        iou = iou_matrix(predicted, det_boxes)
        if iou.size:
            pairs = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
            for ti, di in pairs:
                if iou[ti, di] < self.cfg.iou_threshold:
                    break
                if ti in matched_t or di in matched_d:
                    continue
                self.tracks[ti].update(detections[di])
                matched_t.add(ti)
                matched_d.add(di)

        for di, det in enumerate(detections):
            if di not in matched_d:
                self.tracks.append(Track(det, frame_idx, self.cfg.trace_length))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.cfg.max_age]

        return [t for t in self.tracks
                if t.time_since_update == 0 and (t.hits >= self.cfg.min_hits or frame_idx < self.cfg.min_hits)]
