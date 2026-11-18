"""Module 2 - Phát hiện làn đường (đề bài mục 3.2: edges / lines).

Hai phương pháp để so sánh (phần mở rộng của đề):
  - "hough": Canny -> mặt nạ ROI -> HoughLinesP
  - "lsd"  : Line Segment Detector (không cần Canny) -> lọc theo ROI

Sau khi có các đoạn thẳng: phân loại trái/phải theo dấu độ dốc, khớp mỗi bên thành MỘT đường
x = a*y + b (trung bình có trọng số theo độ dài đoạn), rồi làm mượt theo thời gian (EMA).
Tham số hoá x theo y (thay vì y theo x) để không bị vô cực với đường gần thẳng đứng.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import LaneConfig


@dataclass
class LaneLine:
    a: float  # x = a*y + b
    b: float

    def x_at(self, y: float) -> float:
        return self.a * y + self.b


@dataclass
class LaneResult:
    left: Optional[LaneLine]
    right: Optional[LaneLine]
    segments: np.ndarray          # (N, 4) các đoạn thẳng thô sau khi lọc
    edges: Optional[np.ndarray]   # ảnh cạnh Canny (None với LSD)

    @property
    def both(self) -> bool:
        return self.left is not None and self.right is not None


class LaneDetector:
    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg
        self._roi_mask: Optional[np.ndarray] = None
        self._roi_shape = None
        self._prev = {"left": None, "right": None}
        self._miss = {"left": 0, "right": 0}
        self._lsd = None
        if cfg.method == "lsd":
            self._lsd = self._create_lsd()

    # ------------------------------------------------------------------ public
    def detect(self, gray: np.ndarray) -> LaneResult:
        h, w = gray.shape[:2]
        mask = self._get_roi_mask(h, w)
        if self.cfg.method == "hough":
            segments, edges = self._segments_hough(gray, mask)
        elif self.cfg.method == "lsd":
            segments, edges = self._segments_lsd(gray, mask), None
        else:
            raise ValueError(f"lane.method không hợp lệ: {self.cfg.method}")

        left, right = self._fit_lanes(segments)
        left = self._smooth("left", left)
        right = self._smooth("right", right)
        return LaneResult(left=left, right=right, segments=segments, edges=edges)

    def roi_polygon(self, h: int, w: int) -> np.ndarray:
        pts = np.array(self.cfg.roi, dtype=np.float32) * np.array([w, h], dtype=np.float32)
        return pts.astype(np.int32)

    def reset(self) -> None:
        self._prev = {"left": None, "right": None}
        self._miss = {"left": 0, "right": 0}

    # ------------------------------------------------------------------ nội bộ
    def _get_roi_mask(self, h: int, w: int) -> np.ndarray:
        if self._roi_mask is None or self._roi_shape != (h, w):
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [self.roi_polygon(h, w)], 255)
            self._roi_mask, self._roi_shape = mask, (h, w)
        return self._roi_mask

    def _segments_hough(self, gray, mask):
        c = self.cfg
        edges = cv2.Canny(gray, c.canny_low, c.canny_high)
        edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=c.hough_threshold,
            minLineLength=c.min_line_length, maxLineGap=c.max_line_gap,
        )
        segs = np.empty((0, 4), dtype=np.float32) if lines is None else lines.reshape(-1, 4).astype(np.float32)
        return segs, edges

    @staticmethod
    def _create_lsd():
        if hasattr(cv2, "createLineSegmentDetector"):
            return cv2.createLineSegmentDetector()
        raise RuntimeError("OpenCV của bạn không có LineSegmentDetector; hãy dùng lane.method = 'hough'.")

    def _segments_lsd(self, gray, mask):
        lines = self._lsd.detect(gray)[0]
        if lines is None:
            return np.empty((0, 4), dtype=np.float32)
        segs = lines.reshape(-1, 4).astype(np.float32)
        length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        segs = segs[length >= self.cfg.lsd_min_length]
        if len(segs) == 0:
            return segs
        # giữ đoạn có trung điểm nằm trong ROI
        mx = ((segs[:, 0] + segs[:, 2]) / 2).astype(int).clip(0, mask.shape[1] - 1)
        my = ((segs[:, 1] + segs[:, 3]) / 2).astype(int).clip(0, mask.shape[0] - 1)
        return segs[mask[my, mx] > 0]

    def _fit_lanes(self, segs: np.ndarray):
        """Phân loại và khớp 1 đường cho mỗi bên (trái/phải)."""
        acc = {"left": [], "right": []}
        for x1, y1, x2, y2 in segs:
            dx, dy = x2 - x1, y2 - y1
            if abs(dy) < 1e-3:
                continue
            if abs(dx) < 1e-3:
                continue  # đoạn thẳng đứng tuyệt đối: trong ảnh phối cảnh không phải vạch làn
            slope = dy / dx
            if abs(slope) < self.cfg.min_abs_slope:
                continue
            a = dx / dy
            b = x1 - a * y1
            length = float(np.hypot(dx, dy))
            # Trong hệ toạ độ ảnh (y hướng xuống): làn trái có slope < 0, làn phải có slope > 0
            acc["left" if slope < 0 else "right"].append((a, b, length))

        out = []
        for side in ("left", "right"):
            if not acc[side]:
                out.append(None)
                continue
            arr = np.array(acc[side])
            wts = arr[:, 2]
            out.append(LaneLine(a=float(np.average(arr[:, 0], weights=wts)),
                                b=float(np.average(arr[:, 1], weights=wts))))
        return out[0], out[1]

    def _smooth(self, side: str, new: Optional[LaneLine]) -> Optional[LaneLine]:
        prev = self._prev[side]
        alpha = self.cfg.smooth_alpha
        if new is None:
            self._miss[side] += 1
            if prev is not None and self._miss[side] > self.cfg.hold_frames:
                self._prev[side] = None
                return None
            return prev
        self._miss[side] = 0
        if prev is not None:
            new = LaneLine(a=alpha * new.a + (1 - alpha) * prev.a,
                           b=alpha * new.b + (1 - alpha) * prev.b)
        self._prev[side] = new
        return new
