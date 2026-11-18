"""Module 3 - Biến đổi phối cảnh / Bird's-Eye View (đề bài mục 3.2: homography, mặt phẳng).

Ý tưởng: mặt đường là một mặt phẳng, nên ảnh mặt đường và ảnh nhìn từ trên xuống liên hệ với
nhau bởi một phép homography H (3x3). Chọn 4 điểm trên ảnh tạo thành hình chữ nhật thật trên
mặt đường (biết kích thước thật: road_width_m x road_length_m) rồi ánh xạ sang hình chữ nhật
trong ảnh BEV. Từ đó:
   pixel ảnh  --H-->  pixel BEV  --/pixels_per_meter-->  mét ngoài đời thật.

Hai chế độ lấy 4 điểm nguồn:
  - manual: cố định (chọn bằng tools/calibrate_bev.py)
  - auto  : suy ra từ 2 đường làn do lane_detection tìm được (cắt tại 2 mốc y)
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import BEVConfig
from .lane_detection import LaneResult


class BEVTransformer:
    def __init__(self, cfg: BEVConfig, frame_size: Tuple[int, int]):
        """frame_size = (width, height) của ảnh SAU tiền xử lý."""
        self.cfg = cfg
        self.w, self.h = frame_size
        self.ppm = cfg.pixels_per_meter
        self.bev_w = int(round(cfg.road_width_m * self.ppm))
        self.bev_h = int(round(cfg.road_length_m * self.ppm))
        self._dst = np.array(
            [[0, 0], [self.bev_w, 0], [self.bev_w, self.bev_h], [0, self.bev_h]], dtype=np.float32
        )
        self.src = (np.array(cfg.src_points, dtype=np.float32) * np.array([self.w, self.h], dtype=np.float32))
        self.H = cv2.getPerspectiveTransform(self.src, self._dst)

    # ------------------------------------------------------------------ cập nhật (mode auto)
    def update_from_lanes(self, lanes: LaneResult) -> bool:
        """Cập nhật 4 điểm nguồn từ 2 làn đường. Trả về True nếu có cập nhật."""
        if self.cfg.mode != "auto" or not lanes.both:
            return False
        y_top = self.cfg.auto_y_top * self.h
        y_bot = self.cfg.auto_y_bottom * self.h
        new_src = np.array([
            [lanes.left.x_at(y_top), y_top],
            [lanes.right.x_at(y_top), y_top],
            [lanes.right.x_at(y_bot), y_bot],
            [lanes.left.x_at(y_bot), y_bot],
        ], dtype=np.float32)
        if not (new_src[1, 0] > new_src[0, 0] and new_src[2, 0] > new_src[3, 0]):
            return False  # hai làn bị chéo nhau -> bỏ qua
        a = self.cfg.auto_smooth_alpha
        self.src = a * new_src + (1 - a) * self.src
        self.H = cv2.getPerspectiveTransform(self.src, self._dst)
        return True

    # ------------------------------------------------------------------ biến đổi
    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Ảnh nhìn từ trên xuống (BEV)."""
        return cv2.warpPerspective(frame, self.H, (self.bev_w, self.bev_h))

    def to_bev(self, pts_px: np.ndarray) -> np.ndarray:
        """Điểm ảnh (N,2) -> điểm trên ảnh BEV (N,2), đơn vị pixel BEV."""
        pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_meters(self, pts_px: np.ndarray) -> np.ndarray:
        """Điểm ảnh (N,2) -> toạ độ mặt đường (N,2) đơn vị MÉT."""
        return self.to_bev(pts_px) / self.ppm

    def bev_px_to_meters(self, pts_bev: np.ndarray) -> np.ndarray:
        return np.asarray(pts_bev, dtype=np.float32) / self.ppm

    # ------------------------------------------------------------------ vùng đường
    def road_polygon(self) -> np.ndarray:
        return self.src.astype(np.int32)

    def contains(self, point: Tuple[float, float], margin: float = 0.0) -> bool:
        """Điểm có nằm trong (hoặc cách vùng đường không quá `margin` px) không."""
        poly = self.src.reshape(-1, 1, 2).astype(np.float32)
        dist = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), True)
        return dist >= -margin
