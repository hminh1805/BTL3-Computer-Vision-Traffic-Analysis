"""Module 7 - Trực quan hóa kết quả (đề bài mục 3.4).

Tạo khung hình kết quả gồm 2 phần ghép ngang:
  - Trái : ảnh gốc + vùng đường, làn đường, bbox, ID, tốc độ, vệt chuyển động
  - Phải : bản đồ BEV (nhìn từ trên xuống) + lưới mét + vị trí / quỹ đạo từng xe
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import OutputConfig
from .detection import Detection
from .homography import BEVTransformer
from .lane_detection import LaneDetector, LaneResult
from .speed_estimation import SpeedEstimator
from .tracking import Track

_PALETTE = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53), (171, 71, 188),
            (0, 172, 193), (255, 112, 67), (124, 179, 66), (240, 98, 146), (121, 134, 203)]


def color_of(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _text(img, txt, org, color=(255, 255, 255), scale=0.5, thick=1, bg=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - base), (x + tw, y + base), bg, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Visualizer:
    def __init__(self, cfg: OutputConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ khung trái
    def draw_frame(self, bgr: np.ndarray, lanes: Optional[LaneResult], lane_det: LaneDetector,
                   bev: BEVTransformer, tracks: Sequence[Track], detections: Sequence[Detection],
                   speeds: Dict[int, float], hud: Dict[str, str]) -> np.ndarray:
        img = bgr.copy()
        h, w = img.shape[:2]

        if self.cfg.draw_roi:  # vùng đường dùng cho BEV
            overlay = img.copy()
            cv2.fillPoly(overlay, [bev.road_polygon()], (0, 200, 255))
            img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
            cv2.polylines(img, [bev.road_polygon()], True, (0, 200, 255), 1, cv2.LINE_AA)

        if self.cfg.draw_lanes and lanes is not None:
            y_bot = h - 1
            y_top = int(h * lane_det.cfg.roi[1][1])
            for line in (lanes.left, lanes.right):
                if line is not None:
                    cv2.line(img, (int(line.x_at(y_bot)), y_bot), (int(line.x_at(y_top)), y_top),
                             (0, 255, 0), 3, cv2.LINE_AA)

        for t in tracks:
            c = color_of(t.id)
            x1, y1, x2, y2 = t.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            sp = speeds.get(t.id, float("nan"))
            txt = f"#{t.id} {t.label}" + ("" if np.isnan(sp) else f" {sp:.0f}km/h")
            _text(img, txt, (x1, max(y1 - 4, 14)), bg=c)
            if self.cfg.draw_trajectory and len(t.trace) > 1:
                pts = np.array(t.trace, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, c, 2, cv2.LINE_AA)

        y = 18
        for k, v in hud.items():
            _text(img, f"{k}: {v}", (8, y), scale=0.5)
            y += 20
        return img

    # ------------------------------------------------------------------ khung phải (BEV)
    def draw_bev(self, bev_img: np.ndarray, bev: BEVTransformer, tracks: Sequence[Track],
                 speeds: Dict[int, float], estimator: SpeedEstimator) -> np.ndarray:
        canvas = bev_img.copy()
        ppm = bev.ppm
        # lưới mỗi 5 m theo chiều dài đường
        for m in range(0, int(bev.cfg.road_length_m) + 1, 5):
            yy = int(m * ppm)
            cv2.line(canvas, (0, yy), (canvas.shape[1], yy), (255, 255, 255), 1)
            _text(canvas, f"{m}m", (2, min(yy + 12, canvas.shape[0] - 2)), scale=0.35, bg=(60, 60, 60))

        for t in tracks:
            c = color_of(t.id)
            if self.cfg.draw_trajectory:
                traj = estimator.trajectory_m(t.id)
                if len(traj) > 1:
                    pts = (np.array(traj) * ppm).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [pts], False, c, 2, cv2.LINE_AA)
            p = bev.to_bev(np.array([t.foot]))[0]
            cv2.circle(canvas, (int(p[0]), int(p[1])), 5, c, -1)
            sp = speeds.get(t.id, float("nan"))
            if not np.isnan(sp):
                _text(canvas, f"{sp:.0f}", (int(p[0]) + 6, int(p[1])), scale=0.4, bg=c)
        return canvas

    # ------------------------------------------------------------------ ghép
    @staticmethod
    def compose(left: np.ndarray, bev_panel: Optional[np.ndarray], min_panel_w: int = 240) -> np.ndarray:
        if bev_panel is None:
            return left
        h = left.shape[0]
        scale = h / bev_panel.shape[0]
        panel = cv2.resize(bev_panel, (max(int(bev_panel.shape[1] * scale), 1), h), interpolation=cv2.INTER_AREA)
        canvas = np.full((h, max(panel.shape[1], min_panel_w), 3), 40, dtype=np.uint8)
        x0 = (canvas.shape[1] - panel.shape[1]) // 2
        canvas[:, x0:x0 + panel.shape[1]] = panel
        _text(canvas, "Bird's-eye view", (6, 16), scale=0.5)
        return np.hstack([left, canvas])

    @staticmethod
    def side_by_side(images: Sequence[np.ndarray], titles: Sequence[str]) -> np.ndarray:
        """Ghép nhiều ảnh cùng chiều cao với tiêu đề - dùng để so sánh trước/sau trong báo cáo."""
        h = min(i.shape[0] for i in images)
        out = []
        for img, title in zip(images, titles):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            s = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * s), h)).copy()
            _text(img, title, (8, 20), scale=0.6)
            out.append(img)
        return np.hstack(out)
