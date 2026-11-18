"""Hàm phụ trợ cho giao diện Streamlit (tách khỏi app.py để dễ kiểm thử, không phụ thuộc streamlit)."""
from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .homography import BEVTransformer
from .lane_detection import LaneDetector
from .preprocessing import Preprocessor

ROOT = Path(__file__).resolve().parents[1]

PRESETS = {
    "Video thật (mặc định)": ROOT / "configs" / "default.json",
    "Video giả lập (synthetic)": ROOT / "configs" / "synthetic.json",
}


def load_preset(name: str) -> PipelineConfig:
    return PipelineConfig.from_json(PRESETS[name])


def read_frame(video_path: str, index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def calibration_preview(frame: np.ndarray, cfg: PipelineConfig):
    """Trả về (ảnh có vùng đường + làn, ảnh BEV) để người dùng chỉnh 4 điểm ngay trên giao diện."""
    pre = Preprocessor(cfg.preprocess)(frame)
    h, w = pre.bgr.shape[:2]
    bev = BEVTransformer(cfg.bev, (w, h))
    img = pre.bgr.copy()
    overlay = img.copy()
    cv2.fillPoly(overlay, [bev.road_polygon()], (0, 200, 255))
    img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
    cv2.polylines(img, [bev.road_polygon()], True, (0, 200, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(bev.road_polygon()):
        cv2.circle(img, tuple(int(v) for v in p), 6, (0, 0, 255), -1)
        cv2.putText(img, ["TL", "TR", "BR", "BL"][i], (int(p[0]) + 8, int(p[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    if cfg.lane.enabled:
        c = deepcopy(cfg.lane)
        c.smooth_alpha = 1.0
        lanes = LaneDetector(c).detect(pre.gray)
        y_bot, y_top = h - 1, int(h * c.roi[1][1])
        for line in (lanes.left, lanes.right):
            if line is not None:
                cv2.line(img, (int(line.x_at(y_bot)), y_bot), (int(line.x_at(y_top)), y_top), (0, 255, 0), 3, cv2.LINE_AA)
        roi = LaneDetector(c).roi_polygon(h, w)
        cv2.polylines(img, [roi], True, (255, 0, 255), 1, cv2.LINE_AA)
    return img, bev.warp(pre.bgr)


def to_browser_mp4(src: str | Path, dst: str | Path) -> Optional[Path]:
    """Chuyển mp4v (OpenCV) sang H.264 để trình duyệt phát được. Trả None nếu không có ffmpeg."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-vcodec", "libx264",
           "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", str(dst)]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except Exception:  # noqa: BLE001
        return None
    return Path(dst) if Path(dst).exists() else None
