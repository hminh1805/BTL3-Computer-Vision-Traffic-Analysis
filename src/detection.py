"""Module 4 - Phát hiện đối tượng (đề bài mục 3.3).

Hai detector cùng giao diện `detect(bgr) -> List[Detection]` để dễ so sánh:
  - YoloDetector  : mô hình học sâu đã huấn luyện sẵn (Ultralytics YOLO, tập COCO)
  - MotionDetector: phương pháp truyền thống - trừ nền MOG2 + hình thái học + contour
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .config import DetectionConfig

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    cls: int
    label: str


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, bgr: np.ndarray) -> List[Detection]:
        ...


class YoloDetector(BaseDetector):
    def __init__(self, cfg: DetectionConfig):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Chưa cài ultralytics. Chạy: pip install ultralytics  "
                "(hoặc đặt detection.method = 'motion' để dùng trừ nền)."
            ) from e
        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.names = getattr(self.model, "names", COCO_NAMES)

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        c = self.cfg
        res = self.model.predict(
            bgr, conf=c.confidence, imgsz=c.imgsz, classes=c.classes or None,
            device=c.device, verbose=False,
        )[0]
        out: List[Detection] = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for b, s, k in zip(xyxy, conf, cls):
            out.append(Detection(bbox=b.astype(np.float32), score=float(s), cls=int(k),
                                 label=str(self.names.get(int(k), k))))
        return out


class MotionDetector(BaseDetector):
    """Trừ nền MOG2: vật thể chuyển động = foreground. Không phân loại được xe/người."""

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.motion_history, varThreshold=cfg.motion_var_threshold, detectShadows=True
        )
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._frames = 0

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        fg = self.bg.apply(bgr)
        self._frames += 1
        if self._frames <= self.cfg.motion_warmup_frames:
            return []
        fg = (fg == 255).astype(np.uint8) * 255  # bỏ vùng bóng (giá trị 127)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k_close)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.cfg.motion_min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            out.append(Detection(bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
                                 score=1.0, cls=-1, label="moving"))
        return out


def build_detector(cfg: DetectionConfig) -> BaseDetector:
    if cfg.method == "yolo":
        return YoloDetector(cfg)
    if cfg.method == "motion":
        return MotionDetector(cfg)
    raise ValueError(f"detection.method không hợp lệ: {cfg.method}")
