"""Module 1 - Tiền xử lý (đề bài mục 3.1).

Các bước: resize -> giảm nhiễu (Gaussian) -> chuyển xám -> cân bằng sáng cục bộ (CLAHE).

Đầu ra có 2 phiên bản ảnh:
  - bgr : ảnh màu đã resize + giảm nhiễu  -> dùng cho detect đối tượng và hiển thị
  - gray: ảnh xám đã cân bằng sáng (CLAHE) -> dùng cho tìm cạnh / đường thẳng
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PreprocessConfig


@dataclass
class PreprocessResult:
    bgr: np.ndarray
    gray: np.ndarray


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
        )

    def resize(self, frame: np.ndarray) -> np.ndarray:
        w = self.cfg.max_width
        h0, w0 = frame.shape[:2]
        if w and w0 > w:
            scale = w / w0
            return cv2.resize(frame, (w, int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
        return frame

    def __call__(self, frame: np.ndarray) -> PreprocessResult:
        bgr = self.resize(frame)
        if self.cfg.denoise and self.cfg.blur_ksize > 1:
            k = self.cfg.blur_ksize | 1  # ép về số lẻ
            bgr = cv2.GaussianBlur(bgr, (k, k), 0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.cfg.clahe:
            gray = self._clahe.apply(gray)
        return PreprocessResult(bgr=bgr, gray=gray)
