"""Đọc / ghi video."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoReader:
    """Đọc video (hoặc webcam nếu source là số nguyên) theo từng frame."""

    def __init__(self, source: str | int, start_frame: int = 0):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if isinstance(source, str) and not Path(source).exists():
            raise FileNotFoundError(f"Không tìm thấy video: {source}")
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {source}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps: float = fps if fps and fps > 1 else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self._index = start_frame

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield self._index, frame
            self._index += 1

    def read_frame_at(self, index: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class VideoWriter:
    """Ghi video mp4; khởi tạo trễ khi nhận frame đầu tiên để biết kích thước."""

    def __init__(self, path: str | Path, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
