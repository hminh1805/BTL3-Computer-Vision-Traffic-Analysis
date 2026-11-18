"""Module 6 - Ước lượng tốc độ từ toạ độ mặt đường (mét).

Với mỗi track ta nhận vị trí (x, y) tính bằng mét trên mặt phẳng đường (đã qua homography).
Tốc độ = quãng đường dịch chuyển trong cửa sổ `window_sec` giây gần nhất / thời gian của cửa sổ
(dùng độ dịch chuyển đầu-cuối nên ít bị nhiễu hơn cộng dồn từng bước nhỏ), sau đó làm mượt EMA.

Giả định / hạn chế (nên nêu trong báo cáo):
  - mặt đường phẳng, tỉ lệ mét/pixel đúng theo kích thước người dùng khai báo;
  - điểm chạm đất = giữa cạnh dưới bbox (sai số lớn khi bbox cắt ra khỏi khung hình);
  - camera cố định.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from .config import SpeedConfig


class SpeedEstimator:
    MS_TO_KMH = 3.6

    def __init__(self, cfg: SpeedConfig):
        self.cfg = cfg
        self._hist: Dict[int, Deque[Tuple[float, float, float]]] = {}  # id -> (t, x_m, y_m)
        self._speed: Dict[int, float] = {}

    def update(self, track_id: int, t_sec: float, pos_m: Tuple[float, float]) -> float:
        h = self._hist.setdefault(track_id, deque(maxlen=600))
        h.append((t_sec, float(pos_m[0]), float(pos_m[1])))

        # lấy các mẫu trong cửa sổ thời gian gần nhất
        win = [s for s in h if t_sec - s[0] <= self.cfg.window_sec]
        if len(win) >= self.cfg.min_samples:
            dt = win[-1][0] - win[0][0]
            if dt >= self.cfg.min_dt_sec:
                dist = float(np.hypot(win[-1][1] - win[0][1], win[-1][2] - win[0][2]))
                kmh = dist / dt * self.MS_TO_KMH
                if kmh <= self.cfg.max_kmh:
                    kmh = 0.0 if kmh < self.cfg.stationary_kmh else kmh
                    prev = self._speed.get(track_id)
                    a = self.cfg.smooth_alpha
                    self._speed[track_id] = kmh if prev is None else a * kmh + (1 - a) * prev
        return self._speed.get(track_id, float("nan"))

    def speed(self, track_id: int) -> float:
        return self._speed.get(track_id, float("nan"))

    def trajectory_m(self, track_id: int, last: int = 40) -> List[Tuple[float, float]]:
        h = self._hist.get(track_id)
        if not h:
            return []
        return [(x, y) for _, x, y in list(h)[-last:]]

    def prune(self, alive_ids) -> None:
        """Xoá dữ liệu của các track đã biến mất (tránh phình bộ nhớ)."""
        alive = set(alive_ids)
        for k in list(self._hist):
            if k not in alive:
                del self._hist[k]
                self._speed.pop(k, None)
