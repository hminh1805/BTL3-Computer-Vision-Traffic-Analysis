"""Đo thời gian từng giai đoạn của pipeline (phục vụ phần đánh giá tốc độ)."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict


class StageTimer:
    def __init__(self) -> None:
        self.total: Dict[str, float] = defaultdict(float)
        self.count: Dict[str, int] = defaultdict(int)
        self.last: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.total[name] += dt
            self.count[name] += 1
            self.last[name] = dt

    def mean_ms(self) -> Dict[str, float]:
        return {k: 1000.0 * self.total[k] / max(self.count[k], 1) for k in self.total}
