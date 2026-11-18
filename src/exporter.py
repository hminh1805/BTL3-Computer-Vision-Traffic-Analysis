"""Module 8 - Xuất kết quả: CSV theo từng track/frame và JSON tóm tắt."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

CSV_FIELDS = ["frame", "time_s", "track_id", "label", "x1", "y1", "x2", "y2",
              "foot_x_px", "foot_y_px", "x_m", "y_m", "speed_kmh"]


class ResultExporter:
    def __init__(self, output_dir: str | Path, name: str):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.rows: List[Dict] = []

    @property
    def video_path(self) -> Path:
        return self.dir / f"{self.name}_result.mp4"

    @property
    def csv_path(self) -> Path:
        return self.dir / f"{self.name}_tracks.csv"

    @property
    def summary_path(self) -> Path:
        return self.dir / f"{self.name}_summary.json"

    def add_row(self, **row) -> None:
        self.rows.append(row)

    def write_csv(self) -> Path:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            wr.writeheader()
            for r in self.rows:
                wr.writerow({k: r.get(k, "") for k in CSV_FIELDS})
        return self.csv_path

    def write_summary(self, summary: Dict) -> Path:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return self.summary_path
