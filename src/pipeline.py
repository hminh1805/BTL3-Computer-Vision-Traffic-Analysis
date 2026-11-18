"""CVPipeline - nơi duy nhất ghép các module thành một hệ thống hoàn chỉnh.

File này KHÔNG chứa thuật toán. Nó chỉ tạo các module và gọi theo thứ tự:

    frame --> Preprocessor --> LaneDetector --> BEVTransformer
                    |                                 |
                    +--> Detector --> Tracker --> SpeedEstimator --> Visualizer / Exporter
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .detection import Detection, build_detector
from .exporter import ResultExporter
from .homography import BEVTransformer
from .lane_detection import LaneDetector, LaneResult
from .preprocessing import Preprocessor
from .speed_estimation import SpeedEstimator
from .tracking import Track, Tracker
from .utils.timer import StageTimer
from .utils.video_io import VideoReader, VideoWriter
from .visualization import Visualizer


@dataclass
class FrameResult:
    frame_idx: int
    time_s: float
    bgr: np.ndarray                       # ảnh sau tiền xử lý
    gray: np.ndarray
    lanes: Optional[LaneResult] = None
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    speeds: Dict[int, float] = field(default_factory=dict)
    bev_image: Optional[np.ndarray] = None


class CVPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.timer = StageTimer()
        # các module "không phụ thuộc kích thước ảnh" tạo ngay
        self.preprocessor = Preprocessor(self.cfg.preprocess)
        self.lane_detector = LaneDetector(self.cfg.lane)
        self.detector = build_detector(self.cfg.detection)
        self.tracker = Tracker(self.cfg.tracking)
        self.speed_estimator = SpeedEstimator(self.cfg.speed)
        self.visualizer = Visualizer(self.cfg.output)
        # module cần biết kích thước ảnh -> tạo ở frame đầu tiên
        self.bev: Optional[BEVTransformer] = None

    # ------------------------------------------------------------------ xử lý 1 frame
    def process_frame(self, frame: np.ndarray, frame_idx: int, time_s: float) -> FrameResult:
        cfg, T = self.cfg, self.timer

        with T.measure("1_preprocess"):
            pre = self.preprocessor(frame)
        h, w = pre.bgr.shape[:2]
        if self.bev is None:
            self.bev = BEVTransformer(cfg.bev, (w, h))
        res = FrameResult(frame_idx=frame_idx, time_s=time_s, bgr=pre.bgr, gray=pre.gray)

        if cfg.lane.enabled:
            with T.measure("2_lane_detection"):
                res.lanes = self.lane_detector.detect(pre.gray)
            with T.measure("3_bev_update"):
                self.bev.update_from_lanes(res.lanes)

        with T.measure("4_detection"):
            dets = self.detector.detect(pre.bgr)
            if cfg.detection.road_only:
                dets = [d for d in dets
                        if self.bev.contains(((d.bbox[0] + d.bbox[2]) / 2, d.bbox[3]),
                                             cfg.detection.road_margin_px)]
            res.detections = dets

        with T.measure("5_tracking"):
            res.tracks = self.tracker.update(res.detections, frame_idx)

        with T.measure("6_speed"):
            if cfg.speed.enabled and res.tracks:
                feet = np.array([t.foot for t in res.tracks], dtype=np.float32)
                pos_m = self.bev.to_meters(feet)
                for t, p in zip(res.tracks, pos_m):
                    res.speeds[t.id] = self.speed_estimator.update(t.id, time_s, (p[0], p[1]))
            self.speed_estimator.prune(t.id for t in self.tracker.tracks)

        with T.measure("7_bev_warp"):
            if cfg.output.show_bev:
                res.bev_image = self.bev.warp(pre.bgr)
        return res

    # ------------------------------------------------------------------ vẽ kết quả
    def render(self, res: FrameResult, fps_text: str = "") -> np.ndarray:
        hud = {"Frame": str(res.frame_idx), "Method": f"{self.cfg.lane.method}/{self.cfg.detection.method}",
               "Vehicles": str(len(res.tracks))}
        if fps_text:
            hud["FPS"] = fps_text
        left = self.visualizer.draw_frame(res.bgr, res.lanes, self.lane_detector, self.bev,
                                          res.tracks, res.detections, res.speeds, hud)
        panel = None
        if self.cfg.output.show_bev and res.bev_image is not None:
            panel = self.visualizer.draw_bev(res.bev_image, self.bev, res.tracks, res.speeds,
                                             self.speed_estimator)
        return self.visualizer.compose(left, panel)

    # ------------------------------------------------------------------ chạy toàn bộ video
    def run(self, source: str, output_dir: str = "output", name: Optional[str] = None,
            max_frames: Optional[int] = None, start_frame: int = 0, show: bool = False,
            progress_cb: Optional[Callable[[int, int, np.ndarray], None]] = None) -> Dict:
        """progress_cb(n_done, n_total, ảnh_kết_quả) được gọi sau mỗi frame (dùng cho giao diện)."""
        reader = VideoReader(source, start_frame)
        name = name or (Path(source).stem if not str(source).isdigit() else "webcam")
        out = ResultExporter(output_dir, name)
        writer = VideoWriter(out.video_path, reader.fps) if self.cfg.output.save_video else None

        t_start = time.perf_counter()
        n = 0
        speeds_all: List[float] = []
        ids_seen = set()
        try:
            for idx, frame in reader:
                if max_frames is not None and n >= max_frames:
                    break
                t_s = idx / reader.fps
                t0 = time.perf_counter()
                res = self.process_frame(frame, idx, t_s)
                inst_fps = 1.0 / max(time.perf_counter() - t0, 1e-6)

                for t in res.tracks:
                    ids_seen.add(t.id)
                    fx, fy = t.foot
                    p = self.bev.to_meters(np.array([[fx, fy]], dtype=np.float32))[0]
                    sp = res.speeds.get(t.id, float("nan"))
                    if not np.isnan(sp):
                        speeds_all.append(sp)
                    x1, y1, x2, y2 = t.bbox
                    out.add_row(frame=idx, time_s=round(t_s, 3), track_id=t.id, label=t.label,
                                x1=round(float(x1), 1), y1=round(float(y1), 1),
                                x2=round(float(x2), 1), y2=round(float(y2), 1),
                                foot_x_px=round(fx, 1), foot_y_px=round(fy, 1),
                                x_m=round(float(p[0]), 3), y_m=round(float(p[1]), 3),
                                speed_kmh="" if np.isnan(sp) else round(sp, 2))

                if writer is not None or show or progress_cb is not None:
                    vis = self.render(res, f"{inst_fps:.1f}")
                    if writer is not None:
                        writer.write(vis)
                    if progress_cb is not None:
                        total = reader.frame_count - start_frame
                        if max_frames is not None:
                            total = min(total, max_frames)
                        progress_cb(n + 1, max(total, 1), vis)
                    if show:
                        cv2.imshow("CV pipeline (q de thoat)", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                n += 1
                if n % 50 == 0:
                    print(f"  ... da xu ly {n} frame")
        finally:
            reader.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - t_start
        summary = {
            "source": str(source),
            "frames_processed": n,
            "elapsed_s": round(elapsed, 2),
            "avg_fps_end_to_end": round(n / max(elapsed, 1e-6), 2),
            "stage_mean_ms": {k: round(v, 2) for k, v in self.timer.mean_ms().items()},
            "unique_tracks": len(ids_seen),
            "mean_speed_kmh": round(float(np.mean(speeds_all)), 2) if speeds_all else None,
            "config": asdict(self.cfg),
        }
        if self.cfg.output.save_csv:
            out.write_csv()
        if self.cfg.output.save_summary:
            out.write_summary(summary)
        summary["outputs"] = {"video": str(out.video_path), "csv": str(out.csv_path),
                              "summary": str(out.summary_path)}
        return summary
