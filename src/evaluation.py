"""Module 9 - Đánh giá (phần mở rộng của đề: so sánh phương pháp + đánh giá định lượng).

  - compare_lane_methods : so sánh Hough vs LSD (tốc độ, tỉ lệ tìm đủ 2 làn, độ ổn định theo thời gian)
  - compare_detectors    : so sánh YOLO vs trừ nền (tốc độ, số detection; precision/recall nếu có GT)
  - evaluate_speed       : so sánh tốc độ ước lượng với tốc độ thật (chỉ với video tổng hợp có GT)
"""
from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from typing import Dict, List

import numpy as np

from .config import PipelineConfig
from .detection import build_detector
from .lane_detection import LaneDetector
from .preprocessing import Preprocessor
from .utils.geometry import iou_matrix
from .utils.video_io import VideoReader


# ---------------------------------------------------------------------- làn đường
def compare_lane_methods(video: str, cfg: PipelineConfig, max_frames: int = 300,
                         methods=("hough", "lsd")) -> Dict[str, Dict]:
    """Chạy từng phương pháp trên cùng một đoạn video, KHÔNG làm mượt để đo độ ổn định thô."""
    results: Dict[str, Dict] = {}
    for m in methods:
        c = deepcopy(cfg)
        c.lane.method = m
        c.lane.smooth_alpha = 1.0
        c.lane.hold_frames = 0
        try:
            det = LaneDetector(c.lane)
        except RuntimeError as e:
            results[m] = {"error": str(e)}
            continue
        pre = Preprocessor(c.preprocess)
        reader = VideoReader(video)
        times, found_both, bottoms, tops = [], 0, [], []
        n = 0
        for _, frame in reader:
            if n >= max_frames:
                break
            gray = pre(frame).gray
            h = gray.shape[0]
            t0 = time.perf_counter()
            r = det.detect(gray)
            times.append((time.perf_counter() - t0) * 1000)
            if r.both:
                found_both += 1
                bottoms.append((r.left.x_at(h - 1), r.right.x_at(h - 1)))
                tops.append((r.left.x_at(0.65 * h), r.right.x_at(0.65 * h)))
            n += 1
        reader.release()

        def jitter(arr):
            a = np.array(arr)
            return float(np.mean(np.abs(np.diff(a, axis=0)))) if len(a) > 1 else float("nan")

        results[m] = {
            "frames": n,
            "mean_ms_per_frame": round(float(np.mean(times)), 2) if times else None,
            "both_lanes_rate": round(found_both / max(n, 1), 3),
            "jitter_bottom_px": round(jitter(bottoms), 2),
            "jitter_top_px": round(jitter(tops), 2),
        }
    return results


# ---------------------------------------------------------------------- detector
def compare_detectors(video: str, cfg: PipelineConfig, gt_json: str | None = None,
                      max_frames: int = 300, methods=("yolo", "motion"), iou_thr: float = 0.3):
    gt = None
    if gt_json:
        with open(gt_json, "r", encoding="utf-8") as f:
            gt = json.load(f)["frames"]
    results: Dict[str, Dict] = {}
    for m in methods:
        c = deepcopy(cfg)
        c.detection.method = m
        try:
            det = build_detector(c.detection)
        except Exception as e:  # noqa: BLE001
            results[m] = {"error": str(e)}
            continue
        pre = Preprocessor(c.preprocess)
        reader = VideoReader(video)
        times, counts = [], []
        tp = fp = fn = 0
        n = 0
        for idx, frame in reader:
            if n >= max_frames:
                break
            bgr = pre(frame).bgr
            t0 = time.perf_counter()
            dets = det.detect(bgr)
            times.append((time.perf_counter() - t0) * 1000)
            counts.append(len(dets))
            if gt is not None and idx < len(gt) and idx >= c.detection.motion_warmup_frames:
                gtb = np.array([g["bbox"] for g in gt[idx]["cars"]], dtype=np.float32).reshape(-1, 4)
                db = np.array([d.bbox for d in dets], dtype=np.float32).reshape(-1, 4)
                iou = iou_matrix(db, gtb)
                matched_g = set()
                for di in range(len(db)):
                    if iou.shape[1] == 0:
                        fp += 1
                        continue
                    gi = int(np.argmax(iou[di]))
                    if iou[di, gi] >= iou_thr and gi not in matched_g:
                        tp += 1
                        matched_g.add(gi)
                    else:
                        fp += 1
                fn += len(gtb) - len(matched_g)
            n += 1
        reader.release()
        r = {"frames": n, "mean_ms_per_frame": round(float(np.mean(times)), 2) if times else None,
             "mean_detections_per_frame": round(float(np.mean(counts)), 2) if counts else None}
        if gt is not None and (tp + fp) and (tp + fn):
            p, rc = tp / (tp + fp), tp / (tp + fn)
            r.update(precision=round(p, 3), recall=round(rc, 3),
                     f1=round(2 * p * rc / max(p + rc, 1e-9), 3))
        results[m] = r
    return results


# ---------------------------------------------------------------------- tốc độ
def evaluate_speed(tracks_csv: str, gt_json: str, max_match_px: float = 60.0,
                   skip_first_sec: float = 1.0) -> Dict:
    """Ghép điểm chạm đất của track với xe GT gần nhất (theo ảnh) rồi so tốc độ."""
    with open(gt_json, "r", encoding="utf-8") as f:
        frames = json.load(f)["frames"]
    errs: List[float] = []
    rel: List[float] = []
    with open(tracks_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["speed_kmh"] == "":
                continue
            fr = int(row["frame"])
            if fr >= len(frames) or float(row["time_s"]) < skip_first_sec:
                continue
            cars = frames[fr]["cars"]
            if not cars:
                continue
            p = np.array([float(row["foot_x_px"]), float(row["foot_y_px"])])
            d = [np.linalg.norm(p - np.array(c["foot"])) for c in cars]
            j = int(np.argmin(d))
            if d[j] > max_match_px:
                continue
            gt_speed = cars[j]["speed_kmh"]
            e = float(row["speed_kmh"]) - gt_speed
            errs.append(abs(e))
            rel.append(abs(e) / gt_speed)
    if not errs:
        return {"samples": 0}
    return {"samples": len(errs), "MAE_kmh": round(float(np.mean(errs)), 2),
            "median_abs_err_kmh": round(float(np.median(errs)), 2),
            "mean_rel_err_percent": round(100 * float(np.mean(rel)), 2)}
