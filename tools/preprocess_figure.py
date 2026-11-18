"""Sinh hình minh hoạ cho báo cáo: ảnh gốc | tiền xử lý | cạnh Canny | làn đường | BEV.

    python tools/preprocess_figure.py --video data/synthetic.mp4 --config configs/synthetic.json --frame 100
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PipelineConfig  # noqa: E402
from src.homography import BEVTransformer  # noqa: E402
from src.lane_detection import LaneDetector  # noqa: E402
from src.preprocessing import Preprocessor  # noqa: E402
from src.utils.video_io import VideoReader  # noqa: E402
from src.visualization import Visualizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="output/figure_pipeline_steps.png")
    args = ap.parse_args()

    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    reader = VideoReader(args.video)
    frame = reader.read_frame_at(args.frame)
    reader.release()

    pre = Preprocessor(cfg.preprocess)(frame)
    det = LaneDetector(cfg.lane)
    lanes = det.detect(pre.gray)
    h, w = pre.bgr.shape[:2]
    bev = BEVTransformer(cfg.bev, (w, h))

    lane_vis = pre.bgr.copy()
    for s in lanes.segments.astype(int):
        cv2.line(lane_vis, (s[0], s[1]), (s[2], s[3]), (0, 0, 255), 1)
    for line in (lanes.left, lanes.right):
        if line is not None:
            cv2.line(lane_vis, (int(line.x_at(h - 1)), h - 1), (int(line.x_at(cfg.lane.roi[1][1] * h)),
                     int(cfg.lane.roi[1][1] * h)), (0, 255, 0), 3)
    cv2.polylines(lane_vis, [bev.road_polygon()], True, (0, 200, 255), 2)

    panels = [frame if frame.shape[:2] == (h, w) else cv2.resize(frame, (w, h)), pre.gray]
    titles = ["1. Anh goc", "2. Tien xu ly (xam + CLAHE)"]
    if lanes.edges is not None:
        panels.append(lanes.edges)
        titles.append("3. Canny + ROI")
    panels += [lane_vis, bev.warp(pre.bgr)]
    titles += ["4. Doan thang & lan duong", "5. Bird's-eye view"]
    fig = Visualizer.side_by_side(panels, titles)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, fig)
    print(f"Da luu {args.out}")


if __name__ == "__main__":
    main()
