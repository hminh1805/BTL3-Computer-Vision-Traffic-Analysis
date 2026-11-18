"""Điểm vào của chương trình.

Ví dụ:
    python main.py --video data/synthetic.mp4 --config configs/synthetic.json
    python main.py --video data/road.mp4 --config configs/my.json --show
    python main.py --video 0                       # webcam
"""
import argparse
import json

from src.config import PipelineConfig
from src.pipeline import CVPipeline


def main():
    ap = argparse.ArgumentParser(description="Phân tích cảnh giao thông: làn + BEV + detect/track + tốc độ")
    ap.add_argument("--video", required=True, help="đường dẫn video (hoặc số thứ tự webcam)")
    ap.add_argument("--config", default=None, help="file cấu hình JSON (mặc định: tham số trong src/config.py)")
    ap.add_argument("--output", default="output", help="thư mục kết quả")
    ap.add_argument("--name", default=None, help="tên file kết quả (mặc định: tên video)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="hiển thị cửa sổ trực tiếp (q để thoát)")
    # một số tham số hay đổi, ghi đè nhanh không cần sửa JSON
    ap.add_argument("--detector", choices=["yolo", "motion"], default=None)
    ap.add_argument("--lane-method", choices=["hough", "lsd"], default=None)
    ap.add_argument("--bev-mode", choices=["manual", "auto"], default=None)
    args = ap.parse_args()

    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    if args.detector:
        cfg.detection.method = args.detector
    if args.lane_method:
        cfg.lane.method = args.lane_method
    if args.bev_mode:
        cfg.bev.mode = args.bev_mode

    pipeline = CVPipeline(cfg)
    summary = pipeline.run(args.video, args.output, args.name, args.max_frames, args.start_frame, args.show)
    summary.pop("config", None)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
