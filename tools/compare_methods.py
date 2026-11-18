"""So sánh phương pháp (phần mở rộng): làn đường Hough vs LSD, detector YOLO vs trừ nền.

Ví dụ:
    python tools/compare_methods.py --video data/synthetic.mp4 --gt data/synthetic_gt.json \
        --config configs/synthetic.json --what lanes detectors
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PipelineConfig  # noqa: E402
from src.evaluation import compare_detectors, compare_lane_methods  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--gt", default=None, help="file *_gt.json (chỉ có với video tổng hợp)")
    ap.add_argument("--what", nargs="+", default=["lanes", "detectors"], choices=["lanes", "detectors"])
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="output/comparison.json")
    args = ap.parse_args()

    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    report = {}
    if "lanes" in args.what:
        print("== So sanh lane: hough vs lsd ==")
        report["lanes"] = compare_lane_methods(args.video, cfg, args.max_frames)
        print(json.dumps(report["lanes"], indent=2))
    if "detectors" in args.what:
        print("== So sanh detector: yolo vs motion ==")
        report["detectors"] = compare_detectors(args.video, cfg, args.gt, args.max_frames)
        print(json.dumps(report["detectors"], indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Da luu {args.out}")


if __name__ == "__main__":
    main()
