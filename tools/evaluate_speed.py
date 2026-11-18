"""Đánh giá sai số tốc độ so với ground truth (chỉ dùng với video tổng hợp).

    python tools/evaluate_speed.py --tracks output/synthetic_tracks.csv --gt data/synthetic_gt.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate_speed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate_speed(args.tracks, args.gt), indent=2))


if __name__ == "__main__":
    main()
