"""Công cụ chọn 4 điểm hiệu chỉnh BEV bằng chuột và ghi vào file cấu hình JSON.

Chạy:
    python tools/calibrate_bev.py --video data/my_video.mp4 --config configs/my.json \
        --road-width 7 --road-length 40

Bạn click 4 điểm trên MẶT ĐƯỜNG theo thứ tự:
    1) trên-trái   2) trên-phải   3) dưới-phải   4) dưới-trái
4 điểm này phải tạo thành một HÌNH CHỮ NHẬT thật ngoài đời (ví dụ 2 mép đường + 2 vạch ngang,
hoặc 2 vạch làn song song cách nhau theo chiều dài đã biết). Nhập kích thước thật qua
--road-width / --road-length (mét). Phím: r = chọn lại, s = lưu, q = thoát.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PipelineConfig  # noqa: E402
from src.preprocessing import Preprocessor  # noqa: E402
from src.utils.video_io import VideoReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True, help="file JSON để ghi (tạo mới nếu chưa có)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--road-width", type=float, required=True, help="bề rộng thật (m) của hình chữ nhật")
    ap.add_argument("--road-length", type=float, required=True, help="chiều dài thật (m) của hình chữ nhật")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = PipelineConfig.from_json(cfg_path) if cfg_path.exists() else PipelineConfig()
    reader = VideoReader(args.video)
    frame = reader.read_frame_at(args.frame)
    reader.release()
    if frame is None:
        raise SystemExit("Không đọc được frame.")
    img = Preprocessor(cfg.preprocess)(frame).bgr
    h, w = img.shape[:2]

    pts = []
    labels = ["1 TL", "2 TR", "3 BR", "4 BL"]

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while True:
        vis = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
            cv2.putText(vis, labels[i], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(pts) == 4:
            cv2.polylines(vis, [np.array(pts, np.int32)], True, (0, 255, 255), 2)
        msg = "Click 4 diem: TL, TR, BR, BL | r=chon lai, s=luu, q=thoat"
        cv2.putText(vis, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("calibrate", vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("q"):
            break
        elif k == ord("s") and len(pts) == 4:
            cfg.bev.src_points = [[round(x / w, 4), round(y / h, 4)] for x, y in pts]
            cfg.bev.road_width_m = args.road_width
            cfg.bev.road_length_m = args.road_length
            cfg.bev.mode = "manual"
            cfg.to_json(cfg_path)
            print(f"Đã lưu vào {cfg_path}: src_points = {cfg.bev.src_points}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
