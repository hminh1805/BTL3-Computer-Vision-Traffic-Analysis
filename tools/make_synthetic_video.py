"""Tạo video đường giả lập (có tốc độ thật + bbox thật) để kiểm thử pipeline khi chưa có video thật.

Cách làm: dựng "mặt đường" nhìn từ trên xuống (đơn vị mét), vẽ xe chạy với tốc độ biết trước, rồi
warp sang ảnh phối cảnh bằng một homography đã biết. Nhờ đó ta biết tốc độ thật để đánh giá.

Chạy:  python tools/make_synthetic_video.py
Sinh ra: data/synthetic.mp4, data/synthetic_gt.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROAD_W, ROAD_L, PPM = 7.0, 40.0, 20      # mét, mét, pixel/mét của "thế giới"
W, H = 960, 540
# 4 góc mặt đường (trên-trái, trên-phải, dưới-phải, dưới-trái) trên ảnh phối cảnh
IMG_CORNERS = np.array([[400, 180], [560, 180], [830, 540], [130, 540]], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.mp4")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ww, wh = int(ROAD_W * PPM), int(ROAD_L * PPM)
    world_corners = np.array([[0, 0], [ww, 0], [ww, wh], [0, wh]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(world_corners, IMG_CORNERS)

    # nền (cỏ + nhiễu cố định)
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:] = (60, 120, 70)
    bg[:150] = (200, 190, 170)  # "bầu trời"
    bg = np.clip(bg.astype(np.int16) + rng.normal(0, 6, bg.shape), 0, 255).astype(np.uint8)

    # mặt đường nhìn từ trên xuống
    road = np.full((wh, ww, 3), 85, np.uint8)
    road = np.clip(road.astype(np.int16) + rng.normal(0, 5, road.shape), 0, 255).astype(np.uint8)
    for x_m in (0.25, ROAD_W - 0.25):                       # vạch biên liền
        cv2.line(road, (int(x_m * PPM), 0), (int(x_m * PPM), wh), (235, 235, 235), 5)
    for y in range(0, wh, int(4 * PPM)):                    # vạch giữa đứt
        cv2.line(road, (int(ROAD_W / 2 * PPM), y), (int(ROAD_W / 2 * PPM), y + int(2 * PPM)), (0, 220, 255), 4)

    # lên lịch xe: làn 0 chạy lại gần camera, làn 1 chạy ra xa
    cars, cid = [], 0
    lane_base = {0: 45.0, 1: 70.0}   # km/h
    for lane in (0, 1):
        t = -rng.uniform(0, 2.5)
        while t < args.seconds:
            v = lane_base[lane] * rng.uniform(0.9, 1.1)
            cars.append(dict(id=cid, lane=lane, v=v, t0=t, color=tuple(int(c) for c in rng.integers(40, 230, 3))))
            cid += 1
            t += rng.uniform(3.0, 4.0)

    CAR_W, CAR_L = 1.8, 4.2
    lane_x = {0: ROAD_W * 0.25, 1: ROAD_W * 0.75}

    def car_y(c, t):  # tâm xe theo trục dọc đường (mét), 0 = xa nhất
        s = c["v"] / 3.6 * (t - c["t0"])
        return -3.0 + s if c["lane"] == 0 else ROAD_L + 3.0 - s

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    gt_frames = []
    for f in range(int(args.seconds * args.fps)):
        t = f / args.fps
        world = road.copy()
        frame_gt = []
        for c in cars:
            cy = car_y(c, t)
            if cy < -CAR_L or cy > ROAD_L + CAR_L:
                continue
            cx = lane_x[c["lane"]]
            p1 = (int((cx - CAR_W / 2) * PPM), int((cy - CAR_L / 2) * PPM))
            p2 = (int((cx + CAR_W / 2) * PPM), int((cy + CAR_L / 2) * PPM))
            cv2.rectangle(world, p1, p2, c["color"], -1)
            cv2.rectangle(world, (p1[0] + 4, p1[1] + int(0.25 * CAR_L * PPM)),
                          (p2[0] - 4, p1[1] + int(0.5 * CAR_L * PPM)), (40, 40, 40), -1)  # kính
            # ground truth trên ảnh
            quad = np.array([[(cx - CAR_W / 2) * PPM, (cy - CAR_L / 2) * PPM],
                             [(cx + CAR_W / 2) * PPM, (cy - CAR_L / 2) * PPM],
                             [(cx + CAR_W / 2) * PPM, (cy + CAR_L / 2) * PPM],
                             [(cx - CAR_W / 2) * PPM, (cy + CAR_L / 2) * PPM]], np.float32)
            q = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), Hm).reshape(-1, 2)
            if -10 < cy < ROAD_L + 10:
                x1, y1 = np.clip(q.min(0), 0, [W - 1, H - 1])
                x2, y2 = np.clip(q.max(0), 0, [W - 1, H - 1])
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    frame_gt.append(dict(id=c["id"], speed_kmh=c["v"], bbox=[float(x1), float(y1), float(x2), float(y2)],
                                         foot=[float((x1 + x2) / 2), float(y2)]))
        img = bg.copy()
        cv2.warpPerspective(world, Hm, (W, H), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        vw.write(img)
        gt_frames.append(dict(frame=f, cars=frame_gt))
    vw.release()

    gt_path = out.with_name(out.stem + "_gt.json")
    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(dict(fps=args.fps, road_width_m=ROAD_W, road_length_m=ROAD_L,
                       corners_px=IMG_CORNERS.tolist(), frames=gt_frames), fh)
    print(f"Da tao {out} ({len(gt_frames)} frame) va {gt_path}")


if __name__ == "__main__":
    main()
