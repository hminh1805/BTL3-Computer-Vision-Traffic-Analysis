# BTL 3 – Computer Vision: Phân tích cảnh giao thông (hướng A)

Hệ thống nhận **video đường giao thông từ camera cố định** và cho ra:
1. **Làn đường** (Canny + Hough hoặc LSD),
2. **Bản đồ nhìn từ trên xuống (Bird's-Eye View)** nhờ homography,
3. **Phát hiện + theo dõi xe** (YOLO hoặc trừ nền; tracker Kalman + IoU),
4. **Tốc độ từng xe (km/h)** tính trên toạ độ mét của mặt đường,
5. Video kết quả (ảnh gốc | BEV), file CSV theo từng xe, file JSON tóm tắt.

---

## 1. Cấu trúc thư mục

```
BTL_3_CV/
├── main.py                  # điểm vào dòng lệnh: python main.py --video ... --config ...
├── app.py                   # giao diện web Streamlit: streamlit run app.py
├── requirements.txt
├── README.md                # file này
├── components/quad_editor/  # component JS kéo thả 4 điểm cho giao diện
├── configs/
│   ├── default.json         # cấu hình cho video thật (phải hiệu chỉnh BEV)
│   └── synthetic.json       # cấu hình cho video giả lập
├── src/                     # TOÀN BỘ hệ thống, mỗi file một chức năng
│   ├── config.py            # dataclass chứa mọi tham số, nạp/lưu JSON
│   ├── preprocessing.py     # Module 1: resize, giảm nhiễu, xám, CLAHE
│   ├── lane_detection.py    # Module 2: Canny/LSD + Hough -> 2 đường làn
│   ├── homography.py        # Module 3: homography, BEV, đổi pixel -> mét
│   ├── detection.py         # Module 4: YoloDetector, MotionDetector
│   ├── tracking.py          # Module 5: Kalman + ghép IoU (kiểu SORT)
│   ├── speed_estimation.py  # Module 6: tốc độ km/h từ toạ độ mét
│   ├── visualization.py     # Module 7: vẽ bbox/làn/BEV/quỹ đạo, ghép khung
│   ├── exporter.py          # Module 8: CSV + JSON
│   ├── evaluation.py        # Module 9: so sánh phương pháp, đo sai số
│   ├── pipeline.py          # CVPipeline: CHỈ gọi các module theo thứ tự
│   ├── app_utils.py         # hàm phụ trợ cho giao diện (không phụ thuộc streamlit)
│   └── utils/               # video_io.py, timer.py, geometry.py (IoU...)
├── tools/                   # script phụ trợ
│   ├── make_synthetic_video.py  # tạo video giả lập có tốc độ thật (ground truth)
│   ├── calibrate_bev.py         # click 4 điểm để hiệu chỉnh BEV cho video thật
│   ├── compare_methods.py       # Hough vs LSD, YOLO vs trừ nền
│   ├── evaluate_speed.py        # sai số tốc độ so với ground truth
│   └── preprocess_figure.py     # sinh hình minh hoạ từng bước cho báo cáo
├── data/                    # đặt video vào đây
└── output/                  # kết quả sinh ra
```

## 2. Cài đặt

```powershell
pip install -r requirements.txt
```
`ultralytics` chỉ cần khi dùng YOLO (lần chạy đầu sẽ tự tải `yolov8n.pt`). Không có nó vẫn chạy được với `--detector motion`.

## 2b. Giao diện web (Streamlit) – cách dễ nhất

```powershell
streamlit run app.py
```
Mở trình duyệt (thường http://localhost:8501), rồi:
1. **Nạp video** bằng khung upload (hoặc bấm *Dùng video giả lập có sẵn*).
2. Tab **1️⃣ Hiệu chỉnh**: **kéo trực tiếp 4 chấm màu trên ảnh** (🔴 TL, 🟠 TR, 🟢 BR, 🔵 BL) cho đến khi vùng vàng nằm trên mặt đường; thả chuột thì ảnh BEV bên phải cập nhật (mất ~1–3 s). Đúng khi vạch kẻ đường trên BEV thẳng và song song. Nhập kích thước thật ở thanh bên. Thanh trượt "Frame xem trước" để chọn frame có vạch đường rõ nhất.
   - Mẹo: đặt 2 cạnh bên dọc theo 2 vạch kẻ đường, 2 cạnh ngang tại đầu 2 đoạn vạch đứt; chiều dài thật = số chu kỳ vạch × độ dài 1 chu kỳ (vạch + khoảng trống).
   - Cơ chế: [components/quad_editor/index.html](components/quad_editor/index.html) là component tự viết (HTML/JS thuần), `src/quad_editor.py` bọc lại. Giá trị trả về Python qua một ô text ẩn vì kênh `setComponentValue` chuẩn của Streamlit không hoạt động trong môi trường đã thử (Streamlit 1.49 và 1.51 trên Edge).
3. Tab **2️⃣ Chạy pipeline**: bấm chạy, xem video kết quả hiện trực tiếp cùng thanh tiến độ.
4. Tab **3️⃣ Kết quả**: video (ảnh gốc | BEV), FPS, số xe, biểu đồ thời gian từng giai đoạn, biểu đồ tốc độ theo thời gian, bảng từng xe và nút tải video/CSV/JSON.

Thanh bên (sidebar) chỉnh: detector (YOLO/trừ nền), phương pháp làn (Hough/LSD), chế độ BEV, kích thước thật, số frame tối đa. Đổi "Cấu hình mẫu" sẽ nạp lại toàn bộ tham số.

Các file liên quan: [app.py](app.py) (chỉ giao diện) và `src/app_utils.py` (hàm phụ trợ: xem trước hiệu chỉnh, chuyển video sang H.264). Giao diện gọi đúng `CVPipeline.run(...)` như `main.py`, chỉ thêm `progress_cb` để cập nhật tiến độ.

## 3. Chạy nhanh bằng dòng lệnh (không cần video thật)

```powershell
python tools/make_synthetic_video.py                       # tạo data/synthetic.mp4 + synthetic_gt.json
python main.py --video data/synthetic.mp4 --config configs/synthetic.json
python tools/evaluate_speed.py --tracks output/synthetic_tracks.csv --gt data/synthetic_gt.json
python tools/compare_methods.py --video data/synthetic.mp4 --gt data/synthetic_gt.json --config configs/synthetic.json
python tools/preprocess_figure.py --video data/synthetic.mp4 --config configs/synthetic.json --frame 200
```

Kết quả đã chạy thử trên video giả lập (20 s, 600 frame):

| Chỉ số | Giá trị |
|---|---|
| FPS end-to-end (CPU, detector trừ nền) | ~27 |
| Sai số tốc độ trung vị | 1.4 km/h |
| MAE tốc độ / sai số tương đối | 4.4 km/h / 7% |
| Hough: 7 ms/frame, jitter đáy 15.7 px | LSD: 44 ms/frame, jitter đáy 1.7 px |
| Detector trừ nền: precision / recall | 0.996 / 0.897 |

> Đây là số trên dữ liệu GIẢ LẬP (nền đơn giản). Với video thật số sẽ khác – nhóm phải tự chạy và ghi số của mình vào báo cáo.

## 4. Chạy với video thật (cách làm chính cho bài nộp)

1. Chọn/quay video **camera cố định**, nhìn đường thẳng từ trên cao (cầu vượt, tầng cao), 20–60 s. Bỏ vào `data/`.
2. **Hiệu chỉnh BEV** (bước quan trọng nhất cho độ chính xác tốc độ):
   ```powershell
   python tools/calibrate_bev.py --video data/road.mp4 --config configs/my.json --road-width 7 --road-length 40
   ```
   Click 4 điểm trên mặt đường theo thứ tự **trên-trái, trên-phải, dưới-phải, dưới-trái**, tạo thành hình chữ nhật có kích thước thật ngoài đời. Cách lấy kích thước: bề rộng ≈ 3.5 m × số làn; chiều dài dùng vạch kẻ đường đứt (thường vạch 4–6 m, khoảng trống 6–9 m tuỳ loại đường) hoặc đo trên Google Maps.
3. Chỉnh `lane.roi` trong `configs/my.json` cho bao trùm vùng đường (đa giác theo tỉ lệ ảnh).
4. Chạy:
   ```powershell
   python main.py --video data/road.mp4 --config configs/my.json --show
   ```
5. Xem `output/road_result.mp4`, `road_tracks.csv`, `road_summary.json`.

Tuỳ chọn dòng lệnh: `--detector yolo|motion`, `--lane-method hough|lsd`, `--bev-mode manual|auto`, `--max-frames N`, `--start-frame N`, `--show`.

## 5. Luồng xử lý (pipeline)

```
frame ─► [1 Preprocessor] ─► ảnh màu (denoise) + ảnh xám (CLAHE)
              │
              ├─► [2 LaneDetector] ─► 2 đường làn ─► [3 BEVTransformer] (mode auto thì cập nhật H)
              │
              └─► [4 Detector] ─► lọc theo vùng đường ─► [5 Tracker] ─► track có ID
                                                              │
                              điểm chạm đất (giữa cạnh dưới bbox)
                                                              ▼
                              [3 BEV: pixel → mét] ─► [6 SpeedEstimator] ─► km/h
                                                              ▼
                                        [7 Visualizer] / [8 Exporter]
```
`pipeline.py` (class `CVPipeline`) chỉ khởi tạo các module và gọi theo thứ tự trên; hàm `process_frame` xử lý 1 frame, hàm `run` lặp cả video, ghi video/CSV/JSON và đo thời gian từng giai đoạn.

## 6. Giải thích từng module (để viết báo cáo)

**1. Tiền xử lý – `preprocessing.py`** *(đề mục 3.1)*: resize giữ tỉ lệ (rộng tối đa 960), Gaussian blur 3×3 giảm nhiễu, chuyển xám, **CLAHE** cân bằng sáng cục bộ → giúp Canny ổn định khi sáng không đều.

**2. Làn đường – `lane_detection.py`** *(mục 3.2: edges, lines)*:
- *Hough*: Canny → nhân mặt nạ ROI → `HoughLinesP`.
- *LSD*: Line Segment Detector, lọc theo ROI và độ dài.
- Lọc đoạn gần nằm ngang (|dy/dx| < 0.4); phân loại **trái/phải theo dấu độ dốc** (trục y của ảnh hướng xuống); khớp mỗi bên thành một đường `x = a·y + b` (trung bình có trọng số theo độ dài); làm mượt EMA, giữ làn cũ tối đa 10 frame khi mất dấu.

**3. Homography/BEV – `homography.py`** *(mục 3.2: phép biến đổi phối cảnh, mặt phẳng)*: mặt đường là mặt phẳng ⇒ ảnh và BEV liên hệ bằng ma trận H 3×3 tính từ 4 điểm (`cv2.getPerspectiveTransform`). Đổi điểm ảnh → điểm BEV → **mét** (chia `pixels_per_meter`). Chế độ `auto` suy ra 4 điểm từ 2 làn đường (cắt tại hai mốc y) và làm mượt EMA.

**4. Phát hiện – `detection.py`** *(mục 3.3)*: `YoloDetector` (YOLOv8n pretrained COCO; lấy car/motorcycle/bus/truck) và `MotionDetector` (MOG2 + morphology + contour). Cùng giao diện nên so sánh được. Lọc detection có điểm chạm đất nằm ngoài vùng đường.

**5. Theo dõi – `tracking.py`** *(mục 3.3)*: mỗi xe một bộ lọc Kalman (`cv2.KalmanFilter`, trạng thái vị trí+kích thước+vận tốc). Mỗi frame dự đoán → ghép detection bằng IoU tham lam → cập nhật / tạo mới / xoá track mất dấu (`max_age`). Chỉ xuất track khớp ≥ `min_hits` lần.

**6. Tốc độ – `speed_estimation.py`**: lấy vị trí mét của điểm chạm đất; tốc độ = quãng đường đầu–cuối trong cửa sổ 1 s / thời gian, nhân 3.6, làm mượt EMA; dưới 3 km/h coi là đứng yên.

**7. Trực quan – `visualization.py`** *(mục 3.4)*: bbox + ID + km/h + vệt chuyển động, làn đường (xanh lá), vùng đường (vàng), khung BEV bên phải với lưới 5 m và quỹ đạo. `side_by_side` ghép ảnh so sánh.

**8. Xuất kết quả – `exporter.py`**: CSV mỗi dòng = một xe ở một frame (bbox, điểm chạm đất px, vị trí mét, km/h); JSON tóm tắt (FPS, ms từng giai đoạn, số xe, cấu hình dùng).

**9. Đánh giá – `evaluation.py`** *(phần mở rộng)*: so sánh Hough vs LSD (ms/frame, tỉ lệ thấy đủ 2 làn, độ rung giữa các frame khi tắt làm mượt), YOLO vs trừ nền (ms, số detection, precision/recall nếu có GT), sai số tốc độ với GT.

## 7. Tham số chính (`src/config.py` hoặc JSON)

| Nhóm | Tham số quan trọng |
|---|---|
| `preprocess` | `max_width`, `blur_ksize`, `clahe_clip_limit` |
| `lane` | `method`, `roi`, `canny_low/high`, `hough_threshold`, `min_line_length`, `smooth_alpha` |
| `bev` | `mode`, `src_points`, `road_width_m`, `road_length_m`, `pixels_per_meter` |
| `detection` | `method`, `model`, `confidence`, `classes`, `road_only`, `motion_min_area` |
| `tracking` | `iou_threshold`, `max_age`, `min_hits` |
| `speed` | `window_sec`, `smooth_alpha`, `stationary_kmh` |

Khoá bắt đầu bằng `_` trong JSON được coi là ghi chú; khoá sai tên sẽ báo lỗi ngay.

## 8. Đối chiếu yêu cầu của đề

| Yêu cầu | Đáp ứng bằng |
|---|---|
| 3.1 Thu thập & tiền xử lý | video tự quay/công khai + `preprocessing.py` |
| 3.2 Hình học (≥1) | Canny/Hough/LSD **và** homography/BEV (2 hướng) |
| 3.3 Nội dung (≥1) | YOLO + trừ nền + tracking Kalman |
| 3.4 Trực quan | `visualization.py`, video kết quả, hình so sánh |
| 4 Mở rộng | so sánh phương pháp, đánh giá định lượng, đo thời gian, có công cụ demo `--show` |
| 5 Báo cáo | gợi ý ở mục 10 |

## 9. Hạn chế cần nêu trong báo cáo (thảo luận trung thực)

- Giả định **mặt đường phẳng, camera cố định**; tốc độ phụ thuộc trực tiếp vào kích thước thật khai báo ở bước hiệu chỉnh – sai 10 % kích thước ⇒ sai ~10 % tốc độ.
- Điểm chạm đất = giữa cạnh dưới bbox; bbox bị cắt ở mép khung hình, hoặc bóng xe/xe che nhau làm sai vị trí.
- Làn đường: Hough dễ rung và có thể khớp nhầm vào mép xe/vỉa hè; LSD ổn định hơn nhưng chậm hơn ~6 lần (CPU). Đường cong hoặc vạch mờ cần mô hình bậc cao hơn (chưa làm).
- Trừ nền: không phân loại xe, xe dừng lâu bị "tan vào nền", xe dính nhau bị gộp một bbox; mô hình YOLO đã huấn luyện sẵn nên lỗi ở điều kiện ban đêm/mưa.
- Tracker tham lam theo IoU, không có re-identification ⇒ ID có thể đổi khi bị che lâu.
- Trên video giả lập, số liệu đẹp hơn video thật; trong báo cáo hãy báo cáo cả hai và nói rõ.

## 10. Gợi ý dàn ý báo cáo & phân công (3–4 người)

**Dàn ý:** (1) Bài toán & dữ liệu → (2) Pipeline tổng thể (dùng sơ đồ mục 5) → (3) Từng module + hình `output/figure_pipeline_steps.png` → (4) Thực nghiệm: bảng so sánh Hough/LSD, YOLO/trừ nền, sai số tốc độ, thời gian từng giai đoạn → (5) Thảo luận, hạn chế (mục 9) → (6) Phân công + link repo.

**Phân công gợi ý:**
| Thành viên | Phần |
|---|---|
| A | Dữ liệu + tiền xử lý + hiệu chỉnh BEV (`preprocessing`, `homography`, `calibrate_bev`) |
| B | Làn đường + so sánh Hough/LSD (`lane_detection`, một phần `evaluation`) |
| C | Detection + tracking + tốc độ (`detection`, `tracking`, `speed_estimation`) |
| D | Trực quan, đánh giá, demo, báo cáo (`visualization`, `evaluation`, `pipeline`) |

## 11. Việc còn lại cho nhóm (code chưa làm thay được)

1. **Tìm/quay video thật** và hiệu chỉnh BEV – nên thử ít nhất 2 video khác nhau.
2. Chạy YOLO vs trừ nền, Hough vs LSD **trên video thật** và điền số liệu thật vào báo cáo.
3. Để có sai số tốc độ trên video thật: tự đo vài xe (ví dụ đếm thời gian xe đi qua 2 vạch có khoảng cách biết trước) rồi so với `*_tracks.csv`.
4. Đẩy code lên repository public (đề bắt buộc) – nhớ thêm `.gitignore` cho `data/*.mp4`, `output/`, `*.pt`.
5. Viết báo cáo PDF.
