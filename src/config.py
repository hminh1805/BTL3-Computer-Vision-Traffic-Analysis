"""Cấu hình toàn hệ thống.

Mọi tham số của từng module nằm ở đây (dưới dạng dataclass) để:
  - không phải sửa code khi muốn thử tham số khác,
  - có thể lưu / nạp từ file JSON (xem thư mục configs/),
  - dễ đưa bảng tham số vào báo cáo.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PreprocessConfig:
    max_width: int = 960          # resize giữ tỉ lệ nếu ảnh rộng hơn giá trị này (0 = không resize)
    denoise: bool = True          # làm mờ Gaussian nhẹ để giảm nhiễu
    blur_ksize: int = 3           # kích thước kernel Gaussian (số lẻ)
    clahe: bool = True            # cân bằng sáng cục bộ (CLAHE) cho ảnh xám dùng ở bước tìm làn
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8


@dataclass
class LaneConfig:
    enabled: bool = True
    method: str = "hough"         # "hough" (Canny + HoughLinesP) hoặc "lsd" (Line Segment Detector)
    canny_low: int = 50
    canny_high: int = 150
    # Vùng quan tâm (đa giác), toạ độ theo TỈ LỆ (x/width, y/height)
    roi: List[List[float]] = field(
        default_factory=lambda: [[0.02, 1.0], [0.40, 0.55], [0.60, 0.55], [0.98, 1.0]]
    )
    hough_threshold: int = 30
    min_line_length: int = 40
    max_line_gap: int = 60
    min_abs_slope: float = 0.4    # bỏ các đoạn gần nằm ngang (|dy/dx| nhỏ)
    lsd_min_length: float = 40.0
    smooth_alpha: float = 0.25    # EMA làm mượt tham số làn (1.0 = không làm mượt)
    hold_frames: int = 10         # giữ làn cũ tối đa bao nhiêu frame khi mất dấu


@dataclass
class BEVConfig:
    # "manual": dùng src_points cố định (chọn bằng tools/calibrate_bev.py)
    # "auto"  : tự suy ra từ 2 đường làn phát hiện được ở module lane_detection
    mode: str = "manual"
    # 4 điểm trên ảnh gốc, theo TỈ LỆ, thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái
    # (là 4 góc của một hình chữ nhật NẰM TRÊN MẶT ĐƯỜNG, ví dụ vạch kẻ đường)
    src_points: List[List[float]] = field(
        default_factory=lambda: [[0.42, 0.62], [0.58, 0.62], [0.90, 1.0], [0.10, 1.0]]
    )
    road_width_m: float = 7.0     # bề rộng thực của hình chữ nhật trên (mét)
    road_length_m: float = 40.0   # chiều dài thực của hình chữ nhật trên (mét)
    pixels_per_meter: float = 15.0  # độ phân giải ảnh BEV
    # Chỉ dùng cho mode "auto": cắt 2 đường làn tại các mốc y này (tỉ lệ chiều cao ảnh)
    auto_y_top: float = 0.65
    auto_y_bottom: float = 0.98
    auto_smooth_alpha: float = 0.1


@dataclass
class DetectionConfig:
    method: str = "yolo"          # "yolo" hoặc "motion" (trừ nền MOG2 - phương pháp truyền thống)
    model: str = "yolov8n.pt"
    confidence: float = 0.35
    imgsz: int = 640
    device: Optional[str] = None  # None = tự chọn; "cpu" / "cuda:0"
    # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
    classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    road_only: bool = True        # chỉ giữ đối tượng có điểm chạm đất nằm trong vùng đường
    road_margin_px: float = 30.0  # cho phép lệch ra ngoài vùng đường bao nhiêu pixel
    # --- tham số riêng cho "motion" ---
    motion_history: int = 200
    motion_var_threshold: float = 32.0
    motion_min_area: int = 400
    motion_warmup_frames: int = 20


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.2
    max_age: int = 15             # số frame tối đa giữ track khi không khớp detection
    min_hits: int = 3             # số lần khớp tối thiểu để track được "xác nhận"
    trace_length: int = 40        # độ dài vệt chuyển động vẽ trên ảnh


@dataclass
class SpeedConfig:
    enabled: bool = True
    window_sec: float = 1.0       # cửa sổ thời gian để tính vận tốc
    min_samples: int = 5
    min_dt_sec: float = 0.3
    smooth_alpha: float = 0.3     # EMA làm mượt tốc độ
    stationary_kmh: float = 3.0   # dưới ngưỡng này coi là đứng yên
    max_kmh: float = 200.0        # trên ngưỡng này coi là nhiễu và bỏ qua


@dataclass
class OutputConfig:
    show_bev: bool = True         # ghép khung BEV bên phải video kết quả
    draw_lanes: bool = True
    draw_roi: bool = True
    draw_trajectory: bool = True
    save_video: bool = True
    save_csv: bool = True
    save_summary: bool = True


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    bev: BEVConfig = field(default_factory=BEVConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _build(cls, data)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


def _build(cls, data: dict):
    """Nạp dict (từ JSON) vào dataclass lồng nhau; khoá bắt đầu bằng '_' được coi là ghi chú."""
    obj = cls()
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if not hasattr(obj, key):
            raise ValueError(f"Khoá cấu hình không hợp lệ: '{key}' (trong {cls.__name__})")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(obj, key, _build(type(current), value))
        else:
            setattr(obj, key, value)
    return obj
