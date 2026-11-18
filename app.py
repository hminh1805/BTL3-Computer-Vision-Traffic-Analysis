"""Giao diện Streamlit cho hệ thống phân tích giao thông.

Chạy:   streamlit run app.py

Luồng dùng: (1) nạp video -> (2) chỉnh vùng đường/BEV và xem trước -> (3) chạy pipeline -> (4) xem kết quả.
File này CHỈ lo giao diện; mọi xử lý nằm trong src/ (CVPipeline).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import pandas as pd
import streamlit as st

from src.app_utils import (PRESETS, ROOT, calibration_preview, load_preset, read_frame,
                           to_browser_mp4, video_info)
from src.pipeline import CVPipeline
from src.preprocessing import Preprocessor
from src.quad_editor import quad_editor

UPLOAD_DIR = ROOT / "output" / "uploads"
OUT_DIR = ROOT / "output"
CORNER_NAMES = ["Trên-trái (TL)", "Trên-phải (TR)", "Dưới-phải (BR)", "Dưới-trái (BL)"]

st.set_page_config(page_title="Phân tích giao thông - CV", page_icon="🚗", layout="wide")


@st.cache_data(show_spinner=False, max_entries=16)
def cached_frame(path: str, idx: int, max_width: int = 960):
    """Đọc 1 frame và thu nhỏ (video 4K đọc/giải mã rất chậm nếu làm lại ở mỗi lần kéo chuột)."""
    frame = read_frame(path, idx)
    if frame is not None and frame.shape[1] > max_width:
        s = max_width / frame.shape[1]
        frame = cv2.resize(frame, (max_width, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return frame


# ---------------------------------------------------------------- trạng thái / preset
def apply_preset() -> None:
    """Nạp tham số của preset vào session_state (gọi khi đổi preset và lần đầu mở app)."""
    cfg = load_preset(st.session_state["preset"])
    for i, (x, y) in enumerate(cfg.bev.src_points):
        st.session_state[f"p{i}x"], st.session_state[f"p{i}y"] = float(x), float(y)
    s = st.session_state
    s["road_w"], s["road_l"], s["ppm"] = cfg.bev.road_width_m, cfg.bev.road_length_m, cfg.bev.pixels_per_meter
    s["det_method"], s["lane_method"], s["bev_mode"] = cfg.detection.method, cfg.lane.method, cfg.bev.mode
    s["conf"], s["road_only"] = cfg.detection.confidence, cfg.detection.road_only
    s["roi_top"] = float(cfg.lane.roi[1][1])
    s["min_area"] = int(cfg.detection.motion_min_area)
    s["window_sec"] = float(cfg.speed.window_sec)


if "preset" not in st.session_state:
    st.session_state["preset"] = list(PRESETS)[0]
    apply_preset()


def build_config():
    """Gộp preset + các giá trị người dùng chỉnh trên giao diện thành PipelineConfig."""
    s = st.session_state
    cfg = load_preset(s["preset"])
    cfg.bev.src_points = [[s[f"p{i}x"], s[f"p{i}y"]] for i in range(4)]
    cfg.bev.road_width_m, cfg.bev.road_length_m, cfg.bev.pixels_per_meter = s["road_w"], s["road_l"], s["ppm"]
    cfg.bev.mode = s["bev_mode"]
    cfg.detection.method, cfg.detection.confidence = s["det_method"], s["conf"]
    cfg.detection.road_only, cfg.detection.motion_min_area = s["road_only"], int(s["min_area"])
    cfg.lane.method = s["lane_method"]
    cfg.lane.roi[1][1] = cfg.lane.roi[2][1] = s["roi_top"]
    cfg.speed.window_sec = s["window_sec"]
    return cfg


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚙️ Tham số")
    st.selectbox("Cấu hình mẫu", list(PRESETS), key="preset", on_change=apply_preset,
                 help="Đổi mẫu sẽ nạp lại toàn bộ tham số bên dưới.")

    st.subheader("Phát hiện đối tượng")
    st.selectbox("Detector", ["yolo", "motion"], key="det_method",
                 help="yolo: mô hình học sâu (cần xe thật). motion: trừ nền MOG2 (truyền thống).")
    if st.session_state["det_method"] == "yolo":
        st.slider("Ngưỡng tin cậy YOLO", 0.1, 0.9, step=0.05, key="conf")
    else:
        st.slider("Diện tích tối thiểu (px²)", 50, 2000, step=50, key="min_area")
    st.checkbox("Chỉ giữ xe nằm trong vùng đường", key="road_only")

    st.subheader("Làn đường")
    st.selectbox("Phương pháp", ["hough", "lsd"], key="lane_method")
    st.slider("Mép trên vùng ROI làn (tỉ lệ chiều cao)", 0.2, 0.9, step=0.01, key="roi_top")

    st.subheader("BEV & tốc độ")
    st.selectbox("Chế độ BEV", ["manual", "auto"], key="bev_mode",
                 help="manual: dùng 4 điểm chỉnh ở tab 1. auto: suy ra từ 2 làn đường.")
    st.number_input("Bề rộng thật của vùng (m)", 1.0, 50.0, step=0.5, key="road_w")
    st.number_input("Chiều dài thật của vùng (m)", 5.0, 300.0, step=1.0, key="road_l")
    st.slider("Độ phân giải BEV (px/m)", 5.0, 40.0, step=1.0, key="ppm")
    st.slider("Cửa sổ tính tốc độ (s)", 0.3, 3.0, step=0.1, key="window_sec")

    st.subheader("Chạy")
    max_frames = st.number_input("Số frame tối đa (0 = cả video)", 0, 100000, 0, step=50)

# ---------------------------------------------------------------- nạp video
st.title("🚗 Phân tích cảnh giao thông: làn đường · BEV · theo dõi xe · tốc độ")

col_up, col_demo = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader("Nạp video", type=["mp4", "avi", "mov", "mkv"])
with col_demo:
    st.write("")
    st.write("")
    use_demo = st.button("Dùng video giả lập có sẵn", disabled=not (ROOT / "data" / "synthetic.mp4").exists())

if uploaded is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / uploaded.name
    if not path.exists() or path.stat().st_size != uploaded.size:
        path.write_bytes(uploaded.getbuffer())
    st.session_state["video_path"] = str(path)
elif use_demo:
    st.session_state["video_path"] = str(ROOT / "data" / "synthetic.mp4")
    if st.session_state["preset"] != list(PRESETS)[1]:
        st.session_state["preset"] = list(PRESETS)[1]
        apply_preset()
        st.rerun()

video_path = st.session_state.get("video_path")
if not video_path:
    st.info("Hãy nạp một video (camera cố định, nhìn đường từ trên cao) hoặc bấm *Dùng video giả lập có sẵn*.")
    st.stop()

info = video_info(video_path)
st.caption(f"📁 {Path(video_path).name} · {info['width']}×{info['height']} · {info['frames']} frame · {info['fps']:.1f} fps")

tab_cal, tab_run, tab_res = st.tabs(["1️⃣ Hiệu chỉnh & xem trước", "2️⃣ Chạy pipeline", "3️⃣ Kết quả"])

# ---------------------------------------------------------------- tab 1: hiệu chỉnh
with tab_cal:
    st.markdown(
        "Kéo 4 chấm sao cho chúng là 4 góc của **một hình chữ nhật thật nằm trên mặt đường**, "
        "rồi nhập đúng **bề rộng / chiều dài thật** của hình chữ nhật đó ở thanh bên. "
        "Mẹo: đặt 2 cạnh bên **dọc theo 2 vạch kẻ đường** và 2 cạnh ngang tại **đầu 2 đoạn vạch đứt** "
        "(khoảng cách giữa chúng = số chu kỳ vạch × chiều dài 1 chu kỳ). Tốc độ chính xác đến đâu phụ thuộc vào bước này."
    )
    idx = st.slider("Frame xem trước", 0, max(info["frames"] - 1, 0), min(60, max(info["frames"] - 1, 0)))
    frame = cached_frame(video_path, idx)
    if frame is None:
        st.error("Không đọc được frame này.")
    else:
        try:
            ss = st.session_state
            pre_bgr = Preprocessor(build_config().preprocess)(frame).bgr
            cur = [[ss[f"p{i}x"], ss[f"p{i}y"]] for i in range(4)]
            left, right = st.columns([3, 1])
            with left:
                st.caption("🖱️ Kéo 4 chấm màu trực tiếp trên ảnh: 🔴 TL (trên-trái) · 🟠 TR (trên-phải) · 🟢 BR (dưới-phải) · 🔵 BL (dưới-trái)")
                ev = quad_editor(pre_bgr, cur, key="quad")
            # chỉ nhận giá trị MỚI hơn lần đã nhận (component nhớ giá trị cũ khi đổi preset)
            if ev and ev.get("rev", 0) > ss.get("quad_rev", 0):
                ss["quad_rev"] = ev["rev"]
                if any(abs(a - b) > 1e-6 for p, q in zip(ev["points"], cur) for a, b in zip(p, q)):
                    for i, (x, y) in enumerate(ev["points"]):
                        ss[f"p{i}x"], ss[f"p{i}y"] = float(x), float(y)
                    st.rerun()
            overlay, bev_img = calibration_preview(frame, build_config())
            with left:
                st.caption("Toạ độ hiện tại (tỉ lệ 0–1): " + " · ".join(
                    f"{n} ({ss[f'p{i}x']:.3f}, {ss[f'p{i}y']:.3f})" for i, n in enumerate(["TL", "TR", "BR", "BL"])))
            with right:
                st.image(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB), caption="Bird's-eye view (cập nhật khi thả chuột)",
                         width="stretch")
                st.caption("Đúng khi vạch kẻ đường **thẳng đứng và song song** nhau.")
            with st.expander("Xem làn đường phát hiện được & vùng ROI"):
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                         caption="Vàng: vùng BEV · Xanh: làn phát hiện · Tím: ROI làn", width="stretch")
        except Exception as e:  # noqa: BLE001
            st.exception(e)

# ---------------------------------------------------------------- tab 2: chạy
with tab_run:
    st.write("Tham số hiện tại lấy từ thanh bên và tab 1.")
    if st.button("▶️ Chạy pipeline", type="primary"):
        cfg = build_config()
        prog = st.progress(0.0, text="Đang khởi tạo...")
        preview = st.empty()
        t0 = time.time()

        def on_progress(done: int, total: int, vis) -> None:
            prog.progress(min(done / total, 1.0), text=f"Frame {done}/{total}")
            if done % 3 == 0 or done == 1:
                preview.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="stretch")

        try:
            with st.spinner("Đang tải mô hình..." if cfg.detection.method == "yolo" else "Đang xử lý..."):
                pipe = CVPipeline(cfg)
            name = Path(video_path).stem
            summary = pipe.run(video_path, str(OUT_DIR), name, max_frames=int(max_frames) or None,
                               progress_cb=on_progress)
            prog.progress(1.0, text=f"Xong trong {time.time() - t0:.1f}s")
            out = summary["outputs"]
            web = to_browser_mp4(out["video"], OUT_DIR / f"{name}_result_web.mp4")
            st.session_state["result"] = {"summary": summary, "outputs": out,
                                          "web_video": str(web) if web else None, "name": name}
            st.success("Hoàn tất! Sang tab **3️⃣ Kết quả** để xem.")
        except Exception as e:  # noqa: BLE001
            st.exception(e)

# ---------------------------------------------------------------- tab 3: kết quả
with tab_res:
    res = st.session_state.get("result")
    if not res:
        st.info("Chưa có kết quả. Hãy chạy pipeline ở tab 2.")
    else:
        sm, out = res["summary"], res["outputs"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frame đã xử lý", sm["frames_processed"])
        m2.metric("FPS end-to-end", sm["avg_fps_end_to_end"])
        m3.metric("Số xe (track) phát hiện", sm["unique_tracks"])
        m4.metric("Tốc độ TB (km/h)", sm["mean_speed_kmh"] if sm["mean_speed_kmh"] is not None else "—")

        if res["web_video"]:
            st.video(res["web_video"])
        else:
            st.warning("Không chuyển được video sang H.264 để phát trong trình duyệt "
                       "(cần `pip install imageio-ffmpeg`). Hãy tải video bên dưới để xem.")

        d1, d2, d3 = st.columns(3)
        for col, key, label, mime in ((d1, "video", "⬇️ Video kết quả", "video/mp4"),
                                      (d2, "csv", "⬇️ CSV theo từng xe", "text/csv"),
                                      (d3, "summary", "⬇️ JSON tóm tắt", "application/json")):
            p = Path(out[key])
            if p.exists():
                col.download_button(label, p.read_bytes(), file_name=p.name, mime=mime)

        st.subheader("Thời gian từng giai đoạn (ms/frame)")
        st.bar_chart(pd.Series(sm["stage_mean_ms"], name="ms"))

        csv_path = Path(out["csv"])
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty and df["speed_kmh"].notna().any():
                st.subheader("Tốc độ theo thời gian")
                piv = df.pivot_table(index="time_s", columns="track_id", values="speed_kmh")
                st.line_chart(piv)
                st.subheader("Tốc độ trung bình mỗi xe")
                per = df.groupby("track_id").agg(label=("label", "first"), frames=("frame", "count"),
                                                 speed_mean_kmh=("speed_kmh", "mean"),
                                                 speed_max_kmh=("speed_kmh", "max")).round(1)
                st.dataframe(per, width="stretch")
            with st.expander("Bảng dữ liệu đầy đủ"):
                st.dataframe(df, width="stretch")
        with st.expander("Cấu hình đã dùng"):
            st.json(sm["config"])

