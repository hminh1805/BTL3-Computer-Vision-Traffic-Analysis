"""Component Streamlit: kéo thả 4 điểm trực tiếp trên ảnh (frontend ở components/quad_editor/index.html).

Hai kênh trả giá trị về Python (dùng kênh nào chạy được):
  1. Ô text ẩn "quad_bridge" (JS trong iframe ghi trực tiếp vào ô này)  -> kênh chính
  2. setComponentValue của giao thức component chuẩn                    -> dự phòng
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

_FRONTEND = Path(__file__).resolve().parents[1] / "components" / "quad_editor"
_component = components.declare_component("quad_editor", path=str(_FRONTEND))

BRIDGE_LABEL = "quad_bridge"
_HIDE_CSS = f"""
<style>
div[data-testid="stTextInput"]:has(input[aria-label="{BRIDGE_LABEL}"]) {{
    position: absolute; left: -9999px; height: 0; overflow: hidden; opacity: 0;
}}
</style>
"""


def _to_data_url(bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def quad_editor(bgr: np.ndarray, points: List[List[float]], key: str) -> Optional[dict]:
    """Hiện ảnh + 4 điểm kéo được. Trả về dict {"points": [[x,y]*4] (tỉ lệ 0..1), "rev": mốc thời gian}
    sau mỗi lần thả chuột, hoặc None nếu người dùng chưa kéo gì."""
    st.markdown(_HIDE_CSS, unsafe_allow_html=True)
    raw = st.text_input(BRIDGE_LABEL, key=BRIDGE_LABEL, label_visibility="collapsed")
    ev = _component(image=_to_data_url(bgr), points=points, bridge_label=BRIDGE_LABEL, key=key, default=None)
    if raw:
        try:
            ev = json.loads(raw)
        except ValueError:
            pass
    return ev
