from __future__ import annotations

import time

import cv2
import numpy as np
import streamlit as st

from predict import get_threshold, predict_from_bgr, predict_score


st.set_page_config(
    page_title="Spot the Fake Photo",
    page_icon="📱",
    layout="centered",
)

THRESHOLD = get_threshold()


def _verdict(score: float) -> tuple[str, str]:
    if score >= THRESHOLD:
        return "PHOTO OF A SCREEN", "🔴"
    return "REAL PHOTO", "🟢"


def _show_result(image_rgb: np.ndarray, score: float, latency_ms: float) -> None:
    label, icon = _verdict(score)
    st.image(image_rgb, caption="Captured frame", use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Screen score", f"{score:.3f}")
    col2.metric("Threshold", f"{THRESHOLD:.3f}")
    col3.metric("Latency", f"{latency_ms:.0f} ms")

    st.progress(float(score), text=f"{icon} {label}")
    st.caption("0 = real photo · 1 = photo of a screen (recapture)")


st.title("Spot the Fake Photo — Live Demo")
st.write("Point your camera at a real scene or at a phone/laptop screen showing an image.")

tab_live, tab_upload = st.tabs(["Live camera", "Upload image"])

with tab_live:
    st.subheader("Live camera")
    camera_photo = st.camera_input("Take a live photo with your webcam")

    if camera_photo is not None:
        raw = camera_photo.getvalue()
        arr = np.frombuffer(raw, np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Could not read camera frame.")
        else:
            t0 = time.perf_counter()
            score = predict_from_bgr(image_bgr)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            _show_result(image_rgb, score, latency_ms)

with tab_upload:
    st.subheader("Upload image")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded is not None:
        raw = uploaded.read()
        arr = np.frombuffer(raw, np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Could not read image.")
        else:
            t0 = time.perf_counter()
            score = predict_from_bgr(image_bgr)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            _show_result(image_rgb, score, latency_ms)

st.divider()
st.caption(
    "Run locally: `streamlit run app.py` · "
    "CLI: `python predict.py some_image.jpg`"
)
