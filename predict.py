from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import cv2
import joblib
import numpy as np

from src.features import extract_features_from_bgr, load_and_extract


MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_THRESHOLD = 0.5


def _load_model_bundle() -> Dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def _heuristic_probability(feature_values: np.ndarray) -> float:
    names = [
        "edge_density",
        "grad_mean",
        "sat_mean",
        "sat_std",
        "val_std",
        "gray_std",
        "fft_high_mid_ratio",
        "grid_peak_strength",
        "lap_entropy",
        "blockiness",
        "highlight_clip",
        "channel_imbalance",
        "local_variance_mean",
    ]
    f = {k: float(v) for k, v in zip(names, feature_values[: len(names)])}
    z = (
        2.8 * f["fft_high_mid_ratio"]
        + 0.35 * f["grid_peak_strength"]
        + 0.7 * f["blockiness"]
        - 0.04 * f["gray_std"]
        - 0.7 * f["sat_std"]
        - 0.2 * f["lap_entropy"]
        + 0.15 * f["edge_density"]
        + 0.0003 * f["grad_mean"]
        + 0.3 * (0.4 - f["sat_mean"])
        + 0.2 * (0.25 - f["val_std"])
        + 1.5 * f["highlight_clip"]
        + 0.4 * f["channel_imbalance"]
        - 0.00005 * f["local_variance_mean"]
    )
    return float(1.0 / (1.0 + np.exp(-z)))


def get_threshold() -> float:
    bundle = _load_model_bundle()
    if isinstance(bundle, dict) and "threshold" in bundle:
        return float(bundle["threshold"])
    return DEFAULT_THRESHOLD


def _score_features(feat: np.ndarray) -> float:
    bundle = _load_model_bundle()
    feat = feat.reshape(1, -1)

    if bundle is None:
        score = _heuristic_probability(feat[0])
    elif isinstance(bundle, dict) and "pipeline" in bundle:
        score = float(bundle["pipeline"].predict_proba(feat)[0, 1])
    else:
        score = float(bundle.predict_proba(feat)[0, 1])

    return max(0.0, min(1.0, score))


def predict_from_bgr(image_bgr: np.ndarray) -> float:
    feat = extract_features_from_bgr(image_bgr).values
    return _score_features(feat)


def predict_score(image_path: str) -> float:
    feat = load_and_extract(image_path).values
    return _score_features(feat)


def predict_label(image_path: str) -> int:
    threshold = get_threshold()
    score = predict_score(image_path)
    return int(score >= threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    args = parser.parse_args()

    score = predict_score(args.image_path)
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
