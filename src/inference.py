from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


INFERENCE_PATH = Path("artifacts/inference.json")

_INFERENCE: Dict[str, Any] | None = None


def _load_inference() -> Dict[str, Any] | None:
    global _INFERENCE
    if _INFERENCE is not None:
        return _INFERENCE
    if not INFERENCE_PATH.exists():
        return None
    try:
        _INFERENCE = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return _INFERENCE


def _scale_features(feat: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    return (feat - mean) / scale


def _tree_value(x: np.ndarray, tree: Dict[str, List[float]]) -> float:
    node = 0
    children_left = tree["children_left"]
    while children_left[node] != -1:
        if x[int(tree["feature"][node])] <= tree["threshold"][node]:
            node = children_left[node]
        else:
            node = tree["children_right"][node]
    return float(tree["value"][node])


def score_from_json(feat: np.ndarray) -> float | None:
    model = _load_inference()
    if model is None:
        return None

    x = _scale_features(feat.reshape(-1), model)
    model_type = model.get("type", "logreg")

    if model_type == "logreg":
        coef = np.asarray(model["coef"], dtype=np.float64)
        intercept = float(model["intercept"])
        z = float(np.dot(x, coef) + intercept)
        return float(1.0 / (1.0 + np.exp(-z)))

    if model_type == "gbm":
        raw = float(model.get("init", 0.0))
        lr = float(model.get("learning_rate", 0.1))
        for tree in model["trees"]:
            raw += lr * _tree_value(x, tree)
        return float(1.0 / (1.0 + np.exp(-2.0 * raw)))

    return None


def threshold_from_json() -> float | None:
    model = _load_inference()
    if model is None:
        return None
    return float(model.get("threshold", 0.5))
