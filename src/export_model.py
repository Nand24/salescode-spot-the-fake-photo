from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def export_inference_json(
    pipeline: Pipeline,
    threshold: float,
    out_path: Path,
) -> None:
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    payload: Dict[str, Any] = {
        "threshold": float(threshold),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    if isinstance(clf, LogisticRegression):
        payload["type"] = "logreg"
        payload["coef"] = clf.coef_[0].tolist()
        payload["intercept"] = float(clf.intercept_[0])
    elif isinstance(clf, GradientBoostingClassifier):
        payload["type"] = "gbm"
        payload["learning_rate"] = float(clf.learning_rate)
        init_val = 0.0
        if hasattr(clf.init_, "constant_"):
            init_val = float(clf.init_.constant_[0][0])
        payload["init"] = init_val

        trees: List[Dict[str, List[float]]] = []
        for estimator in clf.estimators_.ravel():
            tree = estimator.tree_
            trees.append(
                {
                    "children_left": tree.children_left.tolist(),
                    "children_right": tree.children_right.tolist(),
                    "feature": tree.feature.tolist(),
                    "threshold": tree.threshold.tolist(),
                    "value": tree.value.reshape(-1).tolist(),
                }
            )
        payload["trees"] = trees
    else:
        raise TypeError(f"Unsupported classifier for JSON export: {type(clf)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
