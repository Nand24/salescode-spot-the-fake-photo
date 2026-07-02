from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import load_and_extract
from src.export_model import export_inference_json


def _collect_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def build_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    real_paths = _collect_images(data_dir / "real")
    screen_paths = _collect_images(data_dir / "screen")

    if not real_paths or not screen_paths:
        raise ValueError(
            "Dataset missing images. Expected at least one image in data/real and data/screen."
        )

    feats = []
    labels = []
    for p in real_paths:
        feats.append(load_and_extract(str(p)).values)
        labels.append(0)
    for p in screen_paths:
        feats.append(load_and_extract(str(p)).values)
        labels.append(1)

    return np.vstack(feats), np.array(labels, dtype=np.int32)


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        acc = accuracy_score(y_true, (probs >= t).astype(np.int32))
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    return best_t, best_acc


def _make_pipeline(model: str) -> Pipeline:
    if model == "gbm":
        clf = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            random_state=42,
        )
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Folder containing real/ and screen/")
    parser.add_argument("--model-out", default="artifacts/model.joblib")
    parser.add_argument("--metrics-out", default="artifacts/metrics.json")
    parser.add_argument("--inference-out", default="artifacts/inference.json")
    parser.add_argument("--model", choices=["logreg", "gbm"], default="gbm")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    inference_out = Path(args.inference_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    X, y = build_dataset(data_dir)
    pipeline = _make_pipeline(args.model)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    cv_auc = float(roc_auc_score(y, cv_probs))
    cv_threshold, cv_accuracy = _best_threshold(y, cv_probs)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    t0 = perf_counter()
    pipeline.fit(X_train, y_train)
    train_ms = (perf_counter() - t0) * 1000.0

    train_probs = pipeline.predict_proba(X_train)[:, 1]
    holdout_threshold, _ = _best_threshold(y_train, train_probs)

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= holdout_threshold).astype(np.int32)

    acc = float(accuracy_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))

    report = classification_report(y_test, preds, target_names=["real", "screen"], output_dict=True)
    bundle = {
        "pipeline": pipeline,
        "threshold": holdout_threshold,
        "model_type": args.model,
    }
    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "cv_accuracy": cv_accuracy,
        "cv_roc_auc": cv_auc,
        "threshold": holdout_threshold,
        "cv_threshold": cv_threshold,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_time_ms": train_ms,
        "classification_report": report,
    }

    joblib.dump(bundle, model_out)
    export_inference_json(pipeline, cv_threshold, inference_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_out}")
    print(f"Saved inference to: {inference_out}")
    print(f"Saved metrics to: {metrics_out}")
    print(f"Holdout Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
    print(f"CV Accuracy (tuned threshold): {cv_accuracy:.4f} | CV ROC-AUC: {cv_auc:.4f}")
    print(f"Decision threshold: {holdout_threshold:.4f}")


if __name__ == "__main__":
    main()
