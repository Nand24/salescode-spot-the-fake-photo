from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from predict import MODEL_PATH, predict_score


def _load_threshold() -> float:
    if not MODEL_PATH.exists():
        return 0.5
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "threshold" in bundle:
        return float(bundle["threshold"])
    return 0.5


def _collect(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="artifacts/eval.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    real_paths = _collect(data_dir / "real")
    screen_paths = _collect(data_dir / "screen")

    paths = [(p, 0) for p in real_paths] + [(p, 1) for p in screen_paths]
    if not paths:
        raise ValueError("No images found in evaluation dataset.")

    y_true = []
    y_prob = []
    latencies_ms = []
    for p, y in paths:
        t0 = perf_counter()
        prob = predict_score(str(p))
        latencies_ms.append((perf_counter() - t0) * 1000.0)
        y_true.append(y)
        y_prob.append(prob)

    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    threshold = _load_threshold()
    y_pred = (y_prob >= threshold).astype(np.int32)

    accuracy = float(np.mean(y_pred == y_true))
    report = {
        "n_images": int(len(paths)),
        "accuracy": accuracy,
        "threshold": threshold,
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "latency_ms_max": float(np.max(latencies_ms)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
