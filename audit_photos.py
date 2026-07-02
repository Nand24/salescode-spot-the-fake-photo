from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from predict import predict_score, _load_model_bundle
from train import _best_threshold, _collect_images, _make_pipeline, build_dataset


def main() -> None:
    data = Path("data")
    real_paths = sorted(_collect_images(data / "real"))
    screen_paths = sorted(_collect_images(data / "screen"))
    paths = [(p, 0) for p in real_paths] + [(p, 1) for p in screen_paths]
    X, y = build_dataset(data)

    pipe = _make_pipeline("gbm")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    cv_t, cv_acc = _best_threshold(y, cv_probs)
    cv_preds = (cv_probs >= cv_t).astype(np.int32)

    bundle = _load_model_bundle()
    threshold = float(bundle["threshold"]) if bundle else 0.5

    rows = []
    for (p, label), cv_p, cv_pr in zip(paths, cv_probs, cv_preds):
        mp = predict_score(str(p))
        mpr = int(mp >= threshold)
        rows.append(
            {
                "file": p.name,
                "folder": "real" if label == 0 else "screen",
                "true_label": "real" if label == 0 else "screen",
                "cv_prob": float(cv_p),
                "cv_pred": "real" if cv_pr == 0 else "screen",
                "cv_correct": bool(cv_pr == label),
                "model_prob": float(mp),
                "model_pred": "real" if mpr == 0 else "screen",
                "model_correct": bool(mpr == label),
                "confidence_gap": float(abs(cv_p - 0.5)),
            }
        )

    print("=== REPLACE OR RETAKE (CV wrong) ===")
    for r in sorted(rows, key=lambda x: -x["confidence_gap"]):
        if not r["cv_correct"]:
            print(
                f"[{r['true_label'].upper()} -> {r['cv_pred'].upper()}] "
                f"{r['folder']}/{r['file']} | cv_prob={r['cv_prob']:.3f}"
            )

    print("\n=== BORDERLINE (CV correct but uncertain) ===")
    for r in sorted(rows, key=lambda x: abs(x["cv_prob"] - 0.5)):
        if r["cv_correct"] and 0.35 <= r["cv_prob"] <= 0.65:
            print(f"[{r['true_label'].upper()}] {r['folder']}/{r['file']} | cv_prob={r['cv_prob']:.3f}")

    easy_real = sum(1 for r in rows if r["folder"] == "real" and r["cv_correct"] and r["cv_prob"] < 0.15)
    easy_screen = sum(1 for r in rows if r["folder"] == "screen" and r["cv_correct"] and r["cv_prob"] > 0.85)
    print(f"\nEasy real photos: {easy_real}/{len(real_paths)}")
    print(f"Easy screen photos: {easy_screen}/{len(screen_paths)}")
    print(f"CV accuracy: {cv_acc:.4f} at threshold {cv_t:.3f}")

    out = Path("artifacts/photo_audit.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps({"cv_threshold": cv_t, "cv_accuracy": cv_acc, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
