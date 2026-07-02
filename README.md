# SalesCode AI Take-Home: Spot the Fake Photo

This project solves:

> Given one image, output a score in `[0, 1]` where `0 = real photo`, `1 = photo of a screen`.

## Project Structure

- `predict.py` - one-line predictor entry point required by assignment
- `train.py` - trains a lightweight logistic regression model on handcrafted CV features
- `evaluate.py` - computes accuracy and latency stats
- `src/features.py` - feature extraction (frequency, texture, gradients, color stats)
- `app.py` - optional Streamlit demo
- `report.md` - half-page writeup template to submit

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Collection (as asked in assignment)

Create:

```text
data/
  real/
  screen/
```

Take around 100 photos total:
- ~50 real photos of real-world scenes/objects
- ~50 recaptures (phone/laptop screen or printout showing an image)

Keep variety in:
- angle
- lighting
- distance
- display type and brightness

## Train

```bash
python train.py --data-dir data
```

This creates `artifacts/model.joblib` and `artifacts/metrics.json`.

## Predict (required interface)

```bash
python predict.py some_image.jpg
```

Output example:

```text
0.9312
```

## Evaluate Accuracy + Latency

```bash
python evaluate.py --data-dir data
```

This writes:
- `artifacts/eval.json`

Copy these numbers into `report.md`.

## Optional Live Demo

```bash
streamlit run app.py
```

## Practical Notes

- If `artifacts/model.joblib` exists, `predict.py` uses trained model probabilities.
- If no trained model exists, it uses a deterministic fallback heuristic so the script still returns a valid score.
- Target threshold is `0.5` by default; tune this threshold based on fraud tolerance.
