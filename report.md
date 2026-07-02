# Spot the Fake Photo - Submission Note

## 1) Approach

I used a lightweight classical computer vision pipeline designed for mobile-friendly inference:

- Extract handcrafted image features that are useful for recapture detection:
  - Frequency-domain signature (`fft_high_mid_ratio`) to capture display/moire-like patterns.
  - Texture and local contrast (`lap_entropy`, gradients, edge density).
  - Color and illumination statistics (saturation/value spread).
  - Coarse blockiness statistic.
- Train a logistic regression classifier on these features (`real=0`, `screen=1`).
- Output a calibrated score in `[0, 1]` through `predict.py`.

This keeps the model tiny, fast, and easy to deploy on-device.

## 2) Accuracy

Fill after running:

- Validation/Test Accuracy: `____`
- ROC-AUC: `____`
- Dataset size: `____ real`, `____ screen`
- Honest caveats:
  - Likely weaker on edge cases such as heavy motion blur, low-light noise, or glossy reflections that mimic screen artifacts.
  - Performance depends on dataset diversity.

## 3) Latency

Measured with `python evaluate.py --data-dir data`:

- Mean latency per image: `____ ms`
- P95 latency: `____ ms`
- Device: `____` (example: laptop CPU)

Target behavior: near-instant response.

## 4) Cost Per Image

### On-device

- Inference on user phone CPU/GPU: effectively `$0` marginal cloud cost per image.

### Cloud estimate (if server-side)

Assumptions (edit as needed):
- One CPU instance at `$0.10/hour`
- Throughput: `X` images/second
- Images/hour = `3600 * X`
- Cost per image = `0.10 / (3600 * X)`

Example with `X=20 img/s`:
- `72,000` images/hour
- Cost/image `~$0.00000139`
- `~$1.39 per million images`

## 5) If Given More Time

- Add hard negatives (glare, reflections, blur, compression) and larger hold-out sets.
- Add temporal signals for live capture (flicker/refresh artifacts across frames).
- Quantize and benchmark on real mobile hardware.
- Tune decision threshold by fraud vs false-reject cost (ROC/PR-driven operating point).
- Build monitoring + periodic retraining to stay robust as cheating strategies evolve.
