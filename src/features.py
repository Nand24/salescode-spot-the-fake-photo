from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class FeatureResult:
    values: np.ndarray
    names: List[str]
    diagnostics: Dict[str, float]


def _safe_resize(image: np.ndarray, target_long_side: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return image
    scale = target_long_side / float(long_side)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _fft_energy_ratio(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(gray_f))
    mag = np.log1p(np.abs(spectrum))
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = rr.max()
    if max_r <= 0:
        return 0.0

    mid_band = (rr > 0.25 * max_r) & (rr <= 0.55 * max_r)
    high_band = rr > 0.55 * max_r

    mid_energy = float(np.mean(mag[mid_band])) if np.any(mid_band) else 0.0
    high_energy = float(np.mean(mag[high_band])) if np.any(high_band) else 0.0
    return high_energy / (mid_energy + 1e-6)


def _grid_peak_strength(gray: np.ndarray) -> float:
    """Screens often leave periodic peaks along horizontal/vertical FFT axes."""
    gray_f = gray.astype(np.float32) / 255.0
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray_f)))
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    row_profile = spectrum[cy, :]
    col_profile = spectrum[:, cx]
    row_profile[cx] = 0.0
    col_profile[cy] = 0.0

    def _peakiness(profile: np.ndarray) -> float:
        if profile.size < 8:
            return 0.0
        center = profile.mean()
        peaks = np.sort(profile)[-5:]
        return float(peaks.mean() / (center + 1e-6))

    return 0.5 * (_peakiness(row_profile) + _peakiness(col_profile))


def _highlight_clip_ratio(image_bgr: np.ndarray) -> float:
    """Recaptured screens often clip highlights or look unnaturally flat."""
    b, g, r = cv2.split(image_bgr)
    stacked = np.stack([b, g, r], axis=-1)
    near_white = np.all(stacked >= 245, axis=-1)
    return float(np.mean(near_white))


def _channel_imbalance(image_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    means = np.array([b.mean(), g.mean(), r.mean()])
    return float(means.std() / (means.mean() + 1e-6))


def _local_variance_mean(gray: np.ndarray, ksize: int = 15) -> float:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f * gray_f, (ksize, ksize))
    var = np.maximum(sq_mean - mean * mean, 0.0)
    return float(np.mean(var))


def _local_contrast_entropy(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    hist, _ = np.histogram(lap_abs, bins=64, range=(0, np.percentile(lap_abs, 99) + 1e-6), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def extract_features_from_bgr(image_bgr: np.ndarray) -> FeatureResult:
    image_bgr = _safe_resize(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_density = float(np.mean(grad_mag > np.percentile(grad_mag, 85)))
    grad_mean = float(np.mean(grad_mag))

    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    sat_mean = float(np.mean(sat))
    sat_std = float(np.std(sat))
    val_std = float(np.std(val))

    gray_std = float(np.std(gray.astype(np.float32)))
    fft_ratio = _fft_energy_ratio(gray)
    grid_peak = _grid_peak_strength(gray)
    entropy = _local_contrast_entropy(gray)
    highlight_clip = _highlight_clip_ratio(image_bgr)
    channel_imbalance = _channel_imbalance(image_bgr)
    local_var = _local_variance_mean(gray)

    block = 16
    h, w = gray.shape
    h2, w2 = h - (h % block), w - (w % block)
    cropped = gray[:h2, :w2]
    if cropped.size == 0:
        blockiness = 0.0
    else:
        reshaped = cropped.reshape(h2 // block, block, w2 // block, block).swapaxes(1, 2)
        block_means = reshaped.mean(axis=(2, 3))
        blockiness = float(np.mean(np.abs(np.diff(block_means, axis=0))) + np.mean(np.abs(np.diff(block_means, axis=1))))

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
    values = np.array(
        [
            edge_density,
            grad_mean,
            sat_mean,
            sat_std,
            val_std,
            gray_std,
            fft_ratio,
            grid_peak,
            entropy,
            blockiness,
            highlight_clip,
            channel_imbalance,
            local_var,
        ],
        dtype=np.float32,
    )
    diagnostics = {k: float(v) for k, v in zip(names, values)}
    return FeatureResult(values=values, names=names, diagnostics=diagnostics)


def load_and_extract(path: str) -> FeatureResult:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return extract_features_from_bgr(image)
