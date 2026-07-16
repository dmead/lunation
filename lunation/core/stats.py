"""Basic image statistics matching the PJSR Image accessors in use."""

import numpy as np

# Rec.709 luma — fine wherever absolute thresholds don't matter.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# PI default RGBWS: CIE Y weights applied to sRGB-LINEARIZED samples.
# Calibrated against a real getLuminance dump 2026-07-16 (max err 2.4e-7);
# the assemble stage MUST use this — all its absolute thresholds
# (0.04/0.08/0.10 masks, adaptive rim cuts) were tuned on PI's scale.
_LUMA_PI = np.array([0.222491, 0.716888, 0.060621], dtype=np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) -> luminance (H,W); mono passes through."""
    a = np.asarray(img)
    if a.ndim == 2:
        return a
    return a[..., :3].astype(np.float32) @ _LUMA


def luminance_pi(img: np.ndarray) -> np.ndarray:
    """PI Image.getLuminance equivalent (default RGBWS, sRGB gamma)."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        return a
    c = a[..., :3]
    lin = np.where(c <= 0.04045, c / 12.92,
                   ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return lin @ _LUMA_PI


def mad(a: np.ndarray) -> float:
    """Median absolute deviation (unscaled, as PJSR Image.MAD)."""
    m = np.median(a)
    return float(np.median(np.abs(a - m)))


def maximum_position(a: np.ndarray) -> tuple[int, int]:
    """(x, y) of the maximum sample — PJSR maximumPosition convention."""
    y, x = np.unravel_index(int(np.argmax(a)), a.shape)
    return int(x), int(y)
