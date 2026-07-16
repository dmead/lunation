"""Basic image statistics matching the PJSR Image accessors in use."""

import numpy as np

# Rec.709 luma — PI's RGBWS quirks are deliberately not replicated
# (better-than parity decision, 2026-07-16).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) -> luminance (H,W); mono passes through."""
    a = np.asarray(img)
    if a.ndim == 2:
        return a
    return a[..., :3].astype(np.float32) @ _LUMA


def mad(a: np.ndarray) -> float:
    """Median absolute deviation (unscaled, as PJSR Image.MAD)."""
    m = np.median(a)
    return float(np.median(np.abs(a - m)))


def maximum_position(a: np.ndarray) -> tuple[int, int]:
    """(x, y) of the maximum sample — PJSR maximumPosition convention."""
    y, x = np.unravel_index(int(np.argmax(a)), a.shape)
    return int(x), int(y)
