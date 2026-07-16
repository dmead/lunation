"""Synthetic fixtures shared by the M0 tests."""

import numpy as np
import scipy.fft as sfft


def lunar_texture(size: int = 256, seed: int = 7) -> np.ndarray:
    """Band-passed noise inside a disk mask — lunar-ish structure with a
    limb, deterministic."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size))
    f = sfft.fft2(noise)
    ky = sfft.fftfreq(size)[:, None]
    kx = sfft.fftfreq(size)[None, :]
    k = np.hypot(kx, ky)
    band = np.exp(-(((k - 0.08) / 0.06) ** 2))
    tex = np.real(sfft.ifft2(f * band))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2, yy - size / 2)
    disk = np.clip((size * 0.42 - r) / 2.0, 0.0, 1.0)
    return (0.15 + 0.7 * tex) * disk


def fourier_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Exact sub-pixel shift via Fourier phase ramp: content moves by
    (+dx right, +dy down), periodic wraparound."""
    f = sfft.fft2(img)
    ky = sfft.fftfreq(img.shape[0])[:, None]
    kx = sfft.fftfreq(img.shape[1])[None, :]
    ramp = np.exp(-2j * np.pi * (kx * dx + ky * dy))
    return np.real(sfft.ifft2(f * ramp)).astype(np.float32)
