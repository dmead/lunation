"""Derive processing scales from rig optics — 1:1 port of scripts/optics.mjs.

Everything that is a spatial SCALE follows from (aperture, focal length,
pixel size, wavelength): plate scale, sampling ratio, drizzle factor,
deconvolution PSF size, and which wavelet layers carry real detail.
AMOUNTS (bias strength, chroma) remain empirical taste knobs.

Units: aperture/focal_length mm, pixel_size um, wavelength nm, angles arcsec.
"""

import math

RAD_TO_ARCSEC = 206265.0


def plate_scale(focal_length: float, pixel_size: float) -> float:
    return (RAD_TO_ARCSEC * pixel_size) / (focal_length * 1000.0)


def diffraction_fwhm(aperture: float, lam: float) -> float:
    """FWHM of the diffraction PSF (~1.02 lambda/D for a circular aperture)."""
    return 1.02 * ((lam * 1e-9) / (aperture * 1e-3)) * RAD_TO_ARCSEC


def sampling_ratio(pixel_size: float, f_ratio: float, lam: float) -> float:
    """Nyquist-critical pixel at the optical cutoff is lambda*F#/2.
    Q > 1 means undersampled by that factor."""
    critical_pixel = (lam * 1e-3 * f_ratio) / 2.0  # um
    return pixel_size / critical_pixel


def derive_drizzle(q: float) -> int:
    return max(1, min(3, round(q)))


def derive_psf_sigma(aperture: float, lam: float, seeing: float,
                     plate_scale_arcsec: float, drizzle: int) -> dict:
    """Combined seeing+diffraction PSF sigma, in drizzled output pixels."""
    fwhm_arcsec = math.hypot(diffraction_fwhm(aperture, lam), seeing)
    fwhm_px = fwhm_arcsec / (plate_scale_arcsec / drizzle)
    return {
        "fwhmArcsec": fwhm_arcsec,
        "fwhmPx": fwhm_px,
        "sigma": round((fwhm_px / 2.355) * 100) / 100,
    }


def derive_mlt_biases(resolution_px: float, layer_count: int = 5,
                      strength: float = 0.075) -> list[float]:
    """MLT starlet biases: layer k acts at scale 2^(k-1) px.
    - layers whose scale is below half the resolution FWHM hold no real
      detail (bias 0 — amplifying them is pure noise gain)
    - peak bias goes one layer above the resolution scale (seeing tail)
    - ramp 0.53 before the peak, taper [0.8, 0.33, 0.13] after
    `strength` is the taste knob (validated house value: 0.075)."""
    peak = max(2, min(layer_count - 1, round(math.log2(resolution_px)) + 1))
    taper = [0.8, 0.33, 0.13]
    biases = []
    for k in range(1, layer_count + 1):
        scale = 2.0 ** (k - 1)
        if scale < resolution_px / 2.0:
            b = 0.0
        elif k < peak:
            b = 0.53 * strength
        elif k == peak:
            b = strength
        else:
            idx = k - peak - 1
            b = strength * (taper[idx] if idx < len(taper) else 0.0)
        biases.append(round(b * 1000) / 1000)
    return biases


def check_plate_scale(disk_px: float, angular_diameter_arcsec: float,
                      plate_scale_arcsec: float, drizzle: int = 1) -> dict:
    """Sanity check: does the measured disk size match the claimed optics?
    A silent barlow/reducer shows up as ratio far from 1."""
    expected_px = angular_diameter_arcsec / (plate_scale_arcsec / drizzle)
    ratio = disk_px / expected_px
    return {"expectedPx": round(expected_px), "ratio": ratio,
            "ok": abs(ratio - 1) <= 0.1}


def band_center(band: list[float]) -> float:
    return (band[0] + band[1]) / 2.0


def derive_all(eq: dict, seeing: float | None = None,
               strength: float = 0.075) -> dict:
    """One-stop derivation for a rig + filter set (equipment.json schema)."""
    if seeing is None:
        seeing = eq.get("seeingDefault", 2.0)
    f_ratio = eq["focalLength"] / eq["aperture"]
    scale = plate_scale(eq["focalLength"], eq["pixelSize"])
    filters = eq.get("filters") or {}
    lam_l = band_center(filters["L"]) if "L" in filters else 550.0
    q = sampling_ratio(eq["pixelSize"], f_ratio, lam_l)
    drizzle = derive_drizzle(q)
    psf = derive_psf_sigma(eq["aperture"], lam_l, seeing, scale, drizzle)
    biases = derive_mlt_biases(psf["fwhmPx"], 5, strength)
    per_filter = {}
    for name, band in filters.items():
        lam = band_center(band)
        per_filter[name] = {
            "lambda": lam,
            "diffractionFWHM": diffraction_fwhm(eq["aperture"], lam),
            **derive_psf_sigma(eq["aperture"], lam, seeing, scale, drizzle),
        }
    return {"fRatio": f_ratio, "plateScale": scale, "Q": q, "drizzle": drizzle,
            "seeing": seeing, "psf": psf, "biases": biases,
            "perFilter": per_filter}
