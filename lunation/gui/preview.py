"""Preview rendering — ports pjsr/master/Preview.jsh (lpPreviewBitmap).

Decodes the selected entry to a display-ready QImage: SER entries show
frame 0, images load through the pipeline readers (XISF/TIFF/PNG/JPG).
Same recipe as the original: downscale FIRST (cheap), then auto-stretch
to median 0.25 with `m = clamp(mtf_for(med, 0.25), 0.02, 0.92)`
(Preview.jsh:16-27). Preview-only — never bakes working data.
"""

import numpy as np

from ..finish.primitives import mtf, mtf_for

TARGET_MEDIAN = 0.25


def autostretch(img: np.ndarray) -> np.ndarray:
    med = float(np.median(img))
    if med <= 0.0 or med >= 1.0:
        return img
    m = min(max(mtf_for(med, TARGET_MEDIAN), 0.02), 0.92)
    return mtf(m, img)


def preview_array(path: str, is_ser: bool, max_px: int = 700) -> np.ndarray:
    """float32 [0,1], (H,W) or (H,W,3), longest side <= max_px."""
    if is_ser:
        from ..io.ser import SerReader

        r = SerReader(path)  # "mono" = 2x2 superpixel / channel average —
        try:                 # better than the original's first-plane grab
            img = r.read(0)
        finally:
            r.close()
    elif path.lower().endswith(".xisf"):
        from ..io.xisf_io import read_xisf

        img = read_xisf(path)
    else:
        from ..io.images import read_image

        img = read_image(path)
    h, w = img.shape[:2]
    if max(h, w) > max_px:
        import cv2

        s = max_px / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return autostretch(np.clip(img, 0.0, 1.0))


def array_to_qimage(img: np.ndarray):
    """float32 [0,1] (H,W) or (H,W,3) -> detached QImage."""
    from PySide6.QtGui import QImage

    u8 = np.clip(np.rint(np.asarray(img) * 255), 0, 255).astype(np.uint8)
    h, w = u8.shape[:2]
    if u8.ndim == 2:
        q = QImage(u8.data, w, h, w, QImage.Format_Grayscale8)
    else:
        u8 = np.ascontiguousarray(u8)
        q = QImage(u8.data, w, h, 3 * w, QImage.Format_RGB888)
    return q.copy()  # detach from the numpy buffer


def preview_qimage(path: str, is_ser: bool, max_px: int = 700):
    """QImage for the preview pane; None if the decode fails (the caller
    shows a placeholder — Preview.jsh:101-106)."""
    try:
        return array_to_qimage(preview_array(path, is_ser, max_px))
    except Exception:  # noqa: BLE001 — preview is best-effort by contract
        return None
