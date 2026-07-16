"""Local seeing-warp correction (alignment points, AS!3-style, sub-pixel).

Port of pjsr/ser-stack.js:229-324. Measured on TS-70 data: median 0.41px /
p90 0.78px residual after global alignment — correction tightens the PSF
rather than rescuing geometry. Overlapping tiles are phase-correlated
against the reference, translated by their local residual, and reassembled
with raised-cosine weights; a low-weight globally-shifted base layer
guarantees full coverage where tiles are gated off (sky, low signal).
"""

import numpy as np

from ..core.fftreg import PhaseCorrelator
from ..core.warp import translate

BASE_WEIGHT = 0.05
TILE_GATE = 0.12  # tile mean below this fraction of ref mean = sky/shadow


class LocalWarp:
    def __init__(self, ref: np.ndarray, tile_size: int = 256,
                 tile_step: int = 128, clamp_px: float = 3.0,
                 upsample: int = 20):
        self.tile = tile_size
        self.step = tile_step
        self.clamp = clamp_px
        self.H, self.W = ref.shape
        self.tiles = []
        ref_mean = float(ref.mean())
        for y in range(0, self.H - tile_size + 1, tile_step):
            for x in range(0, self.W - tile_size + 1, tile_step):
                patch = ref[y : y + tile_size, x : x + tile_size]
                if float(patch.mean()) < TILE_GATE * ref_mean:
                    continue  # sky/deep shadow tile
                pc = PhaseCorrelator(use_gradient=False, upsample=upsample)
                pc.initialize(np.ascontiguousarray(patch))
                self.tiles.append((x, y, pc))

        # raised-cosine blend weight
        i = (np.arange(tile_size, dtype=np.float32) + 0.5) / tile_size
        w1 = 0.5 - 0.5 * np.cos(2 * np.pi * i)
        self.weight = np.maximum(1e-4, np.outer(w1, w1)).astype(np.float32)

    def apply(self, img: np.ndarray, gdx: float, gdy: float) -> np.ndarray:
        """Return img registered to reference geometry (global + local)."""
        # base layer: globally shifted frame at low weight (coverage guarantee)
        base = translate(img, -gdx, -gdy)
        out = base * BASE_WEIGHT
        wsum = np.full((self.H, self.W), BASE_WEIGHT, dtype=np.float32)

        gix, giy = round(gdx), round(gdy)
        gfx, gfy = gdx - gix, gdy - giy
        t = self.tile
        for x, y, pc in self.tiles:
            # extract source patch at the globally-corrected position
            sx, sy = x + gix, y + giy
            if sx < 0 or sy < 0 or sx + t > self.W or sy + t > self.H:
                continue
            patch = np.ascontiguousarray(img[sy : sy + t, sx : sx + t])
            rx, ry = pc.evaluate(patch)
            # implausible local lock -> fall back to the global residual
            if np.hypot(rx - gfx, ry - gfy) > self.clamp:
                rx, ry = gfx, gfy
            shifted = translate(patch, -rx, -ry)
            out[y : y + t, x : x + t] += shifted * self.weight
            wsum[y : y + t, x : x + t] += self.weight
        return out / wsum
