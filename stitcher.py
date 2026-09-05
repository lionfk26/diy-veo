"""stitcher.py — static homography-based panorama stitching.

Because both cameras are fixed in place (mounted like a Veo rig), we only
need to compute the geometric alignment between them once, offline, with
calibrate_stitch.py. Every frame after that is just a cheap warp + composite
— no per-frame feature matching, which would be far too slow on a Pi.
"""
import cv2
import numpy as np

import config


class Stitcher:
    def __init__(self, homography_path=None):
        self.H = np.load(homography_path or config.STITCH_PARAMS_FILE)
        self.pano_w = config.PANORAMA_WIDTH
        self.pano_h = config.PANORAMA_HEIGHT
        self._blend_mask = None  # built lazily from the first frame pair

    def _build_blend_mask(self, left_shape):
        """A static per-pixel alpha mask that feathers the seam between the
        left frame and the warped right frame, instead of a hard cut."""
        lh, lw = left_shape[:2]
        mask = np.zeros((self.pano_h, self.pano_w), dtype=np.float32)
        mask[0:lh, 0:lw] = 1.0
        feather = min(60, lw)
        for i in range(feather):
            col = lw - feather + i
            if 0 <= col < self.pano_w:
                mask[0:lh, col] = 1.0 - (i / feather)
        self._blend_mask = mask

    def stitch(self, left, right):
        warped_right = cv2.warpPerspective(right, self.H, (self.pano_w, self.pano_h))
        panorama = warped_right.astype(np.float32)

        lh, lw = left.shape[:2]
        if self._blend_mask is None:
            self._build_blend_mask(left.shape)

        alpha = self._blend_mask[0:lh, 0:lw][..., None]
        panorama[0:lh, 0:lw] = (
            left.astype(np.float32) * alpha + panorama[0:lh, 0:lw] * (1 - alpha)
        )
        return np.clip(panorama, 0, 255).astype(np.uint8)
