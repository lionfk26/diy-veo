"""virtual_camera.py — turns a stream of ball positions into a smooth,
broadcast-style crop window over the panorama (pan + basic zoom), the way a
human camera operator (or Veo's software) follows play instead of snapping
the frame to wherever the ball is this instant.
"""
from collections import deque

import numpy as np

import config


class VirtualCamera:
    def __init__(self, panorama_w, panorama_h):
        self.pano_w = panorama_w
        self.pano_h = panorama_h
        self.history = deque(maxlen=config.SMOOTHING_WINDOW)
        self.center = (panorama_w / 2, panorama_h / 2)
        self.crop_w = config.BASE_CROP_WIDTH

    def update(self, ball_pos):
        if ball_pos is not None:
            self.history.append(ball_pos)

        aspect = config.OUTPUT_WIDTH / config.OUTPUT_HEIGHT
        max_by_height = self.pano_h * aspect  # can't crop taller than the panorama
        upper = min(config.MAX_CROP_WIDTH, max_by_height, self.pano_w)
        lower = min(config.MIN_CROP_WIDTH, upper)

        if not self.history:
            target_center = (self.pano_w / 2, self.pano_h / 2)
            target_w = self.crop_w
        else:
            xs = [p[0] for p in self.history]
            ys = [p[1] for p in self.history]
            target_center = (float(np.mean(xs)), float(np.mean(ys)))

            # Zoom out a bit when the ball's been moving around a lot lately
            # (open play), zoom in when it's been contained (e.g. near a box).
            spread = float(np.std(xs)) if len(xs) > 1 else 0.0
            target_w = config.BASE_CROP_WIDTH + spread * 2
            target_w = max(lower, min(upper, target_w))

        # Ease the camera towards the target rather than snapping to it —
        # this is what makes the pan look like a camera operator, not a crop.
        ease = 0.08
        cx = self.center[0] + (target_center[0] - self.center[0]) * ease
        cy = self.center[1] + (target_center[1] - self.center[1]) * ease
        self.center = (cx, cy)
        self.crop_w += (target_w - self.crop_w) * ease
        self.crop_w = max(lower, min(upper, self.crop_w))

        crop_h = self.crop_w / aspect

        x0 = self.center[0] - self.crop_w / 2
        y0 = self.center[1] - crop_h / 2
        x0 = max(0, min(self.pano_w - self.crop_w, x0))
        y0 = max(0, min(self.pano_h - crop_h, y0))

        return int(x0), int(y0), int(self.crop_w), int(crop_h)
