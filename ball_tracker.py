"""ball_tracker.py — hybrid detect + track ball tracking.

A full-frame YOLO pass is expensive, so we only run it every
DETECT_EVERY_N_FRAMES frames. On the frames in between we search a small
region-of-interest (ROI) around the last known ball position instead of the
whole panorama — dramatically cheaper, since the ball can't have moved far
in one frame. A Kalman filter smooths the result and predicts through
frames where the ball isn't found at all (occlusion, motion blur, briefly
off-screen).

Note: the pretrained COCO "sports ball" class this uses works well when the
ball is reasonably large/close in frame. If your cameras cover a full pitch
from far away, the ball may only be a handful of pixels across and the
pretrained model will miss it more often — fine-tuning YOLOv8n on a few
hundred labeled frames from your own footage is the real fix for that, the
pretrained model is a starting point, not the final word.
"""
import cv2
import numpy as np
from ultralytics import YOLO

import config


class BallTracker:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.last_pos = None  # (x, y) in full-panorama coords
        self.missed_frames = 0
        self.frame_idx = 0

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.5
        self._kf_initialized = False

    def _yolo_detect(self, image, offset=(0, 0)):
        """Run YOLO on `image`; return the highest-confidence sports-ball
        center in full-panorama coordinates, or None."""
        results = self.model.predict(
            image,
            conf=config.YOLO_CONF_THRESHOLD,
            classes=[config.COCO_SPORTS_BALL_CLASS],
            verbose=False,
        )
        best, best_conf = None, 0.0
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2 + offset[0]
                    cy = (y1 + y2) / 2 + offset[1]
                    best, best_conf = (cx, cy), conf
        return best

    def update(self, frame):
        """Feed one panorama frame in. Returns the smoothed (x, y) ball
        position to steer the virtual camera with, or None if the ball has
        never been seen yet."""
        h, w = frame.shape[:2]
        detection = None

        use_full_frame = (
            self.last_pos is None
            or self.frame_idx % config.DETECT_EVERY_N_FRAMES == 0
            or self.missed_frames > config.MAX_MISSED_DETECTIONS
        )

        if use_full_frame:
            detection = self._yolo_detect(frame)
        else:
            r = config.ROI_SEARCH_SIZE // 2
            lx, ly = self.last_pos
            x0, y0 = max(0, int(lx - r)), max(0, int(ly - r))
            x1, y1 = min(w, int(lx + r)), min(h, int(ly + r))
            roi = frame[y0:y1, x0:x1]
            if roi.size > 0:
                detection = self._yolo_detect(roi, offset=(x0, y0))
            if detection is None:
                # Not in the ROI either — fall back to a full-frame search
                # this frame so we have a chance to re-acquire the ball.
                detection = self._yolo_detect(frame)

        self.frame_idx += 1

        if detection is not None:
            self.missed_frames = 0
            self.last_pos = detection
            meas = np.array([[np.float32(detection[0])], [np.float32(detection[1])]])
            if not self._kf_initialized:
                self.kf.statePre = np.array(
                    [[detection[0]], [detection[1]], [0], [0]], np.float32
                )
                self.kf.statePost = self.kf.statePre.copy()
                self._kf_initialized = True
            self.kf.predict()
            self.kf.correct(meas)
        else:
            self.missed_frames += 1
            if self._kf_initialized:
                self.kf.predict()
            else:
                return None

        smoothed = self.kf.statePost
        return float(smoothed[0]), float(smoothed[1])
