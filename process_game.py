#!/usr/bin/env python3
"""
process_game.py — offline pipeline: takes the two raw recordings from
record.py, stitches them into a panorama, tracks the ball, and renders a
single auto-directed "virtual camera" video that pans/zooms to follow
play — DIY Veo-style.

Usage:
    python3 process_game.py my_match
      -> reads   recordings/my_match_left.mkv + _right.mkv
      -> writes  output/my_match_directed.mp4

This does NOT need to run in real time. On a Pi 5 with no AI accelerator it
will very likely run slower than the match itself (YOLO inference is the
bottleneck) — that's fine since it's a one-off batch job after the game, not
a live broadcast. Progress prints as it goes. If you have a faster machine
around, you can copy the "recordings" folder over and run this same script
there unmodified for a much quicker turnaround.
"""
import sys
import os
import json
import time
import subprocess

import cv2

import config
from stitcher import Stitcher
from ball_tracker import BallTracker
from virtual_camera import VirtualCamera


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 process_game.py <match_name>")
        sys.exit(1)

    name = sys.argv[1]
    left_path = os.path.join(config.RECORDINGS_DIR, f"{name}_left.mkv")
    right_path = os.path.join(config.RECORDINGS_DIR, f"{name}_right.mkv")
    out_path = os.path.join(config.OUTPUT_DIR, f"{name}_directed.mp4")

    for p in (left_path, right_path):
        if not os.path.exists(p):
            print(f"Missing recording: {p}")
            sys.exit(1)

    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)
    fps = cap_l.get(cv2.CAP_PROP_FPS) or config.CAPTURE_FPS
    total_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))

    # If record.py logged start timestamps for both cameras, use the drift
    # between them to skip leading frames on whichever camera started first
    # — cheap insurance against the two ffmpeg processes not starting in
    # perfect lockstep.
    meta_path = os.path.join(config.RECORDINGS_DIR, f"{name}.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        dt = meta["start_times"]["left"] - meta["start_times"]["right"]
        frame_offset = round(abs(dt) * fps)
        if frame_offset > 0:
            skip_cap = cap_r if dt > 0 else cap_l
            print(f"Skipping {frame_offset} frames to sync cameras (~{abs(dt):.2f}s startup drift)")
            for _ in range(frame_offset):
                skip_cap.read()

    stitcher = Stitcher()
    tracker = BallTracker()
    vcam = VirtualCamera(config.PANORAMA_WIDTH, config.PANORAMA_HEIGHT)

    # Pipe raw frames into ffmpeg for software H.264 encoding — no realtime
    # pressure here since this is a post-processing pass.
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{config.OUTPUT_WIDTH}x{config.OUTPUT_HEIGHT}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            out_path,
        ],
        stdin=subprocess.PIPE,
    )

    start = time.time()
    frame_idx = 0
    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            break

        panorama = stitcher.stitch(frame_l, frame_r)
        ball_pos = tracker.update(panorama)
        x0, y0, cw, ch = vcam.update(ball_pos)

        crop = panorama[y0:y0 + ch, x0:x0 + cw]
        out_frame = cv2.resize(crop, (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT))
        ffmpeg.stdin.write(out_frame.tobytes())

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - start
            pct = 100 * frame_idx / total_frames if total_frames else 0
            print(
                f"{frame_idx}/{total_frames} frames ({pct:.1f}%) — "
                f"{frame_idx / elapsed:.1f} fps processing",
                end="\r",
            )

    cap_l.release()
    cap_r.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
