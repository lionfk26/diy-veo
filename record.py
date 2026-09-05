#!/usr/bin/env python3
"""
record.py — records synced footage from two USB webcams on a Raspberry Pi 5.

Uses ffmpeg in "-c copy" mode, which just packages the MJPEG frames the
webcams already compress in their own hardware — the Pi's CPU barely does
any encoding work at all. This matters because the Pi 5 dropped the
hardware video encoder earlier Pis had, so software H.264 encoding of two
live streams would be a heavy ask; MJPEG copy sidesteps that entirely
(at the cost of larger files than H.264 would give you).

Usage:
    python3 record.py my_match        # writes recordings/my_match_left.mkv / _right.mkv
    (Ctrl+C to stop — this lets ffmpeg finalize the files cleanly)
"""
import subprocess
import sys
import time
import json
import signal
import os

import config


def build_ffmpeg_cmd(device, out_path):
    return [
        "ffmpeg",
        "-y",
        "-f", "v4l2",
        "-input_format", "mjpeg",
        "-video_size", f"{config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT}",
        "-framerate", str(config.CAPTURE_FPS),
        "-i", device,
        "-c", "copy",
        out_path,
    ]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("match_%Y%m%d_%H%M%S")
    left_path = os.path.join(config.RECORDINGS_DIR, f"{name}_left.mkv")
    right_path = os.path.join(config.RECORDINGS_DIR, f"{name}_right.mkv")

    print(f"Recording to:\n  {left_path}\n  {right_path}\nPress Ctrl+C to stop.")

    procs = []
    start_times = {}

    # Launch both as close together as possible. Perfect hardware sync isn't
    # possible with two independent USB webcams, but this gets close enough
    # that process_game.py's frame-offset correction can clean up the rest.
    for device, path, label in (
        (config.CAM_LEFT_DEVICE, left_path, "left"),
        (config.CAM_RIGHT_DEVICE, right_path, "right"),
    ):
        cmd = build_ffmpeg_cmd(device, path)
        start_times[label] = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)

    meta = {
        "name": name,
        "start_times": start_times,
        "capture_width": config.CAPTURE_WIDTH,
        "capture_height": config.CAPTURE_HEIGHT,
        "capture_fps": config.CAPTURE_FPS,
    }
    meta_path = os.path.join(config.RECORDINGS_DIR, f"{name}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    def stop(*_):
        print("\nStopping recording...")
        for p in procs:
            p.send_signal(signal.SIGINT)
        for p in procs:
            p.wait()
        print("Done. Metadata saved to", meta_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while True:
        time.sleep(1)
        for p in procs:
            if p.poll() is not None:
                print("A camera process exited unexpectedly — check the camera connections.")
                stop()


if __name__ == "__main__":
    main()
