#!/bin/bash
# One-shot setup for a fresh Raspberry Pi 5 running Pi OS Lite (64-bit only —
# PyTorch, which ultralytics needs, does not support 32-bit).
#
# Run this once, from inside the cloned repo:
#   git clone <your-repo-url> diy-veo
#   cd diy-veo
#   ./setup.sh
set -e

echo "== Installing system packages =="
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg v4l-utils libgl1 libglib2.0-0 git

echo ""
echo "== Creating Python virtual environment =="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "== Pre-downloading the YOLO ball-detection model (needs internet once) =="
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

mkdir -p recordings output

echo ""
echo "== Connected cameras =="
v4l2-ctl --list-devices || echo "(v4l2-ctl found nothing — check your USB webcams are plugged in)"

cat <<'NEXT_STEPS'

Setup complete.

Next steps:
  1. Edit config.py — set CAM_LEFT_DEVICE / CAM_RIGHT_DEVICE to match the
     device nodes listed above.
  2. In future terminal sessions, activate the environment with:
       source venv/bin/activate
  3. Mount your two cameras (side by side, overlapping views), then run:
       python3 calibrate_stitch.py
  4. On match day:
       python3 record.py <name>        (Ctrl+C to stop)
  5. Afterwards:
       python3 process_game.py <name>
NEXT_STEPS
