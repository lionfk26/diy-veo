# DIY Veo Camera

Two USB webcams + a Raspberry Pi 5 (Pi OS Lite, 64-bit) recording a match,
stitched into one wide panorama, with a YOLO-tracked ball driving a virtual
camera that pans and zooms to follow play — an auto-directed output video,
Veo-style.

## How it fits together

1. **Mount the two cameras side by side**, angled slightly toward each
   other so their fields of view overlap in the middle — the same setup a
   real Veo rig uses to cover a full pitch width.
2. **`calibrate_stitch.py`** — run once after mounting. It computes the
   geometric alignment between the two views and saves it. Re-run any time
   the cameras are bumped or repositioned.
3. **`record.py`** — run during the match. Captures both cameras straight
   to disk with almost no CPU load (see "Why MJPEG" below).
4. **`process_game.py`** — run after the match, whenever you like. Stitches
   the two recordings into a panorama frame by frame, tracks the ball, and
   renders the final auto-directed video.

## First-time setup

```bash
./setup.sh
source venv/bin/activate
v4l2-ctl --list-devices          # find your actual camera device nodes
```

Edit `config.py` — at minimum check `CAM_LEFT_DEVICE` / `CAM_RIGHT_DEVICE`
match what `v4l2-ctl` showed you, and confirm your webcams actually support
the resolution/fps/MJPEG format you've set (also visible via
`v4l2-ctl --list-formats-ext -d /dev/videoN`).

## Match day

```bash
python3 calibrate_stitch.py       # only needed again if cameras moved
python3 record.py saturday_game   # Ctrl+C when the match ends
python3 process_game.py saturday_game
# -> output/saturday_game_directed.mp4
```

## Why MJPEG instead of H.264 for recording

The Pi 5 removed the hardware video encoder that earlier Pi models had, so
software H.264 encoding of two simultaneous camera feeds would load the CPU
heavily and risk dropped frames. Instead, `record.py` uses
`ffmpeg -c copy`, which just packages the MJPEG frames the webcams already
compress in their own onboard hardware — negligible CPU cost. The trade-off
is bigger files (expect roughly several GB per camera per hour at 720p);
you can transcode to H.264 afterwards on a PC if storage is tight.

## Performance expectations on a bare Pi 5

Running YOLO on every frame of a ~2400×720 panorama is too slow for a Pi 5
without an AI accelerator (no Hailo/Coral, etc.) to keep up in real time —
which is exactly why the pipeline is designed to run as a batch job after
the match rather than live. Even so, `process_game.py` may well take longer
than the match itself did. A few things help:

- `config.DETECT_EVERY_N_FRAMES` controls how often the full-frame YOLO
  pass runs; the frames in between use a much cheaper local-region search
  (see `ball_tracker.py`). Raising this speeds things up at the cost of
  slightly coarser tracking between detections.
- Exporting the model to NCNN format runs noticeably faster on ARM CPUs
  than plain PyTorch:
  ```bash
  python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='ncnn')"
  ```
  then point `config.YOLO_MODEL_PATH` at the exported folder instead of
  `yolov8n.pt`.
- If you have a more powerful machine around, `process_game.py` is a plain
  script with no Pi-specific code in it — copy the `recordings/` folder
  over and run it there unmodified for a much faster turnaround.

## Ball detection accuracy — set expectations here

`ball_tracker.py` uses YOLOv8n's pretrained COCO "sports ball" class, which
is a starting point, not a finished solution. It works reasonably well when
the ball is fairly large/close in frame; if your cameras cover a full pitch
from a distance, the ball may only be a handful of pixels across and the
pretrained model will miss it more often than you'd like. If that happens,
the real fix is fine-tuning YOLOv8n on a few hundred frames of your own
footage with the ball manually labeled (tools like
[Roboflow](https://roboflow.com) or [CVAT](https://cvat.ai) make this
fairly quick) — general-purpose sports-ball detection at long range is
genuinely one of the harder parts of building something like this.

## Tuning the virtual camera

`config.py` controls the pan/zoom feel:
- `SMOOTHING_WINDOW` — larger = slower, calmer camera movements.
- `BASE_CROP_WIDTH` / `MIN_CROP_WIDTH` / `MAX_CROP_WIDTH` — how far in/out
  the virtual camera can zoom.
- `PANORAMA_WIDTH` / `PANORAMA_HEIGHT` — must match what your calibration
  actually produces; if you increase camera resolution, increase these too
  for more room to pan.
