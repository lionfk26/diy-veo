"""Central configuration for the DIY Veo camera pipeline.

Edit these values for your specific cameras/hardware before running
anything. Check your actual device nodes with:

    v4l2-ctl --list-devices

Most UVC webcams expose two /dev/videoN nodes each (one for actual video,
one for metadata), so your second camera is often /dev/video2, not
/dev/video1 — don't assume, check.
"""
import os

# --- Camera capture settings ---
CAM_LEFT_DEVICE = "/dev/video0"
CAM_RIGHT_DEVICE = "/dev/video2"
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30

# --- File locations ---
RECORDINGS_DIR = "recordings"
OUTPUT_DIR = "output"
STITCH_PARAMS_FILE = "stitch_homography.npy"

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Stitching ---
# Size of the combined panorama after warping the right camera onto the left.
PANORAMA_WIDTH = 2400
PANORAMA_HEIGHT = 720

# --- Ball detection / tracking ---
YOLO_MODEL_PATH = "yolov8n.pt"      # ultralytics auto-downloads this on first run
YOLO_CONF_THRESHOLD = 0.25
COCO_SPORTS_BALL_CLASS = 32         # COCO's pretrained "sports ball" class id
DETECT_EVERY_N_FRAMES = 4           # full-frame YOLO runs this often; ROI tracking fills the gaps
ROI_SEARCH_SIZE = 360               # px, size of the local crop searched around the last known ball position
MAX_MISSED_DETECTIONS = 45          # frames with no detection before forcing a full-frame re-search every frame

# --- Virtual camera (crop that pans/zooms to follow the ball) ---
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
BASE_CROP_WIDTH = 1000              # crop width in panorama-space at default zoom
MIN_CROP_WIDTH = 700
MAX_CROP_WIDTH = 1400
SMOOTHING_WINDOW = 30                # frames (~1s @30fps) of temporal smoothing for the virtual camera pan
