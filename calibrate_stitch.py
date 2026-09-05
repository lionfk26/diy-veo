#!/usr/bin/env python3
"""
calibrate_stitch.py — one-time calibration: computes the homography that
aligns the right camera's view onto the left camera's, so every frame later
can be stitched into one wide panorama with a cheap warp instead of
recomputing feature matches per frame.

Point both cameras at the pitch the way you plan to mount them for real
(side by side, like a Veo rig), with enough overlap between their two
views that there's visible shared texture (pitch lines, goal, advertising
boards) in the overlap region. Then run:

    python3 calibrate_stitch.py

It saves stitch_homography.npy and a preview.jpg — check the preview by
eye before you trust it for a real match; re-run if the cameras move at all.
"""
import cv2
import numpy as np

import config


def grab_frame(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    # Let auto-exposure/white-balance settle before we grab the real frame.
    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {device}")
    return frame


def compute_homography(img_left, img_right):
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img_left, None)
    kp2, des2 = orb.detectAndCompute(img_right, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)  # right -> left

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 15:
        raise RuntimeError(
            f"Only found {len(good)} good matches between the two cameras. "
            "Point them so their views overlap more, with clear texture "
            "(pitch lines / goal / advertising boards) visible in the "
            "overlap area, then try again."
        )

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    inliers = int(mask.sum())
    print(f"Homography computed from {inliers}/{len(good)} inlier matches.")
    return H


def main():
    print("Grabbing calibration frames...")
    left = grab_frame(config.CAM_LEFT_DEVICE)
    right = grab_frame(config.CAM_RIGHT_DEVICE)

    H = compute_homography(left, right)
    np.save(config.STITCH_PARAMS_FILE, H)
    print(f"Saved homography to {config.STITCH_PARAMS_FILE}")

    # Quick preview panorama so you can eyeball the alignment before relying
    # on it for a real match.
    warped_right = cv2.warpPerspective(
        right, H, (config.PANORAMA_WIDTH, config.PANORAMA_HEIGHT)
    )
    panorama = warped_right.copy()
    lh, lw = left.shape[:2]
    panorama[0:lh, 0:lw] = left
    cv2.imwrite("preview.jpg", panorama)
    print("Wrote preview.jpg — check the seam lines up before recording a real match.")


if __name__ == "__main__":
    main()
