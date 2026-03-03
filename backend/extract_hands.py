"""
Targeted MediaPipe Holistic extraction.
Reads segments_manifest.json, seeks to each segment's time range in the video,
extracts 134-D keypoints (body + BOTH hands), and overwrites data/motion/seg_*.npz.

Run from the backend/ directory:
  python extract_hands.py
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
MANIFEST = BASE / "data" / "segments_manifest.json"
MOTION   = BASE / "data" / "motion"
VIDEO    = BASE.parent / "dataset" / "1413451-11105600-11163240_1a1.mp4"

# ── mediapipe setup ───────────────────────────────────────────────────────────
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# ── keypoint helpers ──────────────────────────────────────────────────────────
def _xy(lm, i, thresh=0.1):
    l = lm[i]
    if getattr(l, "visibility", 1.0) < thresh and getattr(l, "presence", 1.0) < thresh:
        return 0.0, 0.0
    return float(min(max(l.x, 0), 1)), float(min(max(l.y, 0), 1))

BODY25_MAP = [
    (0,  0),   # Nose
    (2,  12),  # RShoulder
    (3,  14),  # RElbow
    (4,  16),  # RWrist
    (5,  11),  # LShoulder
    (6,  13),  # LElbow
    (7,  15),  # LWrist
    (9,  24),  # RHip
    (10, 26),  # RKnee
    (11, 28),  # RAnkle
    (12, 23),  # LHip
    (13, 25),  # LKnee
    (14, 27),  # LAnkle
    (15, 5),   # REye
    (16, 2),   # LEye
    (17, 8),   # REar
    (18, 7),   # LEar
    (19, 31),  # LBigToe
    (20, 31),  # LSmallToe
    (21, 29),  # LHeel
    (22, 32),  # RBigToe
    (23, 32),  # RSmallToe
    (24, 30),  # RHeel
]

def extract_frame(results) -> np.ndarray:
    vec = np.zeros(134, dtype=np.float32)

    # Body (25 joints, indices 0..49)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        for op_i, mp_i in BODY25_MAP:
            x, y = _xy(lm, mp_i, thresh=0.1)
            vec[op_i*2]   = x
            vec[op_i*2+1] = y
        # Neck = mid shoulders
        if len(lm) > 12:
            lx, ly = _xy(lm, 11, 0.1)
            rx, ry = _xy(lm, 12, 0.1)
            vec[1*2]   = (lx+rx)*0.5
            vec[1*2+1] = (ly+ry)*0.5
        # MidHip = mid hips
        if len(lm) > 24:
            lx, ly = _xy(lm, 23, 0.1)
            rx, ry = _xy(lm, 24, 0.1)
            vec[8*2]   = (lx+rx)*0.5
            vec[8*2+1] = (ly+ry)*0.5

    # Left hand (21 joints, indices 50..91)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:21]):
            vec[50 + i*2]   = float(min(max(lm.x, 0), 1))
            vec[50 + i*2+1] = float(min(max(lm.y, 0), 1))

    # Right hand (21 joints, indices 92..133)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:21]):
            vec[92 + i*2]   = float(min(max(lm.x, 0), 1))
            vec[92 + i*2+1] = float(min(max(lm.y, 0), 1))

    return vec

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not VIDEO.exists():
        sys.exit(f"Video not found: {VIDEO}")
    with open(MANIFEST) as f:
        manifest = json.load(f)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    print(f"Video FPS: {fps}  |  Segments: {manifest['num_segments']}")

    MOTION.mkdir(parents=True, exist_ok=True)
    segments = manifest["segments"]

    for seg in segments:
        seg_id    = seg["id"]
        start_ms  = seg["start_ms"]
        end_ms    = seg["end_ms"]

        start_frame = int(start_ms / 1000.0 * fps)
        end_frame   = int(end_ms   / 1000.0 * fps)

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_kp = []
        for fi in range(start_frame, end_frame + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            frames_kp.append(extract_frame(res))

        kp = np.stack(frames_kp).astype(np.float32) if frames_kp else np.zeros((1, 134), dtype=np.float32)

        lhand_nz = np.count_nonzero(kp[:, 50:92])
        rhand_nz = np.count_nonzero(kp[:, 92:134])

        out_path = MOTION / f"{seg_id}.npz"
        np.savez_compressed(out_path,
            keypoints=kp,
            fps=np.array([fps], dtype=np.float32),
            start_ms=np.array([start_ms], dtype=np.int32),
            end_ms=np.array([end_ms],   dtype=np.int32),
        )
        print(f"  {seg_id}: frames={kp.shape[0]:3d}  lhand={lhand_nz:4d}  rhand={rhand_nz:4d}")

    cap.release()
    holistic.close()
    print("\nDone. All segments re-extracted with hand keypoints.")

if __name__ == "__main__":
    main()
