from __future__ import annotations

"""
MediaPipe -> OpenPose-like keypoint features (134-D).

Reused from the original signlanguage-german project so that:
  - Offline extraction and live webcam both use the same mapping.
  - Features are compatible with existing Phase 1 models.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MPConfig:
    min_vis: float = 0.5


def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _extract_pose25_xy(results, *, cfg: MPConfig) -> np.ndarray:
    pose_xy = np.zeros((25, 2), dtype=np.float32)
    if getattr(results, "pose_landmarks", None) is None:
        return pose_xy

    lm = results.pose_landmarks.landmark

    def ok(i: int) -> bool:
        vis = float(getattr(lm[i], "visibility", 1.0))
        pres = float(getattr(lm[i], "presence", 1.0))
        return (vis >= cfg.min_vis) and (pres >= cfg.min_vis)

    def xy(i: int) -> tuple[float, float]:
        return (_clamp01(float(lm[i].x)), _clamp01(float(lm[i].y)))

    direct_map = [
        (0, 0),    # Nose
        (2, 12),   # RShoulder
        (3, 14),   # RElbow
        (4, 16),   # RWrist
        (5, 11),   # LShoulder
        (6, 13),   # LElbow
        (7, 15),   # LWrist
        (9, 24),   # RHip
        (10, 26),  # RKnee
        (11, 28),  # RAnkle
        (12, 23),  # LHip
        (13, 25),  # LKnee
        (14, 27),  # LAnkle
        (15, 5),   # REye
        (16, 2),   # LEye
        (17, 8),   # REar
        (18, 7),   # LEar
        (21, 29),  # LHeel
        (24, 30),  # RHeel
        (19, 31),  # LBigToe (approx)
        (20, 31),  # LSmallToe (approx)
        (22, 32),  # RBigToe (approx)
        (23, 32),  # RSmallToe (approx)
    ]

    for op_i, mp_i in direct_map:
        if ok(mp_i):
            x, y = xy(mp_i)
            pose_xy[op_i, 0] = x
            pose_xy[op_i, 1] = y

    # Neck = midpoint of shoulders
    if ok(11) and ok(12):
        lx, ly = xy(11)
        rx, ry = xy(12)
        pose_xy[1, 0] = (lx + rx) * 0.5
        pose_xy[1, 1] = (ly + ry) * 0.5

    # MidHip = midpoint of hips
    if ok(23) and ok(24):
        lx, ly = xy(23)
        rx, ry = xy(24)
        pose_xy[8, 0] = (lx + rx) * 0.5
        pose_xy[8, 1] = (ly + ry) * 0.5

    return pose_xy


def _extract_hand21_xy(hand_landmarks, *, cfg: MPConfig) -> np.ndarray:
    arr = np.zeros((21, 2), dtype=np.float32)
    if hand_landmarks is None:
        return arr
    for i, h in enumerate(hand_landmarks.landmark[:21]):
        pres = float(getattr(h, "presence", 1.0))
        vis = float(getattr(h, "visibility", 1.0))
        if pres < cfg.min_vis and vis < cfg.min_vis:
            continue
        arr[i, 0] = _clamp01(float(h.x))
        arr[i, 1] = _clamp01(float(h.y))
    return arr


def extract_134_from_holistic_results(results, *, cfg: MPConfig) -> np.ndarray:
    pose_xy = _extract_pose25_xy(results, cfg=cfg)
    left_xy = _extract_hand21_xy(getattr(results, "left_hand_landmarks", None), cfg=cfg)
    right_xy = _extract_hand21_xy(getattr(results, "right_hand_landmarks", None), cfg=cfg)

    return np.concatenate(
        [pose_xy.reshape(-1), left_xy.reshape(-1), right_xy.reshape(-1)],
        axis=0,
    ).astype(np.float32)


def extract_134_from_bgr_frame(holistic, frame_bgr: np.ndarray, *, cfg: MPConfig) -> np.ndarray | None:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    return extract_134_from_holistic_results(results, cfg=cfg)

