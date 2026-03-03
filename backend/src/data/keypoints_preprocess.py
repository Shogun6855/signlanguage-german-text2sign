from __future__ import annotations

"""
Utilities to preprocess OpenPose JSON files into numeric tensors.

Phase 1 uses these tensors as input to the gloss recognition model.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import json
import numpy as np


def _reshape_triplets(raw: np.ndarray) -> np.ndarray:
    """
    Ensure raw is shaped [N, 3] (x, y, conf). Truncates any trailing values.
    """
    if raw.size == 0:
        return raw.reshape(0, 3)
    n = (raw.size // 3) * 3
    raw = raw[:n]
    return raw.reshape(-1, 3)


def _extract_fixed_xy(
    person: Dict[str, Any],
    key: str,
    *,
    expected_points: int,
    width: float,
    height: float,
    conf_threshold: float,
    clamp: bool,
) -> np.ndarray:
    """
    Returns a fixed-length vector [expected_points*2] containing normalized x,y.
    Missing/low-confidence points are filled with zeros.
    """
    raw = np.asarray(person.get(key, []), dtype=np.float32)
    triplets = _reshape_triplets(raw)

    out = np.zeros((expected_points, 2), dtype=np.float32)
    if triplets.shape[0] == 0:
        return out.reshape(-1)

    # truncate or pad to expected length
    triplets = triplets[:expected_points]

    x = triplets[:, 0]
    y = triplets[:, 1]
    c = triplets[:, 2]

    # normalize to [0,1] if width/height available
    if width > 0:
        x = x / width
    if height > 0:
        y = y / height
    if clamp:
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

    # zero out low-confidence points
    keep = c >= conf_threshold
    out[: triplets.shape[0], 0] = np.where(keep, x, 0.0)
    out[: triplets.shape[0], 1] = np.where(keep, y, 0.0)
    return out.reshape(-1)


def load_openpose_sequence(
    json_path: str | Path,
    *,
    expected_pose_points: int = 25,
    expected_hand_points: int = 21,
    conf_threshold: float = 0.2,
    include_empty_frames: bool = True,
    clamp_xy: bool = True,
) -> np.ndarray:
    """
    Load an OpenPose JSON exported as a single big file and return an array
    of shape [num_frames, num_features] with a FIXED feature dimension.

    Fixes common issues:
    - **Normalization**: x/y are divided by (width, height).
    - **Missing keypoints**: low-confidence points become zeros.
    - **Inconsistent frame content**: frames with no people become all-zeros
      (if include_empty_frames=True).
    - **Inconsistent keypoint lengths**: pose/hands are padded/truncated to
      expected_pose_points / expected_hand_points.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Some exports wrap the content in a single-element list.
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty OpenPose JSON list in {json_path}")
        data = data[0]

    width = float(data.get("width", 0.0))
    height = float(data.get("height", 0.0))
    frames = data["frames"]

    frame_ids = sorted(frames.keys(), key=lambda k: int(k))
    if not frame_ids:
        raise ValueError(f"No frames found in {json_path}")

    # If we want a dense timeline, include frames 0..max
    if include_empty_frames:
        max_id = int(frame_ids[-1])
        frame_ids = [str(i) for i in range(max_id + 1)]

    per_frame_dim = (expected_pose_points + 2 * expected_hand_points) * 2
    seq = np.zeros((len(frame_ids), per_frame_dim), dtype=np.float32)

    for i, fid in enumerate(frame_ids):
        frame = frames.get(fid)
        if frame is None:
            continue

        people = frame.get("people", [])
        if not people:
            continue

        person = people[0]

        pose = _extract_fixed_xy(
            person,
            "pose_keypoints_2d",
            expected_points=expected_pose_points,
            width=width,
            height=height,
            conf_threshold=conf_threshold,
            clamp=clamp_xy,
        )
        hand_l = _extract_fixed_xy(
            person,
            "hand_left_keypoints_2d",
            expected_points=expected_hand_points,
            width=width,
            height=height,
            conf_threshold=conf_threshold,
            clamp=clamp_xy,
        )
        hand_r = _extract_fixed_xy(
            person,
            "hand_right_keypoints_2d",
            expected_points=expected_hand_points,
            width=width,
            height=height,
            conf_threshold=conf_threshold,
            clamp=clamp_xy,
        )

        seq[i] = np.concatenate([pose, hand_l, hand_r], axis=0)

    return seq


