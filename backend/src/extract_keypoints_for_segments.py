from __future__ import annotations

"""
Step 3: extract 134-D keypoint features for each segment.

In this text2sign project we reuse the existing OpenPose keypoints from the
original `signlanguage-german` project instead of running MediaPipe again,
because those OpenPose features are known-good and already used for training.

We:
- read backend/data/segments_manifest.json (output of prepare_segments.py)
- load the full OpenPose sequence (shape [num_frames, 134])
- slice the keypoint sequence into per-segment windows
- save each segment's motion to:
    backend/data/motion/<segment_id>.npz
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from src.data.keypoints_preprocess import load_openpose_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"
MANIFEST_PATH = DATA_DIR / "segments_manifest.json"

# Path to existing OpenPose sequence in the original project
LEGACY_ROOT = PROJECT_ROOT.parent / "signlanguage-german"
LEGACY_DATASET_DIR = LEGACY_ROOT / "dataset"
OPENPOSE_JSON = LEGACY_DATASET_DIR / "1413451_openpose.json"

# Output directory for per-segment motion files
MOTION_DIR = DATA_DIR / "motion"


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def time_ms_to_frame(time_ms: int, fps: float) -> int:
    return int(time_ms / (1000.0 / fps))


def _auto_normalize_openpose(json_path: Path) -> np.ndarray:
    """
    Load OpenPose JSON and normalize x/y by the actual coordinate ranges
    found in the data (since width/height are often 0 in this dataset).
    Returns array of shape [num_frames, 134] with values in [0, 1].
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    frame_ids_sorted = sorted(frames.keys(), key=lambda k: int(k))
    max_id = int(frame_ids_sorted[-1])
    all_frame_ids = [str(i) for i in range(max_id + 1)]

    POSE_PTS = 25
    HAND_PTS = 21
    DIM = (POSE_PTS + 2 * HAND_PTS) * 2  # 134

    KEYS = ["pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]
    PTS  = [POSE_PTS, HAND_PTS, HAND_PTS]

    # --- pass 1: find max x and max y across all frames ---
    max_x, max_y = 1.0, 1.0
    for fid in frame_ids_sorted:
        frame = frames.get(fid)
        if not frame:
            continue
        people = frame.get("people", [])
        if not people:
            continue
        person = people[0]
        for key in KEYS:
            raw = np.asarray(person.get(key, []), dtype=np.float32)
            n = (raw.size // 3) * 3
            if n == 0:
                continue
            triplets = raw[:n].reshape(-1, 3)
            conf = triplets[:, 2]
            x = triplets[conf >= 0.2, 0]
            y = triplets[conf >= 0.2, 1]
            if x.size:
                max_x = max(max_x, float(x.max()))
            if y.size:
                max_y = max(max_y, float(y.max()))

    print(f"[motion] auto-detected coordinate range: max_x={max_x:.1f} max_y={max_y:.1f}")

    # --- pass 2: build normalized sequence ---
    seq = np.zeros((len(all_frame_ids), DIM), dtype=np.float32)
    for i, fid in enumerate(all_frame_ids):
        frame = frames.get(fid)
        if not frame:
            continue
        people = frame.get("people", [])
        if not people:
            continue
        person = people[0]

        parts = []
        for key, n_pts in zip(KEYS, PTS):
            raw = np.asarray(person.get(key, []), dtype=np.float32)
            n = (raw.size // 3) * 3
            out = np.zeros((n_pts, 2), dtype=np.float32)
            if n > 0:
                triplets = raw[:n].reshape(-1, 3)[:n_pts]
                conf = triplets[:, 2]
                keep = conf >= 0.2
                nx = triplets.shape[0]
                out[:nx, 0] = np.where(keep, np.clip(triplets[:, 0] / max_x, 0.0, 1.0), 0.0)
                out[:nx, 1] = np.where(keep, np.clip(triplets[:, 1] / max_y, 0.0, 1.0), 0.0)
            parts.append(out.reshape(-1))

        seq[i] = np.concatenate(parts)

    return seq


def load_openpose_keypoints() -> tuple[np.ndarray, float]:
    """
    Load the consolidated OpenPose sequence, auto-normalizing coordinates
    since the JSON has width=0/height=0 for this dataset.
    """
    if not OPENPOSE_JSON.exists():
        raise FileNotFoundError(f"OpenPose JSON not found: {OPENPOSE_JSON}")

    print(f"[motion] Loading OpenPose sequence from {OPENPOSE_JSON}")
    kp = _auto_normalize_openpose(OPENPOSE_JSON)
    fps = 25.0
    print(f"[motion] OpenPose keypoints shape={kp.shape} fps={fps:.3f}")
    return kp.astype(np.float32), fps


def slice_segments(manifest: Dict, keypoints: np.ndarray, fps: float) -> None:
    segments = manifest.get("segments", [])
    MOTION_DIR.mkdir(parents=True, exist_ok=True)

    num_frames = keypoints.shape[0]
    for seg in segments:
        seg_id = seg["id"]
        start_ms = int(seg["start_ms"])
        end_ms = int(seg["end_ms"])

        start_f = time_ms_to_frame(start_ms, fps)
        end_f = time_ms_to_frame(end_ms, fps)
        if end_f <= start_f or start_f >= num_frames:
            continue

        end_f = min(end_f, num_frames - 1)
        seq = keypoints[start_f : end_f + 1]

        out_path = MOTION_DIR / f"{seg_id}.npz"
        np.savez_compressed(
            out_path,
            keypoints=seq.astype(np.float32),
            fps=np.array([fps], dtype=np.float32),
            start_ms=np.array([start_ms], dtype=np.int32),
            end_ms=np.array([end_ms], dtype=np.int32),
        )
        # small console confirmation for a few segments
        if int(seg_id.split("_")[-1]) <= 3:
            print(f"[motion] saved {out_path.name} frames={seq.shape[0]}")


def main() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    print(f"[motion] Loaded manifest from {MANIFEST_PATH} with {manifest.get('num_segments')} segments")

    kp, fps = load_openpose_keypoints()
    slice_segments(manifest, kp, fps)
    print(f"[motion] Per-segment motion files written to {MOTION_DIR}")


if __name__ == "__main__":
    main()

