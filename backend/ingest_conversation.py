#!/usr/bin/env python3
"""
ingest_conversation.py — Append one DGS-Korpus EAF+MP4 conversation to the dataset.

For each EAF + corresponding _1a1.mp4 it:
  1. Parses the EAF for German-text segments and right-hand gloss annotations
  2. Appends new segments to backend/data/segments_manifest.json
  3. Runs MediaPipe Holistic on the MP4 to extract 134-D keypoints
  4. Saves per-segment .npz files to backend/data/motion/
  5. Extracts per-gloss clips and appends to backend/data/gloss_dictionary.json

Usage (run from the backend/ directory):

  # Process one conversation:
  python ingest_conversation.py --eaf ../dataset/1413485.eaf --mp4 ../dataset/1413485_1a1.mp4

  # Auto-discover and process all un-ingested EAF+MP4 pairs in dataset/:
  python ingest_conversation.py --all

Idempotent: segments and clips that already exist are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE         = Path(__file__).parent          # backend/
PROJECT_ROOT = BASE.parent                    # project root
DATASET_DIR  = PROJECT_ROOT / "dataset"
MANIFEST     = BASE / "data" / "segments_manifest.json"
MOTION_DIR   = BASE / "data" / "motion"
CLIPS_DIR    = BASE / "data" / "gloss_clips"
DICT_PATH    = BASE / "data" / "gloss_dictionary.json"

# MediaPipe Tasks model files (downloaded to backend/models/)
MODEL_DIR       = BASE / "models"
POSE_MODEL      = MODEL_DIR / "pose_landmarker_lite.task"
HAND_MODEL      = MODEL_DIR / "hand_landmarker.task"

# Tier names used in all DGS-Korpus EAF files
GERMAN_TIER = "Deutsche_\u00dcbersetzung_A"
GLOSS_TIER  = "Lexem_Geb\u00e4rde_r_A"

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    id: str
    german_text: str
    start_ms: int
    end_ms: int
    gloss_sequence: list[str]
    source_eaf: str  # relative to project root, for traceability


# ── MediaPipe Tasks API setup (mediapipe >= 0.10) ─────────────────────────────

_pose_landmarker: Optional[PoseLandmarker] = None
_hand_landmarker: Optional[HandLandmarker] = None


def _get_pose_landmarker() -> PoseLandmarker:
    global _pose_landmarker
    if _pose_landmarker is None:
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        _pose_landmarker = PoseLandmarker.create_from_options(opts)
    return _pose_landmarker


def _get_hand_landmarker() -> HandLandmarker:
    global _hand_landmarker
    if _hand_landmarker is None:
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        _hand_landmarker = HandLandmarker.create_from_options(opts)
    return _hand_landmarker


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


def _xy(lm, i: int, thresh: float = 0.1) -> tuple[float, float]:
    l = lm[i]
    v = getattr(l, "visibility", 1.0)
    p = getattr(l, "presence",   1.0)
    if (v is not None and v < thresh) and (p is not None and p < thresh):
        return 0.0, 0.0
    return float(min(max(l.x, 0), 1)), float(min(max(l.y, 0), 1))


def extract_frame_keypoints(rgb_frame: np.ndarray) -> np.ndarray:
    """
    Run MediaPipe Tasks (pose + hands) on one RGB frame and return a 134-D vector.
    Layout matches the OpenPose-derived format used throughout this project:
      [0..49]   = 25 body joints (OpenPose BODY_25 ordering)
      [50..91]  = 21 left-hand joints
      [92..133] = 21 right-hand joints
    """
    vec = np.zeros(134, dtype=np.float32)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # ── Pose ──────────────────────────────────────────────────────────────────
    pose_result = _get_pose_landmarker().detect(mp_image)
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]   # first (and only) detected person
        for op_i, mp_i in BODY25_MAP:
            if mp_i < len(lm):
                x, y = _xy(lm, mp_i, thresh=0.1)
                vec[op_i * 2]     = x
                vec[op_i * 2 + 1] = y
        # Neck = mid-shoulders (joints 11, 12)
        if len(lm) > 12:
            lx, ly = _xy(lm, 11, 0.1)
            rx, ry = _xy(lm, 12, 0.1)
            vec[1 * 2]     = (lx + rx) * 0.5
            vec[1 * 2 + 1] = (ly + ry) * 0.5
        # MidHip = mid-hips (joints 23, 24)
        if len(lm) > 24:
            lx, ly = _xy(lm, 23, 0.1)
            rx, ry = _xy(lm, 24, 0.1)
            vec[8 * 2]     = (lx + rx) * 0.5
            vec[8 * 2 + 1] = (ly + ry) * 0.5

    # ── Hands ─────────────────────────────────────────────────────────────────
    hand_result = _get_hand_landmarker().detect(mp_image)
    if hand_result.hand_landmarks:
        for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            # handedness[0].category_name is "Left" or "Right"
            label = handedness[0].category_name if handedness else "Right"
            offset = 50 if label == "Left" else 92
            for i, lm in enumerate(hand_lms[:21]):
                vec[offset + i * 2]     = float(min(max(lm.x, 0), 1))
                vec[offset + i * 2 + 1] = float(min(max(lm.y, 0), 1))

    return vec


# ── EAF parsing ───────────────────────────────────────────────────────────────

def _parse_eaf_timed(eaf_path: Path, tier_id: str) -> list[tuple[int, int, str]]:
    """Return sorted list of (start_ms, end_ms, value) for the named tier."""
    tree = ET.parse(str(eaf_path))
    root = tree.getroot()
    ts_map: dict[str, int] = {
        t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE", 0))
        for t in root.findall("TIME_ORDER/TIME_SLOT")
    }
    tier_el = None
    for t in root.findall("TIER"):
        if t.get("TIER_ID") == tier_id:
            tier_el = t
            break
    if tier_el is None:
        available = [t.get("TIER_ID") for t in root.findall("TIER")]
        print(f"  [WARN] Tier '{tier_id}' not found. Available: {available}")
        return []
    anns = []
    for ann in tier_el:
        aa = ann.find("ALIGNABLE_ANNOTATION")
        if aa is None:
            continue
        t1 = ts_map.get(aa.get("TIME_SLOT_REF1"), 0)
        t2 = ts_map.get(aa.get("TIME_SLOT_REF2"), 0)
        val = (aa.find("ANNOTATION_VALUE").text or "").strip()
        if val:
            anns.append((t1, t2, val))
    anns.sort(key=lambda x: x[0])
    return anns


def build_segments_from_eaf(
    eaf_path: Path,
    start_id: int,
) -> list[Segment]:
    """Parse one EAF and return Segment objects starting from start_id."""
    eaf_path = eaf_path.resolve()
    print(f"[parse] Parsing EAF: {eaf_path.name}")
    german_anns = _parse_eaf_timed(eaf_path, GERMAN_TIER)
    gloss_anns  = _parse_eaf_timed(eaf_path, GLOSS_TIER)
    print(f"[parse] {len(german_anns)} German sentences, {len(gloss_anns)} gloss tokens")

    segments: list[Segment] = []
    for idx, (g_start, g_end, g_text) in enumerate(german_anns, start=start_id):
        text = g_text.strip()
        if not text:
            continue
        gloss_tokens = [
            gloss for (s_ms, e_ms, gloss) in gloss_anns
            if gloss and e_ms > g_start and s_ms < g_end
        ]
        seg = Segment(
            id=f"seg_{idx:04d}",
            german_text=text,
            start_ms=int(g_start),
            end_ms=int(g_end),
            gloss_sequence=gloss_tokens,
            source_eaf=str(eaf_path.relative_to(PROJECT_ROOT)),
        )
        segments.append(seg)

    print(f"[parse] Built {len(segments)} segments")
    return segments


# ── Manifest I/O ──────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"segments": [], "num_segments": 0}


def max_segment_id(manifest: dict) -> int:
    """Return the highest numeric segment ID present (default 0 if empty)."""
    segs = manifest.get("segments", [])
    if not segs:
        return 0
    nums = []
    for s in segs:
        m = re.search(r"(\d+)$", s.get("id", ""))
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else 0


def source_eafs_in_manifest(manifest: dict) -> set[str]:
    """Return the set of known EAF stems already in the manifest (normalised to stem)."""
    raw: set[str] = set()
    if "source_eaf" in manifest:
        raw.add(manifest["source_eaf"])
    for s in manifest.get("segments", []):
        v = s.get("source_eaf", "")
        if v:
            raw.add(v)
    for v in manifest.get("sources", []):
        if v:
            raw.add(v)
    # Normalise: keep only the stem so absolute vs relative paths both match
    return {Path(p).stem for p in raw}


def append_segments_to_manifest(segments: list[Segment]) -> None:
    manifest = load_manifest()
    existing_ids = {s["id"] for s in manifest.get("segments", [])}

    new_segs = [asdict(s) for s in segments if s.id not in existing_ids]
    if not new_segs:
        print("[manifest] Nothing new to add — all segments already present")
        return

    manifest.setdefault("segments", []).extend(new_segs)
    manifest["num_segments"] = len(manifest["segments"])
    # Track all source EAFs in a top-level list for easy lookup
    prev_sources = set(manifest.get("sources", []))
    new_sources   = {s.source_eaf for s in segments}
    manifest["sources"] = sorted(prev_sources | new_sources)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[manifest] +{len(new_segs)} segments → total {manifest['num_segments']}")


# ── MediaPipe keypoint extraction ─────────────────────────────────────────────

def extract_keypoints_for_segments(
    mp4_path: Path,
    segments: list[Segment],
) -> None:
    """Run MediaPipe (Tasks API) on the video and save per-segment NPZ files."""
    if not mp4_path.exists():
        print(f"[mp4] ERROR: Video not found: {mp4_path}")
        return

    cap = cv2.VideoCapture(str(mp4_path))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[mp4] {mp4_path.name}  fps={video_fps:.1f}  frames={total_frames}")

    # Warm up the detectors (lazy init)
    _get_pose_landmarker()
    _get_hand_landmarker()

    MOTION_DIR.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments):
        out_path = MOTION_DIR / f"{seg.id}.npz"
        if out_path.exists():
            print(f"  [{i+1}/{len(segments)}] {seg.id} — SKIP (already exists)")
            continue

        start_f = max(0, int(seg.start_ms / 1000.0 * video_fps))
        end_f   = min(total_frames - 1, int(seg.end_ms / 1000.0 * video_fps))

        if end_f <= start_f:
            print(f"  [{i+1}/{len(segments)}] {seg.id} — SKIP (empty frame range [{start_f}, {end_f}])")
            np.savez_compressed(
                out_path,
                keypoints=np.zeros((1, 134), dtype=np.float32),
                fps=np.array([video_fps], dtype=np.float32),
                start_ms=np.array([seg.start_ms], dtype=np.int32),
                end_ms=np.array([seg.end_ms], dtype=np.int32),
            )
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames_kp = []
        for _ in range(end_f - start_f + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames_kp.append(extract_frame_keypoints(rgb))

        kp = (
            np.stack(frames_kp).astype(np.float32)
            if frames_kp
            else np.zeros((1, 134), dtype=np.float32)
        )
        np.savez_compressed(
            out_path,
            keypoints=kp,
            fps=np.array([video_fps], dtype=np.float32),
            start_ms=np.array([seg.start_ms], dtype=np.int32),
            end_ms=np.array([seg.end_ms], dtype=np.int32),
        )
        lnz = np.count_nonzero(kp[:, 50:92])
        rnz = np.count_nonzero(kp[:, 92:134])
        print(f"  [{i+1}/{len(segments)}] {seg.id}: frames={kp.shape[0]}  lhand={lnz}  rhand={rnz}")

    cap.release()
    print(f"[mp4] Done — {len(segments)} segments processed")


# ── Gloss clip extraction ─────────────────────────────────────────────────────

def _safe_filename(gloss: str) -> str:
    name = gloss.strip().replace("*", "_star").replace("^", "_hat").replace("$", "DOLLAR_")
    name = re.sub(r"[^\w\-]", "_", name)
    return name.strip("_")


def _load_segment_npz_index() -> list[dict]:
    """Load metadata for all existing segment NPZ files sorted by start_ms."""
    segs = []
    for npz in sorted(MOTION_DIR.glob("seg_*.npz")):
        try:
            d = np.load(npz)
            segs.append({
                "id": npz.stem,
                "path": npz,
                "start_ms": int(d["start_ms"][0]),
                "end_ms": int(d["end_ms"][0]),
                "fps": float(d["fps"][0]),
                "keypoints": d["keypoints"],
            })
        except Exception as e:
            print(f"  [WARN] Could not load {npz.name}: {e}")
    segs.sort(key=lambda s: (s["start_ms"], s["id"]))
    return segs


def _find_best_segment(all_segs: list[dict], g_start: int, g_end: int) -> dict | None:
    """Find the segment whose time range best contains the gloss interval."""
    # Filter to segments from the same time neighborhood (avoid cross-conversation matches)
    # We use "best overlap" within segments that share the same source (start_ms ballpark)
    for seg in all_segs:
        if seg["start_ms"] <= g_start and seg["end_ms"] >= g_end:
            return seg        # exact containment
    best, best_ov = None, 0
    for seg in all_segs:
        ov = min(g_end, seg["end_ms"]) - max(g_start, seg["start_ms"])
        if ov > best_ov:
            best_ov = ov
            best = seg
    return best


def extract_and_append_gloss_clips(
    eaf_path: Path,
    segments: list[Segment],
) -> None:
    """
    For each gloss annotation in the EAF that falls within one of `segments`,
    slice the corresponding frames from the segment NPZ and save a clip file.
    Appends new entries to gloss_dictionary.json without touching existing ones.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Build a segment NPZ index restricted to just the ones we just created
    seg_ids = {s.id for s in segments}
    all_npz = _load_segment_npz_index()
    local_npz = [s for s in all_npz if s["id"] in seg_ids]
    if not local_npz:
        print("[gloss] No segment NPZ files found — skipping gloss clip extraction")
        return

    gloss_anns = _parse_eaf_timed(eaf_path, GLOSS_TIER)
    print(f"[gloss] {len(gloss_anns)} gloss annotations, {len(local_npz)} local segments")

    # Load existing dictionary
    existing_dict: dict[str, str] = {}
    if DICT_PATH.exists():
        with DICT_PATH.open("r", encoding="utf-8") as f:
            existing_dict = json.load(f)

    occurrence: dict[str, int] = {}
    # Seed occurrence counts from existing clips on disk to avoid filename collisions
    for npz in CLIPS_DIR.glob("*.npz"):
        m = re.match(r"^(.+)_(\d+)\.npz$", npz.stem)
        if m:
            base, n = m.group(1), int(m.group(2))
            occurrence[base] = max(occurrence.get(base, -1), n) + 1

    new_entries = 0
    skipped = 0
    min_frames = 2

    for (g_start, g_end, gloss) in gloss_anns:
        seg = _find_best_segment(local_npz, g_start, g_end)
        if seg is None:
            skipped += 1
            continue

        fps = seg["fps"]
        seg_start = seg["start_ms"]
        kp = seg["keypoints"]
        total_f = kp.shape[0]

        f1 = max(0, round((g_start - seg_start) * fps / 1000))
        f2 = min(total_f, round((g_end   - seg_start) * fps / 1000))
        if f2 - f1 < min_frames:
            skipped += 1
            continue

        clip = kp[f1:f2]
        safe_name = _safe_filename(gloss)

        n = occurrence.get(safe_name, 0)
        occurrence[safe_name] = n + 1
        clip_filename = f"{safe_name}_{n}.npz"
        clip_path = CLIPS_DIR / clip_filename

        if not clip_path.exists():
            np.savez_compressed(
                clip_path,
                keypoints=clip,
                fps=np.array([fps]),
                start_ms=np.array([g_start]),
                end_ms=np.array([g_end]),
                gloss=np.array([gloss]),
            )

        # Add to dictionary only if this gloss isn't already there
        if gloss not in existing_dict:
            existing_dict[gloss] = f"gloss_clips/{clip_filename}"
            new_entries += 1

    with DICT_PATH.open("w", encoding="utf-8") as f:
        json.dump(existing_dict, f, ensure_ascii=False, indent=2)
    print(f"[gloss] +{new_entries} new dictionary entries | {skipped} annotations skipped")
    print(f"[gloss] Dictionary now has {len(existing_dict)} unique glosses")


# ── Orchestration ─────────────────────────────────────────────────────────────

def _segments_for_eaf(eaf_path: Path) -> list[dict]:
    """Return manifest segment dicts whose source_eaf matches this EAF's stem."""
    manifest = load_manifest()
    stem = eaf_path.resolve().stem
    return [
        s for s in manifest.get("segments", [])
        if Path(s.get("source_eaf", "")).stem == stem
    ]


def ingest_one(eaf_path: Path, mp4_path: Path) -> None:
    """Full pipeline for a single EAF+MP4 pair."""
    eaf_path = eaf_path.resolve()
    mp4_path = mp4_path.resolve()
    print(f"\n{'='*60}")
    print(f"INGESTING: {eaf_path.name}")
    print(f"{'='*60}")

    manifest = load_manifest()
    known_stems = source_eafs_in_manifest(manifest)
    already_in_manifest = eaf_path.stem in known_stems

    if already_in_manifest:
        # Check if keypoints are missing (crashed mid-run)
        existing_segs = _segments_for_eaf(eaf_path)
        missing_npz = [s for s in existing_segs if not (MOTION_DIR / f"{s['id']}.npz").exists()]
        if not missing_npz:
            print(f"[skip] Already fully ingested: {eaf_path.name}")
            return
        # Recovery: re-extract keypoints for segments that have no NPZ
        print(f"[recover] {eaf_path.name} has {len(missing_npz)} segments without keypoints — re-extracting")
        segments = [
            Segment(
                id=s["id"],
                german_text=s["german_text"],
                start_ms=s["start_ms"],
                end_ms=s["end_ms"],
                gloss_sequence=s.get("gloss_sequence", []),
                source_eaf=s.get("source_eaf", str(eaf_path)),
            )
            for s in missing_npz
        ]
        extract_keypoints_for_segments(mp4_path, segments)
        extract_and_append_gloss_clips(eaf_path, segments)
        print(f"\n[done] {eaf_path.name} — recovery complete")
        return

    # Step 1: parse EAF → build segment objects
    next_id = max_segment_id(manifest) + 1
    segments = build_segments_from_eaf(eaf_path, start_id=next_id)
    if not segments:
        print("[skip] No segments found in EAF — skipping")
        return

    # Step 2: save segments to manifest
    append_segments_to_manifest(segments)

    # Step 3: extract MediaPipe keypoints from video
    extract_keypoints_for_segments(mp4_path, segments)

    # Step 4: build per-gloss clips and update dictionary
    extract_and_append_gloss_clips(eaf_path, segments)

    print(f"\n[done] {eaf_path.name} — {len(segments)} segments ingested")


def ingest_all() -> None:
    """Process all EAF+_1a1.mp4 pairs in dataset/. Each conversation is processed
    by ingest_one() which handles new ingestion, skip-if-complete, and NPZ recovery."""
    pairs: list[tuple[Path, Path]] = []
    for eaf in sorted(DATASET_DIR.glob("*.eaf")):
        # Match <stem>_1a1.mp4 (DGS-Korpus signer-A clip convention)
        stem = eaf.stem
        mp4 = DATASET_DIR / f"{stem}_1a1.mp4"
        if not mp4.exists():
            print(f"[all] No _1a1.mp4 for {eaf.name} — skipping")
            continue
        pairs.append((eaf, mp4))

    if not pairs:
        print("[all] No EAF+MP4 pairs found in dataset/.")
        return

    print(f"[all] Processing {len(pairs)} EAF+MP4 pair(s) (skipping fully-ingested ones)")
    for eaf, mp4 in pairs:
        ingest_one(eaf, mp4)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest DGS-Korpus conversations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Auto-discover and ingest all new EAF+MP4 pairs in dataset/")
    group.add_argument("--eaf", type=Path, help="Path to a single EAF file")
    parser.add_argument("--mp4", type=Path,
                        help="Path to the corresponding _1a1.mp4 (required with --eaf)")

    args = parser.parse_args()

    if args.all:
        ingest_all()
    else:
        if not args.mp4:
            parser.error("--mp4 is required when using --eaf")
        ingest_one(args.eaf.resolve(), args.mp4.resolve())

    print("\n[ingest] Finished.")


if __name__ == "__main__":
    main()
