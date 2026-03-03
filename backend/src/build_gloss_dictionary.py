from __future__ import annotations

"""
Step 3b -- build a per-gloss clip dictionary from the EAF + existing segment npz files.

For each of the 741 gloss annotations in Lexem_Gebaerde_r_A we:
  1. Read its absolute start_ms / end_ms from the EAF TIME_SLOT table.
  2. Find which segment npz covers that time window.
  3. Compute the frame-index range inside that npz.
  4. Slice out those keyframes and save as
       backend/data/gloss_clips/<GLOSS_NAME>_<n>.npz
     where n disambiguates multiple occurrences of the same gloss.

We also write backend/data/gloss_dictionary.json:
  {
    "LEBEN1A*":  "gloss_clips/LEBEN1A_star_0.npz",
    "ICH1":      "gloss_clips/ICH1_0.npz",
    ...
  }
  (one canonical entry per unique gloss name -- the first occurrence)

Usage:
  python -m src.build_gloss_dictionary            # from backend/
  python backend/src/build_gloss_dictionary.py    # from project root
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EAF_PATH     = PROJECT_ROOT / "dataset" / "1413451-11105600-11163240.eaf"
MOTION_DIR   = PROJECT_ROOT / "backend" / "data" / "motion"
CLIPS_DIR    = PROJECT_ROOT / "backend" / "data" / "gloss_clips"
DICT_PATH    = PROJECT_ROOT / "backend" / "data" / "gloss_dictionary.json"

GLOSS_TIER   = "Lexem_Gebärde_r_A"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_filename(gloss: str) -> str:
    """Convert a gloss label to a safe filesystem name."""
    name = gloss.strip()
    name = name.replace("*", "_star")
    name = name.replace("^", "_hat")
    name = name.replace("$", "DOLLAR_")
    name = re.sub(r"[^\w\-]", "_", name)
    return name.strip("_")


def _parse_eaf(eaf_path: Path) -> list[tuple[int, int, str]]:
    """Return list of (start_ms, end_ms, gloss_label) for the right-hand gloss tier."""
    tree = ET.parse(str(eaf_path))
    root = tree.getroot()

    # Build TIME_SLOT_ID -> ms map
    ts_map: dict[str, int] = {
        t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE", 0))
        for t in root.findall("TIME_ORDER/TIME_SLOT")
    }

    # Find the gloss tier
    gloss_tier = None
    for t in root.findall("TIER"):
        if t.get("TIER_ID") == GLOSS_TIER:
            gloss_tier = t
            break
    if gloss_tier is None:
        raise ValueError(f"Tier '{GLOSS_TIER}' not found in EAF.")

    annotations: list[tuple[int, int, str]] = []
    for ann in gloss_tier:
        aa = ann.find("ALIGNABLE_ANNOTATION")
        if aa is None:
            continue
        t1 = ts_map[aa.get("TIME_SLOT_REF1")]
        t2 = ts_map[aa.get("TIME_SLOT_REF2")]
        val = (aa.find("ANNOTATION_VALUE").text or "").strip()
        if val:
            annotations.append((t1, t2, val))

    annotations.sort(key=lambda x: x[0])
    return annotations


def _load_segments() -> list[dict]:
    """Load all segment npz metadata (id, start_ms, end_ms, fps, path)."""
    segments = []
    for npz in sorted(MOTION_DIR.glob("seg_*.npz")):
        d = np.load(npz)
        segments.append({
            "id":       npz.stem,
            "path":     npz,
            "start_ms": int(d["start_ms"][0]),
            "end_ms":   int(d["end_ms"][0]),
            "fps":      float(d["fps"][0]),
            "keypoints": d["keypoints"],   # [T, 134]
        })
    segments.sort(key=lambda s: s["start_ms"])
    return segments


def _find_segment(segments: list[dict], gloss_start: int, gloss_end: int) -> dict | None:
    """Return the segment whose time range fully contains (or best overlaps) the gloss."""
    # Exact containment first
    for seg in segments:
        if seg["start_ms"] <= gloss_start and seg["end_ms"] >= gloss_end:
            return seg
    # Fallback: best-overlap (gloss crosses segment boundary -- rare edge case)
    best = None
    best_overlap = 0
    for seg in segments:
        overlap = min(gloss_end, seg["end_ms"]) - max(gloss_start, seg["start_ms"])
        if overlap > best_overlap:
            best_overlap = overlap
            best = seg
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def build_gloss_dictionary(min_frames: int = 2) -> None:
    """
    Extract per-gloss keyframe clips and write gloss_dictionary.json.

    min_frames: skip clips shorter than this (usually bad EAF timestamps).
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[gloss_dict] Parsing EAF: {EAF_PATH}")
    glosses = _parse_eaf(EAF_PATH)
    print(f"[gloss_dict] Found {len(glosses)} gloss annotations")

    print(f"[gloss_dict] Loading {len(list(MOTION_DIR.glob('seg_*.npz')))} segment npz files ...")
    segments = _load_segments()
    print(f"[gloss_dict] Loaded {len(segments)} segments")

    # gloss_name -> canonical clip path (first occurrence)
    dictionary: dict[str, str] = {}
    # gloss_name -> count of occurrences (for unique filenames)
    occurrence: dict[str, int] = {}

    saved = 0
    skipped = 0

    for (g_start, g_end, gloss) in glosses:
        seg = _find_segment(segments, g_start, g_end)
        if seg is None:
            print(f"  [WARN] No segment found for {gloss} @ {g_start}-{g_end} ms -- skipped")
            skipped += 1
            continue

        fps = seg["fps"]
        seg_start = seg["start_ms"]
        kp = seg["keypoints"]  # [T, 134]
        total_frames = kp.shape[0]

        f1 = max(0, round((g_start - seg_start) * fps / 1000))
        f2 = min(total_frames, round((g_end   - seg_start) * fps / 1000))

        if f2 - f1 < min_frames:
            print(f"  [WARN] {gloss} @ {g_start}-{g_end} ms -> only {f2-f1} frames -- skipped")
            skipped += 1
            continue

        clip = kp[f1:f2]  # [clip_T, 134]

        n = occurrence.get(gloss, 0)
        occurrence[gloss] = n + 1
        safe_name = _safe_filename(gloss)
        clip_filename = f"{safe_name}_{n}.npz"
        clip_path = CLIPS_DIR / clip_filename

        np.savez_compressed(
            clip_path,
            keypoints=clip,
            fps=np.array([fps]),
            start_ms=np.array([g_start]),
            end_ms=np.array([g_end]),
            gloss=np.array([gloss]),
        )
        saved += 1

        # Canonical dictionary entry = first occurrence only
        if gloss not in dictionary:
            rel = f"gloss_clips/{clip_filename}"
            dictionary[gloss] = rel

    print(f"[gloss_dict] Saved {saved} clips, skipped {skipped}")
    print(f"[gloss_dict] Unique glosses in dictionary: {len(dictionary)}")

    with open(DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print(f"[gloss_dict] Written: {DICT_PATH}")


if __name__ == "__main__":
    build_gloss_dictionary()
