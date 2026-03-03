from __future__ import annotations

"""
Step 2: build a segments manifest from the DGS-Korpus EAF file.

We read:
- the German translation tier (proper German sentences)
- the gloss tier

and produce a JSON manifest with entries of the form:
{
  "id": "seg_0001",
  "german_text": "...",
  "start_ms": 11105600,
  "end_ms": 11163240,
  "gloss_sequence": ["ICH1", "GEHEN1", ...]
}

For now this script targets the single demo sample
`1413451-11105600-11163240.eaf` in this project's own `dataset/`
directory.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import json

from pympi import Elan


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directory inside the new text2sign project
DATASET_DIR = PROJECT_ROOT / "dataset"

# Default DGS-Korpus sample expected in this project
# (copy the real files here, no synthetic data):
#   dataset/1413451-11105600-11163240.eaf
#   dataset/1413451-11105600-11163240_1a1.mp4
DEFAULT_EAF = DATASET_DIR / "1413451-11105600-11163240.eaf"

# Tier names taken from the existing codebase
GERMAN_TIER = "Deutsche_Übersetzung_A"
GLOSS_TIER = "Lexem_Gebärde_r_A"

# Where to write the manifest in the new project
DATA_DIR = PROJECT_ROOT / "backend" / "data"
MANIFEST_PATH = DATA_DIR / "segments_manifest.json"


@dataclass
class Segment:
    id: str
    german_text: str
    start_ms: int
    end_ms: int
    gloss_sequence: List[str]


def _load_tier_annotations(eaf: Elan.Eaf, tier: str) -> list[tuple[int, int, str]]:
    if tier not in eaf.get_tier_names():
        raise ValueError(
            f"Tier '{tier}' not found in EAF. "
            f"Available tiers: {list(eaf.get_tier_names())}"
        )
    return eaf.get_annotation_data_for_tier(tier)


def build_segments_from_eaf(eaf_path: Path) -> list[Segment]:
    if not eaf_path.exists():
        raise FileNotFoundError(f"EAF file not found: {eaf_path}")

    print(f"[segments] Loading EAF: {eaf_path}")
    eaf = Elan.Eaf(str(eaf_path))

    german_ann = _load_tier_annotations(eaf, GERMAN_TIER)
    gloss_ann = _load_tier_annotations(eaf, GLOSS_TIER)

    print(f"[segments] Loaded {len(german_ann)} German segments, {len(gloss_ann)} gloss segments")

    segments: list[Segment] = []

    # For each German segment, collect all gloss tokens that overlap in time
    for idx, (g_start, g_end, g_text) in enumerate(german_ann, start=1):
        text = (g_text or "").strip()
        if not text:
            continue

        gloss_tokens: list[str] = []
        for s_ms, e_ms, gloss in gloss_ann:
            if not gloss or not str(gloss).strip():
                continue
            # simple interval overlap check
            if e_ms <= g_start or s_ms >= g_end:
                continue
            gloss_tokens.append(str(gloss).strip())

        seg_id = f"seg_{idx:04d}"
        seg = Segment(
            id=seg_id,
            german_text=text,
            start_ms=int(g_start),
            end_ms=int(g_end),
            gloss_sequence=gloss_tokens,
        )
        segments.append(seg)

    print(f"[segments] Built {len(segments)} segments with non-empty German text")
    return segments


def save_manifest(segments: list[Segment], out_path: Path, source_eaf: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_eaf": str(source_eaf),
        "german_tier": GERMAN_TIER,
        "gloss_tier": GLOSS_TIER,
        "num_segments": len(segments),
        "segments": [asdict(s) for s in segments],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[segments] Saved manifest to {out_path}")


def main(eaf_path: Path | None = None) -> None:
    eaf = eaf_path or DEFAULT_EAF
    segments = build_segments_from_eaf(eaf)
    save_manifest(segments, MANIFEST_PATH, eaf)

    # Print a small preview for sanity-checking
    if segments:
        print("[segments] Example segment:")
        example = segments[0]
        print(
            f"  id={example.id} "
            f"start_ms={example.start_ms} end_ms={example.end_ms}\n"
            f"  german_text={example.german_text!r}\n"
            f"  gloss_sequence={example.gloss_sequence}"
        )


if __name__ == "__main__":
    main()

