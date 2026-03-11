from __future__ import annotations

"""
German text -> gloss / segment lookup with two matching modes:

  translate(text)
      Sentence-level: use a pretrained multilingual sentence-embedding model
      (paraphrase-multilingual-MiniLM-L12-v2) for semantic similarity.
      Falls back to Jaccard token overlap when sentence-transformers is not
      installed.

  translate_chained(text)
      Word-level greedy covering: tokenise the input, then greedily select
      segments that cover as many content words as possible, in order of their
      first match position.

GPU Note
--------
sentence-transformers uses PyTorch internally.  If a CUDA-capable GPU is
present the model inference will run on it automatically via:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import json
import re

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "backend" / "data" / "segments_manifest.json"

_PUNCT_RE = re.compile(r"[^\w\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]+", flags=re.UNICODE)

# Common German function/stop words
_STOPWORDS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mein", "dein", "sein",
    "unser", "euer", "ja", "nein", "nicht", "auch", "aber", "und", "oder",
    "weil", "dass", "als", "wenn", "dann", "so", "wie", "was", "wo", "wann",
    "warum", "der", "die", "das", "ein", "eine", "einen", "einem", "einer",
    "ist", "bin", "bist", "sind", "war", "hatte", "hat", "haben", "mir",
    "dir", "ihm", "uns", "euch", "ihnen", "na", "mal", "doch", "noch",
    "schon", "hier", "da", "jetzt", "immer", "sehr", "mehr", "nur", "man",
    "an", "auf", "bei", "bis", "durch", "fur", "gegen", "in", "mit", "nach",
    "ohne", "uber", "um", "unter", "vor", "von", "zu", "zum", "zur",
}


def _normalise_text(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenise(s: str) -> List[str]:
    return [t for t in _normalise_text(s).split() if t]


def _content_tokens(tokens: List[str]) -> List[str]:
    ct = [t for t in tokens if t not in _STOPWORDS]
    return ct if ct else tokens


def _detect_device() -> str:
    """Detect the best available compute device (CUDA dGPU > CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[device] CUDA GPU detected: {name} -- using cuda")
            return "cuda"
    except ImportError:
        pass
    print("[device] No CUDA GPU found -- using cpu")
    return "cpu"


# -- Neural sentence-embedding matcher -----------------------------------------

class _EmbeddingMatcher:
    """
    Semantic sentence similarity using a pretrained multilingual model.
    Uses GPU automatically if CUDA is available.
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, segments) -> None:
        self._segments = segments
        self._ready = False
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            device = _detect_device()
            print(f"[EmbeddingMatcher] Loading {self.MODEL_NAME} on {device} ...")
            self._model = SentenceTransformer(self.MODEL_NAME, device=device)
            texts = [seg.german_text for seg in segments]
            self._embeddings = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )  # [N, D] float32
            self._ready = True
            print(
                f"[EmbeddingMatcher] Ready -- {len(segments)} embeddings"
                f" (dim={self._embeddings.shape[1]}) on {device}."
            )
        except ImportError:
            print(
                "[EmbeddingMatcher] sentence-transformers not installed -- "
                "falling back to Jaccard token matching."
            )

    @property
    def ready(self) -> bool:
        return self._ready

    def best_match(self, query: str):
        q_emb = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )  # [1, D]
        scores = (self._embeddings @ q_emb.T).ravel()  # cosine sim [N]
        best_idx = int(np.argmax(scores))
        return self._segments[best_idx], float(scores[best_idx])


# -- Data model ----------------------------------------------------------------

@dataclass
class SegmentEntry:
    id: str
    german_text: str
    tokens: List[str]
    content_tokens: List[str]
    gloss_sequence: List[str]


# -- Main mapper ---------------------------------------------------------------

class TextToGlossMapper:
    def __init__(self, manifest_path: Path = MANIFEST_PATH) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Segments manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.segments: List[SegmentEntry] = []
        for seg in data.get("segments", []):
            text = seg.get("german_text", "") or ""
            tokens = _tokenise(text)
            self.segments.append(
                SegmentEntry(
                    id=seg["id"],
                    german_text=text,
                    tokens=tokens,
                    content_tokens=_content_tokens(tokens),
                    gloss_sequence=list(seg.get("gloss_sequence", [])),
                )
            )

        self._word_index: dict = {}
        for seg in self.segments:
            for tok in seg.tokens:
                self._word_index.setdefault(tok, []).append(seg)

        self._neural = _EmbeddingMatcher(self.segments)

    def _jaccard_best(self, inp_tokens: List[str]):
        inp_set = set(inp_tokens)
        best_entry: Optional[SegmentEntry] = None
        best_score = 0.0
        for seg in self.segments:
            seg_set = set(seg.tokens)
            if not seg_set:
                continue
            inter = len(inp_set & seg_set)
            union = len(inp_set | seg_set)
            score = inter / union if union > 0 else 0.0
            if score > best_score:
                best_score = score
                best_entry = seg
        return best_entry, best_score

    def translate(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Sentence-level: returns (gloss_sequence, [segment_id]).
        Uses neural similarity if available, else Jaccard.
        """
        if self._neural.ready:
            best_entry, score = self._neural.best_match(text)
            if score > 0.0:
                return list(best_entry.gloss_sequence), [best_entry.id]

        inp_tokens = _tokenise(text)
        best_entry, score = self._jaccard_best(inp_tokens)
        if best_entry is None or score <= 0.0:
            return [], []
        return list(best_entry.gloss_sequence), [best_entry.id]

    def translate_chained(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Word-level greedy segment chaining.
        Returns (combined_gloss_sequence, ordered_segment_ids).
        """
        inp_tokens = _tokenise(text)
        if not inp_tokens:
            return [], []

        content = _content_tokens(inp_tokens)
        remaining: set = set(content)
        selected: List[SegmentEntry] = []
        seen_ids: set = set()

        while remaining:
            best_entry: Optional[SegmentEntry] = None
            best_count = 0
            for seg in self.segments:
                if seg.id in seen_ids:
                    continue
                count = len(remaining & set(seg.tokens))
                if count > best_count:
                    best_count = count
                    best_entry = seg
            if best_entry is None or best_count == 0:
                break
            selected.append(best_entry)
            seen_ids.add(best_entry.id)
            remaining -= set(best_entry.tokens)

        if not selected:
            return self.translate(text)

        tok_pos = {tok: i for i, tok in enumerate(inp_tokens)}

        def _first_pos(seg: SegmentEntry) -> int:
            positions = [tok_pos[t] for t in seg.tokens if t in tok_pos]
            return min(positions) if positions else len(inp_tokens)

        selected.sort(key=_first_pos)

        combined_gloss: List[str] = []
        segment_ids: List[str] = []
        for seg in selected:
            combined_gloss.extend(seg.gloss_sequence)
            segment_ids.append(seg.id)

        return combined_gloss, segment_ids


# -- Singleton -----------------------------------------------------------------

_global_mapper: TextToGlossMapper | None = None


def get_mapper() -> TextToGlossMapper:
    global _global_mapper
    if _global_mapper is None:
        _global_mapper = TextToGlossMapper(MANIFEST_PATH)
    return _global_mapper
