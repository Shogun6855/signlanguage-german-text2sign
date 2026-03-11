from __future__ import annotations

import base64
import json
import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.text_to_gloss_map import get_mapper
from src.nlp_pipeline import analyse_text, get_gloss_lm, get_hmm_tagger


class TranslateRequest(BaseModel):
    text: str
    chained: bool = False  # True → word-level greedy segment chaining


class TranslateResponse(BaseModel):
    gloss: list[str]
    segments: list[str]


class MotionResponse(BaseModel):
    keypoints: list[list[float]]
    fps: float
    start_ms: int
    end_ms: int
    gloss_labels: list[str] = []      # one label per clip (for overlay display)
    frame_boundaries: list[int] = []  # start frame of each clip (after transitions)
    missing_glosses: list[str] = []   # glosses not found in the dictionary


class ChainedMotionRequest(BaseModel):
    segment_ids: list[str]


class GlossByNameRequest(BaseModel):
    glosses: list[str]  # ordered list of gloss labels e.g. ["LEBEN1A*", "ICH1"]


class NLPAnalyzeRequest(BaseModel):
    text: str


class GlossToSentenceRequest(BaseModel):
    glosses: list[str]  # gloss sequence from live recognition e.g. ["ICH1", "LEBEN1A*"]


class GlossToSentenceResponse(BaseModel):
    predicted_sentence: str      # best-match German sentence
    confidence: float            # Jaccard similarity of best match (0–1)
    method: str                  # "retrieval" or "reconstruction"
    top_matches: list[dict]      # top-3 [{german_text, score, glosses}]
    gloss_word_map: list[dict]   # per input gloss: [{gloss, word}]
    reconstruction: str          # simple word-for-word German reconstruction


def _concat_with_transitions(
    clips: list,          # list of np.ndarray [T_i, 134]
    hold_frames: int = 3, # repeat last frame of each clip (natural sign hold)
    lerp_frames: int = 5, # linear interpolation between clips (coarticulation)
) -> "tuple[np.ndarray, list[int]]":
    """
    Concatenate keypoint clips with realistic inter-sign transitions.

    Between every pair of adjacent clips the function inserts:
      1. hold_frames  — last frame of clip A repeated  (~120 ms at 25 fps)
      2. lerp_frames  — linear blend from last-A to first-B (~200 ms at 25 fps)

    Returns:
        (combined_array [T_total, 134], start_frame_of_each_clip)
    The start_frame list lets the frontend know exactly when each sign begins.
    """
    if len(clips) == 1:
        return clips[0], [0]

    parts = []
    boundaries = []   # frame index where each clip starts
    pos = 0
    for i, clip in enumerate(clips):
        boundaries.append(pos)
        parts.append(clip)
        pos += len(clip)
        if i < len(clips) - 1:          # not the last clip
            last  = clip[-1]             # [134]
            first = clips[i + 1][0]     # [134]
            # 1. hold
            hold = np.tile(last, (hold_frames, 1))   # [hold_frames, 134]
            # 2. lerp
            alphas = np.linspace(0, 1, lerp_frames + 2)[1:-1]  # exclude endpoints
            bridge = np.stack([(1 - a) * last + a * first for a in alphas])
            parts.append(hold)
            parts.append(bridge)
            pos += hold_frames + lerp_frames

    return np.concatenate(parts, axis=0), boundaries


def create_app() -> FastAPI:
    app = FastAPI(title="DGS Text-to-Sign API", version="0.1.0")

    # Allow local React dev server to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    mapper = get_mapper()
    backend_root = Path(__file__).resolve().parents[1]
    motion_dir  = backend_root / "data" / "motion"
    clips_dir   = backend_root / "data" / "gloss_clips"
    gloss_dict_path = backend_root / "data" / "gloss_dictionary.json"

    # Load gloss dictionary once at startup (maps gloss label -> relative clip path)
    _gloss_dict: dict[str, str] = {}
    if gloss_dict_path.exists():
        import json as _json
        _gloss_dict = _json.loads(gloss_dict_path.read_text(encoding="utf-8"))
        print(f"[startup] Gloss dictionary loaded: {len(_gloss_dict)} unique glosses")
    else:
        print("[startup] WARNING: gloss_dictionary.json not found -- run build_gloss_dictionary.py")

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/translate", response_model=TranslateResponse)
    async def translate(req: TranslateRequest) -> TranslateResponse:
        if req.chained:
            gloss_seq, segment_ids = mapper.translate_chained(req.text)
        else:
            gloss_seq, segment_ids = mapper.translate(req.text)
        return TranslateResponse(gloss=gloss_seq, segments=segment_ids)

    @app.post("/api/motion/by_glosses", response_model=MotionResponse)
    async def get_motion_by_glosses(req: GlossByNameRequest) -> MotionResponse:
        """
        Concatenate per-gloss keyframe clips into a single animation.
        Each gloss label is looked up in the gloss dictionary.
        Missing glosses are silently skipped (they won't appear in the animation).
        """
        all_clips = []
        fps = None
        found: list[str] = []
        missing: list[str] = []

        for gloss in req.glosses:
            rel = _gloss_dict.get(gloss)
            if rel is None:
                missing.append(gloss)
                continue
            clip_path = backend_root / "data" / rel
            if not clip_path.exists():
                missing.append(gloss)
                continue
            d = np.load(clip_path)
            all_clips.append(d["keypoints"])   # [T, 134]
            if fps is None:
                fps = float(d["fps"][0])
            found.append(gloss)

        if not all_clips:
            raise HTTPException(
                status_code=404,
                detail=f"No clips found for any of the requested glosses: {req.glosses}",
            )

        if missing:
            print(f"[by_glosses] Missing glosses (no clip): {missing}")

        combined, boundaries = _concat_with_transitions(all_clips)  # [T_total, 134]
        fps = fps or 50.0
        return MotionResponse(
            keypoints=combined.tolist(),
            fps=fps,
            start_ms=0,
            end_ms=int(combined.shape[0] / fps * 1000),
            gloss_labels=found,
            frame_boundaries=boundaries,
            missing_glosses=missing,
        )

    @app.post("/api/motion/chained", response_model=MotionResponse)
    async def get_chained_motion(req: ChainedMotionRequest) -> MotionResponse:
        """Concatenate keypoints from multiple segments into a single animation."""
        all_keypoints = []
        fps = None
        for seg_id in req.segment_ids:
            npz_path = motion_dir / f"{seg_id}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            all_keypoints.append(data["keypoints"])  # [T, 134]
            if fps is None:
                fps = float(data["fps"][0])

        if not all_keypoints:
            raise HTTPException(
                status_code=404,
                detail="No motion data found for any requested segment",
            )

        combined, _ = _concat_with_transitions(all_keypoints)  # [T_total, 134]
        fps = fps or 25.0
        return MotionResponse(
            keypoints=combined.tolist(),
            fps=fps,
            start_ms=0,
            end_ms=int(combined.shape[0] / fps * 1000),
        )

    @app.get("/api/motion/{segment_id}", response_model=MotionResponse)
    async def get_motion(segment_id: str) -> MotionResponse:
        npz_path = motion_dir / f"{segment_id}.npz"
        if not npz_path.exists():
            raise HTTPException(status_code=404, detail=f"Motion file not found for segment_id={segment_id}")

        data = np.load(npz_path)
        keypoints = data["keypoints"]  # [T, 134]
        fps_arr = data["fps"]
        start_ms_arr = data["start_ms"]
        end_ms_arr = data["end_ms"]

        return MotionResponse(
            keypoints=keypoints.tolist(),
            fps=float(fps_arr[0]),
            start_ms=int(start_ms_arr[0]),
            end_ms=int(end_ms_arr[0]),
        )

    # ── NLP Pipeline endpoint (Lab 1 / Feature Eng / Ex3 / Ex4) ─────────────
    @app.post("/api/nlp/analyze")
    async def nlp_analyze(req: NLPAnalyzeRequest) -> dict:
        """
        Full NLP pipeline analysis of a German text string.
        Returns tokenization, POS tagging (HMM/Viterbi), feature engineering
        (BoW, TF-IDF, PMI), and N-gram language model statistics.
        """
        return analyse_text(req.text)

    @app.get("/api/nlp/gloss_lm")
    async def gloss_lm_info() -> dict:
        """Return N-gram language model statistics over the gloss corpus."""
        lm = get_gloss_lm()
        return {
            "trained_sequences": len(lm.sequences),
            "unigram_top20": lm.unigram.get_counts_table(20),
            "bigram_top20": lm.bigram.get_counts_table(20),
            "trigram_top20": lm.trigram.get_counts_table(20),
            "bigram_perplexity_train": (
                round(lm.bigram.perplexity(lm.sequences), 2)
                if lm.sequences
                else None
            ),
        }

    @app.post("/api/nlp/score_glosses")
    async def score_glosses(req: GlossByNameRequest) -> dict:
        """Score a gloss sequence with the N-gram language model."""
        lm = get_gloss_lm()
        return lm.score_all(req.glosses)

    @app.get("/api/nlp/pos_tables")
    async def pos_tables() -> dict:
        """Return HMM transition and emission probability tables."""
        tagger = get_hmm_tagger()
        return {
            "transition_table": tagger.get_transition_table(),
            "emission_table": tagger.get_emission_table(top_n=8),
        }

    # ── Gloss-to-Sentence prediction ─────────────────────────────────────────

    import re as _re

    def _norm_gloss(g: str) -> str:
        """Strip DGS variant markers to get a canonical comparison form.

        Examples:
            SEHEN1*  → SEHEN1
            LEBEN1A* → LEBEN1A
            $GEST-OFF^ → GEST-OFF
            ICH1*   → ICH1
        """
        g = g.strip().lstrip("$").rstrip("*^").strip()
        return g

    def _gloss_to_lemma(g: str) -> str:
        """Convert a gloss label to a readable German word fragment.

        Strips trailing numeric variant identifier (e.g. '1A', '3') and
        title-cases the result so it looks like a German word.

        Examples:
            LEBEN1A* → Leben
            ICH1     → Ich
            TAUB-GEHÖRLOS1A* → Taub-Gehörlos
        """
        base = _norm_gloss(g)
        # Remove trailing digit + optional letter variant code (e.g. 1, 1A, 3B)
        base = _re.sub(r"\d+[A-Z]?$", "", base).strip("-_")
        if not base:
            base = _norm_gloss(g)
        parts = base.split("-")
        return "-".join(p.capitalize() for p in parts if p)

    # Pre-build segment index for fast gloss-to-sentence retrieval
    _manifest_segments: list[dict] = []
    _manifest_loaded = False

    def _load_manifest_segments() -> list[dict]:
        nonlocal _manifest_segments, _manifest_loaded
        if _manifest_loaded:
            return _manifest_segments
        manifest_path = backend_root / "data" / "segments_manifest.json"
        if not manifest_path.exists():
            _manifest_loaded = True
            return _manifest_segments
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        segs = raw if isinstance(raw, list) else raw.get("segments", [])
        for seg in segs:
            glosses = seg.get("gloss_sequence", [])
            _manifest_segments.append({
                "id": seg.get("id", ""),
                "german_text": seg.get("german_text", ""),
                "glosses": glosses,
                "norm_glosses": set(_norm_gloss(g) for g in glosses),
            })
        _manifest_loaded = True
        return _manifest_segments

    def _build_gloss_word_map(manifest: list[dict]) -> dict[str, list[str]]:
        """Build reverse map: normalized_gloss → list of German words from
        paired segment texts.  Used for word-level reconstruction."""
        import re as _re2
        german_word_re = _re2.compile(r"[a-zA-ZäöüÄÖÜß]+")
        result: dict[str, list[str]] = {}
        for seg in manifest:
            words = german_word_re.findall(seg["german_text"])
            lower_words = [w.lower() for w in words]
            for g in seg["glosses"]:
                key = _norm_gloss(g)
                if key not in result:
                    result[key] = []
                result[key].extend(lower_words)
        return result

    @app.post("/api/gloss_to_sentence", response_model=GlossToSentenceResponse)
    async def gloss_to_sentence(req: GlossToSentenceRequest) -> GlossToSentenceResponse:
        """
        Predict a German sentence from a sequence of recognized DGS glosses.

        Strategy
        --------
        1. Normalize each input gloss (strip variant markers).
        2. Compute Jaccard similarity between the normalized input set and
           every segment's normalized gloss set from the manifest.
        3. If best Jaccard ≥ 0.25, return the matching segment's German text
           as the predicted sentence (retrieval).
        4. Otherwise fall back to word-for-word reconstruction using a
           gloss→lemma mapping (reconstruction).
        Always also returns a simple lemma-based reconstruction string and a
        per-gloss word map.
        """
        manifest = _load_manifest_segments()
        gloss_word_map_raw = _build_gloss_word_map(manifest)

        input_glosses = [g.strip() for g in req.glosses if g.strip()]
        if not input_glosses:
            return GlossToSentenceResponse(
                predicted_sentence="",
                confidence=0.0,
                method="none",
                top_matches=[],
                gloss_word_map=[],
                reconstruction="",
            )

        norm_input = [_norm_gloss(g) for g in input_glosses]
        input_set = set(norm_input)

        # ── 1. Retrieval: Jaccard similarity against all manifest segments ──
        scored: list[tuple[float, dict]] = []
        for seg in manifest:
            seg_set = seg["norm_glosses"]
            if not seg_set and not input_set:
                sim = 0.0
            elif not seg_set or not input_set:
                sim = 0.0
            else:
                sim = len(input_set & seg_set) / len(input_set | seg_set)
            scored.append((sim, seg))

        scored.sort(key=lambda x: x[0], reverse=True)
        top3_matches = scored[:3]
        best_score, best_seg = top3_matches[0] if top3_matches else (0.0, {})

        # ── 2. Per-gloss word mapping ──────────────────────────────────────
        from collections import Counter
        gloss_word_entries: list[dict] = []
        for g, ng in zip(input_glosses, norm_input):
            candidates = gloss_word_map_raw.get(ng, [])
            if candidates:
                most_common_word = Counter(candidates).most_common(1)[0][0]
            else:
                most_common_word = _gloss_to_lemma(g).lower()
            gloss_word_entries.append({
                "gloss": g,
                "normalized": ng,
                "word": most_common_word,
            })

        # ── 3. Simple lemma reconstruction ────────────────────────────────
        reconstruction_words = [e["word"].capitalize() for e in gloss_word_entries]
        reconstruction = " ".join(reconstruction_words) + ("." if reconstruction_words else "")

        # ── 4. Choose predicted sentence ──────────────────────────────────
        if best_score >= 0.25 and best_seg:
            method = "retrieval"
            predicted = best_seg.get("german_text", reconstruction)
        else:
            method = "reconstruction"
            predicted = reconstruction

        top_matches_out = [
            {
                "german_text": s.get("german_text", ""),
                "score": round(sc, 4),
                "glosses": s.get("glosses", []),
            }
            for sc, s in top3_matches
            if s
        ]

        return GlossToSentenceResponse(
            predicted_sentence=predicted,
            confidence=round(best_score, 4),
            method=method,
            top_matches=top_matches_out,
            gloss_word_map=gloss_word_entries,
            reconstruction=reconstruction,
        )

    # ── WebSocket: Live sign recognition from webcam ─────────────────────────
    # Loads gloss clip keypoint centroids on first connection for fast matching.
    _gloss_centroids: Optional[dict] = None

    def _load_gloss_centroids() -> dict:
        """
        Build a lookup of gloss_label → mean hand keypoint vector from
        gloss_clips. Uses right-hand (42 values) + left-hand (42 values)
        as the 84-D feature vector, averaged over all frames.
        """
        nonlocal _gloss_centroids
        if _gloss_centroids is not None:
            return _gloss_centroids

        result: dict[str, np.ndarray] = {}
        if not clips_dir.exists():
            _gloss_centroids = {}
            return _gloss_centroids

        # Load gloss dictionary to resolve clip paths
        gloss_dict: dict[str, str] = {}
        if gloss_dict_path.exists():
            gloss_dict = json.loads(gloss_dict_path.read_text(encoding="utf-8"))

        for gloss_label, rel_path in gloss_dict.items():
            clip_path = backend_root / "data" / rel_path
            if not clip_path.exists():
                continue
            d = np.load(clip_path)
            kp = d["keypoints"]  # [T, 134]: body25 | lhand21 | rhand21
            # Indices: body 0-74, lhand 75-116, rhand 117-158 (each 3D = 3 values)
            # We use lhand (cols 75:117 → 21 joints × 2? Let's use full vector)
            # Actually stored as [body25*3 | lhand21*3 | rhand21*3] = [75+63+63]=201?
            # Check actual shape via known manifest info: 134 features
            # 134 = 25*2 + 21*2 + 21*2 = 50+42+42 (2D keypoints, x+y per joint)
            hand_vec = kp[:, 50:].mean(axis=0)  # lhand+rhand: 84 values, time-averaged
            result[gloss_label] = hand_vec

        _gloss_centroids = result
        print(f"[webcam] Loaded {len(result)} gloss centroids for live recognition")
        return _gloss_centroids

    @app.websocket("/ws/live_recognition")
    async def live_recognition(websocket: WebSocket) -> None:
        """
        WebSocket endpoint for live sign language recognition from webcam frames.

        Protocol (JSON messages):
          Client → Server:  {"frame": "<base64-JPEG>"}
          Server → Client:  {"gloss": "GLOSS_LABEL", "confidence": 0.87,
                             "top3": [{"gloss": ..., "score": ...}, ...]}
        """
        await websocket.accept()
        centroids = _load_gloss_centroids()

        try:
            import cv2
            import mediapipe as mp

            # Lazy-load MediaPipe hand detector (reuse across frames)
            _hand_model_path = backend_root / "models" / "hand_landmarker.task"
            if not _hand_model_path.exists():
                await websocket.send_json(
                    {"error": "hand_landmarker.task model not found in backend/models/"}
                )
                return

            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            hand_opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(_hand_model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
            )
            hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

            while True:
                try:
                    msg = await websocket.receive_text()
                    data = json.loads(msg)
                except (WebSocketDisconnect, json.JSONDecodeError):
                    break

                if "frame" not in data:
                    continue

                # Decode base64 JPEG frame
                try:
                    img_bytes = base64.b64decode(data["frame"])
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue

                # Run MediaPipe hand detection
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=frame_rgb
                )
                hand_result = hand_landmarker.detect(mp_image)

                if not hand_result.hand_landmarks:
                    await websocket.send_json({"gloss": None, "confidence": 0.0, "top3": []})
                    continue

                # Build 84-D hand feature vector (lhand21*2 + rhand21*2)
                lhand = np.zeros(42)
                rhand = np.zeros(42)
                for i, (lm_list, handedness) in enumerate(
                    zip(hand_result.hand_landmarks, hand_result.handedness)
                ):
                    side = handedness[0].category_name  # "Left" or "Right"
                    vec = np.array([[lm.x, lm.y] for lm in lm_list]).ravel()  # 42
                    if side == "Left":
                        lhand = vec
                    else:
                        rhand = vec

                query_vec = np.concatenate([lhand, rhand])  # 84

                # Cosine similarity against gloss centroids
                if not centroids:
                    await websocket.send_json({"gloss": None, "confidence": 0.0, "top3": []})
                    continue

                q_norm = np.linalg.norm(query_vec)
                if q_norm < 1e-6:
                    await websocket.send_json({"gloss": None, "confidence": 0.0, "top3": []})
                    continue

                scores: list[tuple[str, float]] = []
                for gloss_label, centroid in centroids.items():
                    c_norm = np.linalg.norm(centroid)
                    if c_norm < 1e-6:
                        continue
                    sim = float(np.dot(query_vec, centroid) / (q_norm * c_norm))
                    scores.append((gloss_label, sim))

                scores.sort(key=lambda x: x[1], reverse=True)
                top3 = scores[:3]
                best_gloss, best_score = top3[0] if top3 else (None, 0.0)

                await websocket.send_json({
                    "gloss": best_gloss,
                    "confidence": round(best_score, 4),
                    "top3": [
                        {"gloss": g, "score": round(s, 4)} for g, s in top3
                    ],
                })

        except WebSocketDisconnect:
            pass
        finally:
            try:
                hand_landmarker.close()
            except Exception:
                pass

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

