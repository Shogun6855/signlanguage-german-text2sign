# Handoff — DGS Text-to-Sign Project

**Date:** March 3, 2026  
**Workspace root (new):** `D:\College\Semester 6\Lab\NLIP\Project\signlanguage-german-text2sign\`  
**Python venv (new):** `.venv\` inside this folder (Python 3.10.0)

> The previous workspace root was `D:\College\Semester 6\Lab\NLIP\Project\`. All work is now focused exclusively on the `signlanguage-german-text2sign\` subfolder. The sibling folder `signlanguage-german\` is an old reference project — **do not modify it**.

---

## Project Goal

**German text → German Sign Language (DGS) skeleton animation.**

Pipeline:
1. User types a German sentence in the browser
2. Backend fuzzy-matches it to one of 105 segments from the DGS-Korpus dataset
3. Returns the corresponding per-segment MediaPipe keypoint animation
4. Frontend renders a 2D animated skeleton (body + full 5-finger hands) on a canvas

---

## Architecture

```
signlanguage-german-text2sign/
├── .venv/                        ← Python 3.10 venv (NEW location, freshly created)
├── backend/                      ← FastAPI app
│   ├── src/
│   │   ├── main.py               ← FastAPI app, 3 endpoints
│   │   ├── extract_keypoints_for_segments.py  ← OLD OpenPose extractor (no longer used)
│   │   └── data/
│   │       └── keypoints_preprocess.py
│   ├── data/
│   │   ├── segments_manifest.json  ← 105 segments: german_text, gloss_sequence, start_ms, end_ms
│   │   └── motion/
│   │       └── seg_0001.npz … seg_0105.npz  ← 134-D keypoints WITH hand data (re-extracted)
│   ├── requirements.txt          ← minimal deps list
│   ├── requirements_full.txt     ← full pip freeze (use this to recreate venv)
│   └── extract_hands.py          ← MediaPipe hand extractor (IMPORTANT — see below)
├── dataset/
│   ├── 1413451-11105600-11163240.eaf          ← ELAN annotation file
│   ├── 1413451-11105600-11163240.mp4          ← full video
│   ├── 1413451-11105600-11163240_1a1.mp4      ← MAIN SIGNER camera (used for keypoints)
│   ├── 1413451-11105600-11163240_1b1.mp4      ← side camera
│   └── 1413451-11105600-11163240_1c.mp4       ← interpreter camera
└── web/                          ← React + Vite frontend
    ├── src/
    │   ├── App.jsx               ← main component, calls backend API
    │   └── SkeletonViewer3D.jsx  ← 2D canvas skeleton renderer (misleading name, it is 2D)
    ├── package.json
    └── vite.config.mts
```

---

## Running the Project

**Backend** (run from `backend/`):
```
.\.venv\Scripts\activate   # or use full path to python.exe
python -m uvicorn src.main:app --reload
```
Runs on `http://127.0.0.1:8000`

**Frontend** (run from `web/`):
```
npm run dev
```
Runs on `http://localhost:5173`

> **PowerShell restriction**: npm commands fail due to execution policy. Use `cmd /c "npm run dev"` if needed.

> **Always use** `.venv\Scripts\python.exe` explicitly — never system Python.

---

## API Endpoints

| Method | Path | Description |
|--------|------|--------------|
| GET | `/api/health` | Health check |
| POST | `/api/translate` | `{"text": "German sentence"}` → `{gloss, segments, ...}` |
| GET | `/api/motion/{segment_id}` | Returns `{keypoints: [[134 floats]×T], fps, start_ms, end_ms}` |
| POST | `/api/motion/by_glosses` | Gloss list → concatenated keypoint animation |
| POST | `/api/motion/chained` | Segment IDs → chained keypoint animation |
| POST | `/api/nlp/analyze` | Full NLP pipeline on German text |
| GET | `/api/nlp/gloss_lm` | N-gram LM stats |
| POST | `/api/nlp/score_glosses` | Score gloss sequence with N-gram LM |
| GET | `/api/nlp/pos_tables` | HMM transition/emission tables |
| **POST** | **`/api/gloss_to_sentence`** | **Recognized gloss list → predicted German sentence** |
| WebSocket | `/ws/live_recognition` | Real-time per-frame gloss recognition |

---

## Keypoint Format (134-D per frame)

```
[0:50]   = BODY_25 joints (25 joints × 2 = x,y normalised [0,1])
[50:92]  = Left hand (21 joints × 2, MediaPipe wrist=0, fingertips=4/8/12/16/20)
[92:134] = Right hand (21 joints × 2)
```

All values are normalised to [0,1] image coordinates (y=0 is top, y=1 is bottom).  
Zero values (0.0, 0.0) mean the joint was not detected — the renderer skips these.

---

## Key Files — What They Do

### `backend/extract_hands.py`
**The most important script in the project.** Reads `segments_manifest.json`, seeks to each segment's frame range in `_1a1.mp4`, runs MediaPipe Holistic (body + both hands), and writes per-segment `.npz` files to `data/motion/`. Run this if you ever need to re-extract keypoints.

```
python extract_hands.py   # from backend/ directory
```

Previously the old extractor (`extract_keypoints_for_segments.py`) used OpenPose JSON which had no hand data — `lhand_nonzero=0, rhand_nonzero=0` for all segments. This is now fixed.

### `backend/data/segments_manifest.json`
105 segments from the DGS-Korpus recording `1413451`. Each entry has:
- `id`: `seg_0001` … `seg_0105`
- `german_text`: the German sentence annotation
- `gloss_sequence`: list of DGS glosses
- `start_ms` / `end_ms`: timestamp range in the video

### `backend/src/main.py`
FastAPI app. `/api/translate` does semantic sentence-embedding matching against segment German texts (falls back to Jaccard). `/api/motion/{segment_id}` loads the `.npz` and returns keypoints as JSON. **`/api/gloss_to_sentence`** converts a recognized gloss list into a German sentence via Jaccard-based corpus retrieval (≥ 25% match) or word-level reconstruction fallback.

### `web/src/SkeletonViewer3D.jsx`
2D HTML canvas renderer. Despite the name it is **not** 3D — it was renamed from the Three.js version that was replaced. It draws:
- **White**: body skeleton (BODY_25 edges)
- **Per-finger colour** (thumb=red, index=orange, middle=yellow, ring=green, pinky=blue): each finger on both hands as lines with joint dots

`computeBounds()` auto-fits all joints across all frames to the canvas with proportional padding — no clipping.

---

## Current State / Issues

1. **Animation works**: body + both hands animate with real MediaPipe data.
2. **Text matching**: sentence-transformers (multilingual-MiniLM-L12-v2) provides semantic matching; Jaccard fallback if library not installed.
3. **Gloss-level animation**: `POST /api/motion/by_glosses` plays one clip per gloss with hold+lerp transitions.
4. **Live recognition → sentence prediction**: `POST /api/gloss_to_sentence` converts a detected gloss strip to a German sentence (retrieval or reconstruction).
5. **`SkeletonViewer3D.jsx` is misnamed**: it is a 2D canvas component. Rename to `SkeletonViewer.jsx` if desired (update import in `App.jsx` too).
6. **`extract_keypoints_for_segments.py`** in `backend/src/` is dead code (the old OpenPose extractor). Safe to delete.

---

## What Was Done This Session (summary for continuity)

1. Diagnosed that all hand keypoints were zero in the old `.npz` files (OpenPose was run without hand tracking).
2. Confirmed per-frame OpenPose JSONs had `hand_left_keypoints_2d: len=0`.
3. Confirmed MediaPipe NPZ from the old project was also all-zeros (extraction had failed previously).
4. Switched from Three.js 3D visualisation to a flat 2D canvas renderer (user request — "stick in the frame").
5. Wrote `extract_hands.py` for targeted per-segment MediaPipe extraction (seeks to each segment's timestamp, skips unneeded frames — much faster than full-video scan).
6. Re-extracted all 105 segments. Example: `seg_0001` now has `lhand=3906, rhand=4074` non-zero values.
7. Created new `.venv` inside this project folder (Python 3.10). Install with: `pip install -r backend/requirements_full.txt`.
8. Added full NLP pipeline (Lab 1, Feature Engineering, Exercise 3, Exercise 4) in `nlp_pipeline.py`.
9. Implemented live webcam/video recognition via WebSocket (`/ws/live_recognition`).
10. **Implemented gloss-to-sentence prediction** (`POST /api/gloss_to_sentence`): Jaccard-based corpus retrieval with word-reconstruction fallback; full prediction result panel in `WebcamRecognition.jsx`.

---

## Next Steps (suggested)

- [ ] Rename `SkeletonViewer3D.jsx` → `SkeletonViewer.jsx`
- [ ] Remove dead code: `backend/src/extract_keypoints_for_segments.py`
- [ ] Improve gloss-to-sentence with an n-gram LM re-ranking step over the reconstruction candidates
- [ ] Write project report / demo for coursework submission
