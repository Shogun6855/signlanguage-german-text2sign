## Implementation progress

### Step 1 â€“ Backend scaffolding

- **Status**: Completed
- **Location**: `backend/`
- **What was implemented**:
  - Created backend project structure: `backend/` and `backend/src/`.
  - Added `backend/requirements.txt` with core backend dependencies: `fastapi`, `uvicorn[standard]`, `pydantic`, `numpy`.
  - Implemented minimal FastAPI app in `backend/src/main.py`:
    - `GET /api/health` returning `{"status": "ok"}` for health checks.
    - `POST /api/translate` accepting German sentence text and returning a **dummy** gloss/segment response:
      - `{"gloss": ["ICH1", "GEHEN1"], "segments": ["segment_1"]}`.
  - Added `backend/src/__init__.py` so `src` is a Python package.
- **How to test**:
  1. Activate your existing venv (with your installed libraries).
  2. From `backend/`, run `pip install -r requirements.txt`.
  3. Start the server with `python -m uvicorn src.main:app --reload`.
  4. Verify:
     - `GET http://127.0.0.1:8000/api/health` â†’ `{"status":"ok"}`.
     - `POST http://127.0.0.1:8000/api/translate` with `{"text": "Ich gehe nach Hause."}` â†’ dummy gloss/segment JSON.

### Step 2 â€“ Data extraction from DGS-Korpus sample (segments manifest)

- **Status**: Completed
- **Location**: `backend/src/prepare_segments.py`
- **What was implemented**:
  - A standalone script that reads the **real DGS-Korpus EAF file** from this new projectâ€™s dataset folder:
    - `.\dataset\1413451-11105600-11163240.eaf`
  - Uses two tiers (same names as the old project):
    - German translation tier: `Deutsche_Ãœbersetzung_B`
    - Gloss tier: `Lexem_GebÃ¤rde_r_A`
  - For each German translation segment:
    - Collects all gloss annotations whose time intervals overlap that German segment.
    - Builds a `Segment` with:
      - `id`: e.g. `"seg_0001"`
      - `german_text`: the German sentence/phrase
      - `start_ms`, `end_ms`: times from the EAF
      - `gloss_sequence`: list of gloss strings overlapping that interval.
  - Writes a JSON manifest to:
    - `backend/data/segments_manifest.json`
    - Top-level fields: `source_eaf`, `german_tier`, `gloss_tier`, `num_segments`, and `segments` list.
- **How to run and test**:
  1. Ensure your venv (same one used for step 1) is active and has `pympi` installed (it should already be present from the old project):
     - If needed: `pip install pympi-ling pandas`
  2. Ensure the real dataset files are placed in this project under:
     - `.\dataset\1413451-11105600-11163240.eaf`
     - `.\dataset\1413451-11105600-11163240_1a1.mp4` (used later for motion extraction)
  3. From the **backend** directory, run:
     ```powershell
     cd "D:\College\Semester 6\Lab\NLIP\Project\signlanguage-german-text2sign\backend"
     python -m src.prepare_segments
     ```
  4. Watch the console output for counts, e.g.:
     - `[segments] Loaded N German segments, M gloss segments`
     - `[segments] Built K segments with non-empty German text`
     - A printed â€œExample segmentâ€� with its German text and gloss sequence.
  5. Inspect the generated manifest (now created and verified):
     ```powershell
     type ".\data\segments_manifest.json"
     ```
     or open it in your editor to verify:
     - `segments[0].german_text` looks like a proper German sentence.
     - `segments[0].gloss_sequence` contains the expected gloss labels.


### Step 3 â€“ Keypoint / motion extraction for each segment

- **Status**: Completed
- **Location**:
  - `backend/src/data/mediapipe_keypoints.py`
  - `backend/src/extract_keypoints_for_segments.py`
- **What was implemented**:
  - Reused the existing **MediaPipeâ†’134-D feature mapping** from the old project in `mediapipe_keypoints.py`:
    - Converts MediaPipe Holistic landmarks into a 134-D vector:
      - BODY_25 (25Ã—2) + left hand (21Ã—2) + right hand (21Ã—2).
  - Added an offline extractor script `extract_keypoints_for_segments.py` that:
    - Loads `backend/data/segments_manifest.json` (from Step 2).
    - Uses `dataset/1413451-11105600-11163240_1a1.mp4` as the **primary video** for keypoints.
    - Runs MediaPipe Holistic over the whole video and builds a `[num_frames, 134]` keypoint array.
    - For each segment in the manifest:
      - Converts `start_ms` / `end_ms` into frame indices using the video FPS.
      - Slices the keypoint array for that frame range.
      - Saves a per-segment motion file:
        - `backend/data/motion/<segment_id>.npz` with:
          - `keypoints`: `[T, 134]`
          - `fps`: scalar
          - `start_ms`, `end_ms`
- **How to run and test**:
  1. Ensure your venv is active and has `mediapipe` + `opencv-python` installed:
     ```powershell
     cd "D:\College\Semester 6\Lab\NLIP\Project\signlanguage-german-text2sign\backend"
     pip install -r requirements.txt
     ```
  2. Confirm the dataset files exist (already done in your case):
     - `..\dataset\1413451-11105600-11163240.eaf`
     - `..\dataset\1413451-11105600-11163240_1a1.mp4`
  3. Run the extractor:
     ```powershell
     python -m src.extract_keypoints_for_segments
     ```
     You should see logs like:
     - `[motion] Loaded manifest from ...segments_manifest.json with 105 segments`
     - `[extract] processed=250 frames (stride=1)` (repeated)
     - `[extract] full video keypoints shape=(..., 134) fps=...`
     - `[motion] saved seg_0001.npz frames=...` for a few early segments.
  4. Inspect a sample motion file:
     ```powershell
     dir ".\data\motion"
     ```
     and optionally in Python:
     ```powershell
     python -c "import numpy as np; d=np.load(r'data/motion/seg_0001.npz'); print(d['keypoints'].shape, d['fps'])"
     ```
     (run from `backend/`).

### Step 4 â€“ German text â†’ gloss / segment mapping and motion API

- **Status**: Implemented (backend-side)
- **Location**:
  - `backend/src/text_to_gloss_map.py`
  - `backend/src/main.py`
- **What was implemented**:
  - A simple **German text â†’ gloss / segment lookup**:
    - Loads `backend/data/segments_manifest.json`.
    - Normalises German text (lowercase, strips punctuation but keeps umlauts).
    - Tokenises both input and manifest `german_text`.
    - Computes a Jaccard-like token overlap score and picks the **best-matching segment**.
    - Returns that segmentâ€™s `gloss_sequence` and `id`.
  - Integrated this mapper into the FastAPI app:
    - `POST /api/translate` now:
      - Accepts `{ "text": "<German sentence>" }`.
      - Returns `{ "gloss": [...], "segments": ["seg_XXXX"] }` based on the manifest lookup (no more dummy data).
  - Exposed motion data via a new endpoint:
    - `GET /api/motion/{segment_id}`:
      - Loads `backend/data/motion/<segment_id>.npz`.
      - Returns JSON with:
        - `keypoints`: `[[...134 floats per frame...], ...]`
        - `fps`: float
        - `start_ms`, `end_ms`: ints
- **How to run and test**:
  1. Start the backend API (same as Step 1):
     ```powershell
     cd "D:\College\Semester 6\Lab\NLIP\Project\signlanguage-german-text2sign\backend"
     python -m uvicorn src.main:app --reload
     ```
  2. In another terminal, test **text â†’ gloss / segment**:
     ```powershell
     curl -X POST "http://127.0.0.1:8000/api/translate" ^
       -H "Content-Type: application/json" ^
       -d "{\"text\": \"Wie mein Leben aussieht?\"}"
     ```
     Expected: a JSON response whose `segments` list contains `"seg_0001"` (or similar best match) and `gloss` matches that segmentâ€™s `gloss_sequence` from the manifest.
  3. Test **motion retrieval** for a known segment (e.g. `seg_0001`):
     ```powershell
     curl "http://127.0.0.1:8000/api/motion/seg_0001"
     ```
     Expected: JSON with `keypoints` (array of frames), `fps`, `start_ms`, and `end_ms`.  
     You can also quickly check via Python:
     ```powershell
     python - << "PY"
     import requests
     r = requests.get("http://127.0.0.1:8000/api/motion/seg_0001")
     print(r.status_code)
     data = r.json()
     print("frames:", len(data["keypoints"]), "fps:", data["fps"])
     PY
     ```

### Step 5 – React frontend (text input + animation)

- **Status**: Completed and verified (tested with 5 different sentences, no anomalies — March 4, 2026)
- **Location**:
  - `web/package.json`
  - `web/vite.config.mts`
  - `web/index.html`
  - `web/src/main.jsx`
  - `web/src/App.jsx`
  - `web/src/style.css`
  - `web/src/SkeletonViewer3D.jsx`
- **What was implemented**:
  - A small **Vite + React** SPA for the text-to-sign demo:
    - Dark, modern layout with two panels:
      - Left: German text input + “Translate” button + gloss/segment display.
      - Right: 2D skeleton canvas animation driven by motion data.
  - The app calls the backend API:
    - `POST http://127.0.0.1:8000/api/translate` with `{ text }`.
    - On success, shows the returned gloss sequence and segment IDs.
    - Fetches `GET http://127.0.0.1:8000/api/motion/{segment_id}` for the first segment.
  - Full 2D **skeleton animation** in `SkeletonViewer3D.jsx`:
    - Uses the 134-D keypoint vector per frame (BODY_25 + left hand + right hand).
    - Draws full **BODY_25 skeleton edges** in white.
    - Draws **both hands** with per-finger colouring (thumb=red, index=orange, middle=yellow, ring=green, pinky=blue) including palm arch.
    - Auto-fits all joints across all frames via `computeBounds()` — no clipping.
    - Plays frames at the returned `fps` using `requestAnimationFrame`.
  - Backend CORS updated (`backend/src/main.py`) to allow requests from the Vite dev server (`http://localhost:5173`).
- **Verified (March 4, 2026)**:
  - Tested end-to-end with 5 different German sentences from the manifest.
  - Text matching, gloss display, segment ID display, and skeleton animation all working correctly.
  - No anomalies observed.
- **How to run**:
  1. Install frontend dependencies once:
     ```powershell
     cd "D:\College\Semester 6\Lab\NLIP\Project\signlanguage-german-text2sign\web"
     npm install
     ```
  2. Ensure the backend is running (from a separate terminal):
     ```powershell
     cd "D:\College\Semester 6\Lab\NLIP\Project\signlanguage-german-text2sign\backend"
     python -m uvicorn src.main:app --reload
     ```
  3. Start the React dev server:
     ```powershell
     cd "D:\College\Semester 6\Lab\NLIP\Project\signlanguage-german-text2sign\web"
     npm run dev
     ```
  4. Open `http://localhost:5173/` in your browser.


### Step 6 – Playback controls (play/pause, restart, scrubber, frame counter)

- **Status**: Completed and verified (March 4, 2026)
- **Location**:
  - `web/src/SkeletonViewer3D.jsx` (controls added to component)
  - `web/src/style.css` (`.player-controls`, `.ctrl-btn`, `.scrubber`, `.frame-counter` styles)
- **What was implemented**:
  - Added a **control bar** below the skeleton canvas:
    - **Restart button (⏮)**: resets to frame 0 and resumes playback.
    - **Play/Pause button (⏸/▶)**: toggles animation; button label updates to reflect state.
    - **Scrubber (range input)**: drag to any frame; canvas updates immediately on drag.
    - **Frame counter**: shows current frame and total, e.g. `32 / 97`.
  - Used `useRef` for the actual frame index and playing state (avoids stale closure
    issues in the `requestAnimationFrame` loop) and `useState` only for UI re-renders.
  - Also fixed a **rendering bug** where MediaPipe boundary-clamped joints (y ≈ 1.0)
    caused leg bones to stretch to the canvas bottom edge; joints at `y ≥ 0.999` are
    now treated as undetected and skipped.
- **Known limitation**: For segments where the signer is filmed upper-body only (e.g.
  `seg_0001`), MediaPipe detects lower-body joints in only a few frames — this is a
  data issue (extraction), not a rendering bug. Accepted for now.
- **Verified (March 4, 2026)**:
  - Play/pause, restart, and scrubber all working correctly.
  - Frame counter updates in sync with animation and scrubber.


### Step 7 – 3D/animatronic readiness (motion format documentation)

- **Status**: Completed (March 4, 2026)
- **Location**: `backend/README.md`
- **What was implemented**:
  - Created `backend/README.md` documenting:
    - All three API endpoints with request/response examples.
    - Full **134-D joint schema** with named indices for BODY_25 and both MediaPipe hands.
    - **3D avatar interoperability section** covering:
      - How to extend the schema from 2D (x,y) to 3D (x,y,z) with minimal API changes.
      - Bone target mapping for humanoid rigs (Three.js / Babylon.js / IK chains).
      - Quaternion/rotation extension path using per-bone computation on the client.
    - Data file inventory and re-extraction instructions.
  - No code changes required — the existing `[T, 134]` format is already
    structured to be consumed by a 3D renderer as-is.

### Step 8 — Root README.md
**Status:** Completed

- Created `README.md` at project root
- Covers project overview, architecture diagram (directory tree), prerequisites
- Backend setup: venv creation, `pip install -r requirements.txt`, uvicorn start command
- Frontend setup: `npm install`, `npm run dev` (includes PowerShell workaround)
- End-to-end usage walkthrough with playback controls description
- Keypoint format table (body / left hand / right hand ranges)
- Re-extraction instructions (`extract_hands.py`)
- Known limitations section (105 sentences, lower-body detection, single-segment playback)
- References `german_sentences.md` for full sentence list and `backend/README.md` for joint schema

### Step 9 -- Improved coverage and word-level animation (March 4, 2026)

**Status:** Completed

#### 9a -- Neural sentence-level matching (sentence-transformers)
- Replaced Jaccard token overlap with `paraphrase-multilingual-MiniLM-L12-v2`
  (multilingual semantic sentence embeddings via `sentence-transformers`).
- All 105 segment texts are embedded at server startup and stored in memory.
- Inference: single cosine-similarity matrix multiply (L2-normalised vectors).
- GPU auto-detection added: if a CUDA-capable dGPU is present,
  `SentenceTransformer(model, device="cuda")` is used automatically;
  falls back to CPU otherwise. Detection logged on startup.
- `requirements.txt` updated with `sentence-transformers>=2.6.0`.
- Graceful fallback to Jaccard if `sentence-transformers` is not installed.

#### 9b -- Word-level greedy segment chaining
- New method `translate_chained(text)` in `text_to_gloss_map.py`:
  - Extracts content words (German stopword list applied).
  - Greedy loop: repeatedly picks the segment with the highest overlap with
    remaining unmatched content tokens; marks those tokens as covered.
  - Re-orders selected segments by the position of their first matched word
    in the original input.
  - Returns combined gloss sequence and ordered segment IDs.
- `POST /api/translate` accepts `{ "text": "...", "chained": true/false }`.
- Frontend: "Kettenmodus" checkbox enables chaining mode.
  Multiple segments shown with a green "N verkettet" badge.

#### 9c -- Gloss-level clip dictionary (true word-level animation)
- New script `backend/src/build_gloss_dictionary.py`:
  - Parses all 741 gloss annotations from the EAF `TIME_SLOT` table.
  - For each gloss, finds the parent segment npz and slices the exact frame
    range: `f1 = (gloss_start - seg_start) * fps / 1000`.
  - Saves each clip as `backend/data/gloss_clips/<GLOSS>_N.npz`.
  - Writes `backend/data/gloss_dictionary.json` (325 unique glosses, 731 clips).
  - Skips clips shorter than 2 frames (bad EAF boundary annotations, 10 total).
- New endpoint: `POST /api/motion/by_glosses`
  - Accepts `{ "glosses": ["SEHEN1*", "LEBEN1A*", ...] }`.
  - Concatenates per-gloss keyframe slices into one animation.
  - Missing glosses skipped and logged server-side.
- Frontend priority logic (App.jsx):
  1. Try `/api/motion/by_glosses` -- true gloss-level animation (default).
  2. Falls back to `/api/motion/chained` if no clips found.
  3. Mode badge: blue "Gloss-Ebene" or orange "Segment-Ebene".
- Result: "Wie mein Leben aussieht?" -> 5 glosses -> 43 frames / 0.9s
  (was 97 frames / 1.9s full-segment before).

### Step 10 — Gloss-to-Sentence Prediction from Live Recognition (March 11, 2026)

**Status:** Completed

#### 10a — Backend: `POST /api/gloss_to_sentence`
- New Pydantic models `GlossToSentenceRequest` / `GlossToSentenceResponse` in `backend/src/main.py`.
- New endpoint inside `create_app()` that converts a recognized gloss sequence into a proper German sentence using a **two-tier strategy**:
  1. **Retrieval** — normalizes each input gloss (strips `*`, `^`, `$` variant markers), then computes Jaccard set-similarity against every segment in `segments_manifest.json`. If best score ≥ 25%, returns that segment's German text.
  2. **Reconstruction** (fallback) — maps each gloss to the most frequently co-occurring German word in the corpus (built from paired segment gloss/text data), joins them into a readable sentence.
- Response fields: `predicted_sentence`, `confidence`, `method` (`"retrieval"` | `"reconstruction"`), `top_matches` (top-3 nearest segments), `gloss_word_map` (per-gloss `{gloss, normalized, word}`), `reconstruction`.
- Helper functions (all inside `create_app()` closure):
  - `_norm_gloss(g)` — strips DGS variant suffixes/prefixes.
  - `_gloss_to_lemma(g)` — converts a gloss label to a human-readable German word fragment.
  - `_load_manifest_segments()` — lazy-loads and caches the segment manifest for fast lookup.
  - `_build_gloss_word_map(manifest)` — builds reverse index: normalized_gloss → list of co-occurring German words.

#### 10b — Frontend: sentence prediction UI in `WebcamRecognition.jsx`
- New state variables: `predicting`, `prediction`, `predError`.
- `predictSentence()` callback: calls `POST /api/gloss_to_sentence` with the current `glossHistory`, stores response in `prediction`.
- **"🔤 Predict German Sentence"** button appears inside the gloss history panel once ≥ 1 gloss is collected; disabled during request.
- **Prediction result panel** shows:
  - Predicted sentence (large, bold)
  - Method badge (📚 Retrieval / 🔧 Reconstruction)
  - Match confidence percentage
  - Gloss → Word mapping (chip grid)
  - Word-for-word reconstruction (shown as alternative when retrieval wins)
  - Top-3 nearest corpus segments with their gloss chips and scores
- "Clear" button inside the gloss panel resets both history and prediction.
- State is cleared on `handleStart` and `handleModeSwitch`.

#### 10c — CSS: `web/src/style.css`
New classes added: `.predict-row`, `.btn-predict`, `.prediction-panel`, `.prediction-heading`, `.prediction-method-badge` (`.retrieval` / `.reconstruction`), `.prediction-sentence`, `.prediction-meta`, `.gloss-word-map`, `.gloss-word-map-rows`, `.gloss-word-row`, `.gwm-gloss`, `.gwm-arrow`, `.gwm-word`, `.prediction-alt`, `.top-matches`, `.top-match-row`, `.top-match-rank`, `.top-match-body`, `.top-match-german`, `.top-match-glosses`, `.top-match-score`.
