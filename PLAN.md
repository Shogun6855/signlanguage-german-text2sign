---
name: german-text-to-sign-language-webapp
overview: Design and implement a German sentence–to–German Sign Language (DGS) web app using DGS-Korpus data, with an initial 2D/3D-friendly keypoint animation pipeline that can later drive a 3D avatar.
---

## High-level approach

- **Goal**: Build a **web app** that takes **proper German sentences** as input and outputs a **German Sign Language sequence**, visualized as an animation that is compatible with a future 3D avatar.
- **Key idea**: Internally use a **German text → DGS gloss → sign-motion** pipeline, reusing the DGS-Korpus sample (1a1). We’ll first get a robust **keypoint-based sign animation** in the browser, and keep the representation 3D-avatar-friendly (e.g. normalized joint trajectories) so you can later plug in a real 3D model.
- **Scope**: Keep it focused around the existing annotated sample (and closely related phrases), so the project is realistically completable with your current data.

## Architecture outline

- **Backend (`text-to-sign-api`)**
  - Python (FastAPI) service running in a new folder, e.g. `[signlanguage-german-text2sign/backend]`.
  - Exposes REST endpoints:
    - `POST /api/translate` – **input: German sentence text**; **output: DGS gloss sequence + sign-motion token(s)** (e.g. IDs of known signed sentences or per-gloss motion descriptors).
    - `GET /api/motion/{id}` – returns a compact sign-motion representation (e.g. normalized keypoint trajectories) for a given segment.
  - Uses models / rules:
    - A **German text → DGS gloss** component (rule-based or small model) tuned to your limited domain.
    - A **gloss → motion** lookup using precomputed motion units from DGS-Korpus.
- **Data + model preparation (`data/` + `models/`)**
  - New preprocessing scripts in `[signlanguage-german-text2sign/backend/src/]` that:
    - Parse German text + EAF annotations from the DGS-Korpus sample.
    - Segment the video into **signing units** (per-gloss or per-phrase).
    - Extract **2D keypoint trajectories** (using your existing MediaPipe/OpenPose experience) and store them in a normalized format that’s easy to render in web.
  - Optional small **German text → gloss mapping model** (e.g. simple sequence-to-sequence or lookup-based using your existing gloss2text transformer reversed / fine-tuned).
- **Frontend (`web-ui`)**
  - New React (or minimal JS) single-page app in `[signlanguage-german-text2sign/web]`.
  - Features:
    - Input text box for **German sentence**.
    - “Translate to DGS” button.
    - Canvas/WebGL-based **sign animation player** that:
      - Calls `/api/translate` to get gloss + motion IDs.
      - Calls `/api/motion/{id}` (or receives motion inline) and renders a skeleton animation (2D first, but joint structure chosen to be mappable to 3D later).
  - Optional: show gloss sequence and timing below the animation for debugging/learning.
- **Representation for future 3D avatar**
  - Define a simple, explicit **joint schema** (e.g. upper body + both hands, similar to BODY_25 + hands) and a normalized coordinate system (e.g. root at mid-hip, units in body-lengths).
  - Store motion as:
    - `frames × joints × (x,y)` (2D) now, but with room to add a `z` dimension later.
  - Frontend uses this schema for 2D drawing; a future 3D avatar can consume the same schema as joint targets.

## Step-by-step implementation plan

### 1. Project scaffolding (new folder)

- **Create new project structure** (no code changes to the old repo):
  - `[signlanguage-german-text2sign/]`
    - `backend/` – FastAPI service, data scripts, models.
    - `web/` – frontend app.
    - `README.md` – overview, setup commands, and usage.
- **Set up Python environment** for backend:
  - `requirements.txt` with `fastapi`, `uvicorn`, `pydantic`, `numpy`, `torch` (if needed), `mediapipe`/`opencv-python` for preprocessing only.

### 2. Data extraction from DGS-Korpus sample

- **Reuse existing assets from the old project** (without modifying them):
  - Copy or sym-link:
    - `dataset/1413451-11105600-11163240_1a1.mp4`
    - `dataset/1413451-11105600-11163240.eaf`
- **Implement a data script** (e.g. `[backend/src/prepare_segments.py]`) to:
  - Parse EAF to obtain **aligned gloss and German text segments** for the video.
  - Decide on segment units (e.g. full sentence vs multiple clauses) and assign each a unique `segment_id`.
  - Save a simple manifest JSON (e.g. `[backend/data/segments_manifest.json]`) listing:
    - `segment_id`
    - `german_text`
    - `gloss_sequence`
    - `start_time_ms`, `end_time_ms`

### 3. Sign-motion representation (2D, 3D-friendly)

- **Keypoint extraction** (offline, one-time per dataset sample):
  - Implement `[backend/src/extract_keypoints_for_segments.py]` that:
    - Runs MediaPipe/OpenPose over the video.
    - Resamples frames to a stable fps (e.g. 25fps).
    - For each segment in the manifest, slices the correct frame range.
    - Normalizes coordinates (e.g. root at mid-hip, scale by shoulder distance).
    - Outputs: `[backend/data/motion/{segment_id}.npz]` with `frames × joints × 2` and meta-info (fps, joint schema).
- **Define a joint schema module** `[backend/src/joint_schema.py]` that documents joint indices and will be shared with the frontend animation code for consistency.

### 4. German text → gloss → motion mapping

- **Start with rule/lookup-based approach** (given limited data):
  - Implement `[backend/src/text_to_gloss_map.py]` that:
    - For now, maps a small set of **German input sentence patterns** to known gloss sequences / segment IDs from the manifest.
    - Example: simple string or token pattern matching to decide which segment(s) best match the input sentence.
- **API contract** (even if initially backed by rules):
  - Function `translate_text_to_sign(text: str) -> {gloss: List[str], segments: List[segment_id]}`.
  - Keep this clean so you can later swap in a learned German text→gloss model (e.g. fine-tuned sequence-to-sequence) without changing the frontend.

### 5. Backend API (FastAPI service)

- Implement FastAPI app in `[backend/src/main.py]`:
  - `POST /api/translate`:
    - Request: `{ "text": "<German sentence>" }`.
    - Response: `{ "gloss": [...], "segments": [...segment_ids...] }`.
  - `GET /api/motion/{segment_id}`:
    - Loads the corresponding `motion/{segment_id}.npz`.
    - Returns a compact JSON (possibly time-compressed) with:
      - `fps`
      - `joints_schema` (names or indices)
      - `frames`: list of joint coordinates.
- Add CORS configuration to allow the web frontend to call the API from `localhost` during development.

### 6. Frontend web app (text input + animation)

- **Scaffold frontend** in `[web/]`:
  - Minimal React + Vite (or Create React App) project.
  - Install dependencies: `axios` (or `fetch`), and a simple drawing library or just native Canvas.
- **Implement UI components**:
  - `TextInputPanel` – text box + submit button, calls `/api/translate` with a **German sentence**.
  - `GlossDisplay` – displays returned gloss sequence.
  - `SignPlayer` – canvas component that:
    - Fetches motion JSON for the selected `segment_id`.
    - Renders a 2D skeleton over time by looping through frames at the given fps.
    - Uses the shared joint schema (imported as a small JSON or TS module generated from backend’s `joint_schema.py`).
- **Playback controls**: play/pause, restart, simple progress indicator.

### 7. 3D/animatronic readiness

- While keeping the current implementation 2D, ensure interop with future 3D avatar:
  - Design the motion JSON so that a 3D client can:
    - Interpret each joint as a bone target.
    - Optionally add `z` coordinates and rotation/quaternion info later.
  - Document this format in `[backend/README.md]` so a future 3D renderer (e.g. Three.js avatar or a robot controller) can consume the same API.

### 8. Packaging, docs, and run commands

- Add detailed run instructions in the new root `README.md`, including:
  - **Backend**:
    - Create venv, install `requirements.txt`.
    - Run preprocessing scripts (`prepare_segments.py`, `extract_keypoints_for_segments.py`).
    - Start API with `uvicorn src.main:app --reload`.
  - **Frontend**:
    - `npm install` / `npm run dev`.
    - How to configure API base URL (e.g. environment variable or config file).
  - Simple end-to-end usage steps with screenshots or descriptions.

### 9. Optional enhancements (after MVP)

- Improve German text→gloss coverage by:
  - Extending to more DGS-Korpus samples with similar pipeline.
  - Training a small neural German text→gloss model on top of the manifest entries.
- Add user options in the web UI to:
  - Switch between 2D stick figure vs original video snippet (for teaching).
  - Show side-by-side: real signer video + skeleton overlay (debug mode).
- Explore 3D avatar linkage by:
  - Exporting motion JSON into a small Three.js demo that drives a rigged 3D model.

