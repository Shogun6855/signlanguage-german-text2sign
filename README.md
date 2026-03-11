# DGS Text-to-Sign — German Text to German Sign Language

A web app that takes a **German sentence** as input and renders a **German Sign Language (DGS) skeleton animation** using real motion data from the DGS-Korpus dataset.

---

## Demo

1. Type a German sentence (or pick one from the corpus)
2. Click **"In Gebärdensprache übersetzen"**
3. Watch the 2D skeleton animate the corresponding DGS signing motion — body + both hands with per-finger colour coding

---

## Architecture

```
signlanguage-german-text2sign/
├── .venv/                        ← Python 3.10 virtual environment
├── backend/                      ← FastAPI REST API
│   ├── src/
│   │   ├── main.py               ← API endpoints (/translate, /motion)
│   │   ├── text_to_gloss_map.py  ← fuzzy German text → segment matcher
│   │   └── prepare_segments.py   ← EAF parser → segments_manifest.json
│   ├── data/
│   │   ├── segments_manifest.json  ← 105 segments with German text & glosses
│   │   └── motion/seg_XXXX.npz   ← per-segment MediaPipe keypoints [T×134]
│   ├── extract_hands.py           ← MediaPipe re-extraction script
│   ├── requirements.txt
│   └── README.md                  ← API + joint schema documentation
├── dataset/
│   ├── 1413451-11105600-11163240.eaf         ← ELAN annotation file
│   └── 1413451-11105600-11163240_1a1.mp4    ← main signer video (required)
└── web/                           ← React + Vite frontend
    ├── src/
    │   ├── App.jsx                ← main component, calls backend API
    │   ├── SkeletonViewer3D.jsx   ← 2D canvas skeleton renderer with playback controls
    │   └── style.css
    └── package.json
```

---

## Setup

### Prerequisites

- Python 3.10
- Node.js 18+ and npm

### 1. Python virtual environment

```powershell
cd signlanguage-german-text2sign
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
```

### 2. Frontend dependencies

```powershell
cd web
npm install
```

---

## Running

Open **two terminals**:

**Terminal 1 — Backend:**
```powershell
cd backend
python -m uvicorn src.main:app --reload
```
Runs on `http://127.0.0.1:8000`

**Terminal 2 — Frontend:**
```powershell
cd web
npm run dev        # if execution policy allows
# or:
cmd /c "npm run dev"
```
Runs on `http://localhost:5173`

Open `http://localhost:5173` in your browser.

---

## Usage

1. The default sentence **"Wie mein Leben aussieht?"** is pre-filled.
2. Click **"In Gebärdensprache übersetzen"**.
3. The left panel shows the matched **Segment ID** and **Gloss sequence**.
4. The right panel shows the **skeleton animation** playing automatically.
5. Use the **playback controls** below the canvas:
   - ⏮ Restart — jumps to frame 1 and resumes
   - ⏸/▶ Play/Pause — toggles animation
   - Scrubber — drag to any frame
   - Frame counter — shows current / total frames

### Live Sign Recognition → Sentence Prediction

Open the **Live-Erkennung** tab:
1. Click **▶ Start Camera & Begin Recognition** (or upload a video file).
2. Hold signs in front of the camera — detected glosses appear as chips in the *Detected Gloss Sequence* strip.
3. Once the strip contains at least one gloss, click **🔤 Predict German Sentence**.
4. The system queries `POST /api/gloss_to_sentence` and shows:
   - **Predicted sentence** (corpus retrieval when Jaccard ≥ 25%, word reconstruction otherwise)
   - **Match confidence** and **method badge** (📚 Retrieval / 🔧 Reconstruction)
   - **Gloss → Word mapping** table
   - **Top-3 nearest corpus segments**

---

## Keypoint Format

Each animation frame is a flat array of **134 floats**:

| Range | Content | Joints |
|-------|---------|--------|
| `[0:50]` | Body pose (BODY_25) | 25 joints × (x, y) |
| `[50:92]` | Left hand (MediaPipe) | 21 joints × (x, y) |
| `[92:134]` | Right hand (MediaPipe) | 21 joints × (x, y) |

All coordinates are normalised `[0, 1]`. A joint at `(0, 0)` means undetected.

See [`backend/README.md`](backend/README.md) for the full joint schema and 3D avatar interoperability notes.

---

## Re-extracting Keypoints

If you need to re-run MediaPipe extraction from the source video:

```powershell
cd backend
python extract_hands.py
```

Requires `dataset/1413451-11105600-11163240_1a1.mp4` to be present.

---

## Known Limitations

- Only 460 segments from multiple DGS-Korpus recordings — unmatched input falls back to the closest token match.
- Lower-body joints are partially detected in some segments (upper-body filming angle).
- Gloss-to-sentence prediction uses Jaccard set similarity; accuracy improves when the signer uses glosses that appear in the training corpus.
- Live recognition accuracy depends on lighting and sign clarity — hold each sign steady for ∼ 0.5 s.

---

## Dataset

Data sourced from the **DGS-Korpus** (Hamburg, Germany). The ELAN annotation file and video (`1413451-11105600-11163240`) are not included in the repository and must be obtained separately.
