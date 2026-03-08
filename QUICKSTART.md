# DGS Text-to-Sign — Quick Start Guide

German Sign Language NLP system: translate German text into DGS sign animations,
explore the full NLP pipeline (tokenization → POS tagging → N-gram LM → feature
engineering), and recognise signs live from your webcam.

---

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.10 or 3.11 |
| Node.js | 18 LTS or newer |
| npm | 9 or newer |
| Git | any recent version |
| OS | Windows 10/11, macOS 12+, or Ubuntu 22.04+ |

> **GPU is not required.** Everything runs on CPU.

---

## 1  Clone the repository

```bash
git clone <repo-url>
cd signlanguage-german-text2sign
```

---

## 2  Backend setup (Python)

### 2a  Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If you see a script-execution-policy error on Windows, run this first in an
> **Administrator** PowerShell, then re-open a normal one:
> ```powershell
> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2b  Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

This installs FastAPI, MediaPipe, NLTK, scikit-learn, sentence-transformers,
and all other backend dependencies (~1 GB download on first run due to model weights).

### 2c  Download NLTK data (one-time)

```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
"
```

### 2d  Verify model files are present

The MediaPipe model files must exist at:

```
backend/models/hand_landmarker.task
backend/models/pose_landmarker_lite.task
```

If they are missing (e.g. not tracked by git), download them:

```bash
# hand landmarker
curl -L -o backend/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# pose landmarker (lite)
curl -L -o backend/models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

### 2e  Confirm the data files are present

```
backend/data/segments_manifest.json   ← gloss dictionary (required)
backend/data/motion/seg_XXXX.npz      ← keypoint clips (460 files, required)
```

If `segments_manifest.json` is missing, the backend will start but translation
will return empty results.

---

## 3  Start the backend

Open a terminal, activate the virtual environment, then:

```bash
cd backend
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Expected output (after a few seconds of model loading):

```
[EmbeddingMatcher] Ready -- 460 embeddings (dim=384) on cpu
[startup] Gloss dictionary loaded: 1002 unique glosses
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Leave this terminal running.

---

## 4  Frontend setup (Node.js)

Open a **second** terminal:

```bash
cd web
npm install
npm run dev
```

Expected output:

```
VITE v5.x.x  ready in ~500 ms
  ➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173/** in your browser.

---

## 5  Using the application

The interface has three tabs:

| Tab | What it does |
|---|---|
| ✍️ **Text → Gebärde** | Type German text → get gloss sequence → watch 3D skeleton animation |
| 📷 **Live-Erkennung** | Click **Start** → sign in front of your webcam → real-time gloss recognition appears |
| 🔬 **NLP-Pipeline** | Type any German sentence → see all NLP stages: tokenization, POS tags, TF-IDF, N-gram LM |

---

## 6  API endpoints (optional / for testing)

With the backend running you can test endpoints directly:

```bash
# Full NLP analysis
curl -s -X POST http://127.0.0.1:8000/api/nlp/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Ich lerne Gebaerdensprache."}' | python -m json.tool

# Gloss language model stats
curl -s http://127.0.0.1:8000/api/nlp/gloss_lm | python -m json.tool

# Translate text to glosses
curl -s -X POST http://127.0.0.1:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Wie geht es Ihnen?"}' | python -m json.tool
```

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## 7  Generate the project report (optional)

```bash
# from the repo root, with the virtual environment active
python generate_report.py
```

Produces `team12_project.docx` in the repo root.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Make sure you run uvicorn **from inside the `backend/` directory** |
| `npm: cannot be loaded` (PowerShell) | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as admin, or use `node "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" run dev` |
| Backend port 8000 already in use | `lsof -ti:8000 \| xargs kill` (macOS/Linux) or `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` (Windows) |
| Webcam not detected | Grant camera permission in your browser; only one app can use the webcam at a time |
| `hand_landmarker.task` not found | Download the model file per Step 2d above |
| Low sign recognition accuracy | Ensure good lighting and hold each sign steady for ~0.5 s |
| NLTK `LookupError` | Re-run Step 2c to download missing NLTK data packages |

---

## Project structure (quick reference)

```
signlanguage-german-text2sign/
├── backend/
│   ├── requirements.txt          ← Python dependencies
│   ├── models/                   ← MediaPipe .task files (not in git if large)
│   ├── data/
│   │   ├── segments_manifest.json
│   │   └── motion/seg_XXXX.npz   ← 460 keypoint clip files
│   └── src/
│       ├── main.py               ← FastAPI app + all endpoints
│       ├── nlp_pipeline.py       ← Lab1 + Feature Eng + Ex3 + Ex4
│       └── text_to_gloss_map.py  ← German text → DGS gloss mapper
├── web/
│   ├── package.json
│   └── src/
│       ├── App.jsx               ← 3-tab shell
│       ├── SkeletonViewer3D.jsx  ← Three.js animation viewer
│       ├── WebcamRecognition.jsx ← Live sign recognition UI
│       └── NLPAnalysisPanel.jsx  ← NLP pipeline visualisation
├── generate_report.py            ← Generates team12_project.docx
└── QUICKSTART.md                 ← This file
```
