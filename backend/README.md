# DGS Text-to-Sign — Backend API

FastAPI service that translates German text to German Sign Language (DGS) skeleton animation data from the DGS-Korpus dataset.

---

## Running the server

```powershell
# From the project root — activate venv first
cd backend
python -m uvicorn src.main:app --reload
```

Runs on `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

---

## Endpoints

### `GET /api/health`
Health check.
```json
{ "status": "ok" }
```

---

### `POST /api/translate`
Fuzzy-matches a German sentence to the closest segment in the manifest.

**Request:**
```json
{ "text": "Wie mein Leben aussieht?" }
```

**Response:**
```json
{
  "gloss": ["SEHEN1*", "SELBST1A*", "LEBEN1A*", "SEHEN1", "$GEST-OFF^"],
  "segments": ["seg_0001"]
}
```

- `gloss`: DGS gloss sequence for the matched segment.
- `segments`: list of matched segment IDs (currently always 1 item).

---

### `GET /api/motion/{segment_id}`
Returns the full keypoint animation for a segment.

**Example:** `GET /api/motion/seg_0001`

**Response:**
```json
{
  "keypoints": [[...134 floats...], [...], ...],
  "fps": 25.0,
  "start_ms": 240,
  "end_ms": 2160
}
```

- `keypoints`: `T × 134` array — one 134-D vector per frame.
- `fps`: original video frame rate (used by the frontend player).
- `start_ms` / `end_ms`: source timestamp range in the original video.

---

## Keypoint / Joint Schema (134-D per frame)

Each frame is a flat array of 134 floats representing 2D joint coordinates:

```
Indices   Joints          Count   Description
───────────────────────────────────────────────────────────────────────
[0:50]    BODY_25 pose    25 × 2  OpenPose-compatible body joints (x,y)
[50:92]   Left hand       21 × 2  MediaPipe hand joints (x,y)
[92:134]  Right hand      21 × 2  MediaPipe hand joints (x,y)
```

All coordinates are **normalised to [0, 1] image space** (x=0 left, x=1 right, y=0 top, y=1 bottom).  
A joint stored as `(0.0, 0.0)` means **not detected** — consumers must skip these.

### BODY_25 joint indices (indices 0–24)

| Idx | Joint       | Idx | Joint       | Idx | Joint      |
|-----|-------------|-----|-------------|-----|------------|
| 0   | Nose        | 9   | R-Hip       | 18  | L-Ear      |
| 1   | Neck        | 10  | R-Knee      | 19  | L-BigToe   |
| 2   | R-Shoulder  | 11  | R-Ankle     | 20  | L-SmallToe |
| 3   | R-Elbow     | 12  | L-Hip       | 21  | L-Heel     |
| 4   | R-Wrist     | 13  | L-Knee      | 22  | R-BigToe   |
| 5   | L-Shoulder  | 14  | L-Ankle     | 23  | R-SmallToe |
| 6   | L-Elbow     | 15  | R-Eye       | 24  | R-Heel     |
| 7   | L-Wrist     | 16  | L-Eye       |     |            |
| 8   | Mid-Hip     | 17  | R-Ear       |     |            |

Flat array layout: `x_joint = frame[idx * 2]`, `y_joint = frame[idx * 2 + 1]`.

### Hand joint indices (MediaPipe, relative to hand offset)

Both hands follow the same 21-joint MediaPipe Hands layout:

| Idx | Landmark       | Idx | Landmark        |
|-----|----------------|-----|-----------------|
| 0   | Wrist          | 11  | Middle-PIP      |
| 1   | Thumb-CMC      | 12  | Middle-DIP      |
| 2   | Thumb-MCP      | 13  | Ring-MCP        |
| 3   | Thumb-IP       | 14  | Ring-PIP        |
| 4   | Thumb-Tip      | 15  | Ring-DIP        |
| 5   | Index-MCP      | 16  | Ring-Tip        |
| 6   | Index-PIP      | 17  | Pinky-MCP       |
| 7   | Index-DIP      | 18  | Pinky-PIP       |
| 8   | Index-Tip      | 19  | Pinky-DIP       |
| 9   | Middle-MCP     | 20  | Pinky-Tip       |
| 10  | Middle-MCP     |     |                 |

Left hand occupies flat indices `[50:92]` (joint `j` → `frame[50 + j*2]`, `frame[50 + j*2 + 1]`).  
Right hand occupies flat indices `[92:134]` (joint `j` → `frame[92 + j*2]`, `frame[92 + j*2 + 1]`).

---

## 3D Avatar Interoperability

The current format is **2D only** (`x, y` per joint). It is intentionally designed so that a future 3D renderer or robot controller can consume the same API with minimal changes:

### Adding a `z` dimension
Extend each joint from 2 floats to 3:
```
[0:75]    BODY_25 pose    25 × 3  (x, y, z)
[75:138]  Left hand       21 × 3
[138:201] Right hand      21 × 3
```
The API response shape changes from `[T, 134]` to `[T, 201]`. The frontend and 3D client both read the joint schema module to know which slice to use.

### Bone targets for a rigged 3D avatar
Each joint in the schema maps directly to a bone target in a humanoid rig:
- `NECK (1)` → root of upper body chain
- `MID_HIP (8)` → root of lower body chain
- `WRIST (4 / 7)` → wrist effector for IK
- Hand joints → finger bone chain targets (thumb 1–4, index 5–8, middle 9–12, ring 13–16, pinky 17–20)

A Three.js / Babylon.js client can:
1. Call `GET /api/motion/{segment_id}` to fetch the `[T, 134]` keypoint array
2. For each frame, iterate joints and set bone world positions from normalised coordinates
3. Optionally apply inverse kinematics (IK) using wrist + elbow + shoulder as chain endpoints

### Quaternion / rotation extension
To drive a bone-rotation–based avatar (e.g. SMPL-X, BVH), compute per-bone quaternions from adjacent joint positions at runtime on the client. The joint positions from this API are sufficient inputs for that computation — no server-side changes needed.

---

## Data files

| File | Description |
|------|-------------|
| `data/segments_manifest.json` | 105 segments: `id`, `german_text`, `gloss_sequence`, `start_ms`, `end_ms` |
| `data/motion/seg_XXXX.npz` | Per-segment keypoints: `keypoints [T,134]`, `fps`, `start_ms`, `end_ms` |

---

## Re-extracting keypoints

If you need to re-run MediaPipe extraction (e.g. after updating the dataset):

```powershell
cd backend
python extract_hands.py
```

Requires `dataset/1413451-11105600-11163240_1a1.mp4` at the project root level.
