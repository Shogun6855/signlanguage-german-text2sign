from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.text_to_gloss_map import get_mapper


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


class ChainedMotionRequest(BaseModel):
    segment_ids: list[str]


class GlossByNameRequest(BaseModel):
    glosses: list[str]  # ordered list of gloss labels e.g. ["LEBEN1A*", "ICH1"]


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
        import numpy as np

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

        combined = np.concatenate(all_clips, axis=0)  # [T_total, 134]
        fps = fps or 50.0
        return MotionResponse(
            keypoints=combined.tolist(),
            fps=fps,
            start_ms=0,
            end_ms=int(combined.shape[0] / fps * 1000),
        )

    @app.post("/api/motion/chained", response_model=MotionResponse)
    async def get_chained_motion(req: ChainedMotionRequest) -> MotionResponse:
        """Concatenate keypoints from multiple segments into a single animation."""
        import numpy as np

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

        combined = np.concatenate(all_keypoints, axis=0)  # [T_total, 134]
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

        import numpy as np

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

