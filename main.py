"""NL Video Scanner API.

POST /analyze   stream a 1-FPS scan of an uploaded video for a free-text query (NDJSON)
POST /index     caption an uploaded clip at 1 FPS and add it to the local retrieval index
GET  /search    natural-language top-k search over indexed clips (LlamaIndex + Redis)
GET  /clips/{clip_id}/frames   frame/time index lookup
GET  /metrics   Prometheus metrics

Inference runs in-process (Llama 3.2 11B Vision, 4-bit) behind scanner.vlm.BatchDispatcher.
SERVING_MODE=sequential reproduces the original serving path (one frame per model call, each
request's frames strictly one after another); SERVING_MODE=batched (default) lets frames from
all in-flight requests share batched forward passes.
"""
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from scanner import config
from scanner.winperf import disable_power_throttling
from scanner.pipeline import CAPTION_PROMPT, analyze_prompt, parse_analysis, run_frames

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
FRAMES_DIR = Path(os.getenv("FRAMES_DIR", "frames"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

BATCHED = config.SERVING_MODE == "batched"
MAX_BATCH = config.BATCH_MAX_SIZE if BATCHED else 1
WINDOW = config.STREAM_WINDOW if BATCHED else 1

LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 450, 600, 900, 1800)
INFERENCE_SECONDS = Histogram("scanner_inference_seconds", "Per-frame model latency incl. queueing",
                              ["mode"], buckets=LATENCY_BUCKETS)
SCAN_SECONDS = Histogram("scanner_scan_seconds", "End-to-end /analyze latency", ["mode"], buckets=LATENCY_BUCKETS)
SEARCH_SECONDS = Histogram("scanner_search_seconds", "/search latency",
                           buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2))
FRAMES_TOTAL = Counter("scanner_frames_total", "Frames analyzed", ["mode", "status"])
SCANS_TOTAL = Counter("scanner_scans_total", "Completed /analyze requests", ["mode"])
INFLIGHT = Gauge("scanner_inflight_inference", "Frames submitted and not yet answered", ["mode"])

state = {}


def build_dispatcher():
    if os.getenv("INFERENCE_BACKEND", "llama-vision") == "ollama":
        from scanner.ollama_backend import OllamaSubmit
        return OllamaSubmit(), "ollama:" + config.OLLAMA_MODEL
    from scanner.vlm import BatchDispatcher, VisionEngine
    engine = VisionEngine()
    d = BatchDispatcher(engine, MAX_BATCH, config.BATCH_WINDOW_MS)
    d.start()
    return d, engine.model_name


@asynccontextmanager
async def lifespan(_app):
    state["power"] = disable_power_throttling()
    state["dispatcher"], state["model"] = build_dispatcher()
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")
app.mount("/metrics", make_asgi_app())


def get_index():
    if "index" not in state:
        from scanner.index import CaptionIndex
        state["index"] = CaptionIndex()
    return state["index"]


async def submit(jpeg, prompt, max_tokens):
    mode = config.SERVING_MODE
    INFLIGHT.labels(mode).inc()
    start = time.perf_counter()
    try:
        return await state["dispatcher"].submit(jpeg, prompt, max_tokens)
    finally:
        INFLIGHT.labels(mode).dec()
        INFERENCE_SECONDS.labels(mode).observe(time.perf_counter() - start)


async def save_upload(upload: UploadFile, task_id):
    filename = Path(upload.filename or "upload.mp4").name
    path = UPLOAD_DIR / f"{task_id}_{filename}"
    data = await upload.read()
    await asyncio.to_thread(path.write_bytes, data)
    return path


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {
        "model": state.get("model"),
        "serving_mode": config.SERVING_MODE,
        "max_batch": MAX_BATCH,
        "batch_window_ms": config.BATCH_WINDOW_MS,
        "stream_window": WINDOW,
        "model_max_side": config.MODEL_MAX_SIDE,
        "max_image_tiles": int(os.getenv("VLM_MAX_TILES", "4")),
        "analyze_max_tokens": config.ANALYZE_MAX_TOKENS,
        "preprocess_mode": config.PREPROCESS_MODE,
        "process_power_hints": state.get("power"),
    }


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...), object_str: str = Form(...)):
    request_start = time.perf_counter()
    task_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    video_path = await save_upload(video, task_id)
    frames_dir = FRAMES_DIR / task_id
    frames_dir.mkdir(exist_ok=True)
    mode = config.SERVING_MODE

    async def generate():
        async for (second, _, stats, enhanced, _), res in run_frames(
                submit, video_path, analyze_prompt(object_str), config.ANALYZE_MAX_TOKENS,
                WINDOW, save_dir=frames_dir):
            if isinstance(res, Exception):
                FRAMES_TOTAL.labels(mode, "error").inc()
                is_match, description, confidence, usage, raw = False, "Error occurred", 0, {}, None
            else:
                raw, usage = res
                FRAMES_TOTAL.labels(mode, "ok").inc()
                is_match, description, confidence = parse_analysis(raw)
            yield json.dumps({"status": "success", "frame": {
                "second": second, "is_match": is_match, "description": description,
                "confidence": confidence, "frame_path": f"/frames/{task_id}/frame_{second}.jpg",
                "enhanced": enhanced, "lighting": stats, "usage": usage, "raw_response": raw,
            }}) + "\n"
        SCANS_TOTAL.labels(mode).inc()
        SCAN_SECONDS.labels(mode).observe(time.perf_counter() - request_start)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/index")
async def index_video(video: UploadFile = File(...), clip_id: str = Form(None)):
    task_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    video_path = await save_upload(video, task_id)
    clip_id = clip_id or Path(video.filename or task_id).stem
    frames_dir = FRAMES_DIR / task_id
    frames_dir.mkdir(exist_ok=True)

    async def generate():
        frames = []
        async for (second, _, stats, enhanced, frame_path), res in run_frames(
                submit, video_path, CAPTION_PROMPT, config.CAPTION_MAX_TOKENS, WINDOW, save_dir=frames_dir):
            caption = "" if isinstance(res, Exception) else res[0].strip()
            frames.append({"t": second, "caption": caption, "stats": stats, "enhanced": enhanced,
                           "frame_path": f"/frames/{task_id}/frame_{second}.jpg"})
            yield json.dumps({"status": "frame", "t": second, "caption": caption, "enhanced": enhanced}) + "\n"
        await asyncio.to_thread(get_index().add_clip, clip_id, frames,
                                {"video": str(video_path), "task_id": task_id})
        yield json.dumps({"status": "indexed", "clip_id": clip_id, "frames": len(frames)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/search")
async def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=100)):
    start = time.perf_counter()
    results, timings = await asyncio.to_thread(get_index().search, q, k)
    SEARCH_SECONDS.observe(time.perf_counter() - start)
    return {"query": q, "k": k, "results": results, "timings_ms": timings}


@app.get("/clips/{clip_id}/frames")
async def clip_frames(clip_id: str, t0: float = 0, t1: float = 1e9):
    frames = await asyncio.to_thread(get_index().frames_between, clip_id, t0, t1)
    if not frames:
        return JSONResponse(status_code=404, content={"error": f"no frames for {clip_id}"})
    return {"clip_id": clip_id, "frames": frames}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
