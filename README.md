# FrameSift

Sift through video, frame by frame, in plain English. FrameSift is a fully local natural-language video scanner: nothing leaves the machine at runtime.

- **Sampling:** video is sampled at exactly 1 FPS with OpenCV.
- **Preprocessing:** difficult frames (low light, blown highlights) can be enhanced first.
- **Vision model:** Llama 3.2 11B Vision answers questions about each frame or captions it for search.
- **Search:** captions go into a LlamaIndex retrieval layer backed by a local Redis Stack (NoSQL) instance, so you can search a clip library in plain English and get back the best-matching clips with timestamps.

## Where it started

https://github.com/user-attachments/assets/169077b9-def4-4864-88fd-dc9d940ad0fd

The video shows the original prototype, built for a Llama Stack hackathon: upload a clip, name an object, and the model checks every second of footage for it. That object finder is still here: the web UI and `/analyze`.

Since then the project has grown in three directions:
- **Search:** a clip library you can search in plain English. Per-second captions are indexed in LlamaIndex + Redis (`/index`, `/search`, `/clips/{id}/frames`).
- **Evaluation:** a held-out evaluation on MSR-VTT with confidence intervals and paired significance tests. It is written up in [`docs/report.md`](docs/report.md), including the ideas that did not pan out.
- **Serving:** in-process model serving with a batching dispatcher, load tests, and a Prometheus + Grafana dashboard.

## Architecture

```
upload ──> 1-FPS sampler ──> adaptive preprocessing ──> BatchDispatcher ──> Llama 3.2 11B Vision
           (exact, float fps)  (denoise, gray-world,     (queue + batch      (bnb NF4, in-process,
                                CLAHE; only frames        window; batch 1     transformers + SDPA)
                                flagged by thresholds)    = original path)
                                                                 │
             /analyze  <── NDJSON stream of per-frame answers ───┤
                                                                 │ captions
             /search   <── LlamaIndex: dense frame (default) / dense clip / BM25, weighted RRF ──┐
             /clips/{id}/frames <── frame/time index                                             │
                                                       Redis Stack: HNSW vectors, clip/frame hashes,
                                                       per-clip time-sorted frame index
```

| Path | What it does |
|---|---|
| `main.py` | FastAPI app: `/analyze`, `/index`, `/search`, `/clips/{id}/frames`, `/health`, `/metrics` |
| `scanner/frames.py` | 1-FPS sampling, lighting stats (letterbox-aware), denoise / gray-world / CLAHE |
| `scanner/vlm.py` | Llama 3.2 Vision engine and the dynamic batching dispatcher |
| `scanner/index.py` | LlamaIndex + Redis caption index, frame/time index, fused top-k search |
| `scanner/pipeline.py` | Shared frame pipeline used by the API and the eval scripts |
| `scanner/winperf.py` | Opts the process out of Windows power throttling (see Notes) |
| `eval/` | MSR-VTT eval sets, captioning, Hit@k, fusion tuning, statistics, engine batch check |
| `loadtest/` | Concurrent-stream load test and before/after driver |
| `observability/` | Prometheus scrape config and the provisioned Grafana dashboard |
| `artifacts/` | Every number in `docs/report.md`, as the raw JSON the scripts wrote |
| `docs/report.md` | Evaluation report: protocol, results with confidence intervals, negative findings |

## Results

Full details are in [`docs/report.md`](docs/report.md). Test set: MSR-VTT, 320 held-out clips, 5,123 frames.

| | Value | 95% CI |
|---|---|---|
| Hit@5 / Hit@10 / MRR | 68.8% / 76.9% / 0.538 | Hit@5 [63.4, 73.8] |
| Hit@5 on low-light / high-contrast clips (n = 147) | 62.6% | [55.1, 70.1] |
| Effect of adaptive preprocessing on those clips (Hit@5) | +0.0 pp | [−4.1, +4.1] |
| Search latency (embed + HNSW, 5,123-frame index) | p50 ~40 ms, p99 < 60 ms | |
| Batched vs sequential serving, scan latency | +56% (1 stream) … +24% (4 streams) | |

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate          # Git Bash; see "Activating the venv" below for other shells
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
docker run -d --name scanner-redis -p 6379:6379 redis/redis-stack-server:7.4.0-v3

# Model weights, downloaded once (both are ungated):
huggingface-cli download unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit --local-dir .data/models/llama32-11b-vision-bnb4
huggingface-cli download BAAI/bge-base-en-v1.5 --local-dir .data/models/bge-base-en-v1.5
```

### Activating the venv

`activate` has to be *sourced* into the current shell. Running it as a plain command does nothing.

| Shell | Command |
|---|---|
| Git Bash | `source .venv/Scripts/activate` |
| PowerShell | `.venv\Scripts\Activate.ps1` (if scripts are blocked, first run `Set-ExecutionPolicy -Scope Process Bypass`) |
| cmd | `.venv\Scripts\activate.bat` |
| Any shell, no activation | `.venv/Scripts/python.exe -m <module> ...` |

When the venv is active, the prompt starts with `(.venv)`.

## Usage

All commands below are for Git Bash from the repo root. In PowerShell, write `curl.exe` instead of `curl`.

### Start the server

```bash
docker start scanner-redis                      # Redis must be up for /index, /search, /clips

# Full server: loads Llama 3.2 Vision on the GPU (~10 GB VRAM)
.venv/Scripts/python.exe -m uvicorn main:app --port 8000

# Search-only: no vision model and no GPU; /analyze and /index return 503
INFERENCE_BACKEND=none EMBED_DEVICE=cpu .venv/Scripts/python.exe -m uvicorn main:app --port 8001
```

Check that it is up with `curl -s localhost:8000/health`. Stop it with `Ctrl+C`.

Useful environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `INFERENCE_BACKEND` | `llama-vision` | `none` = search-only, `ollama` = optional Ollama backend |
| `INDEX_NAME` | `scanner` | Which Redis index `/index`, `/search` and `/clips` use |
| `EMBED_DEVICE` | `cuda` | `cpu` keeps query embeddings off the GPU |
| `SERVING_MODE` | `batched` | `sequential` = original one-frame-at-a-time path |
| `BATCH_MAX_SIZE` | `4` | Max frames per model call (2 is the largest that fits in 12 GB) |
| `PREPROCESS_MODE` | `adaptive` | `off` / `adaptive` / `always` |
| `W_FRAME`, `W_CLIP`, `W_BM25` | `1`, `0`, `0` | Retriever fusion weights (frozen from the dev-set grid) |

### Use 1: find an object in one video (web UI)

Open http://localhost:8000, upload a video and type what to look for (e.g. `yellow taxi`). Each 1-FPS frame streams back as YES/NO with a confidence and a timestamp. The same request from the command line:

```bash
curl -N -F "video=@uploads/test.mp4" -F "object_str=yellow taxi" http://localhost:8000/analyze
```

### Use 2: index a library, then search it in plain English

```bash
# Index a video: captions every second (~3.4 s per video-second on an RTX 4080 Laptop)
curl -N -F "video=@uploads/test.mp4" -F "clip_id=test" http://localhost:8000/index

# Search (~10-50 ms per query)
curl -s "http://localhost:8000/search?q=a+yellow+taxi+in+traffic&k=5"

# Captions of one clip between two timestamps
curl -s "http://localhost:8000/clips/test/frames?t0=0&t1=5"
```

### Try search on the 320 MSR-VTT test clips (no GPU)

The eval already built the index `eval_test320_off`, so you can search it without indexing anything:

```bash
INFERENCE_BACKEND=none INDEX_NAME=eval_test320_off EMBED_DEVICE=cpu .venv/Scripts/python.exe -m uvicorn main:app --port 8001
```

From another terminal:

```bash
# Happy path: the query describes something visible. The correct clip video7117 ranks #1 at t=8 s.
curl -s "http://localhost:8001/search?q=people+are+singing+on+the+beach&k=3"
curl -s "http://localhost:8001/clips/video7117/frames?t0=7&t1=9"

# Unhappy path: the query depends on audio. The correct clip video7227 is not in the top 100,
# because its frames only show "a person playing a guitar" and the clips have no audio track.
curl -s "http://localhost:8001/search?q=a+man+is+singing&k=5"
curl -s "http://localhost:8001/clips/video7227/frames?t0=0&t1=5"

# Error cases
curl -s -w " [%{http_code}]\n" "http://localhost:8001/clips/video0000/frames"     # unknown clip -> 404
curl -s -w " [%{http_code}]\n" "http://localhost:8001/search?q="                  # empty query  -> 422
curl -s -w " [%{http_code}]\n" -F "video=@uploads/test.mp4" -F "object_str=taxi" http://localhost:8001/analyze   # no model -> 503
```

To watch a returned clip, open the local file (MSR-VTT videos are under `.data/msrvtt/videos/video/`):

```bash
start "" .data/msrvtt/videos/video/video7117.mp4                       # Git Bash
# PowerShell: Invoke-Item .data\msrvtt\videos\video\video7117.mp4
explorer.exe '.data\msrvtt\videos\video'                               # browse all 10,000 clips
```

### Monitoring (Prometheus + Grafana)

Start only the monitoring containers. They scrape the server running natively on the host at `:8000`:

```bash
docker compose up -d prometheus grafana
```

| URL | What |
|---|---|
| http://localhost:3000/d/framesift/framesift-serving | Grafana dashboard (anonymous access) |
| http://localhost:9090/targets | Prometheus scrape status. `host.docker.internal:8000` should be `up`; `app:8000` is only up when the full compose stack runs |
| http://localhost:8000/metrics/ | Raw metrics from the server |

Dashboard panels:
- `/analyze` scan latency
- Per-frame inference latency
- Throughput
- Batching: batch size, queue depth, in-flight frames
- `/search` latency
- `/search` query rate

The panels only draw once the server has served requests. To fill the search panels quickly:

```bash
for i in $(seq 60); do curl -s "localhost:8000/search?q=a+dog+on+the+beach" >/dev/null; sleep 0.5; done
```

Stop the monitoring containers with `docker compose stop prometheus grafana`.

### Whole stack in Docker

```bash
docker compose up -d --build        # API on GPU, Redis Stack, Prometheus :9090, Grafana :3000
docker compose down
```

The compose Redis also binds port 6379. Run `docker stop scanner-redis` first.

### Tests

```bash
.venv/Scripts/python.exe -m pytest -q tests
```

## Reproducing the numbers

```bash
# Eval sets (MSR-VTT videos + annotations under .data/msrvtt)
python -m eval.clip_stats --annotations .data/msrvtt/msrvtt_test_1k.json --videos .data/msrvtt/videos/video --out .data/msrvtt/clip_stats_test1k.json
python -m eval.build_sets

# Caption at 1 FPS (GPU, ~7.5 h in total), choose fusion weights on dev only, then score test once
python -m eval.caption_set --set eval/sets/dev80.json   --preprocess off --out .data/captions/dev80_off.jsonl --max-batch 1
python -m eval.tune_fusion --set eval/sets/dev80.json   --captions .data/captions/dev80_off.jsonl --out artifacts/eval/dev80_fusion_grid.json
python -m eval.caption_set --set eval/sets/test320.json --preprocess off --out .data/captions/test320_off.jsonl --max-batch 1
python -m eval.caption_set --set eval/sets/test320.json --preprocess adaptive --reuse .data/captions/test320_off.jsonl --out .data/captions/test320_adaptive.jsonl --max-batch 1
python -m eval.hit_at_k --set eval/sets/test320.json --captions .data/captions/test320_adaptive.jsonl --index-name eval_test320_adaptive --out artifacts/eval/test320_adaptive.json
python -m eval.hit_at_k --set eval/sets/test320.json --captions .data/captions/test320_off.jsonl      --index-name eval_test320_off      --out artifacts/eval/test320_off.json

# Bootstrap CIs and paired significance tests (CPU only)
python -m eval.stats --out artifacts/eval/stats.json

# Serving: original sequential path vs batched dispatcher, same model and clip
python loadtest/serve_and_test.py --env SERVING_MODE=sequential -- --label before --concurrency 1 2 4 --out artifacts/loadtest/before.json
python loadtest/serve_and_test.py --env SERVING_MODE=batched BATCH_MAX_SIZE=2 STREAM_WINDOW=4 -- --label after --concurrency 1 2 4 --out artifacts/loadtest/after.json
```

Caption caches are resumable. If a run is interrupted, re-running the same command continues where it stopped. Set `EMBED_DEVICE=cpu` on `hit_at_k` and `tune_fusion` to keep the GPU free.

## Notes

- **Windows power throttling.** Windows 11 throttles background processes (EcoQoS). That starves the thread that launches GPU kernels, and generation drops from ~75 to ~300 ms/token. `scanner/winperf.py` opts the process out at startup. `/health` reports whether this succeeded.
- **No audio.** The MSR-VTT copy used here (friedrichor/MSR-VTT on Hugging Face) has no audio tracks. About a third of the human queries describe sound (singing, talking), which a frame-only system cannot observe.
- **Long GPU runs on a laptop.** Captioning keeps the GPU at full load for hours. Keep the machine plugged in and well ventilated. The search-only mode and the statistics scripts do not use the GPU.
- **Clip-level vectors are truncated.** The clip-level retriever embeds all of a clip's captions as one text. bge-base reads at most 512 tokens, and 77% of test clips are longer, so the clip-level vector mostly covers the first part of the clip. The default configuration does not use this retriever (`W_CLIP=0`).
