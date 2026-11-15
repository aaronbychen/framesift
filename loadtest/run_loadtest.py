"""Load test for the video scanner: N concurrent streams, each POSTing a clip to /analyze.

Every number written to the output file comes from this run. Raw per-scan and per-frame
samples are saved next to the summary stats so the percentiles can be recomputed.

Example:
    python loadtest/run_loadtest.py --label after --concurrency 5 10 20 50 \
        --scans-per-stream 2 --vllm-metrics http://127.0.0.1:8001/metrics \
        --out artifacts/loadtest/after_sweep.json
"""
import argparse
import asyncio
import datetime
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np

SAMPLE_INTERVAL = 0.1


def percentiles(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "mean": round(float(arr.mean()), 3),
        "max": round(float(arr.max()), 3),
    }


def hardware_info():
    info = {"os": platform.platform(), "python": platform.python_version()}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        info["gpu"] = gpu
    except (OSError, subprocess.TimeoutExpired):
        info["gpu"] = None
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name; "
                 "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
                capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
            info["cpu"] = out[0].strip()
            info["ram_gb"] = float(out[1])
        except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
            info["cpu"] = platform.processor()
    else:
        info["cpu"] = platform.processor()
    info["note"] = "Load generator, FastAPI app, and model server all run on this one machine."
    return info


def parse_gauge(text, name):
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith(name) and (len(line) > len(name) and line[len(name)] in "{ "):
            total += float(line.rsplit(" ", 1)[1])
            found = True
    return total if found else None


async def sample_metrics(client, app_url, vllm_metrics_url, samples, stop):
    while not stop.is_set():
        t = time.perf_counter()
        row = {"t": t}
        try:
            app_text = (await client.get(f"{app_url}/metrics/")).text
            row["app_inflight"] = parse_gauge(app_text, "scanner_inflight_inference")
        except httpx.HTTPError:
            pass
        if vllm_metrics_url:
            try:
                text = (await client.get(vllm_metrics_url)).text
                row["vllm_running"] = parse_gauge(text, "vllm:num_requests_running")
                row["vllm_waiting"] = parse_gauge(text, "vllm:num_requests_waiting")
            except httpx.HTTPError:
                pass
        samples.append(row)
        await asyncio.sleep(max(0.0, SAMPLE_INTERVAL - (time.perf_counter() - t)))


def parse_histogram(text, name):
    """Return {"count", "sum", "buckets": {le: cumulative}} for an unlabeled Prometheus histogram."""
    out = {"count": 0.0, "sum": 0.0, "buckets": {}}
    for line in text.splitlines():
        if line.startswith(f"{name}_count "):
            out["count"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith(f"{name}_sum "):
            out["sum"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith(f"{name}_bucket"):
            le = re.search(r'le="([^"]+)"', line).group(1)
            out["buckets"][le] = float(line.rsplit(" ", 1)[1])
    return out


async def scrape_batches(client, app_url):
    try:
        text = (await client.get(f"{app_url}/metrics/")).text
    except httpx.HTTPError:
        return None
    return parse_histogram(text, "scanner_batch_size")


def batch_delta(before, after):
    """Model-call batch sizes observed by the app during one level (from its own histogram)."""
    if not before or not after:
        return None
    calls = after["count"] - before["count"]
    items = after["sum"] - before["sum"]
    les = list(after["buckets"])
    cum = [after["buckets"][le] - before["buckets"].get(le, 0.0) for le in les]
    per_bucket, prev = {}, 0.0
    for le, c in zip(les, cum):
        per_bucket[f"<={le}"] = int(c - prev)
        prev = c
    return {"generate_calls": int(calls), "items": int(items),
            "mean_batch_size": round(items / calls, 3) if calls else None,
            "calls_by_batch_size": per_bucket}


async def run_scan(client, app_url, video_name, video_bytes, query):
    start = time.perf_counter()
    first_frame = None
    frames, http_error = [], None
    try:
        async with client.stream(
                "POST", f"{app_url}/analyze",
                files={"video": (video_name, video_bytes, "video/mp4")},
                data={"object_str": query}) as response:
            if response.status_code != 200:
                http_error = f"HTTP {response.status_code}"
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("status") != "success":
                    continue
                if first_frame is None:
                    first_frame = time.perf_counter()
                f = data["frame"]
                usage = f.get("usage") or {}
                frames.append({
                    "second": f["second"],
                    "is_match": f["is_match"],
                    "confidence": f["confidence"],
                    "error": f["description"] == "Error occurred" and f["confidence"] == 0,
                    "inference_seconds": f.get("inference_seconds"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "raw_response": f.get("raw_response"),
                })
    except httpx.HTTPError as e:
        http_error = repr(e)
    end = time.perf_counter()
    return {
        "start": start,
        "end": end,
        "latency_seconds": end - start,
        "ttff_seconds": (first_frame - start) if first_frame else None,
        "http_error": http_error,
        "frames": frames,
    }


async def run_level(app_url, video_name, video_bytes, query, concurrency, scans_per_stream, vllm_metrics_url):
    limits = httpx.Limits(max_connections=concurrency + 16, max_keepalive_connections=concurrency + 16)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        batches_before = await scrape_batches(client, app_url)
        samples, stop = [], asyncio.Event()
        sampler = asyncio.create_task(sample_metrics(client, app_url, vllm_metrics_url, samples, stop))

        async def stream(stream_id):
            results = []
            for _ in range(scans_per_stream):
                scan = await run_scan(client, app_url, video_name, video_bytes, query)
                scan["stream"] = stream_id
                results.append(scan)
            return results

        wall_start = time.perf_counter()
        per_stream = await asyncio.gather(*(stream(i) for i in range(concurrency)))
        wall = time.perf_counter() - wall_start
        stop.set()
        await sampler
        batches_after = await scrape_batches(client, app_url)

    scans = [s for stream_scans in per_stream for s in stream_scans]
    frames = [f for s in scans for f in s["frames"]]
    ok_frames = [f for f in frames if not f["error"]]
    running = [s["vllm_running"] for s in samples if s.get("vllm_running") is not None]
    inflight = [s["app_inflight"] for s in samples if s.get("app_inflight") is not None]

    def distribution(values):
        if not values:
            return None
        nonzero = [v for v in values if v > 0]
        counts = {}
        for v in values:
            counts[str(int(v))] = counts.get(str(int(v)), 0) + 1
        return {
            "samples": len(values),
            "sample_interval_seconds": SAMPLE_INTERVAL,
            "mean_when_busy": round(float(np.mean(nonzero)), 2) if nonzero else 0.0,
            "p50_when_busy": float(np.percentile(nonzero, 50)) if nonzero else 0.0,
            "max": float(max(values)),
            "histogram": dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
        }

    return {
        "concurrency": concurrency,
        "streams": concurrency,
        "scans_per_stream": scans_per_stream,
        "scans": len(scans),
        "scans_with_http_error": sum(1 for s in scans if s["http_error"]),
        "frames": len(frames),
        "frames_with_inference_error": len(frames) - len(ok_frames),
        "wall_seconds": round(wall, 3),
        "throughput": {
            "frames_per_second": round(len(ok_frames) / wall, 4),
            "clips_per_minute": round(60 * sum(1 for s in scans if not s["http_error"]) / wall, 4),
        },
        "scan_latency_seconds": percentiles([s["latency_seconds"] for s in scans]),
        "time_to_first_frame_seconds": percentiles([s["ttff_seconds"] for s in scans if s["ttff_seconds"]]),
        "frame_inference_seconds": percentiles([f["inference_seconds"] for f in ok_frames
                                                if f["inference_seconds"] is not None]),
        "prompt_tokens_per_frame": sorted({f["prompt_tokens"] for f in ok_frames if f["prompt_tokens"]}),
        "model_batches": batch_delta(batches_before, batches_after),
        "vllm_batch_size_num_requests_running": distribution(running),
        "app_inflight_model_calls": distribution(inflight),
        "raw": {
            "scans": [{
                "stream": s["stream"],
                "start_offset_seconds": round(s["start"] - wall_start, 3),
                "latency_seconds": round(s["latency_seconds"], 3),
                "ttff_seconds": round(s["ttff_seconds"], 3) if s["ttff_seconds"] else None,
                "http_error": s["http_error"],
                "frames": s["frames"],
            } for s in scans],
        },
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--video", default="uploads/test.mp4")
    parser.add_argument("--query", default="yellow taxi cab")
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument("--scans-per-stream", type=int, default=1)
    parser.add_argument("--warmup-scans", type=int, default=1)
    parser.add_argument("--vllm-metrics", default=None, help="vLLM /metrics URL, for batch-size sampling")
    parser.add_argument("--label", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    video_path = Path(args.video)
    video_bytes = video_path.read_bytes()
    async with httpx.AsyncClient(timeout=30) as client:
        server_config = (await client.get(f"{args.url}/health")).json()
    print(f"[{args.label}] server config: {server_config}", flush=True)

    async with httpx.AsyncClient(timeout=None) as client:
        for i in range(args.warmup_scans):
            scan = await run_scan(client, args.url, video_path.name, video_bytes, args.query)
            print(f"[{args.label}] warmup {i + 1}: {scan['latency_seconds']:.1f}s, "
                  f"{len(scan['frames'])} frames", flush=True)

    levels = []
    for n in args.concurrency:
        print(f"[{args.label}] concurrency={n} scans_per_stream={args.scans_per_stream} ...", flush=True)
        level = await run_level(args.url, video_path.name, video_bytes, args.query, n,
                                args.scans_per_stream, args.vllm_metrics)
        levels.append(level)
        lat = level["scan_latency_seconds"]
        batch = level["model_batches"]
        print(f"[{args.label}]   wall={level['wall_seconds']}s frames/s={level['throughput']['frames_per_second']} "
              f"scan p50/p95/p99={lat['p50']}/{lat['p95']}/{lat['p99']}s "
              f"errors={level['frames_with_inference_error']} "
              f"batch_mean_busy={batch['mean_when_busy'] if batch else 'n/a'} "
              f"batch_max={batch['max'] if batch else 'n/a'}", flush=True)

    result = {
        "label": args.label,
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command_args": vars(args),
        "server_config": server_config,
        "hardware": hardware_info(),
        "video": {"path": str(video_path), "bytes": len(video_bytes)},
        "query": args.query,
        "notes": args.notes,
        "levels": levels,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[{args.label}] wrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
