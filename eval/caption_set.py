"""Caption every clip of an eval set at 1 FPS with Llama 3.2 Vision, through the same
pipeline + BatchDispatcher the API uses. Results go to a resumable JSONL cache.

    python -m eval.caption_set --set eval/sets/test320.json --preprocess off \
        --out .data/captions/test320_off.jsonl

--reuse FILE (with --preprocess adaptive): frames that adaptive mode does NOT enhance are
pixel-identical to the --preprocess off run, so their captions are copied from FILE and only the
enhanced frames are sent to the model.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from scanner import config
from scanner.frames import encode_jpeg, preprocess, resize_for_model, sample_1fps
from scanner.pipeline import CAPTION_PROMPT, caption_clip


def load_cache(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["video_id"]] = row
    return done


async def caption_with_reuse(submit, clip_path, reuse_frames, mode):
    """Adaptive mode with reuse: only frames that get enhanced go to the model."""
    reuse = {f["t"]: f for f in reuse_frames}
    tasks, frames = [], []
    for second, frame in await asyncio.to_thread(lambda: list(sample_1fps(clip_path))):
        out, stats, enhanced = preprocess(frame, mode)
        if not enhanced and second in reuse and reuse[second]["caption"]:
            frames.append(dict(reuse[second], stats=stats.to_dict(), enhanced=False, reused=True))
            continue
        jpeg = encode_jpeg(resize_for_model(out))
        row = {"t": second, "stats": stats.to_dict(), "enhanced": enhanced, "reused": False}
        frames.append(row)
        tasks.append((row, asyncio.create_task(submit(jpeg, CAPTION_PROMPT, config.CAPTION_MAX_TOKENS))))
    for row, task in tasks:
        try:
            text, usage = await task
            row.update(caption=text.strip(), usage=usage)
        except Exception as e:  # noqa: BLE001
            row.update(caption="", error=repr(e))
    return sorted(frames, key=lambda f: f["t"])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--preprocess", choices=["off", "adaptive", "always"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse", default=None)
    ap.add_argument("--clip-concurrency", type=int, default=6)
    ap.add_argument("--max-batch", type=int, default=config.BATCH_MAX_SIZE)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from scanner.vlm import BatchDispatcher, VisionEngine
    clips = json.loads(Path(args.set).read_text())["clips"][:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_cache(out)
    reuse = load_cache(Path(args.reuse)) if args.reuse else {}
    todo = [c for c in clips if c["video_id"] not in done]
    print(f"{len(clips)} clips, {len(done)} cached, {len(todo)} to caption", flush=True)
    if not todo:
        return

    engine = await asyncio.to_thread(VisionEngine)
    dispatcher = BatchDispatcher(engine, args.max_batch, config.BATCH_WINDOW_MS)
    dispatcher.start()
    sem = asyncio.Semaphore(args.clip_concurrency)
    lock = asyncio.Lock()
    t0 = time.perf_counter()
    counters = {"clips": 0, "frames": 0, "model_frames": 0}

    async def one(c):
        async with sem:
            start = time.perf_counter()
            if args.reuse and c["video_id"] in reuse:
                frames = await caption_with_reuse(dispatcher.submit, c["path"], reuse[c["video_id"]]["frames"],
                                                  args.preprocess)
            else:
                frames = await caption_clip(dispatcher.submit, c["path"], window=config.STREAM_WINDOW,
                                            preprocess_mode=args.preprocess)
            row = {"video_id": c["video_id"], "preprocess": args.preprocess, "frames": frames,
                   "seconds": round(time.perf_counter() - start, 3)}
            async with lock:
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                counters["clips"] += 1
                counters["frames"] += len(frames)
                counters["model_frames"] += sum(1 for f in frames if not f.get("reused"))
                el = time.perf_counter() - t0
                if counters["clips"] % 10 == 0 or counters["clips"] == len(todo):
                    print(f"{counters['clips']}/{len(todo)} clips, {counters['model_frames']} model frames, "
                          f"{counters['model_frames'] / el:.2f} frames/s, {el / 60:.1f} min", flush=True)

    await asyncio.gather(*(one(c) for c in todo))
    errors = sum(1 for r in load_cache(out).values() for f in r["frames"] if f.get("error"))
    print(f"done: {counters}, errors={errors}, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
