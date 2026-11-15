"""Engine check: throughput of one generate call at batch sizes 1..N, peak VRAM, and whether
batched outputs equal the batch-size-1 outputs for the same frames (correctness gate).

    python -m eval.engine_check --set eval/sets/dev80.json --frames 8 --out artifacts/eval/engine_check.json
"""
import argparse
import datetime
import json
import time
from pathlib import Path

from scanner import config
from scanner.winperf import disable_power_throttling
from scanner.frames import encode_jpeg, preprocess, resize_for_model, sample_1fps
from scanner.pipeline import CAPTION_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval/sets/dev80.json")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print("process power hints:", disable_power_throttling(), flush=True)

    import torch
    from scanner.vlm import VisionEngine
    clips = json.loads(Path(args.set).read_text())["clips"]
    jpegs = []
    for c in clips:
        for _, f in sample_1fps(c["path"]):
            jpegs.append(encode_jpeg(resize_for_model(preprocess(f, "off")[0])))
            break
        if len(jpegs) >= args.frames:
            break

    t = time.perf_counter()
    eng = VisionEngine()
    load_s = time.perf_counter() - t
    items = [(j, CAPTION_PROMPT, config.CAPTION_MAX_TOKENS) for j in jpegs]
    eng.generate(items[:1])  # warmup

    ref = [eng.generate([it])[0] for it in items]
    runs = []
    for bs in args.batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        outs, t = [], time.perf_counter()
        for i in range(0, len(items), bs):
            outs += eng.generate(items[i:i + bs])
        el = time.perf_counter() - t
        same = sum(o[0] == r[0] for o, r in zip(outs, ref))
        runs.append({"batch_size": bs, "frames": len(items), "seconds": round(el, 3),
                     "frames_per_second": round(len(items) / el, 3),
                     "completion_tokens": sum(o[1]["completion_tokens"] for o in outs),
                     "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "exact_match_vs_bs1": f"{same}/{len(items)}",
                     "differing": [{"i": i, "bs1": r[0], "batched": o[0]}
                                   for i, (o, r) in enumerate(zip(outs, ref)) if o[0] != r[0]]})
        print(json.dumps({k: v for k, v in runs[-1].items() if k != "differing"}), flush=True)

    out = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "model": eng.model_name, "load_seconds": round(load_s, 1), "max_tiles": eng.max_tiles,
           "gpu": torch.cuda.get_device_name(0), "reference_bs1": [r[0] for r in ref], "runs": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
