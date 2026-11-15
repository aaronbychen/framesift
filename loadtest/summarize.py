"""Print run_loadtest.py results, and optionally combine a before and after sweep into one file.

    python loadtest/summarize.py artifacts/loadtest/before_sweep.json
    python loadtest/summarize.py --combine artifacts/loadtest/results.json \
        artifacts/loadtest/before_sweep.json artifacts/loadtest/after_sweep.json
"""
import argparse
import json
from pathlib import Path


def summary(level):
    batch = level.get("model_batches")
    return {
        "concurrency": level["concurrency"],
        "scans": level["scans"],
        "frames": level["frames"],
        "frames_with_inference_error": level["frames_with_inference_error"],
        "wall_seconds": level["wall_seconds"],
        "frames_per_second": level["throughput"]["frames_per_second"],
        "clips_per_minute": level["throughput"]["clips_per_minute"],
        "scan_latency_seconds": {k: level["scan_latency_seconds"][k] for k in ("p50", "p95", "p99")},
        "time_to_first_frame_seconds": {k: level["time_to_first_frame_seconds"][k] for k in ("p50", "p95", "p99")},
        "frame_inference_seconds": ({k: level["frame_inference_seconds"][k] for k in ("p50", "p95", "p99")}
                                    if level.get("frame_inference_seconds") else None),
        "mean_batch_size": batch["mean_batch_size"] if batch else None,
        "generate_calls": batch["generate_calls"] if batch else None,
    }


def print_file(path, data):
    print(f"{path}  label={data['label']}  mode={data['server_config'].get('serving_mode', data['server_config'].get('backend'))}  "
          f"model={data['server_config']['model']}  gpu={data['hardware'].get('gpu')}")
    for level in data["levels"]:
        s = summary(level)
        print(f"  c={s['concurrency']:>2} scans={s['scans']:>3} frames={s['frames']:>4} "
              f"err={s['frames_with_inference_error']} wall={s['wall_seconds']:>9.1f}s "
              f"fps={s['frames_per_second']:.4f} clips/min={s['clips_per_minute']:.3f} "
              f"scan p50/p95/p99={s['scan_latency_seconds']['p50']}/{s['scan_latency_seconds']['p95']}/"
              f"{s['scan_latency_seconds']['p99']} ttff p50/p99={s['time_to_first_frame_seconds']['p50']}/"
              f"{s['time_to_first_frame_seconds']['p99']} mean batch={s['mean_batch_size']} "
              f"calls={s['generate_calls']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--combine", help="write a before/after comparison (first file = before, second = after)")
    args = parser.parse_args()

    runs = [(p, json.loads(Path(p).read_text())) for p in args.files]
    for path, data in runs:
        print_file(path, data)

    if args.combine:
        (before_path, before), (after_path, after) = runs[:2]
        before_levels = {lv["concurrency"]: summary(lv) for lv in before["levels"]}
        after_levels = {lv["concurrency"]: summary(lv) for lv in after["levels"]}
        comparison = []
        for c in sorted(set(before_levels) & set(after_levels)):
            b, a = before_levels[c], after_levels[c]
            comparison.append({
                "concurrency": c,
                "before": b,
                "after": a,
                "throughput_ratio_after_over_before": round(a["frames_per_second"] / b["frames_per_second"], 3),
                "scan_p50_ratio_after_over_before": round(
                    a["scan_latency_seconds"]["p50"] / b["scan_latency_seconds"]["p50"], 3),
                "scan_p99_ratio_after_over_before": round(
                    a["scan_latency_seconds"]["p99"] / b["scan_latency_seconds"]["p99"], 3),
            })
        out = {
            "throughput_unit": "frames/s analyzed (1 frame per video-second sampled); also clips/min",
            "before": {"file": before_path, "server_config": before["server_config"],
                       "notes": before["notes"], "generated_at": before["generated_at"]},
            "after": {"file": after_path, "server_config": after["server_config"],
                      "notes": after["notes"], "generated_at": after["generated_at"]},
            "hardware": before["hardware"],
            "comparison": comparison,
        }
        Path(args.combine).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.combine}")


if __name__ == "__main__":
    main()
