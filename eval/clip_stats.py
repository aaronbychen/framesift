"""Per-clip lighting statistics at 1 FPS, used to define the difficult-scene subset.

A clip is "difficult" when at least half of its 1-FPS frames are flagged low-light or
high-contrast by scanner.frames.frame_stats (thresholds from scanner/config.py).

    python -m eval.clip_stats --annotations .data/msrvtt/msrvtt_test_1k.json \
        --videos .data/msrvtt/videos/video --out .data/msrvtt/clip_stats_test1k.json
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scanner import config
from scanner.frames import frame_stats, sample_1fps, video_info


def clip_stats(video_path):
    frames = [frame_stats(f).to_dict() for _, f in sample_1fps(video_path)]
    n = len(frames)
    flagged = sum(f["difficult"] for f in frames)
    return {
        "video": Path(video_path).name,
        "info": video_info(video_path),
        "frames_1fps": n,
        "low_light_frac": round(sum(f["low_light"] for f in frames) / n, 3) if n else None,
        "high_contrast_frac": round(sum(f["high_contrast"] for f in frames) / n, 3) if n else None,
        "noisy_frac": round(sum(f["noisy"] for f in frames) / n, 3) if n else None,
        "mean_luma": round(sum(f["mean_luma"] for f in frames) / n, 2) if n else None,
        "difficult": bool(n) and flagged / n >= 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ann = json.loads(Path(args.annotations).read_text())
    ids = sorted({a["video_id"] for a in ann})
    paths = [str(Path(args.videos) / f"{v}.mp4") for v in ids]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(clip_stats, paths, chunksize=8))
    out = {
        "thresholds": {"LOW_LIGHT_LUMA": config.LOW_LIGHT_LUMA,
                       "HIGH_CONTRAST_CLIP_FRAC": config.HIGH_CONTRAST_CLIP_FRAC,
                       "DENOISE_SIGMA": config.DENOISE_SIGMA,
                       "difficult_rule": ">=50% of 1-FPS frames low-light or high-contrast"},
        "clips": {Path(r["video"]).stem: r for r in rows},
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    d = [r for r in rows if r["difficult"]]
    print(f"{len(rows)} clips, {sum(r['frames_1fps'] for r in rows)} frames at 1 FPS, "
          f"{len(d)} difficult "
          f"(low-light-dominant {sum(r['low_light_frac'] >= 0.5 for r in rows)}, "
          f"high-contrast-dominant {sum(r['high_contrast_frac'] >= 0.5 for r in rows)})")


if __name__ == "__main__":
    main()
