"""Build the fixed eval sets from MSR-VTT (real YouTube clips, human-written captions).

test320: MSR-VTT 1k-A test split (JSFusion), one human caption per clip as the query.
         Contains every clip the lighting stats mark as difficult (low-light / high-contrast)
         plus a seeded random sample of the rest, 320 clips total. Difficult clips are therefore
         over-represented (46% vs 15% in the full 1k-A), which makes overall Hit@k harder, not easier.
dev80:   80 clips from the 9k train split (disjoint from test), one seeded-random caption each.
         Used ONLY for choosing retrieval fusion weights; the test set is run once afterwards.

    python -m eval.build_sets
"""
import json
import random
from pathlib import Path

from eval.clip_stats import clip_stats
from scanner import config

ROOT = Path(".data/msrvtt")
VIDEOS = ROOT / "videos" / "video"
OUT = Path("eval/sets")
SEED = 0
TEST_SIZE = 320
DEV_SIZE = 80


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = json.loads((ROOT / "clip_stats_test1k.json").read_text())
    clips = stats["clips"]
    test_ann = {a["video_id"]: a for a in json.loads((ROOT / "msrvtt_test_1k.json").read_text())}

    difficult = sorted(v for v in test_ann if clips[v]["difficult"])
    rest = sorted(v for v in test_ann if not clips[v]["difficult"])
    rng = random.Random(SEED)
    chosen = difficult + rng.sample(rest, TEST_SIZE - len(difficult))
    test = [{
        "video_id": v, "path": str(VIDEOS / f"{v}.mp4"), "query": test_ann[v]["caption"],
        "difficult": clips[v]["difficult"], "low_light_frac": clips[v]["low_light_frac"],
        "high_contrast_frac": clips[v]["high_contrast_frac"], "frames_1fps": clips[v]["frames_1fps"],
    } for v in sorted(chosen)]

    train = json.loads((ROOT / "msrvtt_train_9k.json").read_text())
    rng = random.Random(SEED)
    dev_rows = rng.sample(train, DEV_SIZE)
    dev = []
    for a in sorted(dev_rows, key=lambda a: a["video_id"]):
        s = clip_stats(str(VIDEOS / f"{a['video_id']}.mp4"))
        dev.append({"video_id": a["video_id"], "path": str(VIDEOS / f"{a['video_id']}.mp4"),
                    "query": rng.choice(a["caption"]), "difficult": s["difficult"],
                    "low_light_frac": s["low_light_frac"], "high_contrast_frac": s["high_contrast_frac"],
                    "frames_1fps": s["frames_1fps"]})

    meta = {"source": "MSR-VTT (friedrichor/MSR-VTT on Hugging Face)", "seed": SEED,
            "thresholds": stats["thresholds"],
            "low_light_luma": config.LOW_LIGHT_LUMA, "high_contrast_clip_frac": config.HIGH_CONTRAST_CLIP_FRAC}
    for name, rows in (("test320", test), ("dev80", dev)):
        (OUT / f"{name}.json").write_text(json.dumps({"meta": meta, "clips": rows}, indent=1))
        print(f"{name}: {len(rows)} clips, {sum(r['difficult'] for r in rows)} difficult, "
              f"{sum(r['frames_1fps'] for r in rows)} frames at 1 FPS")


if __name__ == "__main__":
    main()
