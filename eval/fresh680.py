"""Independent check of the CLIP visual retriever on MSR-VTT 1k-A clips never used in any experiment.

test320 took 320 of the 1,000 1k-A clips (all 147 difficult ones + 173 random others). The remaining
680 have not been touched by any earlier experiment. They are all *not difficult* by construction,
so this checks generalisation of the visual retriever, not the low-light findings. Fusion cannot be
checked here: these clips have no Llama captions (that would need ~6 GPU hours).

Analyses, fixed before running (descriptive estimates with 95% bootstrap CIs; nothing is tuned here):
  A. fresh680 queries vs the 680-clip fresh corpus: CLIP-only Hit@1/5/10, MRR@100.
  B. fresh680 queries vs the full 1,000-clip 1k-A corpus (the standard 1k-A setting, so harder).
  C. Like-for-like with test320: corpus size 320 drawn from fresh680 (200 random draws, seed 0),
     mean CLIP-only Hit@5, compared with CLIP-only Hit@5 on test320's 173 not-difficult queries
     (also at corpus size 320). If test320 generalises, C's value should fall within the
     test320 not-difficult CI.
  D. For reference only: all 1,000 queries vs the full corpus (includes test320's queries).

    python -m eval.fresh680 --out artifacts/eval/fresh680_visual.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from eval.visual import CACHE, CLIP_PATH, embed_frames, embed_texts, load_clip

ROOT = Path(".data/msrvtt")
VIDEOS = ROOT / "videos" / "video"
SEED = 0


def ranks(queries, q_emb, corpus, frame_emb):
    """Rank of each query's own clip among `corpus` by best-frame cosine (None if absent)."""
    mats = [frame_emb[v][1] for v in corpus]
    starts = np.cumsum([0] + [len(m) for m in mats[:-1]])
    best = np.maximum.reduceat(q_emb @ np.concatenate(mats).T, starts, axis=1)  # (queries, clips)
    order = np.argsort(-best, axis=1)
    pos = {v: i for i, v in enumerate(corpus)}
    out = []
    for qi, q in enumerate(queries):
        if q["video_id"] not in pos:
            out.append(None)
            continue
        out.append(int(np.where(order[qi] == pos[q["video_id"]])[0][0]) + 1)
    return out


def summary(rs, rng):
    r = np.array([x if x is not None else 10**9 for x in rs], dtype=float)
    per = {f"hit@{k}": (r <= k).astype(float) for k in (1, 5, 10)}
    per["mrr@100"] = np.where(r <= 100, 1 / r, 0.0)
    out = {"n": len(rs)}
    for k, v in per.items():
        b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
        out[k] = {"value": round(float(v.mean()), 4),
                  "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/eval/fresh680_visual.json")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    ann = {a["video_id"]: a for a in json.loads((ROOT / "msrvtt_test_1k.json").read_text())}
    stats = json.loads((ROOT / "clip_stats_test1k.json").read_text())["clips"]
    test320 = json.loads(Path("eval/sets/test320.json").read_text())["clips"]
    used = {c["video_id"] for c in test320}
    fresh = [{"video_id": v, "path": str(VIDEOS / f"{v}.mp4"), "query": ann[v]["caption"],
              "difficult": stats[v]["difficult"]} for v in sorted(ann) if v not in used]
    assert len(fresh) == 680 and not any(c["difficult"] for c in fresh)

    model, proc = load_clip()
    emb = dict(embed_frames(test320, model, proc, CACHE / "test320_clip_l14.npz"))
    emb.update(embed_frames(fresh, model, proc, CACHE / "fresh680_clip_l14.npz"))
    q_fresh = embed_texts([c["query"] for c in fresh], model, proc)
    q_test = embed_texts([c["query"] for c in test320], model, proc)
    fresh_ids = [c["video_id"] for c in fresh]
    all_ids = sorted(emb)
    assert len(all_ids) == 1000

    out = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "clip_model": CLIP_PATH, "seed": SEED, "fresh_clips": len(fresh),
           "fresh_frames": int(sum(len(emb[v][0]) for v in fresh_ids))}
    out["A_fresh_vs_fresh680"] = summary(ranks(fresh, q_fresh, fresh_ids, emb), rng)
    out["B_fresh_vs_full1000"] = summary(ranks(fresh, q_fresh, all_ids, emb), rng)

    # C: corpus size 320, like test320
    draws = []
    for _ in range(200):
        sub = sorted(rng.choice(fresh_ids, 320, replace=False).tolist())
        idx = [fresh_ids.index(v) for v in sub]
        rs = ranks([fresh[i] for i in idx], q_fresh[idx], sub, emb)
        draws.append(np.mean([1.0 if r <= 5 else 0.0 for r in rs]))
    nd = [i for i, c in enumerate(test320) if not c["difficult"]]
    t_ranks = ranks(test320, q_test, [c["video_id"] for c in test320], emb)
    out["C_corpus320"] = {
        "fresh_draws": 200,
        "fresh_hit5_mean": round(float(np.mean(draws)), 4),
        "fresh_hit5_draw_range_2.5_97.5": [round(float(np.percentile(draws, 2.5)), 4),
                                           round(float(np.percentile(draws, 97.5)), 4)],
        "test320_not_difficult": summary([t_ranks[i] for i in nd], rng),
        "test320_all_visual_only": summary(t_ranks, rng)}
    q_all = np.concatenate([q_test, q_fresh])
    out["D_all1000_vs_full1000_reference"] = summary(ranks(test320 + fresh, q_all, all_ids, emb), rng)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
