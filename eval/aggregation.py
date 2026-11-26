"""Experiment: how should per-frame CLIP scores be pooled into one clip score?

The visual retriever scores a clip by the max cosine over its 1-FPS frames. Max ignores every other
frame; averaging can use more of the clip. This is the cheapest form of temporal modelling (no new
model, no re-encoding), and mean pooling would also shrink the index to one vector per clip.

Protocol, fixed and committed before any result was computed:
- Embeddings: unchanged CLIP ViT-L/14 frame embeddings (eval/visual.py), raw query text.
- Candidates, in simplicity order:
    max        max cosine over frames (current default)
    mean_emb   cosine with the re-normalised mean frame embedding (one vector per clip)
    top2, top3, top5   mean of the k highest frame cosines (all frames if fewer than k)
- Dev set (new): dev1k = 1,000 clips from the 9k train split, disjoint from dev80 and from 1k-A,
  seed 1, one seeded-random human caption each. Corpus = the same 1,000 clips (standard 1k-A size).
  dev80 is too small for this: CLIP-only Hit@5 on 80 candidates is already ~88%.
- Selection on dev1k only: highest R@5, then MRR@100, then simplest.
- Verification on fresh680 only (test320 is not touched), selected vs max, paired:
    primary: 680 fresh queries vs the full 1,000-clip 1k-A corpus, R@5, exact McNemar;
    secondary: R@1, R@10, MRR (Wilcoxon) and the 680-clip fresh corpus; Holm over the 4 primary-setting metrics.
  If max is selected, there is nothing to verify and the result is reported as such.

    python -m eval.aggregation --stage dev  --out artifacts/eval/dev1k_aggregation.json
    python -m eval.aggregation --stage test --select-from artifacts/eval/dev1k_aggregation.json --out artifacts/eval/fresh680_aggregation.json
"""
import argparse
import datetime
import json
import random
from pathlib import Path

import numpy as np

from eval.visual import CACHE, CLIP_PATH, embed_frames, embed_texts, load_clip

ROOT = Path(".data/msrvtt")
VIDEOS = ROOT / "videos" / "video"
DEV_SEED = 1
DEV_SIZE = 1000
CANDIDATES = ["max", "mean_emb", "top2", "top3", "top5"]


def build_dev1k():
    path = Path("eval/sets/dev1k.json")
    if path.exists():
        return json.loads(path.read_text())["clips"]
    dev80 = {c["video_id"] for c in json.loads(Path("eval/sets/dev80.json").read_text())["clips"]}
    test = {a["video_id"] for a in json.loads((ROOT / "msrvtt_test_1k.json").read_text())}
    train = [a for a in json.loads((ROOT / "msrvtt_train_9k.json").read_text())
             if a["video_id"] not in dev80 and a["video_id"] not in test]
    rng = random.Random(DEV_SEED)
    rows = sorted(rng.sample(train, DEV_SIZE), key=lambda a: a["video_id"])
    clips = [{"video_id": a["video_id"], "path": str(VIDEOS / f"{a['video_id']}.mp4"),
              "query": rng.choice(a["caption"])} for a in rows]
    path.write_text(json.dumps({"meta": {"source": "MSR-VTT 9k train split", "seed": DEV_SEED,
                                         "excludes": "dev80 and 1k-A test"}, "clips": clips}, indent=1))
    return clips


def clip_scores(q_emb, corpus, emb, method):
    """(queries, clips) score matrix for one pooling method."""
    if method == "mean_emb":
        m = np.stack([emb[v][1].mean(0) for v in corpus])
        return q_emb @ (m / np.linalg.norm(m, axis=1, keepdims=True)).T
    out = np.empty((len(q_emb), len(corpus)), np.float32)
    for j, v in enumerate(corpus):
        s = q_emb @ emb[v][1].T  # (queries, frames)
        if method == "max":
            out[:, j] = s.max(1)
        else:
            k = min(int(method[3:]), s.shape[1])
            out[:, j] = np.sort(s, axis=1)[:, -k:].mean(1)
    return out


def ranks(queries, q_emb, corpus, emb, method):
    s = clip_scores(q_emb, corpus, emb, method)
    pos = {v: i for i, v in enumerate(corpus)}
    own = s[np.arange(len(queries)), [pos[q["video_id"]] for q in queries]]
    return (s > own[:, None]).sum(1) + 1  # ties resolved in the query's favour, same for every method


def per_query(r):
    m = {f"r@{k}": (r <= k).astype(float) for k in (1, 5, 10)}
    m["mrr@100"] = np.where(r <= 100, 1 / r, 0.0)
    return m


def summary(r):
    return {k: round(float(v.mean()), 4) for k, v in per_query(r).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "test"], required=True)
    ap.add_argument("--select-from", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, proc = load_clip()
    base = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "clip_model": CLIP_PATH}

    if args.stage == "dev":
        clips = build_dev1k()
        emb = embed_frames(clips, model, proc, CACHE / "dev1k_clip_l14.npz")
        q = embed_texts([c["query"] for c in clips], model, proc)
        ids = [c["video_id"] for c in clips]
        rows = [{"method": m, "simplicity_order": i, **summary(ranks(clips, q, ids, emb, m))}
                for i, m in enumerate(CANDIDATES)]
        best = sorted(rows, key=lambda r: (-r["r@5"], -r["mrr@100"], r["simplicity_order"]))[0]
        out = {**base, "set": "eval/sets/dev1k.json", "n_queries": len(clips),
               "frames": int(sum(len(emb[v][0]) for v in ids)),
               "selection_rule": "max r@5, then mrr@100, then simplest", "selected": best["method"],
               "candidates": rows}
        for r in rows:
            print(r)
        print("selected:", best["method"])
    else:
        from scipy import stats
        from eval.stats import holm
        sel = json.loads(Path(args.select_from).read_text())["selected"]
        ann = {a["video_id"]: a for a in json.loads((ROOT / "msrvtt_test_1k.json").read_text())}
        test320 = json.loads(Path("eval/sets/test320.json").read_text())["clips"]
        used = {c["video_id"] for c in test320}
        fresh = [{"video_id": v, "query": ann[v]["caption"]} for v in sorted(ann) if v not in used]
        emb = dict(embed_frames(test320, model, proc, CACHE / "test320_clip_l14.npz"))
        emb.update(embed_frames(fresh, model, proc, CACHE / "fresh680_clip_l14.npz"))
        assert len(fresh) == 680 and len(emb) == 1000
        q = embed_texts([c["query"] for c in fresh], model, proc)
        out = {**base, "selected_from": args.select_from, "selected": sel}
        if sel == "max":
            out["result"] = "dev selected max (the current default); nothing to verify"
        else:
            rng = np.random.default_rng(0)
            for setting, corpus in (("fresh_vs_full1000", sorted(emb)),
                                    ("fresh_vs_fresh680", [c["video_id"] for c in fresh])):
                a = per_query(ranks(fresh, q, corpus, emb, "max"))
                b = per_query(ranks(fresh, q, corpus, emb, sel))
                res, pvals = {}, []
                for met in a:
                    d = b[met] - a[met]
                    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                    row = {"max": round(float(a[met].mean()), 4), sel: round(float(b[met].mean()), 4),
                           "diff": round(float(d.mean()), 4),
                           "diff_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                         round(float(np.percentile(boots, 97.5)), 4)]}
                    if met.startswith("r@"):
                        o_a, o_b = int(((a[met] == 1) & (b[met] == 0)).sum()), int(((a[met] == 0) & (b[met] == 1)).sum())
                        p = 1.0 if o_a + o_b == 0 else float(stats.binomtest(o_b, o_a + o_b).pvalue)
                        row["mcnemar"] = {"only_max": o_a, f"only_{sel}": o_b, "p_value": round(p, 4)}
                    else:
                        nz = d != 0
                        p = float(stats.wilcoxon(b[met][nz], a[met][nz]).pvalue) if nz.any() else 1.0
                        row["wilcoxon"] = {"improved": int((d > 0).sum()), "worse": int((d < 0).sum()),
                                           "p_value": round(p, 4)}
                    pvals.append(p)
                    res[met] = row
                if setting == "fresh_vs_full1000":
                    for met, padj in zip(res, holm(pvals)):
                        res[met]["p_holm"] = padj
                out[setting] = res
        print(json.dumps(out, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
