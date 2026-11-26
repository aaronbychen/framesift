"""Post-hoc: is fusion (Llama captions + CLIP) better than CLIP alone on test320?

The v2 protocol (eval/visual.py) pre-specified only fusion vs caption-only. Fusion vs CLIP-only was
never planned, so this is a post-hoc, exploratory comparison on already-scored data: no new model
choices, no tuning. It matters because captions cost ~10x more to index than CLIP vectors.

Visual-only ranks are recomputed from the cached CLIP frame embeddings and checked against the
visual_only metrics stored in the test artifact. Same tests as eval/stats_visual.py: paired bootstrap
CIs (10k, seed 0), exact McNemar for Hit@k, Wilcoxon for MRR, Holm over 8 tests.

    python -m eval.fusion_vs_clip --out artifacts/eval/fusion_vs_clip.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval.stats_visual import per_query
from eval.visual import CACHE, embed_frames, embed_texts, load_clip, metrics, visual_rankings
from eval.stats import holm
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="artifacts/eval/test320_visual.json")
    ap.add_argument("--set", default="eval/sets/test320.json")
    ap.add_argument("--out", default="artifacts/eval/fusion_vs_clip.json")
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text())
    clips = json.loads(Path(args.set).read_text())["clips"]
    q = run["per_query"]
    assert [c["video_id"] for c in clips] == [x["video_id"] for x in q]

    model, proc = load_clip()
    emb = embed_frames(clips, model, proc, CACHE / f"{Path(args.set).stem}_clip_l14.npz")
    vis = visual_rankings(clips, emb, embed_texts([c["query"] for c in clips], model, proc))
    r_vis = [v.index(c["video_id"]) + 1 for c, v in zip(clips, vis)]
    check = metrics(r_vis)
    assert check == run["visual_only"], (check, run["visual_only"])

    clip_m, fus_m = per_query(r_vis), per_query([x["rank_selected"] for x in q])
    diff_mask = np.array([x["difficult"] for x in q])
    rng = np.random.default_rng(0)
    out = {"run": args.run, "status": "post-hoc, exploratory (not pre-specified)",
           "fusion": run["selected"]["name"], "visual_only_check": "matches artifact"}
    pvals, keys = [], []
    for sub, mask in (("all", np.ones(len(q), bool)), ("difficult", diff_mask), ("not_difficult", ~diff_mask)):
        res = {"n": int(mask.sum())}
        for met in clip_m:
            a, b = clip_m[met][mask], fus_m[met][mask]
            d = b - a
            boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            row = {"clip_only": round(float(a.mean()), 4), "fusion": round(float(b.mean()), 4),
                   "diff": round(float(d.mean()), 4),
                   "diff_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
            if met.startswith("hit"):
                o_c, o_f = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
                p = 1.0 if o_c + o_f == 0 else float(stats.binomtest(o_f, o_c + o_f).pvalue)
                row["mcnemar"] = {"only_clip": o_c, "only_fusion": o_f, "p_value": round(p, 4)}
            else:
                nz = d != 0
                p = float(stats.wilcoxon(b[nz], a[nz]).pvalue) if nz.any() else 1.0
                row["wilcoxon"] = {"improved": int((d > 0).sum()), "worse": int((d < 0).sum()), "p_value": round(p, 4)}
            if sub in ("all", "difficult"):
                pvals.append(p); keys.append((sub, met))
            res[met] = row
        out[sub] = res
    for (sub, met), padj in zip(keys, holm(pvals)):
        out[sub][met]["p_holm"] = padj
    out["multiple_comparisons"] = "Holm over 8 tests (4 metrics x {all, difficult})"
    Path(args.out).write_text(json.dumps(out, indent=1))
    for sub in ("all", "difficult", "not_difficult"):
        for met in clip_m:
            r = out[sub][met]
            print(f"{sub:13s} {met:8s} {r['clip_only']:.3f} -> {r['fusion']:.3f}  diff {r['diff']:+.3f} "
                  f"{r['diff_ci95']}  p {r.get('mcnemar', r.get('wilcoxon'))['p_value']}  p_holm {r.get('p_holm', '-')}")


if __name__ == "__main__":
    main()
