"""Secondary statistics for the CLIP fusion test run (artifacts/eval/test320_visual.json).

Paired bootstrap CIs (10k, seed 0) and exact McNemar / Wilcoxon tests, selected config vs caption only,
for Hit@1/5/10 and MRR on all queries and on the difficult (low-light / high-contrast) subset.
The primary test (Hit@5, all queries) is in the artifact itself; everything here is secondary.

    python -m eval.stats_visual --out artifacts/eval/stats_visual.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from eval.stats import holm


def per_query(ranks):
    r = np.array([x if x is not None else 10**9 for x in ranks], dtype=float)
    m = {f"hit@{k}": (r <= k).astype(float) for k in (1, 5, 10)}
    m["mrr@100"] = np.where(r <= 100, 1 / r, 0.0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="artifacts/eval/test320_visual.json")
    ap.add_argument("--out", default="artifacts/eval/stats_visual.json")
    args = ap.parse_args()
    d = json.loads(Path(args.run).read_text())
    q = d["per_query"]
    base, sel = per_query([x["rank_caption"] for x in q]), per_query([x["rank_selected"] for x in q])
    diff_mask = np.array([x["difficult"] for x in q])
    rng = np.random.default_rng(0)
    out, pvals, keys = {"run": args.run, "selected": d["selected"]["name"]}, [], []
    for sub, mask in (("all", np.ones(len(q), bool)), ("difficult", diff_mask), ("not_difficult", ~diff_mask)):
        res = {"n": int(mask.sum())}
        for met in base:
            a, b = base[met][mask], sel[met][mask]
            dlt = b - a
            boots = dlt[rng.integers(0, len(dlt), (10000, len(dlt)))].mean(1)
            row = {"caption_only": round(float(a.mean()), 4), "selected": round(float(b.mean()), 4),
                   "diff": round(float(dlt.mean()), 4),
                   "diff_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                   "relative_change_pct": round(float(dlt.mean() / a.mean() * 100), 1)}
            if met.startswith("hit"):
                o_b, o_s = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
                p = 1.0 if o_b + o_s == 0 else float(stats.binomtest(o_s, o_b + o_s).pvalue)
                row["mcnemar"] = {"only_caption": o_b, "only_selected": o_s, "p_value": round(p, 4)}
            else:
                nz = dlt != 0
                p = float(stats.wilcoxon(b[nz], a[nz]).pvalue) if nz.any() else 1.0
                row["wilcoxon"] = {"improved": int((dlt > 0).sum()), "worse": int((dlt < 0).sum()), "p_value": round(p, 4)}
            if sub in ("all", "difficult"):
                pvals.append(p); keys.append((sub, met))
            res[met] = row
        out[sub] = res
    for (sub, met), padj in zip(keys, holm(pvals)):
        out[sub][met]["p_holm"] = padj
    out["multiple_comparisons"] = "Holm over 8 tests (4 metrics x {all, difficult}); Hit@5/all is also the pre-specified primary test"
    Path(args.out).write_text(json.dumps(out, indent=1))
    for sub in ("all", "difficult", "not_difficult"):
        for met in base:
            r = out[sub][met]
            print(f"{sub:13s} {met:8s} {r['caption_only']:.3f} -> {r['selected']:.3f}  diff {r['diff']:+.3f} {r['diff_ci95']}  rel {r['relative_change_pct']:+.1f}%  p_holm {r.get('p_holm', '-')}")


if __name__ == "__main__":
    main()
