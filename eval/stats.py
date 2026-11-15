"""Statistical analysis of the MSR-VTT retrieval results (CPU only, reads the eval artifacts).

- 95% bootstrap CIs (percentile, 10k resamples of queries) for Hit@1/5/10 and MRR@100
- Paired comparison adaptive preprocessing vs off on the same queries:
  paired-bootstrap CI of the difference, exact McNemar test on Hit@k, Wilcoxon signed-rank on
  reciprocal rank, Holm correction across the main tests
- Difficult vs not-difficult gap (unpaired bootstrap CI, Fisher exact test)
- Dev vs test Hit@5 (Wilson CIs, Fisher exact test)

    python -m eval.stats --out artifacts/eval/stats.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

KS = (1, 5, 10)
B = 10_000
SEED = 0


def load(path):
    d = json.loads(Path(path).read_text())
    q = sorted(d["per_query"], key=lambda r: r["video_id"])
    rank = np.array([r["rank"] if r["rank"] is not None else 10**9 for r in q], dtype=float)
    return {"ids": [r["video_id"] for r in q], "difficult": np.array([r["difficult"] for r in q]),
            "rank": rank, "fusion": d.get("fusion")}


def per_query(rank):
    m = {f"hit@{k}": (rank <= k).astype(float) for k in KS}
    m["mrr@100"] = np.where(rank <= 100, 1.0 / rank, 0.0)
    return m


def boot_ci(x, rng, b=B):
    idx = rng.integers(0, len(x), size=(b, len(x)))
    means = x[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def wilson(successes, n, z=1.959964):
    p = successes / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(a, b):
    """a, b: 0/1 arrays on the same queries. Returns discordant counts and exact two-sided p."""
    only_b = int(((a == 0) & (b == 1)).sum())
    only_a = int(((a == 1) & (b == 0)).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_b, n, 0.5).pvalue)
    return {"only_off": only_a, "only_adaptive": only_b, "p_value": round(p, 4)}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - j) * pvals[i]))
        adj[i] = round(running, 4)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="artifacts/eval/test320_off.json")
    ap.add_argument("--adaptive", default="artifacts/eval/test320_adaptive.json")
    ap.add_argument("--dev-grid", default="artifacts/eval/dev80_fusion_grid.json")
    ap.add_argument("--out", default="artifacts/eval/stats.json")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    off, ada = load(args.off), load(args.adaptive)
    assert off["ids"] == ada["ids"] and (off["difficult"] == ada["difficult"]).all()
    subsets = {"all": np.ones(len(off["ids"]), bool), "difficult": off["difficult"],
               "not_difficult": ~off["difficult"]}
    m_off, m_ada = per_query(off["rank"]), per_query(ada["rank"])

    out = {"method": {"bootstrap_resamples": B, "seed": SEED, "ci": "95% percentile bootstrap over queries",
                      "paired_tests": "exact McNemar (Hit@k), Wilcoxon signed-rank (reciprocal rank)",
                      "multiple_comparisons": "Holm over the 8 primary paired tests (4 metrics x {all, difficult})"},
           "conditions": {}, "paired_adaptive_vs_off": {}}

    for name, m in (("off", m_off), ("adaptive", m_ada)):
        out["conditions"][name] = {
            s: {met: {"value": round(float(v[mask].mean()), 4), "ci95": boot_ci(v[mask], rng)}
                for met, v in m.items()} | {"n": int(mask.sum())}
            for s, mask in subsets.items()}

    primary = []
    for s, mask in subsets.items():
        res = {"n": int(mask.sum())}
        for met in m_off:
            a, b = m_off[met][mask], m_ada[met][mask]
            d = b - a
            row = {"off": round(float(a.mean()), 4), "adaptive": round(float(b.mean()), 4),
                   "diff": round(float(d.mean()), 4), "diff_ci95": boot_ci(d, rng),
                   "relative_change_pct": round(float(d.mean() / a.mean() * 100), 1)}
            if met.startswith("hit"):
                row["mcnemar"] = mcnemar_exact(a, b)
                p = row["mcnemar"]["p_value"]
            else:
                nz = d[d != 0]
                w = stats.wilcoxon(b[d != 0], a[d != 0]) if len(nz) else None
                row["wilcoxon"] = {"n_nonzero": int(len(nz)), "improved": int((d > 0).sum()),
                                   "worse": int((d < 0).sum()),
                                   "p_value": round(float(w.pvalue), 4) if w else 1.0}
                p = row["wilcoxon"]["p_value"]
            if s in ("all", "difficult"):
                primary.append((s, met, p))
            res[met] = row
        out["paired_adaptive_vs_off"][s] = res
    for (s, met, _), padj in zip(primary, holm([p for *_, p in primary])):
        out["paired_adaptive_vs_off"][s][met]["p_holm"] = padj

    # The resume claim: +40% Hit@5 on difficult clips, i.e. 0.626 -> 0.876 (+0.25 absolute).
    base = out["paired_adaptive_vs_off"]["difficult"]["hit@5"]
    out["claim_check_40pct"] = {
        "required_diff_abs": round(0.4 * base["off"], 4), "observed_diff": base["diff"],
        "diff_ci95": base["diff_ci95"], "rejected": base["diff_ci95"][1] < 0.4 * base["off"]}

    # Difficult vs not-difficult (off condition), unpaired.
    h = m_off["hit@5"]
    dmask, nmask = subsets["difficult"], subsets["not_difficult"]
    diffs = [rng.choice(h[nmask], nmask.sum()).mean() - rng.choice(h[dmask], dmask.sum()).mean()
             for _ in range(B)]
    table = [[int(h[nmask].sum()), int(nmask.sum() - h[nmask].sum())],
             [int(h[dmask].sum()), int(dmask.sum() - h[dmask].sum())]]
    out["difficulty_gap_off_hit5"] = {
        "not_difficult": round(float(h[nmask].mean()), 4), "difficult": round(float(h[dmask].mean()), 4),
        "gap": round(float(h[nmask].mean() - h[dmask].mean()), 4),
        "gap_ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
        "fisher_p": round(float(stats.fisher_exact(table).pvalue), 4)}

    # Dev (80 train clips, frozen config) vs test (320) Hit@5.
    dev = json.loads(Path(args.dev_grid).read_text())["best"]
    dev_hits = round(dev["hit@5"] * dev["n_queries"])
    test_hits = int(m_off["hit@5"].sum())
    n_test = len(off["ids"])
    out["dev_vs_test_hit5"] = {
        "dev": {"n": dev["n_queries"], "hit@5": dev["hit@5"], "wilson_ci95": wilson(dev_hits, dev["n_queries"])},
        "test": {"n": n_test, "hit@5": round(test_hits / n_test, 4), "wilson_ci95": wilson(test_hits, n_test)},
        "fisher_p": round(float(stats.fisher_exact(
            [[dev_hits, dev["n_queries"] - dev_hits], [test_hits, n_test - test_hits]]).pvalue), 4),
        "note": "dev = MSR-VTT train clips with the frozen config chosen on dev itself (optimistic); "
                "test = 1k-A clips, oversampled for low-light/high-contrast (147/320 vs 147/1000 in 1k-A)"}

    # Hit@5 the test set would have at the natural 1k-A difficulty mix (reweighting the two strata).
    w_d = 147 / 1000
    rw = w_d * h[dmask].mean() + (1 - w_d) * h[nmask].mean()
    rws = [w_d * rng.choice(h[dmask], dmask.sum()).mean() + (1 - w_d) * rng.choice(h[nmask], nmask.sum()).mean()
           for _ in range(B)]
    out["test_hit5_reweighted_to_1kA_mix"] = {
        "difficult_share": w_d, "hit@5": round(float(rw), 4),
        "ci95": [round(float(np.percentile(rws, 2.5)), 4), round(float(np.percentile(rws, 97.5)), 4)]}

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("conditions",)}, indent=1))
    for s in subsets:
        c = out["conditions"]["off"][s]
        print("off", s, {k: (c[k]["value"], c[k]["ci95"]) for k in m_off})


if __name__ == "__main__":
    main()
