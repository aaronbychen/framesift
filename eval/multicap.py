"""Larger query samples: use all 20 MSR-VTT captions per clip as queries.

Every comparison so far used one query per clip, so 1-2 pp differences were below what 320 or 680
queries can resolve. MSR-VTT has 20 human captions per clip. Using all of them multiplies the queries
~20x at no GPU cost (queries are only text). Queries from the same clip are not independent, so all
inference is clustered by clip.

Protocol, fixed and committed before any result was computed:
- Queries: the 20 MSR-VTT sentences of each clip (MSRVTT_data.json); exact duplicates within a clip
  (after lowercasing and stripping) are dropped. Corpora and embeddings are unchanged.
- Statistics, per comparison: per-query metrics R@1/5/10 and MRR@100, averaged over all queries.
  95% CI by cluster bootstrap (resample clips, 10k, seed 0). p-value by a clustered sign-flip
  permutation test on the per-clip mean difference (10k, seed 0). Holm over the 4 metrics.
  Primary metric R@5.
- Analysis 1 (pre-specified follow-up to report 3.8, whose primary test failed with one query per
  clip): fresh680 clips' queries vs the full 1,000-clip 1k-A corpus, top5 pooling vs max pooling.
  This does not overturn 3.8; it is reported next to it as a larger-sample follow-up.
- Analysis 2 (post-hoc, test320, third look at that set): fusion (caption + CLIP, w = 1, RRF k = 60,
  depth 100, as served) vs CLIP only (max), corpus = the 320 clips. Caption-only is reported for
  reference. Requires the Redis index eval_test320_off.

    python -m eval.multicap --out artifacts/eval/multicap.json
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np

from eval.aggregation import clip_scores
from eval.visual import CACHE, CLIP_PATH, embed_frames, fuse, load_clip

ROOT = Path(".data/msrvtt")
B = 10000


def queries_for(video_ids):
    sents = json.loads((ROOT / "MSRVTT_data.json").read_text())["sentences"]
    wanted, seen, out = set(video_ids), set(), []
    for s in sents:
        key = (s["video_id"], s["caption"].strip().lower())
        if s["video_id"] in wanted and key not in seen:
            seen.add(key)
            out.append({"video_id": s["video_id"], "query": s["caption"].strip()})
    return sorted(out, key=lambda q: q["video_id"])


def embed_texts_batched(texts, model, proc, bs=256):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            tok = proc(text=texts[i:i + bs], return_tensors="pt", padding=True, truncation=True)
            out.append(torch.nn.functional.normalize(model.get_text_features(**tok), dim=-1).numpy())
    return np.concatenate(out)


def visual_ranks_and_lists(queries, q_emb, corpus, emb, method, depth=None, chunk=1000):
    pos = {v: i for i, v in enumerate(corpus)}
    ranks, lists = [], []
    for i in range(0, len(queries), chunk):
        s = clip_scores(q_emb[i:i + chunk], corpus, emb, method)
        own = s[np.arange(len(s)), [pos[q["video_id"]] for q in queries[i:i + chunk]]]
        ranks.extend(((s > own[:, None]).sum(1) + 1).tolist())
        if depth:
            order = np.argsort(-s, axis=1)[:, :depth]
            lists.extend([[corpus[j] for j in row] for row in order])
    return np.array(ranks), lists


def per_query(r):
    r = np.asarray(r, dtype=float)
    m = {f"r@{k}": (r <= k).astype(float) for k in (1, 5, 10)}
    m["mrr@100"] = np.where(r <= 100, 1 / r, 0.0)
    return m


def clustered_compare(queries, ra, rb, name_a, name_b):
    from eval.stats import holm
    clips = sorted({q["video_id"] for q in queries})
    cidx = {v: i for i, v in enumerate(clips)}
    g = np.array([cidx[q["video_id"]] for q in queries])
    n_q = np.bincount(g, minlength=len(clips))
    ma, mb = per_query(ra), per_query(rb)
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(clips), (B, len(clips)))
    signs = rng.choice([-1.0, 1.0], (B, len(clips)))
    out, pvals = {"n_queries": len(queries), "n_clips": len(clips)}, []
    for met in ma:
        sa, sb = np.bincount(g, ma[met], len(clips)), np.bincount(g, mb[met], len(clips))
        d_sum = sb - sa  # per-clip summed difference
        obs = d_sum.sum() / n_q.sum()
        bs = (d_sum[boot_idx].sum(1)) / n_q[boot_idx].sum(1)
        perm = (signs * d_sum).sum(1) / n_q.sum()
        p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
        p = max(p, 1 / B)
        out[met] = {name_a: round(float(ma[met].mean()), 4), name_b: round(float(mb[met].mean()), 4),
                    "diff": round(float(obs), 4),
                    "diff_ci95_cluster": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                    "p_sign_flip": round(p, 4)}
        pvals.append(p)
    for met, padj in zip(ma, holm(pvals)):
        out[met]["p_holm"] = padj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/eval/multicap.json")
    ap.add_argument("--index-name", default="eval_test320_off")
    args = ap.parse_args()
    os.environ.setdefault("EMBED_DEVICE", "cpu")
    model, proc = load_clip()
    test320 = json.loads(Path("eval/sets/test320.json").read_text())["clips"]
    ann = json.loads((ROOT / "msrvtt_test_1k.json").read_text())
    used = {c["video_id"] for c in test320}
    fresh_ids = sorted(a["video_id"] for a in ann if a["video_id"] not in used)
    emb = dict(embed_frames(test320, model, proc, CACHE / "test320_clip_l14.npz"))
    emb.update(embed_frames([{"video_id": v} for v in fresh_ids], model, proc, CACHE / "fresh680_clip_l14.npz"))
    assert len(fresh_ids) == 680 and len(emb) == 1000
    out = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "clip_model": CLIP_PATH, "bootstrap": B, "seed": 0}

    # Analysis 1
    qf = queries_for(fresh_ids)
    ef = embed_texts_batched([q["query"] for q in qf], model, proc)
    corpus = sorted(emb)
    r_max, _ = visual_ranks_and_lists(qf, ef, corpus, emb, "max")
    r_top5, _ = visual_ranks_and_lists(qf, ef, corpus, emb, "top5")
    out["A1_fresh680_top5_vs_max_corpus1000"] = clustered_compare(qf, r_max, r_top5, "max", "top5")
    print(json.dumps(out["A1_fresh680_top5_vs_max_corpus1000"], indent=1), flush=True)

    # Analysis 2
    from eval.visual import DEPTH
    from scanner.index import CaptionIndex
    qt = queries_for([c["video_id"] for c in test320])
    et = embed_texts_batched([q["query"] for q in qt], model, proc)
    ids = [c["video_id"] for c in test320]
    r_clip, vis_lists = visual_ranks_and_lists(qt, et, ids, emb, "max", depth=DEPTH)
    ix = CaptionIndex(name=args.index_name)
    assert set(ids) <= set(ix.clip_ids())
    fusion = {"w_frame": 1.0, "w_clip": 0.0, "w_bm25": 0.0, "w_visual": 0.0}
    r_cap, r_fus = [], []
    for i, (q, vl) in enumerate(zip(qt, vis_lists)):
        cl = [r["clip_id"] for r in ix.search(q["query"], k=DEPTH, fusion=fusion)[0]]
        r_cap.append(cl.index(q["video_id"]) + 1 if q["video_id"] in cl else 10**9)
        fl = fuse(cl, vl, 1.0, 1.0)
        r_fus.append(fl.index(q["video_id"]) + 1 if q["video_id"] in fl else 10**9)
        if (i + 1) % 500 == 0:
            print(f"  caption search {i + 1}/{len(qt)}", flush=True)
    a2 = clustered_compare(qt, r_clip, r_fus, "clip_only", "fusion")
    a2["caption_only_reference"] = {k: round(float(v.mean()), 4) for k, v in per_query(r_cap).items()}
    a2["status"] = "post-hoc; third look at test320"
    out["A2_test320_fusion_vs_clip_corpus320"] = a2
    print(json.dumps(a2, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
