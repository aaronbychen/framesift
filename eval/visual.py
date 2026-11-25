"""Experiment: does a CLIP image-embedding retriever, fused with caption retrieval, improve Hit@5?

Motivation (docs/report.md, error analysis): captions are a lossy bottleneck. Anything the one-sentence
caption omits cannot be retrieved. A CLIP retriever compares the query with the pixels directly.

Protocol, fixed before any result was computed:
- Visual retriever: CLIP ViT-L/14 (openai/clip-vit-large-patch14), standard processor (224 px, center crop).
  Every 1-FPS frame is embedded without preprocessing. A clip's score is the max cosine between the raw
  query text and its frames.
- Caption retriever: the frozen default (dense frame captions only) from the existing Redis index.
- Fusion: weighted reciprocal-rank fusion, rrf_k = 60, caption weight 1, visual weight w.
- Candidates: caption only (= current system), visual only, and fusion with w in {0.25, 0.5, 1, 2}.
- Selection on dev80 only: highest Hit@5, then MRR@100; any remaining tie goes to the simpler config
  (caption only < visual only < fusion with smaller w).
- Test: test320 is scored ONCE with the config named in the dev artifact (read from the file, not typed
  by hand). Primary comparison: selected config vs caption only, paired, exact McNemar on Hit@5.
  This is the second scoring of test320 (the first was the v1 report), and the report says so.

    python -m eval.visual --set eval/sets/dev80.json   --index-name eval_dev_tune    --out artifacts/eval/dev80_visual_grid.json
    python -m eval.visual --set eval/sets/test320.json --index-name eval_test320_off --select-from artifacts/eval/dev80_visual_grid.json --out artifacts/eval/test320_visual.json
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np

CLIP_PATH = os.getenv("CLIP_MODEL", ".data/models/clip-vit-large-patch14")
CACHE = Path(".data/visual")
RRF_K = 60
CANDIDATES = [{"name": "caption_only", "w_caption": 1.0, "w_visual": 0.0},
              {"name": "visual_only", "w_caption": 0.0, "w_visual": 1.0}] + \
             [{"name": f"fusion_w{w}", "w_caption": 1.0, "w_visual": w} for w in (0.25, 0.5, 1.0, 2.0)]
DEPTH = 100


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    torch.set_grad_enabled(False)
    return CLIPModel.from_pretrained(CLIP_PATH).eval(), CLIPProcessor.from_pretrained(CLIP_PATH)


def embed_frames(clips, model, proc, cache_file):
    """{video_id: (ts, unit-norm embeddings)} cached in an npz."""
    import cv2
    import torch
    from scanner.frames import sample_1fps
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        return z["emb"].item()
    out, t0 = {}, time.perf_counter()
    for i, c in enumerate(clips, 1):
        ts, imgs = [], []
        for t, bgr in sample_1fps(c["path"]):
            ts.append(t)
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        embs = []
        for j in range(0, len(imgs), 32):
            px = proc(images=imgs[j:j + 32], return_tensors="pt")["pixel_values"]
            e = model.get_image_features(pixel_values=px)
            embs.append(torch.nn.functional.normalize(e, dim=-1).numpy())
        out[c["video_id"]] = (np.array(ts), np.concatenate(embs).astype(np.float32))
        if i % 20 == 0 or i == len(clips):
            print(f"  embedded {i}/{len(clips)} clips, {time.perf_counter() - t0:.0f}s", flush=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, emb=np.array(out, dtype=object))
    return out


def embed_texts(texts, model, proc):
    import torch
    tok = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
    e = model.get_text_features(**tok)
    return torch.nn.functional.normalize(e, dim=-1).numpy()


def visual_rankings(clips, frame_emb, q_emb):
    ids = [c["video_id"] for c in clips]
    out = []
    for q in q_emb:
        best = {vid: (float((frame_emb[vid][1] @ q).max()), int(frame_emb[vid][0][(frame_emb[vid][1] @ q).argmax()]))
                for vid in ids}
        out.append(sorted(ids, key=lambda v: -best[v][0]))
    return out


def caption_rankings(clips, index_name):
    from scanner.index import CaptionIndex
    ix = CaptionIndex(name=index_name)
    have = set(ix.clip_ids())
    missing = [c["video_id"] for c in clips if c["video_id"] not in have]
    if missing:
        raise SystemExit(f"index {index_name} lacks {len(missing)} clips, e.g. {missing[:3]}")
    fusion = {"w_frame": 1.0, "w_clip": 0.0, "w_bm25": 0.0}
    return [[r["clip_id"] for r in ix.search(c["query"], k=DEPTH, fusion=fusion)[0]] for c in clips]


def fuse(cap, vis, w_caption, w_visual):
    score = {}
    for w, ranking in ((w_caption, cap), (w_visual, vis)):
        if w <= 0:
            continue
        for r, cid in enumerate(ranking[:DEPTH], 1):
            score[cid] = score.get(cid, 0.0) + w / (RRF_K + r)
    return sorted(score, key=lambda c: -score[c])


def ranks_for(clips, cap, vis, cfg):
    out = []
    for c, cr, vr in zip(clips, cap, vis):
        fused = fuse(cr, vr, cfg["w_caption"], cfg["w_visual"])
        out.append(fused.index(c["video_id"]) + 1 if c["video_id"] in fused else None)
    return out


def metrics(ranks):
    r = np.array([x if x is not None else 10**9 for x in ranks], dtype=float)
    m = {f"hit@{k}": round(float((r <= k).mean()), 4) for k in (1, 5, 10)}
    m["mrr@100"] = round(float(np.where(r <= 100, 1 / r, 0).mean()), 4)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--index-name", required=True)
    ap.add_argument("--select-from", default=None, help="dev grid artifact; test mode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("EMBED_DEVICE", "cpu")

    clips = json.loads(Path(args.set).read_text())["clips"]
    model, proc = load_clip()
    frame_emb = embed_frames(clips, model, proc, CACHE / f"{Path(args.set).stem}_clip_l14.npz")
    q_emb = embed_texts([c["query"] for c in clips], model, proc)
    vis = visual_rankings(clips, frame_emb, q_emb)
    cap = caption_rankings(clips, args.index_name)
    base = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "set": args.set, "index_name": args.index_name, "clip_model": CLIP_PATH, "rrf_k": RRF_K,
            "n_queries": len(clips), "frames": int(sum(len(v[0]) for v in frame_emb.values()))}

    if args.select_from is None:  # dev: evaluate all candidates, apply the selection rule
        rows = []
        for order, cfg in enumerate(CANDIDATES):
            ranks = ranks_for(clips, cap, vis, cfg)
            rows.append({**cfg, "simplicity_order": order, **metrics(ranks)})
        best = sorted(rows, key=lambda r: (-r["hit@5"], -r["mrr@100"], r["simplicity_order"]))[0]
        out = {**base, "selection_rule": "max hit@5, then mrr@100, then simplest", "selected": best["name"],
               "candidates": rows}
        for r in rows:
            print(r)
        print("selected:", best["name"])
    else:  # test: score the dev-selected config once, plus caption_only for the paired comparison
        from scipy import stats
        sel_name = json.loads(Path(args.select_from).read_text())["selected"]
        sel = next(c for c in CANDIDATES if c["name"] == sel_name)
        base_cfg = CANDIDATES[0]
        r_sel, r_base = ranks_for(clips, cap, vis, sel), ranks_for(clips, cap, vis, base_cfg)
        h = lambda rs: np.array([1 if (x is not None and x <= 5) else 0 for x in rs])
        a, b = h(r_base), h(r_sel)
        only_base, only_sel = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p = 1.0 if only_base + only_sel == 0 else float(stats.binomtest(only_sel, only_base + only_sel).pvalue)
        rng = np.random.default_rng(0)
        d = (b - a).astype(float)
        boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        diff = {"diff_hit5": round(float(d.mean()), 4),
                "diff_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                "mcnemar": {"only_caption": only_base, "only_selected": only_sel, "p_value": round(p, 4)}}
        diff_subsets = {}
        for name, mask in (("difficult", np.array([c["difficult"] for c in clips])),):
            diff_subsets[name] = {"caption_only": metrics([x for x, m in zip(r_base, mask) if m]),
                                  "selected": metrics([x for x, m in zip(r_sel, mask) if m])}
        out = {**base, "selected_from": args.select_from, "selected": sel,
               "note": "second scoring of test320 (v2); v1 = artifacts/eval/test320_off.json",
               "caption_only": metrics(r_base), "selected_metrics": metrics(r_sel),
               "visual_only": metrics(ranks_for(clips, cap, vis, CANDIDATES[1])),
               "paired_hit5_selected_vs_caption": diff, "subsets": diff_subsets,
               "per_query": [{"video_id": c["video_id"], "query": c["query"], "difficult": c["difficult"],
                              "rank_caption": rb, "rank_selected": rs}
                             for c, rb, rs in zip(clips, r_base, r_sel)]}
        print(json.dumps({k: v for k, v in out.items() if k != "per_query"}, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
