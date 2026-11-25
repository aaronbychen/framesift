"""Text-to-clip retrieval eval: Hit@k over an eval set, through the real retrieval layer
(LlamaIndex + Redis). Every clip of the set is indexed from a caption cache, then each clip's
human-written MSR-VTT caption is issued as a query; a hit means that clip is in the top k.

    python -m eval.hit_at_k --set eval/sets/test320.json --captions .data/captions/test320_off.jsonl \
        --index-name eval_test320_off --out artifacts/eval/test320_off.json
"""
import argparse
import datetime
import json
import platform
from pathlib import Path

import numpy as np

from eval.caption_set import load_cache


def pct(values):
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "p99": round(float(np.percentile(a, 99)), 3),
            "mean": round(float(a.mean()), 3), "max": round(float(a.max()), 3)}


def metrics(ranks, ks=(1, 5, 10)):
    """ranks: 1-based rank of the correct clip, or None if not in the returned list."""
    n = len(ranks)
    out = {"n_queries": n}
    for k in ks:
        out[f"hit@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4) if n else None
    out["mrr@100"] = round(sum(1.0 / r for r in ranks if r is not None) / n, 4) if n else None
    return out


def build_index(ix, clips, captions):
    ix.clear()
    for c in clips:
        row = captions[c["video_id"]]
        ix.add_clip(c["video_id"], [{"t": f["t"], "caption": f.get("caption", ""), "stats": f.get("stats", {}),
                                     "enhanced": f.get("enhanced", False)} for f in row["frames"]],
                    {"path": c["path"], "preprocess": row["preprocess"]})


def evaluate(ix, clips, fusion=None, depth=100):
    ranks, latencies, per_query = [], [], []
    for c in clips:
        results, timings = ix.search(c["query"], k=depth, fusion=fusion)
        ids = [r["clip_id"] for r in results]
        rank = ids.index(c["video_id"]) + 1 if c["video_id"] in ids else None
        ranks.append(rank)
        latencies.append(timings)
        per_query.append({"video_id": c["video_id"], "query": c["query"], "difficult": c["difficult"],
                          "rank": rank, "top5": ids[:5],
                          "best_t": next((r["best_t"] for r in results if r["clip_id"] == c["video_id"]), None)})
    return ranks, latencies, per_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--index-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion", default=None, help="JSON dict overriding scanner.index.FUSION")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    from scanner.index import FUSION, CaptionIndex
    clips = json.loads(Path(args.set).read_text())["clips"]
    captions = load_cache(Path(args.captions))
    missing = [c["video_id"] for c in clips if c["video_id"] not in captions]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing from {args.captions}, e.g. {missing[:3]}")
    # Caption retrieval only (the v1 system); the CLIP retriever is evaluated by eval/visual.py.
    fusion = {"w_visual": 0.0, **(json.loads(args.fusion) if args.fusion else {})}

    ix = CaptionIndex(name=args.index_name)
    build_index(ix, clips, captions)
    ix.search("warmup query", k=5, fusion=fusion)  # load BM25 + CUDA kernels before timing
    ranks, lat, per_query = evaluate(ix, clips, fusion)

    diff = [r for r, c in zip(ranks, clips) if c["difficult"]]
    normal = [r for r, c in zip(ranks, clips) if not c["difficult"]]
    frames = [f for c in clips for f in captions[c["video_id"]]["frames"]]
    result = {
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "set": args.set, "captions": args.captions, "index_name": args.index_name,
        "fusion": dict(FUSION, **(fusion or {})), "notes": args.notes,
        "corpus": {"clips": len(clips), "frames_1fps": len(frames),
                   "frames_enhanced": sum(1 for f in frames if f.get("enhanced")),
                   "frames_caption_error": sum(1 for f in frames if f.get("error"))},
        "all": metrics(ranks), "difficult": metrics(diff), "not_difficult": metrics(normal),
        "query_latency_ms": {key: pct([t[key] for t in lat]) for key in lat[0]},
        "platform": {"os": platform.platform(), "python": platform.python_version()},
        "per_query": per_query,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("corpus", "all", "difficult", "not_difficult")}))
    print("latency ms total:", result["query_latency_ms"]["total_ms"])


if __name__ == "__main__":
    main()
