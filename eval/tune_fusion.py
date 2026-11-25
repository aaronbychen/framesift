"""Choose retrieval fusion weights on the DEV set only (MSR-VTT train clips).

Grid over (w_frame, w_clip, w_bm25, rrf_k); the winner is the config with the best dev Hit@5,
ties broken by MRR. The test set is never touched here.

    python -m eval.tune_fusion --set eval/sets/dev80.json --captions .data/captions/dev80_adaptive.jsonl \
        --out artifacts/eval/dev80_fusion_grid.json
"""
import argparse
import itertools
import json
from pathlib import Path

from eval.caption_set import load_cache
from eval.hit_at_k import build_index, evaluate, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from scanner.index import CaptionIndex
    clips = json.loads(Path(args.set).read_text())["clips"]
    ix = CaptionIndex(name="eval_dev_tune")
    build_index(ix, clips, load_cache(Path(args.captions)))

    rows = []
    weights = [0.0, 0.5, 1.0]
    for wf, wc, wb, k in itertools.product(weights, weights, weights, [10, 60]):
        if wf == wc == wb == 0:
            continue
        fusion = {"w_frame": wf, "w_clip": wc, "w_bm25": wb, "rrf_k": k, "w_visual": 0.0}
        ranks, _, _ = evaluate(ix, clips, fusion)
        rows.append({"fusion": fusion, **metrics(ranks)})
    rows.sort(key=lambda r: (-r["hit@5"], -r["mrr@100"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"set": args.set, "captions": args.captions,
                                          "best": rows[0], "grid": rows}, indent=1))
    for r in rows[:8]:
        print(r)
    single = [r for r in rows if sum(v > 0 for k2, v in r["fusion"].items() if k2.startswith("w_")) == 1]
    print("single-retriever baselines:", [(r["fusion"], r["hit@5"]) for r in single if r["fusion"]["rrf_k"] == 60])


if __name__ == "__main__":
    main()
