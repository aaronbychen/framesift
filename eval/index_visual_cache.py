"""Load cached CLIP frame embeddings (from eval/visual.py) into a Redis index's visual store, so the
API's /search can use the visual retriever on an eval index without re-embedding.

    python -m eval.index_visual_cache --cache .data/visual/test320_clip_l14.npz --index-name eval_test320_off
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--index-name", required=True)
    args = ap.parse_args()
    from scanner.index import CaptionIndex
    ix = CaptionIndex(name=args.index_name)
    have = set(ix.clip_ids())
    emb = np.load(args.cache, allow_pickle=True)["emb"].item()
    n = 0
    for vid, (ts, e) in emb.items():
        if vid not in have:
            continue
        ix.add_visual(vid, [int(t) for t in ts], e)
        n += len(ts)
    print(f"added {n} frame vectors for {len(set(emb) & have)} clips to {args.index_name}")


if __name__ == "__main__":
    main()
