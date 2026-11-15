"""Compare per-frame scanner outputs across run_loadtest.py result files.

The first file is the baseline; its first scan is the reference. For every scan in every file,
reports agreement with that reference on:
  - the match decision (is_match), and
  - the exact raw model text, when both sides recorded it.
Also lists every frame whose raw text differs, so divergence can be inspected directly.

Example (gate: batched vLLM vs one-at-a-time vLLM, same query):
    python loadtest/compare_outputs.py --out artifacts/correctness/gate_taxi.json \
        artifacts/correctness/vllm_seq_taxi.json artifacts/correctness/vllm_batched_taxi.json
"""
import argparse
import json
from pathlib import Path


def by_second(scan):
    return {f["second"]: f for f in scan["frames"] if not f["error"]}


def compare(ref, scan):
    common = sorted(set(ref) & set(scan))
    decision_same = [s for s in common if ref[s]["is_match"] == scan[s]["is_match"]]
    text_common = [s for s in common
                   if ref[s].get("raw_response") is not None and scan[s].get("raw_response") is not None]
    text_diff = [s for s in text_common if ref[s]["raw_response"] != scan[s]["raw_response"]]
    return {
        "frames_compared": len(common),
        "decision_agreement": round(len(decision_same) / len(common), 4) if common else None,
        "decision_disagreeing_seconds": [s for s in common if s not in decision_same],
        "text_frames_compared": len(text_common),
        "text_exact_match_rate": round(1 - len(text_diff) / len(text_common), 4) if text_common else None,
        "text_differing": [{
            "second": s,
            "reference": ref[s]["raw_response"],
            "this_scan": scan[s]["raw_response"],
        } for s in text_diff],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    loaded = []
    for path in args.files:
        data = json.loads(Path(path).read_text())
        scans = [s for level in data["levels"] for s in level["raw"]["scans"]]
        loaded.append((path, data, scans))

    base_path, base_data, base_scans = loaded[0]
    ref = by_second(base_scans[0])
    queries = {data["query"] for _, data, _ in loaded}
    if len(queries) != 1:
        raise SystemExit(f"files use different queries: {queries}")

    runs = []
    for path, data, scans in loaded:
        results = [compare(ref, by_second(s)) for s in scans]
        decisions = [r["decision_agreement"] for r in results if r["decision_agreement"] is not None]
        texts = [r["text_exact_match_rate"] for r in results if r["text_exact_match_rate"] is not None]
        runs.append({
            "file": path,
            "label": data["label"],
            "server_config": data["server_config"],
            "concurrency_levels": [lv["concurrency"] for lv in data["levels"]],
            "scans": len(scans),
            "min_decision_agreement_vs_reference": min(decisions) if decisions else None,
            "mean_decision_agreement_vs_reference": round(sum(decisions) / len(decisions), 4) if decisions else None,
            "min_text_exact_match_vs_reference": min(texts) if texts else None,
            "mean_text_exact_match_vs_reference": round(sum(texts) / len(texts), 4) if texts else None,
            "per_scan": results,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "query": queries.pop(),
        "reference": f"{base_path} (scan 0)",
        "reference_decisions": {str(s): f["is_match"] for s, f in sorted(ref.items())},
        "runs": runs,
    }, indent=2))
    for run in runs:
        print(f"{run['label']:>28} scans={run['scans']:>3} decision min/mean="
              f"{run['min_decision_agreement_vs_reference']}/{run['mean_decision_agreement_vs_reference']} "
              f"text-exact min/mean={run['min_text_exact_match_vs_reference']}/"
              f"{run['mean_text_exact_match_vs_reference']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
