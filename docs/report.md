# Does low-light preprocessing help a vision-language video search engine? A held-out evaluation

**Project:** FrameSift. This is a fully local natural-language video search system built from Llama 3.2 11B Vision, LlamaIndex and Redis.
**Data:** MSR-VTT (public YouTube clips with human captions).
**Hardware:** one laptop: RTX 4080 Laptop GPU (12 GB), i9-13980HX, 16 GB RAM.
Every number in this report comes from a JSON file under `artifacts/`. The commands that regenerate those files are in the README.

## Summary

Earlier write-ups of this project made three claims:
- OpenCV preprocessing (denoising, CLAHE, color normalization) gives a 40%+ retrieval gain on low-light and high-contrast footage.
- Retrieval reaches 85%+ Hit@5.
- Request batching cuts end-to-end latency by about 50%.

I rebuilt the evaluation to test these claims properly. The split between tuning data and test data was fixed in advance, the model-selection rule was written down before any results existed, and the test set was scored once. Under that protocol, none of the three claims holds:

| Claim | Measured on held-out test (n = 320 queries) | 95% CI |
|---|---|---|
| +40% Hit@5 on difficult clips from preprocessing | **+0.0 pp** (62.6% → 62.6%), McNemar p = 1.0 | [−4.1, +4.1] pp |
| 85%+ Hit@5 | **68.8%** | [63.4, 73.8] |
| ~50% lower latency from batching | **+56%** scan latency at 1 stream, +24% at 4 streams | (n = 1 scan per level; no CI) |

The 95% confidence interval for the preprocessing effect excludes the +25 pp that a 40% gain would require. The negative result therefore rests on a test with enough power to have caught the claimed effect. It is not just a failure to reach significance. Preprocessing slightly *hurts* ranking quality: MRR falls by 0.024 (CI [−0.043, −0.006]), and in a paired comparison 77 queries rank worse against 44 that rank better.

**Follow-up (v2): a CLIP visual retriever.** The error analysis pointed at the caption step as the bottleneck: anything a one-sentence caption omits cannot be retrieved. So I added a retriever that matches the query against the pixels directly (CLIP ViT-L/14) and fused it with caption retrieval. The protocol was committed before any result was computed (`eval/visual.py`, commit `5654aa6`), and the configuration was selected on dev only. This is the second scoring of the test set.

| Test320, caption only → caption + CLIP fusion | Δ [95% CI] | Holm-adjusted p |
|---|---|---|
| Hit@5: 68.8% → **74.4%** | +5.6 pp [+1.6, +9.7] | 0.029 (primary test, raw McNemar p = 0.010) |
| Hit@1: 42.2% → 51.3% | +9.1 pp [+4.7, +13.4] | < 0.001 |
| MRR: 0.538 → 0.618 | +0.080 [+0.051, +0.110] | < 0.001 |
| Difficult clips, Hit@1: 38.1% → 50.3% | +12.2 pp [+6.1, +19.1] (+32% relative) | 0.003 |

The gain the preprocessing was supposed to deliver on difficult footage came instead from changing *what* is matched (pixels, not captions). Even so it is +32% relative on Hit@1 and +13% on Hit@5, not +40%. The headline Hit@5 (74.4%) is still below 85%.

Most of that gain comes from CLIP itself. Post-hoc, with all 20 captions per clip as queries (5,945 queries, section 3.9), fusion beats CLIP alone by Hit@5 +2.9 pp [+1.6, +4.3]: 74.7% vs 71.7%. Captions add a small, real gain, at about 10× the indexing cost of a CLIP-only index.

## 1. System

```
video ─► 1-FPS sampler ─► lighting stats ─► (adaptive) denoise → gray-world → CLAHE
                                                     │
                                                     ▼
                        Llama 3.2 11B Vision (4-bit NF4, in-process) ─► one-sentence caption per frame
                                                     │
                                                     ▼
          bge-base-en-v1.5 embeddings ─► Redis Stack (HNSW, cosine) + BM25, via LlamaIndex
                                                     │
                       natural-language query ─► top-k clips with the best-matching timestamp
```

- **Sampling.** Each clip is sampled at exactly one frame per second, at frame index `round(t · fps)`, so variable frame rates do not drift.
- **Frame flags.** Each frame gets a letterbox-aware luminance profile. It is flagged *low-light* if the mean luma of the active area is below 50, and *high-contrast* if at least 5% of pixels are crushed (Y ≤ 10) **and** at least 5% are blown out (Y ≥ 245).
- **Adaptive preprocessing.** Only flagged frames are enhanced. Unflagged frames are left byte-identical, which keeps the comparison clean.
- **Captioning.** Every frame is captioned by the vision-language model.
- **Retrieval.** Three retrievers are available: dense search over frame captions, dense search over per-clip caption summaries, and BM25. They can be fused with weighted reciprocal-rank fusion.

## 2. Data and protocol

### 2.1 Defining "difficult" before looking at retrieval

I computed lighting statistics for all 1,000 clips in MSR-VTT's 1k-A test split (16,048 frames). A clip counts as difficult if at least 50% of its frames are flagged.

My first thresholds flagged 459 of the 1,000 clips. When I inspected them, most were letterboxed: black bars were being counted as crushed shadows. I therefore crop to the active picture area before computing statistics. With that fix the rule flags **147 clips (14.7%)**: 120 are mostly low-light and 22 mostly high-contrast.

The thresholds were set from the pixel-statistics distribution alone. No retrieval result had been computed at that point.

### 2.2 Splits

| Set | Source | Clips | Frames | Difficult | Used for |
|---|---|---|---|---|---|
| dev80 | MSR-VTT *train* split | 80 | 1,275 | 13 | choosing retrieval fusion weights |
| test320 | MSR-VTT 1k-A *test* split | 320 | 5,123 | 147 (all of them) + 173 random others | final numbers, scored once |

- The difficult stratum is **oversampled** in test320 (46% of clips, against 14.7% in 1k-A). Section 4.3 reweights the result back to the natural mix.
- **Queries.** Each clip's human-written MSR-VTT caption is used as a query. A hit at k means that clip appears in the top k of the 320-clip corpus.
- **Pre-specified selection rule.** On dev, the configuration with the best Hit@5 wins, with MRR as the tie-breaker. The grid covered 3 × 3 × 3 retriever weights × 2 RRF constants. After selection the weights were frozen in code (commit `5a62b2a`) before the test set was scored.

### 2.3 Controlled comparison

The test set was captioned twice:
- **off:** no preprocessing.
- **adaptive:** the 1,944 flagged frames (38%) were enhanced and re-captioned. The other 3,179 frames reuse the *off* captions, because their pixels are identical.

Model, prompt, greedy decoding and index configuration are the same in both runs. The comparison is fully paired: the same 320 queries run against two indexes that differ only in the enhanced frames.

### 2.4 Statistics

`eval/stats.py` computes the following:
- 95% percentile-bootstrap CIs over queries (10,000 resamples, seed 0).
- Paired-bootstrap CIs for the difference between conditions.
- An exact McNemar test for each Hit@k.
- A Wilcoxon signed-rank test on reciprocal rank.
- A Holm correction across the 8 primary tests (4 metrics × {all, difficult}).

## 3. Results

### 3.1 Retrieval quality (test320, *off*)

| Subset | n | Hit@1 | Hit@5 | Hit@10 | MRR@100 |
|---|---|---|---|---|---|
| All | 320 | 42.2 [36.9, 47.5] | **68.8 [63.4, 73.8]** | 76.9 [72.2, 81.3] | 0.538 [0.493, 0.585] |
| Difficult | 147 | 38.1 [30.6, 46.3] | 62.6 [55.1, 70.1] | 72.8 [66.0, 80.3] | 0.490 [0.422, 0.561] |
| Not difficult | 173 | 45.7 [38.2, 53.2] | 74.0 [67.1, 80.4] | 80.4 [74.6, 86.1] | 0.579 [0.518, 0.640] |

Difficult clips really are harder. Their Hit@5 is 11.4 pp lower (bootstrap CI [1.1, 21.5] pp; Fisher exact p = 0.030). So there was a real gap for preprocessing to close.

### 3.2 Effect of preprocessing (paired, adaptive − off)

| Subset | Metric | off | adaptive | Δ [95% CI] | Test | p (raw / Holm) |
|---|---|---|---|---|---|---|
| Difficult | Hit@5 | 62.6 | 62.6 | **0.0 [−4.1, +4.1] pp** | McNemar, 4 vs 4 discordant | 1.00 / 1.00 |
| Difficult | Hit@1 | 38.1 | 33.3 | −4.8 [−10.2, 0.0] pp | McNemar, 11 vs 4 | 0.12 / 0.59 |
| Difficult | MRR | 0.490 | 0.454 | −0.036 [−0.072, −0.003] | Wilcoxon, 28 better / 40 worse | 0.088 / 0.53 |
| All | Hit@5 | 68.8 | 68.4 | −0.3 [−2.5, +1.9] pp | McNemar, 7 vs 6 | 1.00 / 1.00 |
| All | Hit@1 | 42.2 | 39.1 | −3.1 [−5.9, −0.3] pp | McNemar, 16 vs 6 | 0.053 / 0.37 |
| All | MRR | 0.538 | 0.515 | −0.024 [−0.043, −0.006] | Wilcoxon, 44 better / 77 worse | 0.007 / 0.058 |

- **The headline claim is rejected.** A 40% relative gain on difficult clips means +25.0 pp of Hit@5. The upper end of the CI is +4.1 pp.
- **The direction is, if anything, negative.** Every point estimate for Hit@1 and MRR is below zero. The all-query MRR drop is nominally significant (p = 0.007) but only marginal after the Holm correction (p = 0.058). I report it as "no benefit, and weak evidence of a small harm", not as a confirmed harm.
- **Clips that were not difficult also moved.** Some of their frames were enhanced, and every query competes against the whole corpus. Changing the captions of difficult clips therefore also changes which distractors outrank the correct clip.

### 3.3 Dev vs test: a calibration lesson

| | n | Hit@5 | 95% Wilson CI |
|---|---|---|---|
| dev80 (config chosen on it) | 80 | 90.0 | [81.5, 94.9] |
| test320 | 320 | 68.8 | [63.5, 73.6] |

The intervals do not overlap (Fisher p < 0.001). Three things contribute, and the dev number mixes them together:
1. **Selection bias.** The dev score belongs to the configuration that won on dev.
2. **Corpus size.** Hit@5 among 80 candidates is a much easier task than among 320 (random ranking gives 6.3% against 1.6%).
3. **Difficulty mix.** Dev has 16% difficult clips; test has 46%.

The lesson, also stated in the README, is that the dev number must never be quoted as the system's accuracy.

### 3.4 Retriever ablation (dev80)

| Retriever | Hit@5 | MRR |
|---|---|---|
| Dense, frame captions | **90.0** | 0.771 |
| Dense, clip summaries | 83.8 | 0.684 |
| BM25 | 82.5 | 0.712 |
| Best 3-way RRF fusion | 88.8 | 0.774 |

Fusion did not beat the best single retriever on dev Hit@5. Under the pre-specified rule, the shipped default is dense frame retrieval alone. The other retrievers stay available through configuration.

### 3.5 Serving performance

| | Result |
|---|---|
| Query latency (embed + HNSW search, 5,123-frame index, CPU embeddings) | p50 37–45 ms, p99 < 60 ms |
| Captioning throughput, batch 1 | ~0.29 frames/s (3.4 s/frame) |
| Batched serving vs sequential, scan latency | +56% (1 stream), +28% p50 (2 streams), +24% p50 (4 streams) |
| Batched vs sequential agreement, YES/NO decisions | 92.9% |
| Batched vs sequential agreement, exact text | 52% |
| **vLLM 0.10.0 vs in-process transformers** (same NF4 checkpoint, one frame per call), scan latency p50 | **−43%** at 1, 2 and 4 streams (73.5 → 41.9 s; 145.0 → 81.1 s; 287.8 → 162.8 s) |
| vLLM vs transformers throughput | 0.19 → 0.33 frames/s (+75%) |
| vLLM vs transformers agreement, YES/NO decisions / exact text | 92.9% (13/14 frames) / 36% |
| vLLM, 4 concurrent vs one at a time (560 px, 1 tile) | ×1.04 throughput |

**Why vLLM is faster, and why batching still does not help.**
- vLLM runs the *same* checkpoint about 1.75× faster per request.
- Its request concurrency never exceeded 1 during the load test (sampled from `vllm:num_requests_running`). At 1120 px one image occupies 6,460 of the 7,728 KV-cache tokens that fit on 12 GB.
- Even at 560 px, where four requests fit, concurrency gained only 4%.
- The one decision disagreement is a frame where the transformers output contained a second, contradictory answer block (YES, then NO); the parser keeps the last one, while vLLM stopped after the first.
- This was measured on a single 14-frame clip, the same day, back to back (`artifacts/loadtest/backend_*.json`).

**Why batching lost.**
- The 4-bit (bitsandbytes NF4) weights only get a fast fused kernel for single-token, batch-1 matrix-vector products. At batch ≥ 2 every decoding step de-quantizes the weights instead.
- At batch 4 the 12 GB GPU overflows into system RAM, and throughput falls to 0.01 frames/s.
- The latency comparison has one scan per concurrency level, so it is a large, consistent effect rather than a precisely estimated one.

### 3.6 v2: adding a CLIP visual retriever

**Protocol.** The protocol was committed before any result existed (`eval/visual.py`, commit `5654aa6`).
- **Visual retriever.** CLIP ViT-L/14 embeds every 1-FPS frame with the standard processor (224 px, center crop) and no preprocessing. A clip's score is the max cosine between the raw query and its frames.
- **Fusion.** Weighted RRF (k = 60) of the frozen caption retriever (weight 1) and the visual retriever (weight w).
- **Candidates.** Caption only, visual only, and fusion with w ∈ {0.25, 0.5, 1, 2}.
- **Selection rule (dev80).** Highest Hit@5, then MRR, then the simplest configuration.
- **Test scoring.** The test run reads the selected configuration from the dev artifact. Primary test: Hit@5, selected vs caption only, exact McNemar.

**Dev80** (`artifacts/eval/dev80_visual_grid.json`):

| Config | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| Caption only (v1 system) | 67.5 | 90.0 | 92.5 | 0.771 |
| Visual only | 68.8 | 87.5 | 96.3 | 0.778 |
| Fusion w = 0.25 | 72.5 | 90.0 | 95.0 | 0.802 |
| Fusion w = 0.5 | 75.0 | 93.8 | 95.0 | 0.822 |
| **Fusion w = 1 (selected)** | 77.5 | 93.8 | 95.0 | **0.844** |
| Fusion w = 2 | 72.5 | 90.0 | 96.3 | 0.807 |

**Test320** (`artifacts/eval/test320_visual.json` and `stats_visual.json`). Paired, n = 320. Holm correction over 8 tests (4 metrics × {all, difficult}):

| Subset | Metric | Caption only | + CLIP fusion | Δ [95% CI] | Relative | p (Holm) |
|---|---|---|---|---|---|---|
| All | Hit@1 | 42.2 | 51.3 | +9.1 [4.7, 13.4] | +21.5% | 0.0005 |
| All | **Hit@5** | 68.8 | **74.4** | +5.6 [1.6, 9.7] | +8.2% | 0.029 |
| All | Hit@10 | 76.9 | 82.8 | +5.9 [2.5, 9.4] | +7.7% | 0.005 |
| All | MRR | 0.538 | 0.618 | +0.080 [0.051, 0.110] | +14.9% | < 0.001 |
| Difficult (147) | Hit@1 | 38.1 | 50.3 | +12.2 [6.1, 19.1] | +32.1% | 0.003 |
| Difficult | Hit@5 | 62.6 | 70.8 | +8.2 [2.0, 15.0] | +13.0% | 0.043 |
| Difficult | MRR | 0.490 | 0.595 | +0.104 [0.061, 0.149] | +21.3% | < 0.001 |
| Not difficult (173) | Hit@5 | 74.0 | 77.5 | +3.5 [−1.2, 8.7] | +4.7% | (not tested) |

On Hit@5, the two retrievers each find clips the other misses: fusion recovers 31 queries the caption system missed and loses 13.

Notes:
- **Where the idea came from.** The hypothesis behind v2 (captions are the bottleneck) came from error analysis of the v1 *test* results. Selection and protocol stayed on dev, but the choice of experiment was informed by the test set. That is a softer form of test-set reuse, and it is one reason for the independent check in 3.7.
- **Fusion vs CLIP alone (post-hoc).** The pre-specified test compares fusion with caption only. Fusion vs CLIP alone was checked afterwards on the same per-query ranks (`eval/fusion_vs_clip.py`, `artifacts/eval/fusion_vs_clip.json`). It is exploratory, with no new choices and Holm correction over 8 tests:

  | test320 (320) | CLIP only | Fusion | Diff [95% CI] | p (Holm) |
  |---|---|---|---|---|
  | Hit@1 | 46.9 | 51.3 | +4.4 [−0.6, 9.7] | 0.80 |
  | Hit@5 | 73.4 | 74.4 | +0.9 [−3.4, 5.3] | 1.0 |
  | Hit@10 | 82.2 | 82.8 | +0.6 [−3.1, 4.4] | 1.0 |
  | MRR | 0.577 | 0.618 | +0.041 [0.005, 0.078] | **0.031** |

  With one query per clip only MRR differs, and the Hit@5 CI (±4 pp) is too wide to say more. With all 20 MSR-VTT captions per clip (3.9), fusion does beat CLIP alone: Hit@5 +2.9 pp [+1.6, +4.3]. So captions add a small but real gain, at about 10× the indexing cost. On dev, fusion had led by 6 pp (93.8 vs 87.5), which was too optimistic.
- **Indexing cost.** This makes the cost question concrete.
  - CLIP indexing: 0.19 s per frame on the CPU (5,123 frames in 951 s, including video decoding).
  - Llama captions: about 2.0 s per frame on the GPU with vLLM, and 3.4 s per frame in-process.

  A CLIP-only index is therefore roughly 10× cheaper and needs no GPU. Captions still provide the human-readable "best frame" text and the `/analyze` feature.
- **Natural difficulty mix.** Reweighted to 1k-A's 14.7% difficult share, Hit@5 is 72.3% [66.6, 77.9] for caption only and 76.5% [70.9, 81.8] for fusion.
- **The gain is larger on difficult clips.** This is consistent with low light hurting the captioner more than CLIP's embedding. It is a subgroup observation, not a pre-specified hypothesis.
- **Serving.** The visual retriever is now the default in the API.
  - `/index` stores CLIP frame vectors in a second Redis vector store; `captions=false` indexes CLIP vectors only, with no vision LLM.
  - `/search` fuses the two retrievers with the dev-selected weights.
  - Implementation check: with the cached test320 vectors loaded, the API's search path gives Hit@5 74.38%, identical to the evaluation, and the same rank on 315/320 queries. The rest differ because HNSW search is approximate.
  - Cost: search latency rises from p50 ~40 ms to 106 ms (p99 138 ms), because the CLIP text encoder runs on the CPU.

### 3.7 Independent check on untouched clips

The protocol was committed before running (`eval/fresh680.py`). The 680 1k-A clips outside test320 had never been used in any experiment. By construction none of them is difficult, so this checks how well the visual retriever generalises, not the low-light findings. Fusion is not checked: these clips have no Llama captions, which would take about 6 GPU hours. All numbers are CLIP visual only, with 95% bootstrap CIs (`artifacts/eval/fresh680_visual.json`).

| Analysis | Queries / corpus | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| A. Fresh queries, fresh corpus | 680 / 680 | 39.4 | 64.4 [60.7, 67.9] | 72.9 | 0.510 |
| B. Fresh queries, full 1k-A corpus | 680 / 1,000 | 35.6 | 59.1 [55.4, 62.8] | 68.4 | 0.467 |
| C. Fresh queries, 320-clip corpora (200 random draws) | 320 / 320 | | **73.0** (draws: 69.4–76.3) | | |
| C. test320 not-difficult, for comparison | 173 / 320 | | 74.0 [67.1, 80.4] | | |
| D. All 1k-A queries, full corpus (reference; includes test320) | 1,000 / 1,000 | 35.2 | 58.5 | 69.0 | 0.463 |

- **The test320 result generalises.** At the same corpus size, fresh clips give 73.0% against 74.0% on test320's not-difficult queries, well inside its CI. As a sanity check, recomputing CLIP-only on test320 with this script reproduces 73.4%.
- **Corpus size dominates the headline number.** The same retriever drops from 73.0% (320 candidates) to 64.4% (680) and 59.1% (1,000). Hit@5 on a 320-clip library is therefore not comparable to published 1k-A results. Analysis D (R@1 35.2, R@5 58.5, R@10 69.0) is the number that sits in the standard 1k-A setting, for CLIP alone and zero-shot.

### 3.8 Pooling frame scores: max vs top-k mean

The visual retriever scores a clip by its single best frame. Averaging the top few frames is the cheapest way to use more of the clip: it needs no new model and no re-encoding.

The protocol was committed before running (`eval/aggregation.py`, commit `ec584e9`):
- Candidates: max, mean embedding, and the mean of the top 2, 3 or 5 frame scores.
- Selection used a new dev set, **dev1k**: 1,000 train-split clips (15,874 frames), disjoint from dev80 and 1k-A. dev80 was retired for this because on 80 candidates CLIP is near ceiling, and its earlier leads did not hold on test.
- Verification on fresh680 against the full 1,000-clip corpus; test320 was not used. Primary test: R@5, exact McNemar, Holm over 4 metrics.

On dev1k all candidates were within 0.7 pp on R@5 (max 59.2%, top5 59.9%). The rule selected **top5**.

| fresh680 queries, 1,000-clip corpus | max | top5 | Diff [95% CI] | p (Holm) |
|---|---|---|---|---|
| R@1 | 35.6 | 38.1 | +2.5 [+0.1, +4.7] | 0.14 |
| **R@5 (primary)** | 59.1 | 60.6 | +1.5 [−0.4, +3.5] | **0.37** |
| R@10 | 68.4 | 69.0 | +0.6 [−1.0, +2.4] | 0.61 |
| MRR | 0.467 | 0.486 | +0.019 [+0.004, +0.033] | < 0.001 |

- **The primary test failed.** top5 does not significantly improve R@5, so the default stays max.
- **Ranking order improves.** MRR is higher, with 232 queries ranking better and 117 worse. Small, consistent ranking gains without a recall gain are the same pattern as fusion in 3.6.
- On the 680-clip fresh corpus the picture is the same: R@5 +0.7 pp (p = 0.57), MRR +0.020.
- **Second use of fresh680.** Its first use (3.7) did not inform any choice made here.
- **Larger-sample follow-up (3.9):** with 20 queries per clip, top5 beats max on every metric (R@5 +1.4 pp [+0.8, +2.0]). The effect is real but small. The 680-query test was not powered to see it.

### 3.9 Twenty queries per clip

One query per clip leaves CIs of about ±4 pp on paired differences, so effects of 1–3 pp are invisible. MSR-VTT has 20 human captions per clip. The follow-up uses all of them as queries, with exact duplicates dropped. Queries from the same clip are correlated, so all inference is clustered by clip:
- CIs from a cluster bootstrap;
- p-values from a clustered sign-flip permutation test (10k each);
- Holm correction over 4 metrics.

The protocol was committed before running (`eval/multicap.py`, commit `906010a`); results are in `artifacts/eval/multicap.json`.

| Comparison | Queries / clips | Metric | A | B | Diff [cluster 95% CI] | p (Holm) |
|---|---|---|---|---|---|---|
| fresh680, 1,000-clip corpus: max (A) vs top5 (B) | 12,675 / 680 | R@1 | 35.5 | 36.8 | +1.3 [+0.6, +2.0] | < 0.001 |
| | | **R@5** | 59.2 | 60.6 | +1.4 [+0.8, +2.0] | < 0.001 |
| | | R@10 | 68.2 | 69.3 | +1.1 [+0.6, +1.6] | < 0.001 |
| | | MRR | 0.466 | 0.478 | +0.011 [+0.006, +0.017] | < 0.001 |
| test320, 320-clip corpus: CLIP only (A) vs fusion (B), post-hoc | 5,945 / 320 | Hit@1 | 46.9 | 48.4 | +1.5 [−0.2, +3.2] | 0.095 |
| | | **Hit@5** | 71.7 | 74.7 | +2.9 [+1.6, +4.3] | < 0.001 |
| | | Hit@10 | 79.7 | 82.6 | +2.9 [+1.5, +4.3] | < 0.001 |
| | | MRR | 0.582 | 0.603 | +0.021 [+0.008, +0.034] | 0.004 |

On the same 5,945 queries, caption-only retrieval reaches Hit@5 66.6% (reference only).

- **Both earlier "no difference" results were a lack of power, not a lack of effect.** The point estimates hardly moved: top5 +1.5 → +1.4 pp, fusion +0.9 → +2.9 pp, both inside the earlier CIs. With 20× the queries, the CIs shrank from about ±4 pp to about ±1 pp. Absence of significance at n = 320 or 680 meant "not measurable here", not "zero".
- **Status.** The top5 comparison is a pre-specified follow-up to 3.8. It does not change 3.8's failed primary test; it is reported next to it. The fusion comparison is post-hoc and the third look at test320. The 20 queries are not new clips, so the corpus-level findings still rest on 320 and 680 clips.
- **Effects are small.** Neither top5 pooling (+1.4 pp) nor fusion over CLIP (+2.9 pp) comes close to the gain of adding CLIP in the first place (+5.6 pp over captions).
- **Serving default unchanged.** `/search` still pools frames with max. Switching to top5 is now supported by evidence, but that evidence comes from fresh680, so the switch has not been verified on clips that played no part in the decision.

## 4. Discussion

### 4.1 Why preprocessing did not help

A modern vision-language model is already fairly robust to underexposure.
- Enhancement mostly *perturbs* captions rather than improving them. Of the 1,944 enhanced frames, 1,818 (94%) received a different caption. The median word-set Jaccard overlap with the original caption is only 0.50. In typical diffs the scene stays the same while details move: a "cell phone" becomes a "camera", or clothing colors change.
- Lighting vocabulary did **not** increase: words such as dim, dark, grainy or shadow appear in 1,077 original captions against 1,032 enhanced ones. The enhancement therefore did not make the model "see" darkness differently. It simply re-rolled the details, and that helps and hurts individual queries about equally.
- A claim like "+40%" plausibly comes from older detectors or captioners that are far less robust to lighting. It may also come from an evaluation without a held-out test set.

I did not try to tune the preprocessing until it looked good on test. Doing that would turn the test set into a training set.

### 4.2 A measurement confound I had to remove first

Midway through captioning, throughput fell from about 0.3 to 0.1 frames/s. Two causes were involved:
- **The main one:** Windows 11 power throttling (EcoQoS) was demoting the background process. With its kernel-launch thread starved, the GPU sat at idle clocks (throttle reason `0x1`), and decoding took about 300 ms/token instead of about 75 ms/token. The process now opts itself out of throttling (`scanner/winperf.py`).
- **A smaller one:** the WSL virtual machine had bloated after Docker builds and was holding system memory.

Every performance number above was measured after this fix. Without it, any latency comparison would have measured the OS scheduler, not the system under test.

### 4.3 How representative is 68.8%?

Reweighting the two strata to the natural 1k-A mix (14.7% difficult) gives a Hit@5 of **72.3% [66.4, 77.8]**.

This is still well below 85%, and it is measured on a 320-clip corpus. Standard full 1k-A benchmarks rank against 1,000 clips, where the same system would very likely score lower (not measured). The number is therefore not comparable to published MSR-VTT text-to-video results. Those systems are trained end-to-end for retrieval; this pipeline is zero-shot, captioning with a general model and matching text against text.

### 4.4 Limitations

- **One query per clip in the main tests.** This yields 320 queries. The CIs are about ±5 pp overall and ±7.5 pp on the difficult subset. Section 3.9 uses all 20 captions per clip, which narrows paired CIs to about ±1 pp. The number of distinct clips (320, 680) is unchanged, though.
- **"Difficult" is a pixel-statistics proxy,** not a human judgment of visibility.
- **One captioning model, one prompt and greedy decoding.** The preprocessing result may not transfer to weaker models.
- **The serving benchmark is small:** one 14-second clip, one scan per level, on a single laptop where the load generator shares the machine.
- **The test set has now been scored twice:** v1, and the pre-specified v2 CLIP experiment. Every other choice was made on dev80, which has only 80 queries. The dev grid differences (one or two queries) are within noise, so the choice between w = 0.5 and w = 1 is not meaningful. The test gain of w = 1 over caption only is.

## 5. Reproducing

```bash
python -m eval.clip_stats ...                     # lighting statistics for 1k-A
python -m eval.build_sets                         # dev80 / test320 (seed 0)
python -m eval.caption_set ...                    # captions, off and adaptive
python -m eval.tune_fusion ...                    # dev-only selection
python -m eval.hit_at_k ...                       # test320, each condition
python -m eval.stats --out artifacts/eval/stats.json
python -m eval.visual --set eval/sets/dev80.json   --index-name eval_dev_tune    --out artifacts/eval/dev80_visual_grid.json
python -m eval.visual --set eval/sets/test320.json --index-name eval_test320_off --select-from artifacts/eval/dev80_visual_grid.json --out artifacts/eval/test320_visual.json
python -m eval.stats_visual --out artifacts/eval/stats_visual.json
python -m eval.fusion_vs_clip --out artifacts/eval/fusion_vs_clip.json   # post-hoc
python -m eval.fresh680 --out artifacts/eval/fresh680_visual.json
python -m eval.aggregation --stage dev  --out artifacts/eval/dev1k_aggregation.json
python -m eval.aggregation --stage test --select-from artifacts/eval/dev1k_aggregation.json --out artifacts/eval/fresh680_aggregation.json
python -m eval.multicap --out artifacts/eval/multicap.json        # needs Redis index eval_test320_off
```

Raw outputs:
- `artifacts/eval/test320_off.json` and `test320_adaptive.json` (per-query ranks).
- `artifacts/eval/dev80_fusion_grid.json`.
- `artifacts/eval/stats.json`.
- `artifacts/loadtest/*.json`.
- `artifacts/eval/engine_check.json`.
