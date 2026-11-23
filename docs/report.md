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
- **Pre-registered selection rule.** On dev, the configuration with the best Hit@5 wins, with MRR as the tie-breaker. The grid covered 3 × 3 × 3 retriever weights × 2 RRF constants. After selection the weights were frozen in code (commit `5a62b2a`) before the test set was scored.

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

Fusion did not beat the best single retriever on dev Hit@5. Under the pre-registered rule, the shipped default is dense frame retrieval alone. The other retrievers stay available through configuration.

### 3.5 Serving performance

| | Result |
|---|---|
| Query latency (embed + HNSW search, 5,123-frame index, CPU embeddings) | p50 37–45 ms, p99 < 60 ms |
| Captioning throughput, batch 1 | ~0.29 frames/s (3.4 s/frame) |
| Batched serving vs sequential, scan latency | +56% (1 stream), +28% p50 (2 streams), +24% p50 (4 streams) |
| Batched vs sequential agreement, YES/NO decisions | 92.9% |
| Batched vs sequential agreement, exact text | 52% |

**Why batching lost.**
- The 4-bit (bitsandbytes NF4) weights only get a fast fused kernel for single-token, batch-1 matrix-vector products. At batch ≥ 2 every decoding step de-quantizes the weights instead.
- At batch 4 the 12 GB GPU overflows into system RAM, and throughput falls to 0.01 frames/s.
- The latency comparison has one scan per concurrency level, so it is a large, consistent effect rather than a precisely estimated one.

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

- **One query per clip.** This yields 320 queries. The CIs are about ±5 pp overall and ±7.5 pp on the difficult subset.
- **"Difficult" is a pixel-statistics proxy,** not a human judgment of visibility.
- **One captioning model, one prompt and greedy decoding.** The preprocessing result may not transfer to weaker models.
- **The serving benchmark is small:** one 14-second clip, one scan per level, on a single laptop where the load generator shares the machine.

## 5. Reproducing

```bash
python -m eval.clip_stats ...                     # lighting statistics for 1k-A
python -m eval.build_sets                         # dev80 / test320 (seed 0)
python -m eval.caption_set ...                    # captions, off and adaptive
python -m eval.tune_fusion ...                    # dev-only selection
python -m eval.hit_at_k ...                       # test320, each condition
python -m eval.stats --out artifacts/eval/stats.json
```

Raw outputs:
- `artifacts/eval/test320_off.json` and `test320_adaptive.json` (per-query ranks).
- `artifacts/eval/dev80_fusion_grid.json`.
- `artifacts/eval/stats.json`.
- `artifacts/loadtest/*.json`.
- `artifacts/eval/engine_check.json`.
