# PARSeq experiments and measured reference points

This repository is an experiment reference **and** a runnable training/export workflow. PARSeq Base, Tiny and the custom Fast Tiny recipe belong to one family; WATERec, Florence and CCT-S are separate families. The figures below describe particular trained checkpoints and cohorts, not promises for a fresh dummy-data run.

## Benchmark summary

RTX 5090; parameter counts are learned model parameters, not weight-file bytes or process memory. **NM means not measured in the current matched reference**, not zero. H×W is the configured fixed recognizer input.

| Variant / task | Parameters | H×W | Exact result / cohort | Mean normalized Lev. | RGB / encoded mean ms | GPU / host peak MiB |
|---|---:|---|---|---:|---:|---:|
| Base, ALPR | 23.980127M | 64×256 | 2,415/2,693 (89.68%), historical full cohort | NM | NM / NM | NM / NM |
| Tiny, ALPR | 6.091871M | 64×256 | NM | NM | NM / NM | NM / NM |
| Fast Tiny, ALPR | 6.048863M | 48×192 | **203/256 (79.30%), mixed timing subset only** | 96.50% | 0.686226 / 1.658176 | 682 / 533 |
| Base, price | 23.980896M | 64×256 | 148/206 (71.84%) | 87.47% | NM / NM | NM / NM |
| Tiny, price | 6.092256M | 64×256 | 118/206 (57.28%), native BF16 | 81.94% | NM / NM | NM / NM |
| Fast Tiny, price | 6.049248M | 48×192 | 112/206 (54.37%), TensorRT | 81.15% | 0.889019 / 1.094589 | 688 / 502 |

Price reference checkpoints include a SPACE-token extension, accounting for their slightly larger parameter counts; the pinned default upstream vocabulary remains non-space. Parameter counts also include positional embeddings at the stated geometry.

The 256-image Fast Tiny result is **not a full 2,693-image accuracy result**. Its normalized edit similarity can be high while whole-string accuracy is lower. Older aggregate metadata contains other historical Fast Tiny counts; this scoped table does not qualify those counts or attach them to the measured timing cohort. Standard Tiny is not the same recipe as Fast Tiny, and its missing ALPR result cannot be filled with a Fast Tiny score.

RGB-to-text begins with decoded RGB images. Encoded-to-text additionally includes local file reading and image decoding. Both include resize/normalization, transfer, OCR and text decoding; detection, network, model loading and first-use compilation are excluded. Fast Tiny uses five warmups, three complete passes and fresh processes at batch 1. The existing engine has a max-batch-32 profile; these B1 rows do not establish B32 throughput. GPU memory is the sampled driver-accounted process peak. Host RSS is from a **separate encoded-input functional probe**, including libraries, host buffers and transient inputs; it is neither minimum required RAM nor simultaneous timing evidence.

Base's older 3.997 ms B1 median / 914 MiB and approximately 941 crops/s at B32 remain historical snapshots in [reference_metrics.json](../results/reference_metrics.json). They are not new matched means and must not be subtracted from the Fast Tiny means above.

## What the price comparison says

Separate task checkpoints were evaluated on the same 206 price crops. Training pools, pretraining, resolution and decoding differ, so this is not an equal-training-budget architecture experiment.

| Family / task checkpoint | Exact text /206 | Whole-value numeric /206 | AI-reviewed numeric /144 |
|---|---:|---:|---:|
| PARSeq Base | 148 | 175 | 125 |
| PARSeq Tiny | 118 | 146 | 98 |
| Custom Fast Tiny | 112 | 135 | 88 |
| WATERec R2048 | 153 | 180 | 132 |
| Florence reference | 148 | 176 | 133 |
| PP-derived medium reference | 161 | 187 | 131 |

The reviewed 144 is a frozen subset of the 206 images, **not the 144-image internal training holdout**. AI review is not human certification. Some supplied labels omit or disagree with visible fractions, so supplied-label and reviewed metrics are both retained; neither is silently treated as new human truth. Exact text retains punctuation/currency after outer-space trimming. Numeric comparison parses the complete output, permits the declared decimal point/comma and optional `DH`, and rejects malformed or partial values. No convenient substring is counted as success.

These results make Fast Tiny a speed-oriented reference, not a Florence-equivalent complex-price reader. A lower-resolution image may lose small raised cents; an autoregressive or parallel decoder can also make recognition/ordering errors. The results do not isolate which cause dominates. PARSeq Base matching Florence's raw 148 does not imply identical predictions or equal reviewed-numeric accuracy.

## Recipes and difficulties

- **Base and Tiny:** official PARSeq permutation training with a pretrained visual encoder and one query decoder layer. Standard inference uses autoregressive decoding plus refinement. Tiny reduces encoder width from 384 to 192; it is not a smaller WATERec.
- **Fast Tiny:** the custom staged recipe first warms up at 64×256, adapts parallel prediction/refinement, then trains at 48×192 with generic synthetic replay. Selection uses only a label-disjoint holdout made from TRAIN. Each stage's full-data refit initializes from the previous full-data refit, rather than mixing held-out-selected and full-data states. The synthetic replay is image/text supervision, not box annotation.
- **Generalization:** high in-domain plate accuracy did not establish unseen-country, unfamiliar-font, cluttered-price or selective-reading ability. The existing separate 443-image international comparison gave legacy PARSeq 327/443; US was 164/222, stronger than WATE's 151/222 on that particular subset. These are distinct cohorts/checkpoints, not evidence of universal country coverage or pretraining novelty.
- **Rotation:** historical raw PARSeq and Fast Tiny outputs degrade severely under quarter-turns. See [capabilities](capabilities.md). No orientation correction is silently included in their scores.
- **Training is not solved by a speed result:** parallel refinement can produce a fast graph while losing difficult price accuracy. More capacity, altered resolutions or synthetic replay require their own trained checkpoints and held-out tests; do not inherit a previous engine's performance or accuracy.

The broader experiments also tried compact spatial/query readers, auxiliary heads and distillation. Several could fit small TRAIN sets yet failed generalization or preserved words while dropping fractions. Those negative experiments motivate keeping recognition prior, layout/order and input-detail tests separate. They do not prove that all compact models or all distillation methods fail. No unqualified custom architecture is offered as a supported PARSeq preset here.

## Checked workflow versus inherited evidence

The implementation's earlier reference workflow had actual native/ONNX/TensorRT and notebook checks; [validation_summary.json](../results/validation_summary.json) records that historical scope. The refreshed dummy-data/config/notebook documentation passed six CPU data/config/AST tests. That is **not** a fresh Base/Tiny/Fast Tiny training or GPU-export matrix. Dummy fixtures are fictional workflow examples, not a replacement for the private evaluation datasets. Private trained weights, images, labels, prediction rows and caches are not distributed.

For a new run, preserve split membership, image hashes, label policy, model initialization, software, precision, selection policy and output records. Validate native versus exported decoded strings on your own fixed images, including EOS and non-divisible batches. Record failures rather than relaxing numerical or text-retention gates after seeing results.

## Evidence and related repositories

[scenario_evidence.json](../results/scenario_evidence.json) contains only public-safe aggregates, method scope and SHA256 identifiers for the frozen source summaries. A digest is an evidence identifier, not a claim that private raw data is publicly reproducible. [reference_metrics.json](../results/reference_metrics.json) retains older snapshots rather than silently replacing them. Vendored PARSeq source is pinned to upstream commit `1902db043c029a7e03a3818c616c06600af574be` under Apache-2.0.

Related reference repositories:

- [PARSeq Base / Tiny / Fast Tiny](https://github.com/algofly-oss/parseq-ocr-training)
- [WATERec R / RS / optional speculative workflow](https://github.com/algofly-oss/waterec-ocr-training)
- [Florence-2 OCR training reference](https://github.com/algofly-oss/vllm-ocr-training)
- [Original compact CCT-S plate baseline](https://github.com/algofly-oss/license-plate-ocr-training)
