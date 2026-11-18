# PARSeq experiments and measured reference points

This repository is an experiment reference **and** a runnable training/export workflow. PARSeq Base, Tiny and the custom Fast Tiny recipe belong to one family; WATERec, Florence and CCT-S are separate families. The figures below describe particular trained checkpoints and cohorts, not promises for a fresh dummy-data run.

## Current corrected-reference development comparison

New price experiments use 676 fitting images and 144 development images under the **full amount plus visible currency** label convention. **Literal whole-string exact** preserves case, symbols, spaces and sequence order. All 144 images remain, including 45 unresolved images. These are development-selected results, not the historical 206-image cohort and not a new PARSeq evaluation.

| Current task checkpoint | Literal exact /144 |
|---|---:|
| WATERec R2048 | 106 (73.61%) |
| Florence-2 | 105 (72.92%) |
| Compact WATERec | 102 (70.83%) |
| Custom multiscale reader with context distillation | 98 (68.06%) |
| Custom multiscale reader with synthetic-layout stage | 97 (67.36%) |
| CCT, 64×128 | 66 (45.83%) |

A separate custom ALPR checkpoint scores929/1,167 (79.61%) on its development cohort; clean CCT selects953/1,167 (81.66%), with own TensorRT0.846 ms RGB /1.032 ms encoded,542 MiB. It is a38-class checkpoint, not the public37-class plate default. Pretraining, adaptation and selection budgets differ. Compact WATE's runtime retained its 102 correct strings without losing a previously correct output, but changed one already-incorrect string and failed its strict arithmetic qualification; it is an empirical retention result, not universal native equality. No existing PARSeq, historical price or ALPR score should be assigned to these new checkpoints.

The maintained [comparison tables](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md) and [exact-only PDF](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/reports/ocr_comparison_2026-11-17.pdf) join resolution, parameter count, measured RGB/encoded latency, GPU memory and available host RSS, with runtime and cohort limits. Current Florence is a native Torch BF16-autocast baseline: 35.366 ms RGB /35.527 ms encoded and 1,768 MiB, not the historical optimized TensorRT runtime; current-checkpoint TensorRT is unmeasured. CCT64 now measures 0.746 /0.937 ms and 542 MiB. CCT128 selected 65/144 versus 66/144 at64, with microbatch/RNG differences and no inference cost. A separate WATE B8 experiment retained all native correct outputs but was slower per image than static B1; neither unsuccessful operating-point change is promoted. The clean CCT ALPR result is now complete; no historical timing is assigned to it. Same-source WATE R1024 selected98/144 at epoch4; its qualified FP32 encoder / FP16 decoder costs2.859 /3.067 ms,668 MiB, slower and eight matches belowR2048. PPmedium104/144 now measures0.884 ms RGB /1.087 ms encoded,654 MiB under CTC-semantic retention; its failed numeric/frame gates remain disclosed. Global FP32 passes those stricter gates at1.413 /1.688 ms,588 MiB. Separate task checkpoints are permitted; approximately 1M parameters with WATE-like exact accuracy and easy 50/100/250-example fine-tuning remains an unmet research goal. Training inputs remain images and raw expected text only.

The sections below preserve historical reference measurements and their original reporting conventions; they are not current corrected-policy results or the selection/reporting policy for new training.

## Benchmark summary

RTX 5090; parameter counts are learned model parameters, not weight-file bytes or process memory. **NM means not measured in the current matched reference**, not zero. H×W is the configured fixed recognizer input.

| Variant / task | Parameters | H×W | Exact result / cohort | RGB / encoded mean ms | GPU / host peak MiB |
| --- | ---: | --- | --- | ---: | ---: |
| Base, ALPR | 23.980127M | 64×256 | 2,415/2,693 (89.68%), historical full cohort | NM / NM | NM / NM |
| Tiny, ALPR | 6.091871M | 64×256 | NM | NM / NM | NM / NM |
| Fast Tiny, ALPR | 6.048863M | 48×192 | **203/256 (79.30%), mixed timing subset only** | 0.686226 / 1.658176 | 682 / 533 |
| Base, price | 23.980896M | 64×256 | 148/206 (71.84%) | NM / NM | NM / NM |
| Tiny, price | 6.092256M | 64×256 | 118/206 (57.28%), native BF16 | NM / NM | NM / NM |
| Fast Tiny, price | 6.049248M | 48×192 | 112/206 (54.37%), TensorRT | 0.889019 / 1.094589 | 688 / 502 |

Price reference checkpoints include a SPACE-token extension, accounting for their slightly larger parameter counts; the pinned default upstream vocabulary remains non-space. Parameter counts also include positional embeddings at the stated geometry.

The 256-image Fast Tiny result is **not a full 2,693-image accuracy result**. Older aggregate metadata contains other historical Fast Tiny counts; this scoped table does not qualify those counts or attach them to the measured timing cohort. Standard Tiny is not the same recipe as Fast Tiny, and its missing ALPR result cannot be filled with a Fast Tiny score.

RGB-to-text begins with decoded RGB images. Encoded-to-text additionally includes local file reading and image decoding. Both include resize/normalization, transfer, OCR and text decoding; detection, network, model loading and first-use compilation are excluded. Fast Tiny uses five warmups, three complete passes and fresh processes at batch 1. The existing engine has a max-batch-32 profile; these B1 rows do not establish B32 throughput. GPU memory is the sampled driver-accounted process peak. Host RSS is from a **separate encoded-input functional probe**, including libraries, host buffers and transient inputs; it is neither minimum required RAM nor simultaneous timing evidence.

Base's older 3.997 ms B1 median / 914 MiB and approximately 941 crops/s at B32 remain historical snapshots in [reference_metrics.json](../results/reference_metrics.json). They are not new matched means and must not be subtracted from the Fast Tiny means above.

## What the price comparison says

Separate task checkpoints were evaluated on the same 206 price crops. Training pools, pretraining, resolution and decoding differ, so this is not an equal-training-budget architecture experiment.

| Family / task checkpoint | Exact text /206 |
| --- | ---: |
| PARSeq Base | 148 |
| PARSeq Tiny | 118 |
| Custom Fast Tiny | 112 |
| WATERec R2048 | 153 |
| Florence reference | 148 |
| PP-derived medium reference | 161 |

The historical206 price cohort is different from the internal144-image development cohort above. Historical checkpoints may have different training, label and selection policies; current training/scoring is literal whole-string exact only. Historical alternative scores remain in private evidence rather than public comparisons.

These results make Fast Tiny a speed-oriented reference, not a Florence-equivalent complex-price reader. A lower-resolution image may lose small raised cents; an autoregressive or parallel decoder can also make recognition/ordering errors. The results do not isolate which cause dominates. PARSeq Base matching Florence's raw 148 does not imply identical predictions or equal predictions on difficult images.

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

## Current price rotation diagnostic

On the same corrected price DEV144, native clockwise 90°/180°/270° literal counts are WATE:26/22/66, compact WATE:20/16/29, measured custom 97 model:1/4/1, native 98 candidate:0/3/2, CCT64 price:0/0/0 and current price Florence:45/4/49. Upright counts are 106/102/97/98/66/105 respectively; all upright saved strings/tokens match. No correction or angle oracle was applied. This is not ALPR, final-test or TensorRT rotation evidence.

Quarter-turn preprocessing differs: WATE preserves area at 256×128, custom models shrink to 128×64 (one-quarter of upright area), CCT stretches 64×128 and Florence uses 384×384;180° retains each upright geometry. These whole-pipeline results do not isolate encoder invariance. Tiny-model intrinsic price rotation remains unmet. The98 checkpoint gains six exact cases and loses five versus97; its own upright deployment measures1.539 /1.768 ms,552 MiB. The97 checkpoint remains a distinct measured point, not a borrowed timing. See the [complete native rotation table](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md#native-quarter-turn-reading-current-price-dev144) and [exact-only PDF](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/reports/ocr_comparison_2026-11-17.pdf).


An **optional learned orientation pipeline** (1,962,662 learned parameters, including a275,524-parameter head) gives **97/97/97/98 of144** atCW0/90/180/270, compared with the default98/0/3/2. It loses one previously correct upright output. Its own TensorRT pipeline reproduces the576 native views; all-angle RGB mean is **1.961 ms,560 MiB**, with upright mean1.920 ms. Encoded lossless-PNG mean2.734 ms is not a matched JPEG comparison. The head was selected on these same144 DEV images, not a blind rotation test; the default upright workflow is unchanged.

Variable-shape support is not scale-invariant accuracy: the same98 checkpoint gives **98,88,89,48,9 exact matches of144** at bounds128×256,96×192,160×320,192×384,256×512 respectively. These inference-only trials do not rule out higher-resolution training; no speed claim or default change follows.

Measured few-shot development curves for the synthetic-stage reader are **77/144 (53.47%) with50 real FIT images,83/144 (57.64%) with100, and87/144 (60.42%) with250**, versus97/144 (67.36%) for its676-image control. These nested single-seed runs each used13,520 task exposures and144 DEV labels to select among21 checkpoints. They also inherit public pretraining,384,000 natural-image exposures from20,000 unique crops, and32,000 synthetic exposures. They are not cold50-label adaptation, do not establish a simple user workflow, and have no own deployment timings.


Current PPmedium104/144 has two qualified operating points: mixed TensorRT **0.884 ms RGB /1.087 ms encoded,654 MiB** preserves canonical CTC output tokens and literal strings while two internal frame alignments differ; original numeric/frame failures remain recorded. The global FP32 control retains frames and passes its frozen numeric gate at **1.413 /1.688 ms,588 MiB**. Neither is a universal equivalence claim. A newer1.687M photometric-replay checkpoint scores **101/144 (70.14%)**, selected before further adaptation; its own B1 FP32 TensorRT/Graph4 deployment preserves all144 native strings/tokens/EOS at1.545 ms RGB /1.737 ms encoded,552 MiB. Similar runtime to98 is not a speed-win claim. All are single-seed, development-selected results on the same144 images, not final-test evidence.


A separate20-epoch matched training pair starting from the original98 checkpoint also did not favor160×320: the128 control selected98/144 at epoch0 (terminal95), while160 selected95/144 at epoch16 (terminal93), with13 gains and16 losses versus128. It improves over that frozen checkpoint’s160-bound89/144, but does not beat the128 control; it is not promoted.

The actual101 checkpoint also has a separately qualified **fixed128×256 batch1/2/4/8** TensorRT path, retaining all144 native strings/tokens/EOS at every capacity. At batch8, RGB is **8.373 ms per batch,1.047 ms per image amortized,955.5 images/s,634 MiB**; encoded is10.028 ms per batch. This is throughput, not a1 ms single-request latency. Keep the separate static-B1 **1.545 ms,552 MiB** point for single requests; the batch path atB1 is slower/larger (1.877 ms,618 MiB). There is no mixed-resolution or OOD qualification.


## Current fixed-checkpoint evaluation

The [maintained exact-only comparison](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md#current-fixed-checkpoint-heldout-results) now separates current checkpoints from historical models. Current ALPR WATE/PP/CCT/custom score2,476/2,321/2,226/2,180 exact of2,693 and855/606/628/438 of984, with all native/TRT strings and tokens retained. Clean Florence224 scores2,002/2,693 and797/984 natively; its own deployment cost is unmeasured and it is not the supplied strong Florence model. These historically reused cohorts are not newly blind tests;984 labels are AI-reviewed, not human-certified.

Current price206 exact counts are163 WATE,163 PP,152 custom101,159 Florence and91 CCT. All9 external photos remain, with6 accepted literal references and3 null/unscorable: WATE/custom each match1 of6; others0. Bounds acrossall9 are1–4 or0–3, not known nine-image accuracy. Mixed PP changes one already-incorrect206 string; other applicable backends retain every native string. Costs remain separately measured DEV results. No labels, weights, precision or input policies are changed using final results. See the [exact-only PDF](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/reports/ocr_comparison_2026-11-17.pdf).


## Reproduction boundary: public workflow versus measured checkpoint

The commands in these notebooks call public package modules and repository-relative configs; no private experiment directory is required to run a **new** workflow. They require your own image/TXT data, downloaded hash-verified pretrained assets, compatible dependencies and target-specific engine builds. Source archives contain code and fictional dummy data, **not the measured trained weights or private benchmark data**. Matching a command does not reproduce a published checkpoint or its counts automatically.

The legacy WATE notebook8 uses its own public R/RS training/holdout recipe; use the standalone WATE repository for the maintained standard workflow. PP notebook9 defaults to the public historical15-epoch preset with fit-only BN calibration and an auxiliary reader. Its separate optional corrected recipe now reproduces the newer20-epoch fixed-BN/micro1 operations; it still requires an explicit compatible initializer and does not supply the measured104 weights/data. Fast Tiny notebooks1–7 select the preset via the same `PARSEQ_CONFIG` everywhere and run warm start, parallel adaptation and small-input replay; the one-epoch dummy preset demonstrates software, not historical accuracy. Literal-only checkpoint selection differs from some historical recipes.

The current1.687M multiscale reader, its natural alignment/photometric replay/context-distillation stages, other current corrected-policy reference training runs and sealed aggregate evaluation are **not yet complete portable notebook replays** in these public workflows. The public compact WATE workflow is a different17.9M architecture, not this1.687M reader. Exact replay still needs released compatible training adapters/configs, lawful data or an explicitly different dummy substitute, frozen split/pixel/label/weight hashes, recorded launch/environment versions and actual native/deployment qualification. Do not substitute historical weights, alphabets, input policies or latency numbers. Full release-to-fresh-training-to-final-score reproduction has not been demonstrated.


## Optional corrected-price PP recipe

The default `configs/ppocr_price.yaml` is unchanged. `configs/ppocr_price_corrected.yaml` and CLI mode `--recipe corrected_fixed_bn_v1` are a distinct recipe: **20epochs, effective batch16, sequential microbatch1**, FP32 stored weights, CUDA BF16 training, FP32 evaluation, TF32 off, AdamW learning rate1e-4/weight decay3e-5, constant schedule, global gradient clipping5. All parameters—including BN affine weights—train; source BN running means, variances and counters remain fixed. No auxiliary reader, extra loss, hidden holdout or final refit is used. The source BN is reconstructed before strict state loading, without fresh calibration. A training-only BN output hook makes backward gradients contiguous without changing their values; native inference/export loads do not install it.

Input is **OpenCV linear stretch128×320, RGB→BGR, float/127.5−1**, producing40 CTC frames and96 output classes: blank0 plus the ordered95 printable ASCII characters, including literal spaces. Case/spacing are preserved; unsupported currency/scripts and targets needing more than40 frames (including repeated-character blanks) raise, rather than filter or truncate. Each image's unreduced CTC loss is divided by its target length, then micro losses are summed over the actual effective-batch image count; the676-image case has43 updates/epoch with a4-image tail,860updates and13,520 exposures overall. No batch-equivalence claim extends to stochastic training trajectories.

The base seed is20261114 by default. Epoch order uses a local `Random(seed+epoch)` permutation; each image uses seed `seed+epoch*10000+position` for independent brightness/contrast probabilities0.4 with factors0.8–1.2, then optional0.25 probability ±2° PIL bilinear rotation with gray127 fill. The same seed is reset before the student's stochastic forward. There is no geometry crop or text change. Evaluation writes durable raw predictions before a literal GT join. Epoch0 is diagnostic; epochs1–20 select the largest literal exact count, earliest tie, on the configured `val` **development** set. A later untouched test is required. Selected and terminal checkpoints are reloaded into fresh models/optimizers and fully replayed on that DEV set. Existing directories are refused; automatic resume is not implemented.

**Required ancestry:** the measured104 checkpoint started from an earlier proper-fitting price BN+auxiliary selection checkpoint, not directly from generic official weights. That ancestor used676 real fitting images, a separate144-image selection pool,128×320 input, batch16, BN calibration, a PARSeq Tiny auxiliary reader and an older selection/text policy; it selected epoch15 of30. Public default training shares components but uses its own internal split/current exact-only selection, so it does not reproduce that ancestor byte-for-byte. The historical initializer and exact private data are not bundled. Supply `initial_checkpoint` and its actual `initial_checkpoint_sha256`, with explicit `initial_codec: ppocr_ascii96_v1`; loading rejects a hash/shape mismatch before training and never silently substitutes generic initialization. You are responsible for the checkpoint's lawful source, character-row identities and validation/test ancestry.

```bash
# First edit dataset_root, initial_checkpoint and its SHA in this separate config.
# train/val are literal same-stem Image/TXT; val selects epochs here.
python -m parseq_ocr.ppocr_cli prepare --config configs/ppocr_price_corrected.yaml
python -m parseq_ocr.ppocr_cli train --config configs/ppocr_price_corrected.yaml --smoke-steps 2
python -m parseq_ocr.ppocr_cli train --config configs/ppocr_price_corrected.yaml
python -m parseq_ocr.ppocr_cli evaluate --config configs/ppocr_price_corrected.yaml
python -m parseq_ocr.ppocr_cli export --config configs/ppocr_price_corrected.yaml
python -m parseq_ocr.ppocr_cli build --config configs/ppocr_price_corrected.yaml
python -m parseq_ocr.ppocr_cli evaluate --config configs/ppocr_price_corrected.yaml --runtime tensorrt
```

The two-update smoke has its own `smoke` directory and is mandatory before full training with the same source/config/input/initial-state contract; full training writes `full_fit/model.pt` and `last.pt`. Optional `train_manifest`/`validation_manifest` JSONL preserves supplied record order and literal `text`, with `image_path` relative to each manifest; without manifests, image filenames are sorted. Config paths resolve relative to the repository, as for other PP presets. Matching row order, initial bytes, labels, augmentation, dependencies and hardware matters. CPU FP32 is a software check, not CUDA BF16 parity. Private GPU lease/memory-sampler governance and all original environment locks are not recreated by this portable trainer; no runtime or memory result is inherited. **Recipe reproduction does not supply the measured weights/data or promise104/144.** Qualify newly exported engines on your own data; the public builder's precision policy is not silently identical to every private benchmark branch.


A bounded private compatibility check of the optional corrected PP implementation used the actual15.58M initializer and two FIT images: initialized states, pixels, FP32 forward/loss, gradients, one Adam update and public inference/export-loader roundtrip matched the historical helper exactly, with BN buffers unchanged. This is CPU/global-batch2 evidence, not the full effective16 CUDA BF16 training run, measured-weight availability or an accuracy guarantee.


New PP ALPR48/128 checkpoints have independent empirical DEV operating points:1001/1167 at0.760 ms RGB /0.939 ms encoded,650 MiB;1027/1167 at0.964 /1.094 ms,654 MiB. Both use15.580M parameters and batch1; spatial inputs are48×320 and128×320. Exact DEV counts equal native, but two strings change per model; strict output/numeric gates remain failed. The48 point gains one correct case and loses one;128 changes two already-incorrect strings. Separate strict-FP32 controls are slower. **Their own heldout counts now differ by backend**: PP48 native/mixed gives2341/2339 of2693 and624/626 of984; PP128 gives2367/2368 and533/534. PP128 improves DEV/final but worsens reviewed984 versus48, so no blanket recommendation follows. Mixed strict-output failures remain; none of the older983-checkpoint results or costs is inherited. See the [full comparison and qualification scopes](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md).


A separate PLM/control study remains pending and has no completed result in this report. General-purpose few-shot transfer/usability and the approximately1M-parameter WATE-like accuracy/speed goal remain unproven. Source archives contain code, not measured trained weights.


Completed studies now add a negative matched natural-CTC result (custom control/CTC:ALPR933/927,price98/92) and all three official-generic PP few-shot outcomes:66/11/90 selected of144 at50/100/250 labels, including PP100 epoch0 selection/terminal0. Custom few-shot values77/83/87 inherit different pretraining; N also uses144 DEV annotations and is not N total. No causal family superiority or unique failure cause is established. Own control933 FP32 TRT costs1.833/2.063 ms,552 MiB; clean Florence224 native BF16 costs34.966/34.947 ms,1696 MiB, on full1167 DEV. Do not inherit older929 or historical Florence costs. A separate PLM/control study remains pending, without a quality claim. See the maintained comparison for exact scope and limitations.

### Optional official-weight PP precursor

`python -m parseq_ocr.ppocr_cli train --config configs/ppocr_price_precursor.yaml` runs the distinct `selection_only_precursor_v1` preset. Configure ordered FIT and DEV JSONL manifests (`image_path`, literal `text`) first. It starts from the pinned official PP conversion and public PARSeqTiny auxiliary weights, calibrates BN using at most128 FIT images in batches of32, then trains BN with CTC + auxiliary CE (weight1),30 epochs/B16, AdamW1e-4/wd3e-5/clip5. Workers default0; CPU requires explicit `device: cpu`, `precision: fp32`. Conflicting fixed recipe settings are rejected; use a fresh output directory. Auxiliary labels have a12-character limit and the ASCII96 CTC codec preserves spaces/case. No images are silently dropped.

The result is `selection/model.pt`, never a FIT+DEV full-refit checkpoint. `precursor_handoff.json` records its SHA, training configuration/source/environment and ordered FIT/DEV image-byte/RGB/literal-label hashes. Set `initial_checkpoint`, `initial_checkpoint_sha256`, `precursor_receipt` in the corrected config from this result, and reuse the identical manifests. Corrected training rejects membership/order/pixel/label changes when that receipt is supplied. Deliberately changing data is a separate experiment, not the same handoff. Notebook9 shows the calls, corrected two-update smoke and export. An explicit hash-bound private/user warm start remains supported without a precursor receipt.

This makes the recipe executable with user-supplied images/text and official weights; it does **not** reproduce the historical initializer or its measured accuracy. Historical labels changed (368 FIT/71 DEV), complete per-update provenance is unavailable, and conversion/worker/environment differences remain. The new precursor and subsequent corrected recipe require their own benchmark. The legacy demo still defaults to its original fresh full refit; never use that returned full-fit checkpoint as a clean-DEV initializer.

Use a fresh `pretrained_dir` for this precursor, or explicitly bind existing `model.onnx` / `trainable.onnx` with `expected_model_onnx_sha256` / `expected_trainable_onnx_sha256`. Converted-cache hashes are not proof of a trusted original conversion; review their provenance before opting in. Official source bytes are hash-checked, and actual converted bytes/source hashes are checked before and after training. The corrected smoke/full replay contract also binds the handoff receipt SHA. Restart the notebook kernel if a foreign `strhub` package was already imported.

### Main Fast Tiny notebooks: portable support versus measured-model replay

Notebooks 0–7 call the public config/data/train/inference/export/TensorRT/evaluation modules; they do not require private experiment paths. They support the Base/Tiny/Fast Tiny choice, with fixed configured input size and image/TXT labels. The saved [validation summary](../results/validation_summary.json) records a Fast Tiny synthetic end-to-end exercise: six fits (selection and full refit over three stages), one epoch per stage, 12 FIT/4 validation crops; eight notebook fresh-kernel checks; checked ONNX export; actual TensorRT 10.15.1.29 build and 17 predictions at batches 1/3/8 with changed inputs/revisited shapes. These are historical functional receipts, not a fresh GPU replay of every current source revision. Current exact-only/cache/selection changes have additional CPU regressions; no GPU execution is claimed for this documentation review.

Remaining replay gaps: measured task weights/private training data are not bundled; the dummy workflow does not reproduce 2377/2693 or 715/984, establish few-shot accuracy, or measure full-cohort latency. Docker execution was not performed in the recorded preparation environment. TensorRT depends on compatible NVIDIA hardware/software and needs local rebuilding; supported batch profiles do not imply arbitrary resolution. Check vocabulary and length limits before training. Historical aggregates whose decoder output was normalized before saving must not be presented as current literal exact accuracy. The maintained [corrected-policy comparison](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md) keeps different checkpoints/cohorts/backends separate.
