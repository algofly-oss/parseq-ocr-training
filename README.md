# PARSeq OCR Training

[Download the public energy-meter example](notebooks/00_download_ocr_example.ipynb), then open [1_dataset_preparation.ipynb](notebooks/1_dataset_preparation.ipynb). Set `PARSEQ_CONFIG=configs/energy_meter.yaml` in the kernel environment (for example, `%env PARSEQ_CONFIG=configs/energy_meter.yaml`) before running dataset preparation. Run the numbered notebooks in the same kernel environment.

[Train in Jupyter](notebooks/2_model_training.ipynb) · [Copy the TensorRT runtime](notebooks/6_trt_inference.ipynb). The notebooks expose the training loop, predictions, and automatic engine build.

## Compare the tested TensorRT profiles

### Price tags

![Price crop example](dummy_data/prices/val/000.png)

Expected output is `107.25`. On real tags, cents can be raised or smaller, a currency symbol can split the amount, and unit prices can compete with the headline price. [Open a real benchmark price tag](https://prices.openfoodfacts.org/img/price-tags/000/123/000123564.webp)

![Real benchmark price tag](https://prices.openfoodfacts.org/img/price-tags/000/123/000123564.webp)

| Model | Params | Input | Corrected tags | Public photos | Latency¹ | GPU¹ |
|---|---:|---|---:|---:|---:|---:|
| WATERec rank-192 | 20.84M | ≤896 patches | **163/206** | **49/71** | 1.17 ms | 724 MiB |
| [PP-OCRv6](https://github.com/algofly-oss/ppocrv6-ocr-training) Medium | 15.58M | 128×320 | **163/206** | 6/71 | 0.77 ms | 656 MiB |
| PP-OCRv6 Small | 3.04M | 128×320 | 160/206 | 13/71 | **0.60 ms** | 652 MiB |
| [Florence-2](https://github.com/algofly-oss/florence2-ocr-training) Base | 231.41M | 384×384 | 157/206 | 3/71 | 4.95 ms | 1,452 MiB |
| [PARSeq](https://github.com/algofly-oss/parseq-ocr-training) Fast Tiny | 6.05M | 48×192 | 44/206 | 2/71 | 0.89 ms | 688 MiB |

¹ Mean from decoded RGB on the corrected price crops, RTX 5090, batch one. GPU memory is the sampled process peak. Ranking prioritizes corrected exact matches, then public exact matches. The corrected set includes 68 provisional labels.

### License plates

![Plate crop example](dummy_data/plates/val/000.png)

Expected output is `AB100CD223`. On real crops the reader must keep the character order across unfamiliar fonts, stacked rows, glare, and blur.

![Plate crop used for OCR evaluation](docs/images/openalpr-ayo9034-crop.png)

The international evaluation uses this crop from the OpenALPR benchmark. The recognizer receives the cropped plate and the expected text is `AYO9034`. [Source photograph](https://github.com/openalpr/benchmarks/blob/9790ed20d475be1e940a7e2a3f492a97c2ebed9b/endtoend/br/AYO9034.jpg)

| Model | Params | Input | Reviewed unseen | Local validation | Latency² | GPU² |
|---|---:|---|---:|---:|---:|---:|
| WATERec speculative | 27.51M | ≤512 patches | **879/984** | **2,476/2,693** | 1.07 ms | 700 MiB |
| Florence-2 Base | 231.41M | 224×224 | 857/984 | 2,381/2,693 | 4.88 ms | 1,130 MiB |
| PARSeq Fast Tiny | 6.05M | 48×192 | 715/984 | 2,377/2,693 | 0.69 ms | 682 MiB |
| [CCT-S](https://github.com/algofly-oss/cct-ocr-training) | 0.98M | 64×128 | 628/984 | 2,226/2,693 | 0.85 ms | 542 MiB |
| PP-OCRv6 Medium | 15.58M | Height 48 | 626/984 | 2,339/2,693 | 0.76 ms | 650 MiB |

² Mean from decoded RGB, RTX 5090, batch one. WATERec and Florence-2 timing used the 984 reviewed crops. Other timings come from separate workloads. Ranking uses reviewed unseen exact matches.

All accuracy scores count complete exact strings. Use WATERec when unfamiliar plates or price layouts matter most. PP-OCRv6 Small is the fastest tested option on the corrected price crops. Expand the detailed tables below for more variants, encoded timings, and protocol notes.

<details>
<summary>All benchmark variants and historical tables</summary>

<!-- ocr-family-comparison:start -->


### Naturally tiny plate crops

These 32 visually scorable crops were originally 32–47 pixels wide. Labels were transcribed before inference and score the complete alphanumeric registration in reading order. Printed separators are omitted from the references. The small subset includes 31 distinct registrations. [Source dataset](https://huggingface.co/datasets/sonnetechnology/license-plate-text-recognition-full)

| TensorRT model | Parameters | Model input | Exact match original / 2× Lanczos | Mean latency original / 2× Lanczos | Sampled GPU memory |
| --- | ---: | --- | ---: | ---: | ---: |
| WATERec | 27.51M | Up to 512 patches | 11/32 / 21/32 | 1.17 / 1.23 ms | 692 MiB |
| PARSeq Fast Tiny | 6.05M | 48×192 | 13/32 / 15/32 | 0.52 / 0.58 ms | 682 MiB |
| Florence-2 | 231.41M | 224×224 | 14/32 / 13/32 | 4.88 / 5.01 ms | 1,130–1,132 MiB |
| CCT-S | 0.98M | 64×128 | 4/32 / 5/32 | 0.48 / 0.54 ms | 742 MiB |

Batch-one timings start with decoded RGB on the RTX 5090 and include any upscaling. Each profile used eight warmup images and five passes. This subset measures OCR on known plate crops.


## Choose an OCR model

### Price-tag OCR

#### Price development

| Rank | Model family | Variant | Parameters | Input H×W | Exact match | RGB / encoded mean | GPU peak |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | Florence-2 | Base, optimized TensorRT price | 231.41M | 384×384 | 106/144 (73.61%) | 5.37 / 5.62 ms | 1,454 MiB |
| 1 | [WATERec](https://github.com/algofly-oss/waterec-ocr-training) | R2048, price | 26.12M | 128×256 | 106/144 (73.61%) | 1.97 / 2.20 ms | 716 MiB |
| 3 | [Florence-2](https://github.com/algofly-oss/florence2-ocr-training) | Base, PyTorch price | 231.41M | 384×384 | 105/144 (72.92%) | 35.37 / 35.53 ms | 1,768 MiB |
| 4 | [PP-OCRv6](https://github.com/algofly-oss/ppocrv6-ocr-training) | Medium mixed, price | 15.58M | 128×320 | 104/144 (72.22%) | 0.88 / 1.09 ms | 654 MiB |
| 4 | PP-OCRv6 | Medium FP32, price | 15.58M | 128×320 | 104/144 (72.22%) | 1.41 / 1.69 ms | 588 MiB |
| 6 | CCT | CCT64, price | 0.98M | 64×128 | 66/144 (45.83%) | 0.75 / 0.94 ms | 542 MiB |
| 7 | CCT | CCT128, price | 1.15M | 128×256 | 65/144 (45.14%) | — | — |

#### Broader corrected price

| Rank | Model family | Variant | Parameters | Input H×W | Exact match | RGB / encoded mean | GPU peak |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | WATERec | Original TensorRT | 26.12M | ≤1,024 patches | 167/206 (81.07%) | 1.11 / 1.31 ms | 702 MiB |
| 2 | [WATERec](https://github.com/algofly-oss/waterec-ocr-training) | R2048, price | 26.12M | 128×256 | 163/206 (79.13%) | 1.97 / 2.20 ms | 716 MiB |
| 2 | PP-OCRv6 | Currency-adapted TensorRT | 15.58M | 128×320 | 163/206 (79.13%) | 0.77 / 1.00 ms | 656 MiB |
| 4 | WATERec | Photo-adapted TensorRT | 26.12M | 896 patches | 162/206 (78.64%) | 1.07 / 1.29 ms | 696 MiB |
| 5 | PP-OCRv6 | Currency-adapted native | 15.58M | 128×320 | 161/206 (78.16%) | 5.43 / 5.43 ms | 724 MiB |
| 6 | PP-OCRv6 | Small, currency-adapted TensorRT | 3.04M | 128×320 | 160/206 (77.67%) | 0.60 / 0.82 ms | 652 MiB |
| 7 | Florence-2 | Base, PyTorch price | 231.41M | 384×384 | 159/206 (77.18%) | 33.37 / 35.30 ms | 1,768 MiB |
| 8 | Florence-2 | Base, optimized TensorRT price | 231.41M | 384×384 | 157/206 (76.21%) | 4.95 / 5.22 ms | 1,452 MiB |
| 9 | WATERec | Rank 128, photo-adapted TensorRT | 17.87M | 896 patches | 154/206 (74.76%) | 1.08 / 1.34 ms | 696 MiB |
| 10 | PARSeq | Fast Tiny, TensorRT price | 6.05M | 48×192 | 44/206 (21.36%) | 0.89 / 1.11 ms | 688 MiB |

#### Unseen retail photos

The three public photo cohorts contain 71 scorable references. Rows are ordered by full-string exact matches across the three cohorts.

| Model and TensorRT variant | Parameters | Input | Earlier | Confirmation | Fresh | Total exact | Tiled WebP mean | Tiled GPU peak |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| WATERec rank 192 + paired-format head (diagnostic) | 20.84M | ≤896 patches | 23/31 | 15/23 | 11/17 | 49/71 (69.01%) | 2.95–5.36 ms | 702 MiB |
| WATERec full + paired-format head (experimental) | 26.15M | ≤896 patches | 23/31 | 14/23 | 10/17 | 47/71 (66.20%) | 2.98–5.43 ms | 702 MiB |
| WATERec full + formatting head (experimental) | 26.15M | ≤896 patches | 23/31 | 9/23 | 11/17 | 43/71 (60.56%) | 2.92–5.33 ms | 702 MiB |
| WATERec rank 128 + formatting head (experimental) | 17.89M | ≤896 patches | 18/31 | 9/23 | 8/17 | 35/71 (49.30%) | 2.95–5.25 ms | 702 MiB |
| WATERec full, photo-adapted | 26.12M | ≤896 patches | 18/31 | 5/23 | 5/17 | 28/71 (39.44%) | — | — |
| WATERec rank 128, photo-adapted | 17.87M | ≤896 patches | 17/31 | 7/23 | 4/17 | 28/71 (39.44%) | — | — |
| PP-OCRv6 Small, currency-adapted | 3.04M | 128×320 | 8/31 | 2/23 | 3/17 | 13/71 (18.31%) | — | — |
| PP-OCRv6 Medium, currency-adapted | 15.58M | 128×320 | 3/31 | 0/23 | 3/17 | 6/71 (8.45%) | — | — |
| Florence-2 Base, optimized | 231.41M | 384×384 | 3/31 | 0/23 | 0/17 | 3/71 (4.23%) | — | — |

The timing range spans the three cohort means from encoded bytes in RAM at batch one. The paired-format head reads the visible separator and currency from image features. The full paired model matched stock at 162/206 on corrected crops. The rank-192 diagnostic read 163/206. Its encoder engine was 23% smaller and averaged 1.174 ms from decoded crops versus 1.135 ms for the full paired model. Preparing lossless, model-sized BMPs at ingestion preserved all 71 predictions and ran in 1.07–1.13 ms at 698 MiB. The prepared-input timings exclude ingest conversion.

#### Historical price

| Rank | Model family | Variant | Parameters | Input H×W | Exact match | RGB / encoded mean | GPU peak |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | [WATERec](https://github.com/algofly-oss/waterec-ocr-training) | Compact restart2, static batch 1 | 17.91M | 2,048 → 512 selected patches | 161/206 (78.16%) | 1.39 ms / — | 708 MiB |
| 1 | [PP-OCRv6](https://github.com/algofly-oss/ppocrv6-ocr-training) | Medium, historical price | 15.58M | 128×320 | 161/206 (78.16%) | 1.18 ms / — | 654 MiB |
| 3 | WATERec | R rank128 | 17.86M | 2,048 patches | 159/206 (77.18%) | 1.62 ms / — | 714 MiB |
| 4 | WATERec | R1024 | 26.12M | 88×180 | 154/206 (74.76%) | 1.09 ms / — | 700 MiB |
| 5 | WATERec | R2048 | 26.12M | 128×256 | 153/206 (74.27%) | 1.68 ms / — | 716 MiB |
| 6 | WATERec | R512 | 26.12M | 64×128 | 149/206 (72.33%) | 0.82 ms / — | 688 MiB |
| 7 | [PARSeq](https://github.com/algofly-oss/parseq-ocr-training) | Base, price | 23.98M | 64×256 | 148/206 (71.84%) | — | — |
| 8 | PARSeq | Tiny, native BF16 price | 6.09M | 64×256 | 118/206 (57.28%) | — | — |
| 9 | PARSeq | Fast Tiny, price | 6.05M | 48×192 | 112/206 (54.37%) | 0.89 / 1.09 ms | 688 MiB |

### License-plate recognition (LPR)

#### LPR development

| Rank | Model family | Variant | Parameters | Input H×W | Exact match | RGB / encoded mean | GPU peak |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | [WATERec](https://github.com/algofly-oss/waterec-ocr-training) | Adapted 512-patch TensorRT, plates | 26.12M | 64×128 dev | 1,075/1,167 (92.12%) | 1.27 / 1.44 ms | 686 MiB |
| 2 | PP-OCRv6 | Medium height 128, plates | 15.58M | 128×320 bound | 1,027/1,167 (88.00%) | 0.96 / 1.09 ms | 654 MiB |
| 3 | PP-OCRv6 | Medium height 48, plates | 15.58M | Height 48 | 1,001/1,167 (85.78%) | 0.76 / 0.94 ms | 650 MiB |
| 4 | [CCT](https://github.com/algofly-oss/cct-ocr-training) | CCT-S, plates | 0.98M | 64×128 | 953/1,167 (81.66%) | 0.85 / 1.03 ms | 542 MiB |

#### LPR validation

| Rank | Model family | Variant | Parameters | Input H×W | Exact match | RGB / encoded mean | GPU peak |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | [WATERec](https://github.com/algofly-oss/waterec-ocr-training) | Adapted 512-patch TensorRT, plates | 26.12M | 64×128 dev | 2,477/2,693 (91.98%) | 1.27 / 1.44 ms | 686 MiB |
| 2 | WATERec | Speculative TensorRT with orientation correction | 27.76M | ≤512 patches | 2,476/2,693 (91.94%) | — | — |
| 3 | WATERec | RS, plates | 26.12M | Aspect-preserving, ≤512 patches | 2,475/2,693 (91.90%) | 1.26 / 2.26 ms | 694 MiB |
| 4 | Florence-2 | Base, optimized TensorRT plates | 231.41M | 224×224 | 2,381/2,693 (88.41%) | — | — |
| 5 | PARSeq | Fast Tiny, plates | 6.05M | 48×192 | 2,377/2,693 (88.27%) | 0.69 / 1.66 ms | 682 MiB |
| 6 | [PP-OCRv6](https://github.com/algofly-oss/ppocrv6-ocr-training) | Medium height 128, plates | 15.58M | 128×320 bound | 2,368/2,693 (87.93%) | 0.96 / 1.09 ms | 654 MiB |
| 7 | [PP-OCRv6](https://github.com/algofly-oss/ppocrv6-ocr-training) | Medium height 48, plates | 15.58M | Height 48 | 2,339/2,693 (86.85%) | 0.76 / 0.94 ms | 650 MiB |
| 8 | [CCT](https://github.com/algofly-oss/cct-ocr-training) | CCT-S, plates | 0.98M | 64×128 | 2,226/2,693 (82.66%) | 0.85 / 1.03 ms | 542 MiB |

#### Reviewed unseen plates

| Rank | Model family | Variant | Parameters | Input H×W | Exact match | RGB / encoded mean | GPU peak |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | WATERec | Speculative TensorRT | 27.51M | ≤512 patches | 879/984 (89.33%) | 1.07 / 2.71 ms | 700 MiB |
| 2 | WATERec | Speculative TensorRT with orientation correction | 27.76M | ≤512 patches | 878/984 (89.23%) | 1.17 / 2.84 ms | 700 MiB |
| 3 | Florence-2 | Base, optimized TensorRT plates | 231.41M | 224×224 | 857/984 (87.09%) | 4.88 / 6.56 ms | 1,130 MiB |
| 4 | [WATERec](https://github.com/algofly-oss/waterec-ocr-training) | Adapted 512-patch TensorRT, plates | 26.12M | 64×128 dev | 855/984 (86.89%) | 1.27 / 1.44 ms | 686 MiB |
| 5 | WATERec | Rank 128, TensorRT plates | 17.86M | ≤512 patches | 849/984 (86.28%) | 1.18 / 2.82 ms | 696 MiB |
| 6 | PARSeq | Fast Tiny, plates | 6.05M | 48×192 | 715/984 (72.66%) | 0.69 / 1.66 ms | 682 MiB |
| 7 | [CCT](https://github.com/algofly-oss/cct-ocr-training) | CCT-S, plates | 0.98M | 64×128 | 628/984 (63.82%) | 0.85 / 1.03 ms | 542 MiB |
| 8 | PP-OCRv6 | Medium height 48, plates | 15.58M | Height 48 | 626/984 (63.62%) | 0.76 / 0.94 ms | 650 MiB |
| 9 | PP-OCRv6 | Medium height 128, plates | 15.58M | 128×320 bound | 534/984 (54.27%) | 0.96 / 1.09 ms | 654 MiB |

The optional RGB-array API for speculative WATERec averaged 0.981 ms on the same 984 upright plates, preserving all output strings at 700 MiB. BMP reached 1.050 ms from in-memory bytes with about twice the original PNG payload.

- **Fast recognition on a tight latency budget:** start with PP-OCRv6. Its currency-adapted TensorRT build reached 163/206 in the corrected-crop comparison.
- **Unseen plate layouts:** start with WATERec. The 512-patch speculative checkpoint reached 879/984 on reviewed unseen crops. With orientation correction it read 880/984 at 90°, 880/984 at 180°, and 881/984 at 270°.
- **Smallest model:** CCT-S is about 1M parameters. Use it when the input resembles its training distribution.
- **Custom text selection and ordering:** benchmark Florence-2 on your own annotation policy. Its faster TensorRT path lost two exact matches versus PyTorch on the broader price-tag evaluation.

Timings are batch 1 on an RTX 5090, with preprocessing, recognition and text decoding included. Encoded scope includes image decoding and may include file reads in historical runs. GPU peak is driver-accounted memory. Exact match uses the literal full string. The LPR validation table includes separate WATERec checkpoints identified in the variant column. Compare accuracy within the same label version and split: historical PARSeq price scores use an earlier label policy, and the broader corrected price cohort includes 68 provisional labels. A dash means a comparable result is unavailable. The corrected price timings use the same 206 images for PP-OCRv6, the photo-adapted WATERec profiles, Florence-2 and Fast Tiny. The two original WATERec rows retain development-crop timings. The reviewed unseen plate timings for speculative WATERec, rank-128 WATERec and Florence-2 use the same 984 images. Other timing rows use their original benchmark workloads. The latest Florence values are research measurements; see its repository for the executable versions.

<!-- ocr-family-comparison:end -->

</details>

A standalone, notebook-first workflow for **PARSeq Base, PARSeq Tiny and the custom Fast Tiny recipe**: image/TXT data → training → PyTorch inference → ONNX → TensorRT. The numbered workflow follows [vllm-ocr-training](https://github.com/algofly-oss/florence2-ocr-training). No other checkout is needed to use these PARSeq workflows.

**WATERec is a separate model family and a separate standalone repository.** Use the sibling [`waterec-ocr-training`](https://github.com/algofly-oss/waterec-ocr-training) checkout for WATERec R/RS and its optional speculative decoder. Fast Tiny is PARSeq-derived; it is not a WATERec preset. Existing legacy WATE/PP entrypoints and notebooks 8/9 remain for compatibility, while notebooks **0–7 are the main PARSeq workflow**.

## Measured reference points first

For the separately trained corrected-policy checkpoints and their current DEV/heldout/backend results, use the maintained [current comparison](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md). In particular, current Florence price 159/206 belongs to a different corrected-policy checkpoint from the historical 148/206 below; neither its accuracy nor timing transfers between checkpoints.

This is also a record of experiments, benchmarks and failures—not just a training product. See [experiment methods/results](docs/experiments.md), the [scenario capability matrix](docs/capabilities.md) and [aggregate evidence metadata](results/scenario_evidence.json). Separate trained checkpoints and cohorts are explicit; dummy-data execution is not accuracy evidence.

| PARSeq reference | Params / H×W | Exact matches / cohort (accuracy) | RGB / encoded mean ms | GPU / host peak MiB |
|---|---|---|---:|---:|
| Base, ALPR | 23.980M /64×256 | Not verified for literal output | NM /NM | NM /NM |
| Fast Tiny, ALPR | 6.049M /48×192 | **2,377/2,693 (88.27%)** | 0.686226 /1.658176 | 682 /533 |
| Base, price | 23.981M /64×256 | 148/206 (71.84%) | NM /NM | NM /NM |
| Tiny, price (native BF16) | 6.092M /64×256 | 118/206 (57.28%) | NM /NM | NM /NM |
| Fast Tiny, price | 6.049M /48×192 | 112/206 (54.37%) | 0.889019 /1.094589 | 688 /502 |

NM means unmeasured in the current matched runtime reference. Fast Tiny’s fresh literal replay verifies **2,377/2,693 validation and 715/984 reviewed unseen matches** ([receipt](results/literal_alpr_replay_2026_11_17.json)); the historical timing measurement uses 256 crops and is not a new full-cohort latency measurement. Current accuracy reporting uses literal full-string exact match only. RTX5090 timing includes preprocessing/transfer/OCR/text decoding; encoded adds local file/image decode. Host peaks come from separate functional probes, including libraries and buffers, not minimumRAM. Standard Tiny ALPR is unmeasured. The historical Base ALPR aggregate is not currently certified against unnormalized decoder output. See the linked methods before comparing rows or old snapshots.

The price comparison exposes a limitation: Fast Tiny is fast but has only 112/206 exact matches, versus Florence’s 148/206. Two-line ordering, raised cents, clutter, country/font shifts and rotation must be tested separately; no universal90% or50-shot capability is claimed. Related references: [WATERec](https://github.com/algofly-oss/waterec-ocr-training), [Florence](https://github.com/algofly-oss/florence2-ocr-training), [CCT-S](https://github.com/algofly-oss/cct-ocr-training).

| Option | Starting representation | Decode / training |
| --- | --- | --- |
| PARSeq Base | Official 384-wide encoder | Permutation training; autoregressive decode plus refinement |
| PARSeq Tiny | Official 192-wide encoder | Same standard objective, smaller backbone |
| Custom Fast Tiny | PARSeq Tiny | Warm start → parallel adaptation → smaller-input replay; parallel decode plus two refinements |

Supply cropped words, plates or price tags. This is a recognizer, not a full-frame detector. Dummy fixtures establish the software workflow only: they do not establish accuracy, arbitrary reading order, rotation robustness, a 50-example capability or a latency/memory target. Compare trained weights on your own untouched validation set before choosing a model.

## Start with bundled dummy data

The repository includes `dummy_data/plates/{train,val}` and `dummy_data/prices/{train,val}`: 16 TRAIN and 4 validation image/TXT pairs per task, generated synthetic examples rather than a benchmark. Notebooks default to `configs/dummy_tiny.yaml`; choose `dummy_base.yaml` or `dummy_fast_tiny.yaml` for the other options. All three use short CPU/FP32 settings. Training still downloads the official generic PARSeq initialization with a checked SHA; no pretrained or private weights are bundled.

```bash
python tools/make_dummy_data.py  # verifies existing fixtures; never overwrites user data
# After installing dependencies below:
parseq-ocr train --config configs/dummy_tiny.yaml
parseq-ocr infer --config configs/dummy_tiny.yaml --image dummy_data/plates/val/000.png
parseq-ocr export --config configs/dummy_tiny.yaml
parseq-ocr infer --config configs/dummy_tiny.yaml --backend onnx --image dummy_data/plates/val/000.png
```

Change `dataset_root` to `dummy_data/prices` to exercise the same interface on decimal strings, or point it at your own folder. `train/` supplies the internal holdout and optional full-training refit; `val/` is read only for reporting, never epoch selection. Every image has a same-stem UTF-8 `.txt` expected transcript. Labels outside the model vocabulary or length capacity raise an error rather than being silently truncated. Label normalization is explicit in each YAML; choose `none` to preserve case, punctuation and spacing (unsupported vocabulary characters still fail validation).

The original `alpr_base.yaml`, `alpr_tiny.yaml` and `alpr_fast_tiny.yaml` are longer GPU training recipes. The one-epoch dummy presets are not production recipes. TensorRT is GPU-only; exported engines use the selected fixed input height/width and a configured batch profile, not arbitrary-resolution inference.

## Start with Docker

Install Docker Compose and the NVIDIA Container Toolkit, then run:

```bash
cp .env.example .env
docker compose up --build
```

- JupyterLab: http://localhost:8780/lab?token=parseq-local
- Optional crop-label editor: http://localhost:8000

Change `JUPYTER_TOKEN` in `.env` if desired. Both ports bind to localhost. The GPU image uses PyTorch 2.10 and CUDA 12.8, rather than the older CUDA image in the Florence reference. Model/data downloads remain on disk between restarts.

Open `notebooks/` in JupyterLab and follow the numbered notebooks. To run only JupyterLab, use `docker compose up --build jupyterlab`.

## Native environment

Python 3.10–3.12 is supported by the pinned environment. Use a compatible CUDA-enabled PyTorch build for your GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[notebooks,onnx]'
# Optional NVIDIA deployment dependencies:
pip install -e '.[tensorrt]'
jupyter lab notebooks
```

TensorRT is optional until notebook 6. CPU use is supported for PyTorch/ONNX by setting `device: cpu` and `precision: fp32`; TensorRT requires CUDA. Training on CPU is primarily useful for functional checks.

## Notebook workflow

| Notebook | Purpose |
| --- | --- |
| `0_model_options.ipynb` | Choose Base/Tiny/Fast Tiny and understand their tradeoffs |
| `1_dataset_preparation.ipynb` | Validate image/TXT pairs, check overlap, optionally split annotated crops |
| `2_model_training.ipynb` | Download verified pretrained weights, select epochs, refit on all training data |
| `3_inference.ipynb` | Load your trained checkpoint and run PyTorch inference |
| `4_onnx_export.ipynb` | Export split graphs (Base/Tiny) or one complete graph (Fast Tiny) |
| `5_onnx_inference.ipynb` | Run those graphs with ONNX Runtime |
| `6_trt_inference.ipynb` | Build target-specific TensorRT engines and recognize crops |
| `7_evaluation_and_benchmark.ipynb` | Evaluate accuracy/generalization and benchmark fresh processes |
| `8_waterec_workflow.ipynb` | Legacy compatibility; prefer the separate WATERec repository |
| `9_ppocr_price.ipynb` | Legacy optional PP-derived workflow, separate from PARSeq |

Each notebook initializes its own paths and configuration. It does not require variables left in another notebook's kernel.

## Legacy WATERec integration (separate family)

For new WATERec work, use the separate standalone sibling repository. The following retained section documents compatibility with older releases; it is not a dependency of notebooks 0–7.

### WATERec for varied plate layouts

[Notebook 8](notebooks/8_waterec_workflow.ipynb) covers the separate WATERec model family. Use `configs/waterec_alpr.yaml` for RS initialization and plate transcription, or `configs/waterec_price.yaml` for R initialization with an explicit space token. Both default to a 512-patch, aspect-preserving input budget. These are not PARSeq presets; use the dedicated entry point:

The ALPR recipe adds 2048 known-label synthetic single-/two-line examples to training only. Install DejaVu fonts (included in the Dockerfile), or set `synthetic_font_dir` to their directory. The generator records font hashes, fallback use and Pillow version; font availability can change rendered examples across hosts. Price training uses no synthetic plate replay. `synthetic_plate_count: 0` disables this augmentation but changes the measured recipe.

The WATERec heads cover printable ASCII, with space added for price tags, and labels up to 25 characters. Non-ASCII scripts or currency symbols require an expanded vocabulary and new training. The plate benchmarks establish performance on their tested transcriptions and layouts, not universal multilingual OCR.

```bash
python -m parseq_ocr.waterec_cli train --config configs/waterec_alpr.yaml
python -m parseq_ocr.waterec_cli export --config configs/waterec_alpr.yaml
python -m parseq_ocr.waterec_cli build --config configs/waterec_alpr.yaml
python -m parseq_ocr.waterec_cli infer --config configs/waterec_alpr.yaml --backend tensorrt --image path/to/crop.jpg
```

Install `.[onnx,waterec-tensorrt]` for WATERec deployment. The tested runtime uses TensorRT 10.15.1.29, PyCUDA 2026.1 and CUDA bindings 12.9.4; engine building and kernel compilation require a compatible NVIDIA CUDA development environment. Engines are target-specific. The cached decoder runtime uses static batch 1; lists are processed sequentially. It does not use PyTorch for TensorRT inference.

Our final trained WATERec TensorRT plate model scored **91.90% (2475/2693)** on validation and **89.43% (880/984)** on reviewed unseen plates, compared with optimized Florence's 88.41% and 87.09%. The default cached runtime measured **2.71 ms/image including file decoding**, or **1.64 ms from decoded RGB**, at **688 MiB** driver-accounted GPU memory on an RTX5090. This exceeds the strict 1 ms target. FP16 TensorRT differed from native FP32 on two of 3677 predictions; revalidate conversion on your data. The source release includes neither private data nor trained weights, and a different training dataset will produce different results.

On a separate 443-crop Brazil/Europe/US benchmark, the frozen WATERec engine scored **332/443 (74.94%)**, versus the current Florence engine's **191/443 (43.12%)** and an older trained PARSeq model's 327/443. PARSeq remained stronger on the US subset (164/222 versus WATERec's 151/222), so the improvement is not uniform across regions. None of these 443 crops exactly match the ALPR training or primary 984-crop set by RGB content; original public pretraining exposure is unknown.

Price-tag performance needs a separate decision. The historical static-B1 WATERec R512 engine scored149/206 literal exact and matched all 206 native strings and prepared patch arrays. It measured1.069 ms including file decoding /0.822 ms decoded RGB at688 MiB. This is the original autoregressive reader, not the speculative ALPR draft. Different checkpoints, training pools and reference policies must remain separate; every space, symbol and sequence position matters in current literal scoring.

Automatic rotation correction is not enabled by default. The experimental orientation head recovered quarter-turn plate accuracy but slightly reduced upright accuracy. Treat it as a separate pipeline requiring its own validation.

Notebook 8 also covers `train-orientation`, `export-orientation` and `build-orientation`. The head must be trained for the exact final OCR encoder; export checks its checkpoint hash. Supply disjoint training-only fit/holdout manifests, then explicitly pass `--orientation --backend tensorrt` to inference, evaluation or benchmarking. The orientation deployment lives in `deployment_orientation/`, separately from the default engines. Corrected crops require a second encoder pass. If the final OCR encoder was refitted on the orientation holdout images, disclose that inherited text-training exposure when reporting head selection results.

### Optional exact GPU preprocessing

CPU Pillow preprocessing remains the default. Add `--gpu-preprocess --backend tensorrt` to WATERec inference, evaluation or benchmarking to move bicubic resizing, normalization and patch packing to CUDA. Image-file decoding and RGB conversion still run on the CPU. This option uses the existing 512-patch engines; it requires no retraining or re-export and can be combined with the separately validated `--orientation` option. It is not supported by the PyTorch backend.

```bash
waterec-ocr infer --config configs/waterec_alpr.yaml --backend tensorrt --gpu-preprocess --image crop.jpg
waterec-ocr evaluate --config configs/waterec_alpr.yaml --backend tensorrt --gpu-preprocess --folder dataset/val --output artifacts/waterec_alpr/evaluation_gpu_preprocess
waterec-ocr benchmark --config configs/waterec_alpr.yaml --backend tensorrt --gpu-preprocess --folder dataset/val --predecoded --output artifacts/waterec_alpr/benchmark_gpu_rgb.json
```

Use a Linux CUDA development environment with a working C compiler (`CC`, or `cc` by default) and `nvcc`; the Docker image installs the C compiler. On first use, the coefficient helper compiles into `${XDG_CACHE_HOME:-~/.cache}/parseq-ocr/resize`, keyed by source/compiler identity. CUDA kernels compile locally through PyCUDA. Source distributions contain only C/CUDA text and the Pillow license, with no compiled binaries. Exclude compilation/setup from steady-state timings and report whether image-file decoding was included. Benchmark JSON identifies CPU/GPU preprocessing and hashes its source files.

Public validation found bitwise-identical prepared inputs and predictions on all **4,120** tested plate crops, unchanged orientation predictions/angles on **4,136** cases, and exact results through **80** queued buffer/cache lifecycle checks. These tests establish parity on those inputs, not every image/runtime combination. The paired measurements below compare this opt-in path directly with CPU preprocessing. Keep instances alive to reuse bounded coefficient/geometry caches and call `close()` when finished; the recognizer remains single-crop and is not thread safe.

| Pipeline | Preprocessing | File-to-text mean ms | Predecoded RGB-to-text mean ms | GPU MiB |
| --- | --- | ---: | ---: | ---: |
| Default OCR | CPU Pillow | 2.709 | 1.643 | 688 |
| Default OCR | GPU opt-in | 2.273 | 1.261 | 694 |
| Optional orientation | CPU Pillow | 2.825 | 1.805 | 688 |
| Optional orientation | GPU opt-in | 2.461 | 1.411 | 694 |

Fresh isolated processes on RTX 5090, the same 256 mixed plate crops, batch 1 and five warmups; the public one-step cached decoder is unchanged. Both scopes include resize, normalization, patch packing, inference and text decode. File-to-text additionally includes local image-file read/JPEG decode; network, detection and model loading are excluded. These are measured values, not universal guarantees. Orientation timing depends on the frequency of corrections; rotated crops may need a second encoder pass.

## PP-derived price recognition

For current corrected-reference results, use the [maintained comparison and PDF](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md). Current fixed price Florence scores159/206; the historical 148/206 below belongs to different labels/checkpoints and is not overwritten. Current task-specific/native/TRT scopes are reported separately.

A separate opt-in [corrected-price recipe](docs/experiments.md#optional-corrected-price-pp-recipe) is available as `configs/ppocr_price_corrected.yaml` / `--recipe corrected_fixed_bn_v1`: fixed source BN, effective16/micro1,20epochs and literal DEV selection. It **requires your compatible initial checkpoint and SHA256**; neither the measured104 warm start nor its private data is bundled. The default recipe below is unchanged.

[Notebook 9](notebooks/9_ppocr_price.ipynb) covers the custom price reader. This is a PP-OCRv6-medium-derived model with an ASCII output head, recovered batch normalization and an auxiliary reader used only during training. Deployment retains the compact CTC model. It is not an unchanged upstream PP-OCR preset.

```bash
pip install -e '.[ppocr,ppocr-tensorrt]'
python -m parseq_ocr.ppocr_cli prepare --config configs/ppocr_price.yaml
python -m parseq_ocr.ppocr_cli train --config configs/ppocr_price.yaml
python -m parseq_ocr.ppocr_cli export --config configs/ppocr_price.yaml
python -m parseq_ocr.ppocr_cli build --config configs/ppocr_price.yaml
python -m parseq_ocr.ppocr_cli infer --config configs/ppocr_price.yaml --runtime tensorrt --image path/to/price.jpg
```

Preparation verifies pinned official assets and converts the Paddle inference model through ONNX into the trainable graph. It requires the extra Paddle/ONNX conversion dependencies; the tested conversion environment is Python 3.10. Training calibrates batch normalization using fit images only, selects epochs on a grouped training holdout, then refits from the same generic initialization. The historical winning recipe selected by exact holdout text; the preset preserves that choice. Blank labels are excluded explicitly. Decimal punctuation, inner spaces and currency text are preserved.

The historical PP price TensorRT model scored161/206 literal exact. It measured1.326 ms file-to-text /1.125 ms decoded RGB at670 MiB on RTX5090. Native and TensorRT predictions can differ; evaluate the actual engine and batch size you deploy. The managed runtime supports batches1–32; throughput and B1 latency are different measurements.

## Choose Base, Tiny or Fast Tiny

Start with [notebook 0: model options](notebooks/0_model_options.ipynb). It explains the available models, settings, measured tradeoffs, and how standard Tiny differs from the custom Fast Tiny recipe now included here.

| Option | `model` value | Config | Output directory |
| --- | --- | --- | --- |
| PARSeq-base, approximately 24M parameters | `parseq` | [`alpr_base.yaml`](configs/alpr_base.yaml) | `artifacts/parseq_base` |
| PARSeq-tiny, approximately 6M parameters | `parseq-tiny` | [`alpr_tiny.yaml`](configs/alpr_tiny.yaml) | `artifacts/parseq_tiny` |
| Fast Tiny, approximately 6M parameters | `parseq-tiny` + `recipe: fast_tiny` | [`alpr_fast_tiny.yaml`](configs/alpr_fast_tiny.yaml) | `artifacts/parseq_fast_tiny` |

Base and standard Tiny use the 64×256 training pipeline. Fast Tiny ends at 48×192 with parallel prediction and two refinement passes. In each notebook's first code cell, choose a config with `CONFIG_PATH`; the optional `PARSEQ_CONFIG` environment variable overrides its default. Use the same config throughout notebooks 1–7 and restart the kernel when changing models. The original `configs/alpr.yaml` remains the default for compatibility.

```bash
python -m parseq_ocr.cli train --config configs/alpr_base.yaml
python -m parseq_ocr.cli train --config configs/alpr_tiny.yaml
python -m parseq_ocr.cli train --config configs/alpr_fast_tiny.yaml
```

Every size/recipe needs its own checkpoint, export and target-specific engines. **Fast Tiny is our custom recipe built on official PARSeq Tiny, not a separate upstream model.** Choosing the standard Tiny preset does not enable it. CCT-S and Florence-2 are comparison references and are not trained in this repository.

### Fast Tiny workflow

Choose `configs/alpr_fast_tiny.yaml` in every numbered notebook, or set `PARSEQ_CONFIG=configs/alpr_fast_tiny.yaml` before launching JupyterLab. The same `train`, `infer`, `export`, `build`, `evaluate` and `benchmark` commands work for all three presets:

```bash
python -m parseq_ocr.cli train --config configs/alpr_fast_tiny.yaml
python -m parseq_ocr.cli export --config configs/alpr_fast_tiny.yaml
python -m parseq_ocr.cli build --config configs/alpr_fast_tiny.yaml --max-batch 32
python -m parseq_ocr.cli infer --config configs/alpr_fast_tiny.yaml --backend tensorrt --image path/to/crop.jpg
```

Training performs three stages:

1. Standard Tiny warm start at 64×256, using the permutation objective (up to 20 epochs).
2. Parallel adaptation at 64×256, supervising the initial prediction and both refinements (up to 5 epochs).
3. Transfer to 48×192 by interpolating image positional embeddings, with parallel supervision and generic synthetic replay (up to 5 epochs). One random uppercase/digit synthetic image is added per four real images, about 20% of the effective batch. System fonts are used when available; Pillow provides a fallback.

All stages use the same training-only, label-disjoint holdout. A selection stage starts from the preceding **selection** checkpoint. Its full-data refit starts from the preceding **full-data** checkpoint and uses the selected epoch count. Validation and external images are never read during training. Refinement prefixes use model predictions rather than target characters. Fast Tiny defaults to a 12-character limit plus EOS; long labels fail explicitly. The output alphabet is unchanged, so it does not add space support automatically.

The current Fast Tiny config validator requires `refit_all_training: true`: staged selection and full-refit initialization remain separate. For a deliberate selection-checkpoint inference/export audit, pass its path with `--checkpoint`; this does not change the standard staged recipe. Completed-stage reuse verifies normalized labels, decoded RGB pixels, geometry and fit/held membership before writing the split. Changed inputs and legacy runs without this fingerprint require a new output directory.

The final model and `selection/` remain at the usual artifact paths. Intermediate stages live under `warm_start/` and `parallel_adaptation/`; `recipe_summary.json` records the stages. ONNX export writes **`model.onnx`**, embedding image normalization, encoder, 13 parallel queries and both refinements. TensorRT writes **`model.engine`**. PyTorch, ONNX Runtime and TensorRT share the same `infer(images)` interface. The default TensorRT build allows FP16 in the encoder and constrains decoder/refinement arithmetic to FP32. `trt_precision: fp32` enables a full-FP32 control; all builds still require accuracy validation.

`cuda_graph: true` enables replay with persistent GPU buffers and a separate execution context per used batch size. First use of a batch size warms and captures it; exclude that setup from steady-state timings. Instances are not thread safe. Torch provides the CUDA graph API, so no PyCUDA or extra CUDA Python package is required. Set `cuda_graph: false` and re-export to use ordinary TensorRT enqueue.

Historical ~0.99 ms latency used an additional all-FP16/OpenCV-area runtime variant. This portable preset retains Pillow-bicubic preprocessing and conservative decoder precision; it does **not** promise that exact latency or the same accuracy. Measure on your target hardware and dataset. Price-tag experiments showed that Base was substantially more accurate than Tiny/Fast Tiny, and Florence handled complex price layouts better on a visual-review subset. Fast Tiny prioritizes speed, not universal Florence-level accuracy.

## Data and configuration

Edit [`configs/alpr.yaml`](configs/alpr.yaml). Paths resolve relative to the repository root, so the same config works from a notebook or CLI.

```text
dataset/
  train/
    example.jpg
    example.txt
  val/
    another.png
    another.txt
```

**Training contract: image + expected output text only.** Supply the intended input image and complete desired string, including text selection and output order. The WATERec, PARSeq and PP recognition loaders use these pairs; bounding boxes, character positions, region labels and separate ordering annotations must not enter training, initialization or selection thresholds. The model must learn selection and ordering from image/text supervision; this is a capability to validate, not a guarantee of the data format. Causal/padding masks and learned internal attention are model computations, not supplied annotations. Optional crop utilities prepare existing crop datasets; recognition training does not require bounding boxes.

Each UTF-8 text file contains that image's complete expected output. Supported image extensions are JPG, JPEG, PNG, BMP and WebP. Image stems must be unique within a split.

Alternatively, put cropped images into `annotator/static/images/`, label them in the editor, and let notebook 1 create train/val folders. The editor writes matching `.txt` files and does not delete originals.

The ALPR default uppercases labels and removes whitespace while preserving punctuation. There are no country-specific regex corrections or character substitutions. The released character head covers 94 printable ASCII characters excluding spaces, and supports labels up to 25 characters for Base/standard Tiny, or 12 by default for Fast Tiny. Unsupported characters and oversized labels cause an error; labels are never silently truncated. `normalization: none` preserves case and punctuation for compatible word datasets.

The default recipe uses a 64×256 input, full-model fine-tuning, AdamW at 1e-4, BF16, mild brightness/contrast/rotation augmentation, and gradient clipping. An internal holdout groups identical text labels and uses only the training split. The selected epoch count is then refitted from the same pretrained initialization on all training images. The supplied `val/` and any external dataset remain outside training/epoch selection.

Change `output_dir` for a new recipe. Resume restores locally generated model/optimizer state; CUDA and data-worker randomness can make resumed augmentation trajectories differ from an uninterrupted run.

## CLI equivalents

Run from the repository root after installing the package:

```bash
python -m parseq_ocr.cli train --config configs/alpr.yaml
python -m parseq_ocr.cli infer --config configs/alpr.yaml --image path/to/crop.jpg
python -m parseq_ocr.cli export --config configs/alpr.yaml
python -m parseq_ocr.cli build --config configs/alpr.yaml --max-batch 32
python -m parseq_ocr.cli infer --backend tensorrt --image path/to/crop.jpg
python -m parseq_ocr.cli evaluate --backend tensorrt --folder dataset/val --output artifacts/evaluation
python -m parseq_ocr.cli benchmark --backend tensorrt --folder dataset/val --batch-size 1 --repeats 3
```

Artifacts are written under the configured output directory:

```text
artifacts/parseq_alpr/
  config.json
  split.json
  selection/{best.pt,last.pt,summary.json,history.jsonl}
  full_fit/{model.pt,last.pt,summary.json,history.jsonl}
  deployment/{encoder.onnx,decoder.onnx,refine.onnx,metadata.json,...}
```

For the standard split-graph exporter, parallel decoding adds `parallel.onnx`; refinement is omitted when `refinements: 0`. Encoder and decoder permit FP16, while refinement stays FP32 because TensorRT 10.15 crashed building that graph with FP16 enabled during the reference experiment. Fast Tiny enables TensorRT runtime activation resizing where available and allocates scratch per captured batch shape. A fresh batch-1 instance therefore avoids reserving batch-32 activation memory. Capturing additional batch sizes retains their buffers until `close()`, so measure memory in a fresh process at your deployment batch size. Engines are hardware/runtime-specific. Fast Tiny instead exports a complete single graph, with its own decoder precision constraints and optional CUDA graph replay. The cache checks source hashes and target settings; use `build --force` to deliberately rebuild.

## Historical model comparison on both tasks

These are historical task-specific trained checkpoints: a family may use different plate and price weights, resolution, decoder or backend. Training and selection histories differ; these runs are not retroactively claimed to use today’s literal-only selection policy. Current selection/reporting compares the complete output string without case conversion, trimming, whitespace removal or amount equivalence. The internal144-image development cohort is separate from the206 historical price crops.

**Archived comparison, not the current literal-exact leaderboard.** Some legacy ALPR predictions were normalized before persistence; the Base/PP legacy aggregates below are not certified literal-output counts. Historical numbers are retained as reported, not relabeled as fresh exact results. Fast Tiny’s later literal replay is 2377/2693 and 715/984 ([receipt](results/literal_alpr_replay_2026_11_17.json)); its 203/256 entry below describes the older timing subset only.

| Historical model/checkpoint | Legacy ALPR reported count (2,693) | Legacy reviewed reported count (984) | Historical price exact (206) |
| --- | ---: | ---: | ---: |
| Florence-2 optimized | 2381/2693 (88.41%) | 857/984 (87.09%) | 148/206 (71.84%) |
| WATERec final (512 patches) | 2475/2693 (91.90%) | 880/984 (89.43%) | 149/206 (72.33%) |
| PP medium (separate plate/price models) | 2310/2693 (85.78%) | 571/984 (58.03%) | 161/206 (78.16%) |
| PARSeq Base 64×256 | 2415/2693 (89.68%) | 747/984 (75.91%) | 148/206 (71.84%) |
| PARSeq Tiny 64×256 | Not measured | Not measured | 118/206 (57.28%) |
| Fast Tiny 48×192, archived timing snapshot | Full cohort not qualified in this snapshot;203/256 timing subset | Not qualified in this snapshot; later literal715/984 above | 112/206 (54.37%) |
| CCT-S notebook model | 2315/2693 (85.96%) | 644/984 (65.45%) | 100/206 (48.54%) |
| TrOCR-small (separate task models) | 2350/2693 (87.26%) | 274/984 (27.85%) | 115/206 (55.83%) |

Plate results use the evaluated TensorRT checkpoints except TrOCR (native). Price Florence, WATERec, Base, Fast Tiny, PP and CCT-S use TensorRT with separate task-specific checkpoints; standard Tiny uses native BF16 and TrOCR uses its separate native price checkpoint. CCT-S prices use a separate 39-symbol head, not the original plate head. The standard Tiny ALPR cells remain unmeasured because a verified full-cohort result was not available; Fast Tiny is a distinct custom training recipe, not a substitute for that missing result. The 984 reviewed unseen plates and the separate 443 international crops are different cohorts; their counts are never combined.

For these historical checkpoints, WATERec has the highest listed exact counts on both primary plate cohorts, and PP has the highest listed price exact count. This does not establish universal dominance across fonts, countries or layouts. WATERec, PARSeq and PP workflows are provided here; comparison families are not additional trainable presets.

### Same 50-label price adaptation (research comparison)

These runs use the **same 50 original price-training images**, seed 15001, 200 optimizer updates and batch 8. All 50 images receive gradients. Results are measured once at the fixed terminal checkpoint on the separate **144-image internal holdout**; this is **not** a subset of the external206-image cohort. Labels and update counts are matched; computation, resolution, objectives and trainable parameter counts are different. This single-seed experiment does not establish a general 50-shot guarantee.

| Initialization and adaptation | Internal price exact (144) | Trainable parameters | Input / objective |
| --- | ---: | ---: | --- |
| Generic Florence-2 base-ft, decoder only | 92/144 (63.89%) | 57,500,160 | 384×384; native BPE cross-entropy; beam 5 |
| Provided ALPR-trained Florence-2, decoder only | 98/144 (68.06%) | 57,500,160 | 384×384; native BPE cross-entropy; beam 5 |
| Generic WATERec R, decoder only | 71/144 (49.31%) | 4,807,680 | Aspect-preserving 512-patch budget; native AR cross-entropy |
| Generic WATERec R, Q/V LoRA + space rows | 53/144 (36.81%) | 25,344 | Same 512-patch budget and AR objective; rank 8, scale 1 |
| Generic PP medium + training-only reader auxiliary | 65/144 (45.14%) | 16,253,954 including auxiliary | 128×320; CTC plus training-only auxiliary |

The provided Florence initialization already had ALPR training; it is not a generic initialization or evidence of learning ALPR from only 50 labels. These adapted checkpoints have **no measured deployment latency or memory here**. Do not assign them the runtime measurements of the separately trained full-data models below. In particular, the existing full-820-image PP price model remains **161/206 literal exact**, about **1.125 ms decoded-RGB-to-text / 670 MiB**; the full-820-image WATERec price model remains **149/206 exact / 688 MiB**. Their performance is not a result of these 50-label runs.

Research evidence is distributed separately from this source repository. Paths relative to that evidence bundle are `natural_ocr_pretraining/{florence50_price,wate50_price,wate50_price_lora,ppmedium50_price}/` (protocols, terminal results and independent audits); the comparison PDF is `report/ocr_model_comparison_2026-11-15.pdf`. Private crops, labels and trained checkpoints are not included in this source release.

A separate fresh generic WATERec price refit used all 820 training images and its original internal selection policy. Its fixed checkpoint scored150/206 literal exact, versus149/206 for the existing model, with11 exact gains and10 losses. It was closed without export, timing or promotion and inherits neither0.823 ms timing nor688 MiB memory. Initialization and augmentation differ, so this is not a selection-policy-only causal comparison. Its historical selection ancestry is not the current exact-only policy.

### Private exact-order Florence rank optimization

A separate private Florence runtime combines exact GPU conversion of resized RGB pixels with parallel full-vocabulary beam ranking, preserving the original normalization reduction and tie ordering. It matches **all 4,326 frozen strings**, and all 96 tested preprocessing cases match the reference pixel tensors. The model remains **231.414 million parameters**, with the same 224×224 plate / 384×384 price inputs and beam 5. Final combined, fresh-process original-rank/CPU→parallel-rank/GPU measurements were: plates **6.350→5.066 ms decoded RGB** and **7.326→6.092 ms file-to-text**, at **1,130 MiB** for both; prices **7.258→5.359 ms RGB** and **8.447→5.689 ms file-to-text**, at **1,176→1,178 MiB**. Each scope uses separate fresh processes, five warmups and one pass over the same 256 plate/unseen or 206 price images. These are complete-pipeline comparisons, not sums of improvements measured in separate snapshots; the historical Florence rows below remain historical references. This still does **not** meet the 1 ms / 700 MiB target. The private inference tool adds no Florence training preset, weights or runtime implementation to this repository. Evidence-bundle paths: `florence/exact_preprocessing/gpu_conversion/{independent_full_replay_audit,independent_combined_timing_audit}.json`.

### Private batch prototype and official Mirage audit

The public WATERec runtime still executes **batch 1 per image**, including when `infer()` receives a list. A separate private, true-batch TensorRT prototype was tested at B8/B32. On a repeated 32-image **training diagnostic**, mean batch times were 7.267/27.358 ms, throughput 1,101/1,170 images/s and process GPU memory 790/1,118 MiB. These are batch-throughput measurements, not single-image latency or public supported-batch claims. Full-cohort replay changed four B8 and five B32 strings relative to the qualified public B1 engine: B8 scored 2474/2693, 880/984 and 332/443; B32 scored 2475/2693, 879/984 and 332/443. Strict full-string parity failed. The prototype remains private and does not inherit the public engine's accuracy qualification. Evidence-bundle paths: `waterec/dynamic_batch/{independent_benchmark_audit,independent_full_replay_audit,independent_localization_audit}.json`.

[Mirage](https://huggingface.co/Cactus-Compute/mirage) was also tested through its official Cactus API, without a reference architecture port. The audited [official source snapshot](https://github.com/cactus-compute/cactus/tree/925b716cb0d823666a92ed71d64eb9c1f4596d68) provided Mirage `describe`/`extract` inference, but no Mirage-specific public training workflow was found; documented Needle/Whistle training is not Mirage training. Client 4.2.0 with its pinned native 4.1.1 engine ran on Linux CPU. Small task-specific smoke tests did not qualify it for these OCR tasks; they do not prove inherent model incapability, and no matching published golden fixture was found. Mirage is not a trainable preset or TensorRT deployment in this repository. Evidence-bundle path: `official_mirage/official_audit_summary.json`.

The following batch-1 measurements use an RTX5090 and fresh processes. File-to-text includes decoding the encoded image; decoded-RGB-to-text starts with RGB pixels already loaded. Process GPU memory includes runtime context and buffers and is **not** the model's weight-file size. A missing measurement is not zero and should not be inferred from another task or checkpoint.

| Model and task | File-to-text mean, batch 1 | Decoded-RGB-to-text mean, batch 1 | Process GPU memory |
| --- | ---: | ---: | ---: |
| Florence optimized, plates | 7.339 ms | Not measured | 1,130 MiB |
| Florence optimized, prices | 7.796 ms | Not measured | 1,176 MiB |
| WATERec plates, default CPU preprocessing | 2.709 ms | 1.643 ms | 688 MiB |
| WATERec plates, opt-in GPU preprocessing | 2.260 ms | 1.256 ms | 694 MiB |
| WATERec plates, experimental speculative | 1.965 ms | 0.977 ms | 698 MiB |
| WATERec plates, speculative + orientation | 2.115 ms | 1.081 ms | 698 MiB |
| WATERec prices, static batch-1 TensorRT | 1.090 ms | 0.823 ms | 688 MiB |
| PP medium prices | 1.326 ms | 1.125 ms | 670 MiB |
| Fast Tiny plates | 1.658 ms | 0.686 ms | 682 MiB |
| Fast Tiny prices | 1.095 ms | 0.889 ms | 688 MiB |
| CCT-S plates, original notebook TensorRT | 1.542 ms | 0.455 ms | 742 MiB |
| CCT-S prices, static batch-1 TensorRT | 0.783 ms | 0.555 ms | 540 MiB |
| PARSeq Base / standard Tiny / TrOCR, current matched protocol | Not measured | Not measured | Not measured |

The ordinary AR WATERec GPU-input rows above use fresh November 15 source replays with unchanged engines and frozen-string parity; prior timing snapshots remain in the aggregate reference JSON. Evidence-bundle paths: `waterec/current_public_runtime_2026_11_15/validation_summary.json` and `waterec/current_price_runtime_2026_11_15/validation_summary.json`. Current WATERec/PP measurements use five warmups and explicitly recorded scopes: WATERec plates use 256 mixed plate/unseen crops; both price measurements use 206 price crops. CCT-S plates use the same 256 plate/unseen crops as WATERec, five warmups and the unchanged original TensorRT engine. Fast Tiny uses 256 plate/unseen crops or 206 price crops, three repeats and five warmups; every timed string matches its original reference. CCT-S prices use 206 crops, a distinct 39-symbol head and a static batch-1 CUDA graph. WATERec's GPU preprocessing is opt-in and preserves the measured predictions; the default remains CPU preprocessing. Batch-32 throughput is not batch-1 latency. The older PARSeq/Florence table below uses a different historical protocol and units, so its median timings and MB memory must not be compared directly with these mean/MiB rows. The experimental speculative row below is a separate opt-in decoding mode, not a replacement for the default AR row. Details and missing values are recorded in [`results/reference_metrics.json`](results/reference_metrics.json).

### Experimental speculative ALPR reader (opt-in)

A separately trained draft head proposes text tokens; the original causal reader verifies a prefix, then its cached decoder handles any remaining tokens. The measured pipeline includes drafting, verification, cache handoff and all required fallback. This optional public implementation preserves the original autoregressive default.

The final-refit model scored **2476/2693** ALPR validation, **879/984** reviewed unseen plates and **332/443** international crops. The public implementation matches all **4,120** frozen speculative strings. It agrees with the native FP32 reference on **3677/3677** primary-cohort strings, and with the original cached FP16 engine on **4118/4120** across the three cohorts. One plate improves and one unseen plate regresses relative to that engine; empirical agreement is not a universal numerical-equivalence guarantee.

The validated public shared-workspace runtime measured **0.977 ms from decoded RGB** (p95 **1.460 ms**) and **1.965 ms including file decoding**, at **698 MiB** process GPU memory. This uses 256 mixed plate/unseen crops, five warmups and three repeats in fresh isolated processes, including exact GPU preprocessing. It meets the sub-1 ms **mean RGB** and 700 MiB goals in this measurement, while p95 and encoded-file latency remain above 1 ms. The earlier 704 MiB and private 700 MiB research measurements are separate controls, not these public timings.

Combined speculative/orientation inference also passed **4,136** prediction and angle checks against the native orientation reference. Correct results on 788 crops at 0/90/180/270 degrees are **703/705/705/706**; upright unseen accuracy is **878/984**. It matches **4131/4136** cached-AR orientation strings; the same numerically sensitive plate accounts for the differences. Orientation remains optional because occasional false flips reduce upright accuracy. On the same mixed 256 timing crops it measured **1.081 ms RGB**, **2.115 ms encoded** and **698 MiB**; this is not a latency measurement on an all-rotated cohort.

### Train and export the optional drafter

Complete original WATERec training and its standard ONNX/TensorRT export/build first. The optional workflow uses its **fit-only** checkpoint for internal draft selection, then initializes a fresh draft from the corresponding full-data target for the selected fixed epoch count. It never updates the original encoder/AR reader. It reads only the training pool and known training replay; full refit includes the internal training holdout, while the external validation/test images remain excluded. Supervision is the teacher's free greedy output, not dataset text labels. A draft is bound to the exact original checkpoint SHA256 and cannot be swapped onto another encoder with matching shapes.

```bash
python -m parseq_ocr.waterec_cli train-speculative --config configs/waterec_alpr.yaml
python -m parseq_ocr.waterec_cli export-speculative --config configs/waterec_alpr.yaml --draft-checkpoint artifacts/waterec_alpr/speculative/full_fit/draft.pt
python -m parseq_ocr.waterec_cli build-speculative --config configs/waterec_alpr.yaml
```

The validated profile is plate OCR, 512 patch tokens, maximum text length 25 and 95 output classes. Selection uses up to 10 epochs and the largest internal teacher-verified prefix, with the earliest tie; full refit uses that fixed number of epochs with the original ten-epoch learning-rate schedule prefix. It does not select by external accuracy. Public teacher content/split fingerprints are checked when available; legacy/custom teacher provenance is explicitly unverified. Do not use a full-fit teacher for held-out draft selection.

If you only have a final original checkpoint, use the explicit fixed-epoch path with no held-out selection:

```python
from parseq_ocr.waterec_config import load_config
from parseq_ocr.waterec_speculative_training import train_speculative
config = load_config("configs/waterec_alpr.yaml")
config["speculative"] = {"fixed_epoch": 8, "cache_mode": "ram"}
draft_checkpoint = train_speculative(config)
```

The teacher cache uses about 0.79 MB of host RAM per crop (about 10 GB for 13,000 crops), independently of inference GPU memory. Set `cache_mode: disk` in the nested `speculative` mapping to store the cache under the ignored artifact directory instead. Training defaults to CUDA BF16 or CPU FP32 and allows nested `epochs`, `batch_size`, `cache_batch_size` and `precision`. The original ten-epoch schedule permits 1–10 epochs. Changed training pixels, labels, split membership, targets or settings reject completed-run reuse. Interrupted optimizer training is not resumed automatically; use a new speculative output directory. Runtime use is opt-in with `--backend tensorrt --speculative --gpu-preprocess`; original AR inference remains the default. Validate your actual target/draft/engine pair before deployment.

### Price model trained with 1024 patch tokens

A separate historical R1024 price checkpoint scored154/206 literal exact. Its private TensorRT runtime with exact GPU preprocessing/shared scratch measured1.340 ms including file decoding /1.090 ms decoded RGB at700 MiB, on206 crops with five warmups and three repeats. All 206 strings and prepared arrays match its native reference.

This is a trained-resolution research option; the historical R512 row remains unchanged. Increasing an existing512 checkpoint’s inference budget without matched training did not help on the internal holdout. The public export profile validates512 patches; this1024 measurement is not an additional packaged export preset. Separate task training and runtime policies prevent a resolution-only causal comparison.

### Host RAM, measured separately

These are fresh-process encoded-input functional probes, with five warmups and one full cohort. They record Linux process high-water RSS, including libraries, runtime/host buffers, parsed metadata and transient decoded images. They do not preload the whole RGB cohort, are separate from the isolated latency runs and are not GPU VRAM or weight-file sizes. All probe predictions match the corresponding frozen reference.

| Model/task | Recorded host high-water RSS | Images |
| --- | ---: | ---: |
| CCT-S plates | 498.7 MiB | 256 |
| CCT-S prices | 464.8 MiB | 206 |
| Fast Tiny plates | 533.0 MiB | 256 |
| Fast Tiny prices | 501.5 MiB | 206 |
| PP prices | 516.6 MiB | 206 |
| WATERec AR plates, GPU preprocessing | 540.9 MiB | 256 |
| WATERec speculative plates | 551.7 MiB | 256 |
| WATERec R512 prices | 519.0 MiB | 206 |
| WATERec R1024 prices (research) | 518.3 MiB | 206 |
| Florence plates | 760.4 MiB | 256 |
| Florence prices | 762.6 MiB | 206 |
| Standard PARSeq Base/Tiny and TrOCR, current matched RAM probe | Not measured | — |

## Legacy reference measurements

These are previously measured results from a private ALPR dataset, **not a promise of accuracy on your data**. Private data and fine-tuned weights are not distributed here. Downloading the official weights gives the generic pretrained model, not the fine-tuned checkpoint below.

RTX 5090; 10,990 training crops; 2,693 validation crops; identical crop-to-text timing protocol:

| Metric | Existing Florence-2 TensorRT | Fine-tuned PARSeq TensorRT |
| --- | ---: | ---: |
| Exact validation accuracy | 88.04% | 89.68% |
| Median latency, batch 1 | 9.12 ms | 4.00 ms |
| Sampled process VRAM, batch 1 | 1,629 MB | 958 MB |
| Throughput, batch 32 | 376 crops/s | 941 crops/s |

Florence's best beam-5 PyTorch score was 88.34%; PARSeq's PyTorch score was 89.75%. On 443 independently sourced [OpenALPR](https://github.com/openalpr/benchmarks) crops from Brazil, Europe and the US, strict exact accuracy was 42.89% for that Florence checkpoint and 73.81% for the fine-tuned PARSeq model.

Generic reading ability still has tradeoffs: the original PARSeq at its native 32×128 input scored 82.62% on a punctuation-insensitive international metric; the fine-tuned 64×256 model scored 74.04%. A resized-pretrained control scored 68.85%, so resolution and fine-tuning effects should not be conflated. External images were excluded from our fine-tuning and tuning, but unknown original pretraining exposure cannot be ruled out. These results do not establish performance on every alphabet or country.

Only aggregate reference numbers are included in [`results/reference_metrics.json`](results/reference_metrics.json). Re-evaluate your checkpoint after export. Use fresh CLI processes for memory/speed comparisons; a notebook kernel can retain other loaded models.

## Publish or contribute

```bash
python scripts/clean_notebooks.py
python scripts/check_release.py
pytest
git status --short
```

Datasets, pretrained/fine-tuned weights, engines, local environments, checkpoints and notebook outputs must stay out of Git. Ignore rules exclude generated data/artifacts; `check_release.py` also checks notebook outputs and local-path leakage. The notebooks are committed without execution results. No private images or labels are included.

## Attribution

Apache-2.0. The required PARSeq reference subset is vendored at commit `1902db043c029a7e03a3818c616c06600af574be`, with its original license and notice. See [`NOTICE`](NOTICE) and `parseq_ocr/_vendor/parseq/source_pin.json`. The notebook workflow, helper APIs and label editor are implemented in this repository. TensorRT is an optional NVIDIA dependency with its own terms.
