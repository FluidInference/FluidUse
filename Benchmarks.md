# Benchmarks

Both decision models FluidUse runs on device, measured on the same machine: **Apple M5 Pro,
24 GB, macOS 27.0**, September 2026. Every number below has a checked-in report; the conversion
pipelines, verification harnesses and raw reports live in
[mobius `models/computer-use/`](https://github.com/FluidInference/mobius/tree/main/models/computer-use)
(`cua-s1-forms/coreml/reports`, `laya/coreml/reports`; the laya suites and PyTorch reference rows
are in `laya/coreml/benchmark`).

| | CUA-S1-FORMS | laya-multilingual |
| --- | --- | --- |
| What it decides | one form-element action out of the supplied options (fill / check / click / skip) | typed `choice` / `score` / `noul` questions about any text state |
| Architecture | byte-level 2-layer encoder + option attention, 706,048 params | mmBERT-base encoder + 2-layer decision head, 322M params |
| Core ML package | 1.5 MB fp16 | 614 MB fp16 per bucket, 448 MB with int8 embedding (`e8`) |
| Warm latency, CPU+ANE | **0.9 ms** | **3.6 ms** (128 tokens) |
| Accuracy vs PyTorch | identical (24,370-row synthetic test) | identical (3,899 questions, 10 suites) |
| Ops on the Neural Engine | 149 / 173 | 973 / 978 |
| Swift API | `CuaS1FormsManager` (FluidAudio) | `LayaManager` (this repo) |

## CUA-S1-FORMS

Cua's from-scratch form scorer (`cua-ai/cua-s1-forms`, MIT), converted to a fixed-shape FP16 Core ML
package with 32 option slots. Context is truncated at 224 UTF-8 bytes and each option at 96, exactly
as the checkpoint was trained.

### Accuracy

| Test | Rows | Upstream PyTorch | Core ML fp16 | Core ML int8 weights | Core ML int4 weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned demo (3 forms, 3 PDFs) | 196 | 196 / 196 | 196 / 196 | 196 / 196 | – |
| Full published synthetic test | 24,370 | 24,359 (0.99955) | 24,359 (0.99955) | 24,359 (0.99955) | 24,353 (0.99930) |

The synthetic test is the complete released file with no filtering or resampling; the model card
claims 0.9995 on "about 15,000 decisions" and the full file reproduces that. The upstream Cua
evaluator reports 0 wrong actions and 0 wrong targets on the demo; its 23.5% coverage is because it
counts `skip` as abstention (150 of the 196 decisions are skips). Max per-row probability error vs
PyTorch: 0.020 (fp16), 0.068 (int8), 0.754 (int4); the fp32 export adapter alone is within 1.1e-6,
so all of that is compute precision.

### Latency and placement

Synchronous `predict` on pre-encoded inputs, 30 timed passes after warm-up (`ane-profile.json`);
Swift figures are `CuaS1FormsManager.score` end to end, release build, 200 calls
(`swift-variant-comparison.json`).

| Configuration | Warm median | p95 | Load | Ops on ANE |
| --- | ---: | ---: | ---: | ---: |
| CPU only | 1.53 ms | 1.60 ms | 71 ms | 0 |
| CPU + GPU | 0.93 ms | 2.38 ms | 70 ms | 0 |
| CPU + ANE | **0.93 ms** | 0.97 ms | 567 ms | 149 / 173 |
| All units | 0.91 ms | 1.23 ms | 116 ms | – |
| Swift, CPU + ANE | **0.91 ms** | 0.93 ms | 600 ms | |

The 24 CPU ops are integer/mask preparation and the byte-embedding gathers. Host encoding is
0.03 ms per decision. Live browser validation (WKWebView DOM observation, real Core ML choices,
dispatched events, DOM readback) reproduced 100/100 of the original decisions across six runs.

### Compression

int8 weights keep accuracy identical (24,359 / 24,370) and int4 loses 6 rows, but at 1.5 MB there
is nothing to gain; the shipped package stays fp16.

## laya-multilingual

Convai Innovations' open Jev-style decision model (`convaiinnovations/laya` `multilingual/`,
Apache-2.0), converted to fixed-length FP16 buckets of 128 / 256 / 512 / 1024 tokens with 32 option
slots. `LayaManager` runs a prompt on the smallest loaded bucket that fits.

### Accuracy on laya's published suites

The 3,899 questions are rebuilt from laya's own research scripts (same datasets, seed 13, 400 cases
per task, 300 MASSIVE cases with 20 options; banking77 excluded because its 77 labels exceed the 32
slots). "Upstream" is the laya-multilingual column of laya's BENCHMARKS.md (Tesla T4, PyTorch).

| Suite | n | Upstream (T4) | PyTorch CPU (M5 Pro) | Core ML fp16 | Core ML e8 | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AG News (4 labels) | 400 | 0.930 | 0.935 | **0.935** | 0.935 | 3.8 ms |
| DAIR Emotion (6) | 400 | 0.530 | 0.537 | **0.537** | 0.535 | 3.7 ms |
| MASSIVE intent (20 options) | 300 | 0.657 | 0.657 | **0.657** | 0.653 | 5.2 ms |
| Support triage (10 queues) | 400 | 0.522 | 0.540 | **0.542** | 0.537 | 5.3 ms |
| Email spam (noul) | 400 | 0.993 | 0.993 | **0.993** | 0.993 | 5.8 ms |
| Phishing (noul) | 400 | 0.993 | 0.993 | **0.993** | 0.993 | 9.0 ms |
| Guardrails / jailbreak (noul, held out) | 400 | 0.755 | 0.805 | **0.805** | 0.810 | 3.8 ms |
| Moderation / toxicity (noul, held out) | 400 | 0.525 | 0.535 | **0.535** | 0.535 | 3.8 ms |
| RAG passage relevance (noul) | 400 | 0.657 | 0.672 | **0.672** | 0.675 | 5.3 ms |
| Model routing domain (6) | 399 | 0.123 | 0.441 | **0.441** | 0.454 | 5.3 ms |

Core ML fp16 matches the PyTorch reference on every suite, with 100% row-level argmax agreement on
eight suites and 99.8% / 99.5% on the other two. Whole run from Swift: **3,899 questions in 22.9 s,
p50 5.2 ms, p95 18.0 ms** (bucket picked per prompt), versus 61.6 ms per question for PyTorch FP32
on the same CPU and 32.8 ms per question upstream reports on a T4. The PyTorch column reproduces
upstream's table except model routing, where the published 0.123 looks like an upstream run
artefact (their own script gives 0.441 here).

### Latency and placement per bucket

16 fixture questions vs PyTorch (`verification-multilingual-L*.json`) and `coreml-cli` profiles
(`ane-profile-L*.json`, 20 iterations). Argmax agreement is 16/16 everywhere; Δprob is the max
per-row probability error.

| Bucket | CPU only | CPU + GPU | CPU + ANE | All units | Δprob ANE / ALL | Cold compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L128 | 14.1 ms | 4.6 ms | **3.6 ms** | 3.9 ms | 0.013 / 0.002 | 4.9 s |
| L256 | 27.1 ms | 5.3 ms | 9.9 ms | **5.2 ms** | 0.013 / 0.002 | 5.5 s |
| L512 | 58.9 ms | 8.8 ms | 27.5 ms | **9.0 ms** | 0.013 / 0.002 | 6.0 s |
| L1024 | 149.6 ms | 18.1 ms | 80.1 ms | **17.9 ms** | 0.013 / 0.002 | 8.5 s |

973 of 978 ops are placed on the ANE for every bucket (the five CPU ops are int32 casts and the
embedding gather), but `all` runs 100% on the GPU and the ANE only wins at 128 tokens because the
L×L attention cost grows faster on it. `LayaManager` therefore defaults the 128 bucket to CPU+ANE
and longer buckets to all units. A single short question from Swift, release build, including
tokenization: 3.7 ms.

### Compression

`quantize.py` applied post-training compression to the 2-D linear weights and/or the embedding
table and re-ran the parity gates (L128):

| Variant | Package | CPU+ANE argmax · Δprob | ALL argmax · Δprob | Result |
| --- | ---: | ---: | ---: | --- |
| fp16 | 644 MB | 16/16 · 0.013 | 16/16 · 0.002 | reference |
| **e8** int8 embedding table | **448 MB** | 16/16 · 0.014 | 16/16 · 0.015 | **published**, within 0.5 pt on every suite above |
| int8 encoder + head linears | 519 MB | 11/16 · 0.666 | 16/16 · 0.029 | fails on the ANE |
| int8 everything | 324 MB | 11/16 · 0.668 | 16/16 · 0.042 | fails on the ANE |
| 6-bit k-means palette | 488 MB | 15/16 · 0.118 | 15/16 · 0.114 | fails |
| 4-bit k-means palette | 456 MB | 12/16 · 0.728 | 12/16 · 0.727 | fails |

Only the embedding table tolerates compression; the encoder stays fp16 where it does its compute.
Per-block int8 needs an iOS 18 deployment target and was not tried.

### Tetris demo

`LayaTetrisDemo` / `FluidUseLaya tetris` score every legal landing with one `noul` question.
Sustained rate on the 128 bucket: **~15,800 decisions per minute at 3.8 ms median** (seeds 1, 2,
3, 7), against the 1,799 per minute in the original laya Tetris post (~27 ms on an M1 Max GPU).
Zero-shot laya clears 13–32 lines before topping out; the feature-weighted heuristic policy clears
71–77 in 200 pieces. It is a latency demo, not a Tetris player.

## Reproduce

```bash
# laya: suites.jsonl + reference-rows.jsonl come from mobius models/computer-use/laya/coreml/benchmark
# (generated by its benchmark.py); the buckets download on first run
swift run -c release FluidUseLaya benchmark --suites <mobius>/benchmark/suites.jsonl \
    --reference <mobius>/benchmark/reference-rows.jsonl --report /tmp/laya.json
swift run -c release FluidUseLaya benchmark --suites <mobius>/benchmark/suites.jsonl \
    --reference <mobius>/benchmark/reference-rows.jsonl --precision e8

# CUA-S1-FORMS and the laya conversion / PyTorch reference, from mobius
cd models/computer-use/cua-s1-forms/coreml && uv sync --frozen && uv run python verify.py && uv run python benchmark-synthetic.py
cd models/computer-use/laya/coreml && uv sync --frozen && uv run python verify.py --length 128 && uv run python benchmark.py
```
