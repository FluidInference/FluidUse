# laya benchmarks

Data and reports behind the numbers in the top-level README, so the run can be repeated from
this repository alone. Apple M5 Pro (24 GB), macOS 27.0, September 21, 2026. Checkpoint:
`convaiinnovations/laya` `multilingual/` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, converted to Core ML in
[mobius `models/computer-use/laya/coreml`](https://github.com/FluidInference/mobius/tree/main/models/computer-use/laya/coreml).

## Accuracy on laya's published suites

`suites.jsonl` holds the 3,899 questions that laya's own research scripts build (same datasets,
seed 13, 400 cases per task, 300 MASSIVE cases with 20 options; banking77 is excluded because its
77 labels exceed the 32 option slots). `reference-rows.jsonl` is the unmodified PyTorch FP32 model's
answer to every row on this Mac's CPU at `max_len` 1024 (`benchmark-reference.json` summarises it).
`benchmark-coreml.json` is the Core ML run from Swift with the 128/256/512/1024 buckets, smallest
bucket that fits each prompt.

| Suite | n | Upstream (T4, PyTorch) | PyTorch CPU (M5 Pro) | Core ML from Swift | Row agreement | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jev.ag_news | 400 | 0.930 | 0.935 | **0.935** | 1.000 | 3.8 ms | 6.6 ms |
| jev.emotion | 400 | 0.530 | 0.537 | **0.537** | 1.000 | 3.7 ms | 3.8 ms |
| massive_intent.en | 300 | 0.657 | 0.657 | **0.657** | 1.000 | 5.2 ms | 5.5 ms |
| app.support_triage | 400 | 0.522 | 0.540 | **0.542** | 0.998 | 5.3 ms | 8.9 ms |
| app.email_spam | 400 | 0.993 | 0.993 | **0.993** | 1.000 | 5.8 ms | 18.6 ms |
| app.phishing | 400 | 0.993 | 0.993 | **0.993** | 1.000 | 9.0 ms | 19.5 ms |
| app.guardrails_jailbreak | 400 | 0.755 | 0.805 | **0.805** | 0.995 | 3.8 ms | 9.6 ms |
| app.moderation_toxicity | 400 | 0.525 | 0.535 | **0.535** | 1.000 | 3.8 ms | 9.4 ms |
| app.rag_relevance | 400 | 0.657 | 0.672 | **0.672** | 1.000 | 5.3 ms | 6.3 ms |
| app.model_routing_domain | 399 | 0.123 | 0.441 | **0.441** | 1.000 | 5.3 ms | 5.9 ms |

Whole run: **3,899 questions in 22.9 s, p50 5.2 ms, p95 18.0 ms** from Swift,
versus 61.5 ms per question for PyTorch FP32 on the same CPU (4 threads) and 32.8 ms per
question upstream reports on a Tesla T4. "Upstream" is the laya-multilingual column of laya's
BENCHMARKS.md; the PyTorch column reproduces it here except model routing, where the published
0.123 looks like an upstream run artefact (the same script gives 0.441).

## Bucket parity and latency

`verification-multilingual-L*.json`: 16 fixture questions (Tetris placements, a form field, support
triage, zh/ja/de, 20 options, prompt injection, a long meeting note) against the PyTorch reference.
`ane-fallback-L128.json`: compute placement, 973 of 978 ops on the Neural Engine.

| Bucket | Argmax (CPU+ANE) | Max Δprob (CPU+ANE) | p50 (CPU+ANE) | Max Δprob (all units) | p50 (all units) |
| --- | ---: | ---: | ---: | ---: | ---: |
| L128 | 16/16 | 0.0126 | 3.64 ms | 0.0021 | 3.87 ms |
| L256 | 16/16 | 0.0126 | 9.88 ms | 0.0021 | 5.19 ms |
| L512 | 16/16 | 0.0126 | 27.45 ms | 0.0021 | 9.00 ms |
| L1024 | 16/16 | 0.0126 | 80.07 ms | 0.0023 | 17.89 ms |

`LayaManager` therefore runs the 128 bucket on CPU+ANE and longer buckets on all units.

## Reproduce

```bash
swift run -c release FluidUseLaya benchmark \
    --suites Benchmarks/laya/suites.jsonl \
    --reference Benchmarks/laya/reference-rows.jsonl \
    --report /tmp/benchmark-coreml.json
```

The first run downloads the four buckets (about 2.5 GB). To regenerate `suites.jsonl` and the
PyTorch reference, run `uv run python benchmark.py` in the mobius conversion directory.
