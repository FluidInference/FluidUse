# Official benchmark inventory and plan (working document, 2026-09-22)

Goal: evaluate each FluidUse model/variant from PR #9 on **its authors' own published benchmark**, with the official data and scorer unchanged, calling the model through FluidUse from Swift. Third-party suites (Jev Decision Index, JevBench) are listed separately and are not substitutes.

Execution path labels:
- **Swift/Core ML**: inference runs in-process in Swift (Laya, GLiNER2, GLiClass, Verdict).
- **Bridge**: Swift calls `PublishedCoreMLManager`; a Python worker performs tokenization, Core ML inference (coremltools `predict`), and decoding. Timings measured in Swift cover the whole bridge call.

Each `Tools/official-benchmarks/<model>/run.py` documents the pinned official checkout and environment it expects.

## Inventory

| Model / variants | Official benchmark (author) | Kind | Data (pinned) | Scorer (pinned) | Reference score (checkpoint, settings) | Fits our Core ML | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Kev 0.6B** fp16, w8 (Bridge) | `kev.benchmark` on frozen suites `evals/v7/decision-v7` (1,204 records / 1,468 q) and `evals/v4/transfer-v4` (764 / 764); locked test partitions via `--allow-test` | Full | jaredpalmer/kev repo; suite files identical at `37cb2f61` (scored) and `90990a5` (our pin); manifests `a8f50e48…`, `31677c22…` | `kev/benchmark.py` + `kev/metrics.py`; `--remote` scores any System One endpoint | dev in-dist 0.801 acc (ECE 0.086), transfer 0.620; locked test 0.808 / 0.642. `jaredpalmer/kev-0.6b@dece6dba`, fp32, state ≤384 / branch ≤1024 tokens, packed questions | decision-v7 dev: 863/1,264 clean fit L128 untruncated, 321 need state truncation, 80 exceed 32 options; transfer-v4: 514/656 fit | **Reproducible** on a matched subset only |
| **Kev 0.5B** fp16, e8 (Bridge) | (a) card eval: `kev/evaluate.py`, 6 sources × 150, seed 1, generated at run time; (b) frozen `transfer-v4` dev | (a) not frozen, (b) full | (a) HF datasets, revisions not pinned; (b) as above | as above | (a) 0.799 acc; (b) transfer-v4 0.561. `jaredpalmer/kev-0.5b@9ce2fd39` | (b) 514/656 fit | (b) **Reproducible** on matched subset; (a) not pinnable. The 0.6B card's "0.712 decision-v4 dev" for 0.5B was actually run on decision-v2; no 0.5B decision-v7 run found |
| **Decision 1.0 Kai** fp16, w8 (Bridge) | "Kai 0.6B fixed decision benchmark", 54 tasks, 3,766 q | Full | **Not publicly available** (only counts/hashes; transfer_v9 panel likely = `jaredpalmer/kev-suites` v9 test, our inference) | **Not found** (checked HF repo, semantic-router blog, gists, GitHub org, code search) | 53.52 weighted, FP32, 1,024-token budget, `7185f514` | transfer-v9: 507/1,156 clean fit L128 and **3 choice/score slots** | **Not reproducible** |
| **Decision 1.0 Lex** fp16, w8 (Bridge) | typed-decisions TEST (2,000 decisions) | Full | `LocalLLaMA/typed-decisions@ea930645` test (Apache-2.0) | **Not publicly available** (method in prose only) | 78.15% hard accuracy (`ee8e74d9`) | **71/2,000** (all noul); choice/score have 4–5 options > 3 slots | **Not reproducible** (no scorer; <4% fit) |
| **LFM2.5-350M-RLCD** fp16 (Bridge) | RLCD diagnostic suite (12 cases) + 5 stress probes | Diagnostic; author states "not a population benchmark" | inline in `rlcd/tasks.py`, `stress_tasks.py` @ `deb589d8` | `rlcd/benchmark.py` `evaluate()` + `summarize()`; 2 warmups, 3 repeats, seed 42 | constrained: field acc 77.8%, exact 50.0% (M2 Max/L40S/H100); `LiquidAI/LFM2.5-350M@9e6c6ccf` bytes | diagnostic 12/12 (routing needs 2 calls); stress 0/5 (490–1,636 tokens > L256) | **Reproducible** (diagnostic); stress not runnable |
| **Jeff** fp16, w8 (Bridge) | `bench/eval_accuracy.py`, 8 tasks × 200 | Full | committed `bench/data/*.jsonl` @ `34b32f99` | same script, `summarize --full` | gliformer-large-v1, `original` profile, T=3.2: AG News 75.5%, emotion 47.0%, Amazon MAE 0.600, SST-5 MAE 0.611, AUROC BoolQ 0.750 / irony 0.714 / spam 0.918 / SST-2 0.991 | 1,429/1,600 fit L128 (boolq 107/200) | **Reproducible** on matched subset |
| (Jeff checkpoint) gliformer-large-v1 | GLiClass zero-shot classification suite, 13 datasets | Full | HF datasets, revisions not pinned | `gliformer_eval/eval_classification.py` @ `b5c0a0fd` | mean macro-F1 75.03 | 8/13 datasets by label count; token fit not measured | Secondary; checkpoint-level, not Jeff's prompt path |
| **Verdict** fp16 L128/L512 (Swift/Core ML) | held-out Banking77/CLINC-OOS calibration eval (1,000 test) + challenge slices (missing option, distant OOS, K=3…25) | Full | `data/real_banking_{cal,test}.jsonl`, `data/slice_*.jsonl` @ `30f15564` | `scripts/evaluate.py` (`compute_metrics`, temperature fit on cal) | calibrated top-1 95.00%, NLL 0.1768, Brier 0.0785, ECE 3.35%, OOS recall 97.50% / precision 89.45% | test K=5, <~71 tokens → fits L128 | **Reproducible**. Official harness fits a single temperature; we ship the per-K calibrator — report both |
| **GLiNER 2.5 small/base/multi** W8 (Swift/Core ML) | Fastino zero-shot suite (6 classification + 10 extraction datasets) | Claimed full | **Not publicly available** (splits, revisions, settings unstated) | **Not publicly available** (`benchmarks/multitask/` referenced by tests but absent on all branches and PyPI) | base cls avg 69.86, multi 72.44; **small: none published** | classification only; xnli/clinc (150)/imdb exceed L128/K8 | **Not reproducible** |
| **GLiClass Edge Apps v2** (Swift/Core ML) | none: FluidInference fine-tune of `knowledgator/gliclass-edge-v3.0@df03993a` | — | — | — | only our own numbers (application suite 72.75% macro) | — | **No author benchmark exists**; parent's `test_gliclass.py` scores the parent, with train/test overlap |
| **NanoJev** fp16 local (Bridge) | 548-episode closed-loop game benchmark (Maze, Snake, ViZDoom Basic, Predict Position) | Full | `C-Tianyu/NanoJev-Data@fd4d07b6` (`unified-games-v1`) | `unified_game_pipeline.py` rollout + replay/summarize | weighted macro 66.85% test / 45.47% OOD (`047b927b`) | **0 steps fit L128** (shortest candidate path 203 tokens) | **Not runnable** on the L128/K4 package |
| **Laya multilingual** (Swift/Core ML, existing) | author application suites (`bench_apps.py`, N=400, seed 13) + T4 notebook suites | Full | HF datasets, **unpinned** upstream; our `suites.jsonl` stores evaluated rows | `bench_local.py` metrics | e.g. AG News 0.930, Emotion 0.530 (multilingual column) | 9 app suites + MASSIVE-en (banking77 excluded, 77 > 32 options) | **Already reproduced** (3,899 q); gaps: typed-decisions, XNLI, other MASSIVE languages; unexplained deltas on model routing/guardrails/support triage |

Third-party references (not official): Jev Decision Index v0.1 (Kev 0.5B 30.34, Kev 0.6B 31.30, Kai 18.37, Lex 19.57, LFM 25.79, Jeff 27.23, Verdict 13.38, GLiNER small/base/multi 23.93/24.70/22.42, NanoJev 26.19, Laya English 16.39; frozen suite requires rebuild, HLE gated for this account); JevBench v1.3.0 (Verdict 1.4 38.9, GLiNER 24.0/16.6/13.8).

## Attribution method

Every reproducible suite is run in up to four arms so a gap can be attributed rather than assumed:

1. **Official reference** as published (full suite, native context).
2. **Native PyTorch on the matched subset**: the pinned checkpoint through the official predictor, restricted to rows our Core ML can take, with the same request adaptation (e.g. one question per call). Isolates coverage + adaptation from conversion.
3. **FluidUse Core ML, strict**: no truncation; rows that do not fit are recorded as unsupported with a reason.
4. **FluidUse Core ML, published runtime behavior** where it differs (Kev's runtime shortens state to fit L128): reported separately, labelled truncated.

Differences 1→2 = coverage/adaptation; 2→3 = conversion (precision, compute units). Every row is kept in results, including failures; official denominators and answered-only scores are both reported; startup and steady-state latency are separated.

## Implementation

- `Sources/FluidUseOfficialBench` (Swift executable, `run`): loads one FluidUse model, reads one official request per line, and writes one result row per line in order (status, answer or error, Swift-measured latency); model load time is recorded separately in `meta.json`. Bridge models receive the request bytes unchanged. Verdict lines carry `{question, context, labels}` and use `VerdictManager.logits`, which renders the author's `build_model_input` exactly.
- `Tools/official-benchmarks/<model>/run.py`: builds the official request stream from the pinned checkout, runs the Swift runner, and scores its outputs with the **unchanged** official functions:
  - RLCD: `rlcd.benchmark.evaluate`; constrained block aggregated as in `summarize` (the autoregressive method is not converted).
  - Verdict: `extract_logits_and_targets` with a stand-in model returning FluidUse logits per prompt, then `TemperatureCalibrator` and `compute_metrics` as in `evaluate_pipeline` steps 1–3.
  - Kev: `kev.benchmark.evaluate_records` with a predictor that returns FluidUse probabilities (unrounded) per question; capacity rejections go through the scorer's own `skip_overlong` path. Native arms use the official `LocalPredictor`.
  - Jeff: the official `jeff.core.Engine` with FluidUse standing in for `Backend.score` (record → score → replay); rows as in `run_jeff`, scored by `task_metrics`.
- FluidUse changes made for fidelity: `ConstrainedField.description` (rendered into the RLCD prompt via `json.dumps(schema)`, member order as RLCD's schemas); `VerdictManager.logits(question:context:labels:)`; Kev worker `FLUIDUSE_STRICT_CONTEXT=1` (reject rather than shorten state) and unrounded `probabilities`/`option_keys` for Kev 0.6B (0.5B already returned them). Documented: Kai/Lex Core ML choice/score packages have 3 candidate slots.

## Results so far (Apple M5 Pro, macOS 27.0)

### LFM2.5-350M-RLCD — RLCD diagnostic suite (complete run)
Official protocol: 12 cases, 2 warmups, 3 repeats with the seeded per-repeat shuffle, scored by the unchanged `rlcd.benchmark.evaluate`. 60 requests, 0 failures.

| | FluidUse Core ML (bridge, fp16, `.all`) | Author reference (native) |
| --- | ---: | ---: |
| Field accuracy | **77.8%** | 77.8% |
| Exact objects | **50.0%** | 50.0% |
| Syntax / schema valid | 100% / 100% | 100% / 100% |
| Output objects identical to the author's published M2 Max outputs | **12/12 cases, 36/36 fields** | — |
| Request latency (median / p95) | 57 ms / 112 ms | 55.3 ms mean (M2 Max, native MPS; different hardware and path) |
| Startup | 7.3 s | — |

Stress probes (5) are not runnable: all exceed L256.

### Verdict — author's held-out evaluation (complete run)
2,500 requests (cal 500, test 1,000, 7 slices 1,000), 0 unsupported. Checkpoint identical to the author's evaluated `artifacts/v2/model.safetensors` (sha256 d2528239…). Temperature fitted with the author's `TemperatureCalibrator` on FluidUse calibration logits; metrics from the unchanged `compute_metrics`.

| Test (1,000) | FluidUse Core ML (Swift, fp16) | Author reference |
| --- | ---: | ---: |
| Fitted temperature | 1.418 | 1.427 |
| Top-1 accuracy | **95.00%** | 95.00% |
| NLL | 0.1765 | 0.1768 |
| Brier | 0.0783 | 0.0785 |
| ECE (equal width) | 3.38% | 3.35% |
| OOS abstention recall / precision | 97.50% / 89.45% | 97.50% / 89.45% |
| Predictions identical to `predictions_v2.jsonl` | **998/1,000** | — |
| Latency median / p95 | 10.9 ms / 48.7 ms | — |

All seven challenge slices match the reference accuracy exactly (missing option 75.5, distant OOS 98.0, K=3/5/9/17/25: 97/96/91/78/72); Brier and ECE within 0.02.

Finding: the per-K `calibrator.json` shipped in the author's HF repo (and used by `VerdictManager.answer`) is fitted for open-domain use; on this test set it gives the same accuracy but NLL 0.367 and ECE 21.6% (under-confident). The author's card numbers use a single fitted temperature.

## Kev and Jeff results (development runs completed 2026-09-22 22:47–23:08)

Scoring units. Kev accuracy is per **clean question** (the official `report["clean"]["acc"]`); decision-v7 dev has 1,204 records / 1,468 questions / 1,264 clean questions (1,032 one-question, 80 two-, 92 three-question records; 180 variant records feed flip-rate/abstention metrics, not accuracy); transfer-v4 dev has 764 records / 656 clean questions. Records are accepted or rejected whole (official `evaluate_records`); no record is partially answered. Kev's scorer states rejected records count as wrong in any headline number, so the **official-protocol headline = correct / all clean questions**; answered-only accuracy is a diagnostic. Jeff's scorer has no rejection rule: its metrics are reported on answered items with coverage beside them, without a penalty.

### Kev — official protocol (strict: no truncation)

| Model · suite | Native (this Mac) | Published | Core ML answered | Headline (rejected = wrong) | Answered-only acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B fp16 · decision-v7 dev | 0.8006 | 0.801 | 834 / 1,264 | **0.524** | 0.794 |
| 0.6B w8 · decision-v7 dev | — | — | 834 / 1,264 | **0.521** | 0.790 |
| 0.6B fp16 · transfer-v4 dev | 0.6204 | 0.620 | 514 / 656 | **0.491** | 0.627 |
| 0.6B w8 · transfer-v4 dev | — | — | 514 / 656 | **0.489** | 0.625 |
| 0.5B fp16 · transfer-v4 dev | 0.5610 | 0.561 | 514 / 656 | **0.447** | 0.570 |
| 0.5B e8 · transfer-v4 dev | — | — | 514 / 656 | **0.448** | 0.572 |

Native also reproduces the published ECE/Brier (decision-v7: 0.086 / 0.297). Rejections (strict): 421 decision-v7 records and 164 transfer-v4 records, all because the untruncated request exceeds 128 tokens (Banking77's 77-option questions are among them; the length check fires first).

### Kev — conversion parity (identical inputs: one question per call, same rows and IDs, no truncation)

| Model · suite | Questions | Argmax agreement | Accuracy Core ML vs native (clean) | Max Δprob |
| --- | ---: | ---: | --- | ---: |
| 0.6B fp16 · decision-v7 | 968 | **100%** | 0.7938 vs 0.7938 | 0.025 |
| 0.6B w8 · decision-v7 | 968 | 99.48% | 0.7902 vs 0.7938 | 0.108 |
| 0.6B fp16 · transfer-v4 | 600 | **100%** | 0.6265 vs 0.6265 | 0.006 |
| 0.6B w8 · transfer-v4 | 600 | 98.5% | 0.6245 vs 0.6265 | 0.075 |
| 0.5B fp16 · transfer-v4 | 600 | 99.83% | 0.5700 vs 0.5681 | 0.016 |
| 0.5B e8 · transfer-v4 | 600 | 99.67% | 0.5720 vs 0.5681 | 0.012 |

Question grouping has no effect: native one-question-per-call and native packed predictions are identical on the same rows (max Δ 4e-6). Attribution: the gap between the published score and the Core ML headline is coverage (questions over 128 tokens); on the questions Core ML takes, conversion changes accuracy by ≤0.4 pt.

### Kev — locked test partitions (run once, 2026-09-22 23:21–23:44)

Run once after the adapters and scoring rules were frozen (no changes after the development runs), with `--allow-test`, kev @ `90990a5fac2995b9faa3190f7d437e84f2067768`, the same checkpoints and FluidUse packages as above. Nothing was tuned against these results. The Verdict app-path evaluation ran concurrently with part of the Kev 0.6B decision-v7 runs, so their latencies are not reported.

| Model · suite (test) | Native (this Mac) | Published | Core ML answered | Headline (rejected = wrong) | Answered-only | Parity vs native, same rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.6B fp16 · decision-v7 | 0.8083 | 0.808 | 796 / 1,200 | **0.535** | 0.807 | 100% of 959 q; acc 0.8065 = 0.8065 |
| 0.6B w8 · decision-v7 | — | — | 796 / 1,200 | **0.536** | 0.808 | 99.58%; 0.8078 vs 0.8065 |
| 0.6B fp16 · transfer-v4 | 0.6418 | 0.642 | 503 / 656 | **0.500** | 0.652 | 100% of 596 q; 0.6521 = 0.6521 |
| 0.6B w8 · transfer-v4 | — | — | 503 / 656 | **0.494** | 0.644 | 97.32%; 0.6441 vs 0.6521 |
| 0.5B fp16 · transfer-v4 | 0.5747 | none published | 503 / 656 | **0.427** | 0.557 | 99.83%; 0.5567 = 0.5567 |
| 0.5B e8 · transfer-v4 | — | — | 503 / 656 | **0.427** | 0.557 | 99.66%; 0.5567 = 0.5567 |

Native also reproduces the published locked Brier (transfer-v4 0.483). Rejections (strict): 405 decision-v7 and 168 transfer-v4 records over 128 tokens. Grouping again has no effect (one question per call vs packed: identical). Adapted shortened-state protocol on the locked partitions: 0.6B fp16 0.750 / 0.633, w8 0.749 / 0.628; 0.5B 0.576 (fp16 and e8).

### Kev — ADAPTED protocol: published runtime shortens the state to fit L128

Not the official protocol (the suites are frozen with `truncate: false`); no native reference exists for shortened inputs.

| Model · suite | Answered | Correct / all clean | Answered-only |
| --- | ---: | ---: | ---: |
| 0.6B fp16 · decision-v7 | 1,184 / 1,264 | 0.737 | 0.786 |
| 0.6B w8 · decision-v7 | 1,184 / 1,264 | 0.731 | 0.780 |
| 0.6B fp16 · transfer-v4 | 654 / 656 | 0.604 | 0.606 |
| 0.6B w8 · transfer-v4 | 654 / 656 | 0.605 | 0.607 |
| 0.5B fp16 · transfer-v4 | 654 / 656 | 0.564 | 0.566 |
| 0.5B e8 · transfer-v4 | 654 / 656 | 0.566 | 0.567 |

Remaining rejections (116 decision-v7, 2 transfer-v4) are questions whose instruction and options alone exceed 128 tokens (mostly Banking77).

### Jeff — official `bench/eval_accuracy.py` (original profile, T=3.2)

Native TorchBackend on this Mac reproduces every published number (AG News 0.755, emotion 0.470, Amazon MAE 0.600, SST-5 MAE 0.611, AUROC BoolQ 0.750 / irony 0.714 / spam 0.918 / SST-2 0.991). Core ML answered 1,429 / 1,600 items; unsupported (over 128 tokens): BoolQ 93, Amazon 45, AG News 26, emotion 7.

| Task | Core ML answered | Core ML fp16 (answered) | Core ML w8 | Native (all 200) |
| --- | ---: | ---: | ---: | ---: |
| AG News acc | 174 | 0.747 | 0.741 | 0.755 |
| Emotion acc | 193 | 0.477 | 0.487 | 0.470 |
| Amazon MAE | 155 | 0.631 | 0.633 | 0.600 |
| SST-5 MAE | 200 | 0.611 | 0.611 | 0.611 |
| BoolQ AUROC | 107 | 0.726 | 0.726 | 0.750 |
| Irony AUROC | 200 | 0.714 | 0.714 | 0.714 |
| SMS spam AUROC | 200 | 0.919 | 0.919 | 0.918 |
| SST-2 AUROC | 200 | 0.991 | 0.991 | 0.991 |

Parity on identical items: fp16 decisions agree with native on **1,428 / 1,429**, w8 on 1,417 / 1,429. Differences on AG News, Amazon and BoolQ are the answered subset (shorter items), not conversion; the four fully covered tasks match native.

### Verdict — which path was evaluated, and the app path

The reproduction above used **raw logits** (`VerdictManager.logits`, the author's `build_model_input` rendering) with the author's single temperature fitted on the calibration split. It did **not** use `VerdictManager.answer`, which is what an app calls: `answer` renders "It is {description}", always appends abstention last, and applies the shipped per-K `calibrator.json`. The app path on the same 1,000 test records (`answer_path.py`, 1,500 requests incl. calibration, 0 unsupported):

| Test (1,000) | Accuracy | NLL | Brier | ECE | OOS recall / precision |
| --- | ---: | ---: | ---: | ---: | --- |
| `answer` as shipped (per-K calibrator) | 93.1% | 0.395 | 0.167 | **21.2%** | 96.5% / 83.9% |
| same logits, uncalibrated (T=1) | 93.1% | 0.213 | 0.100 | 1.9% | same |
| same logits, one temperature refit on cal | 93.1% | 0.209 | 0.099 | 3.0% | same |
| author's protocol (raw rendering, fitted T) | 95.0% | 0.177 | 0.078 | 3.4% | 97.5% / 89.5% |

Investigation of the difference (all against the author's code at `30f15564`):

| Check | Finding |
| --- | --- |
| Artifact selection | The author's serving engine (`core/engine_encoder.py`) auto-loads `artifacts/v2/calibrator.json`, byte-identical (sha256 af2a8769…) to the `calibrator.json` FluidUse downloads. The evaluation script (`scripts/evaluate.py`) does **not** use it; it fits its own single temperature (1.427) on the calibration split. |
| Option-count handling | Engine keys `per_k` by the candidate count including abstention; K without an entry (e.g. 8) falls back to the global 2.8039 (`calibrator(logits, k=…)` raises `TypeError`, then `forward` applies `exp(log_temperature)`). FluidUse does the same. |
| Temperature application | Engine divides logits by T before softmax; FluidUse the same. |
| Double application | None. Core ML logits match the engine's raw FP32 ONNX logits within 0.10 on the pinned cases (not scaled by ~1/T), and `answer` probabilities match the engine's calibrated probabilities within 0.0034 (same argmax, 4/4). |
| Prompt rendering | **Mismatch found and fixed.** The engine validates score levels as floats (pydantic `FiniteFloat`), so integer levels render `(Value: 0.0)`; FluidUse followed the Mobius reference port and rendered `(Value: 0)`. `VerdictQuestion.Level.value` is now `Double`, rendered like Python `repr(float)` (the earlier int/float `VerdictNumber` distinction is removed). Choice ("It is …") and noul rendering already matched. |

Regression test `VerdictCalibrationTests` replays six cases pinned from the author's engine on real held-out contexts (`Fixtures/verdict-upstream-calibration.json`, generated by `Tools/official-benchmarks/verdict/pin_upstream_calibration.py`): choice K=3/5/8 (fallback)/25, noul, score with integer and fractional levels. It asserts byte-identical prompts, the applied temperature, and calibrated probabilities within 1e-5 from the pinned logits. Nothing is fitted on the held-out set.

Conclusion: `VerdictManager.answer` reproduces the author's **serving** path. The 21.2% ECE on this test set is a property of the shipped open-domain per-K calibrator (under-confident in-domain), not of the port; the author's card numbers come from the evaluation script's separately fitted single temperature. `VerdictManager.Configuration.calibration` now makes the choice explicit — `.shipped` (default, unchanged), `.uncalibrated`, or `.temperature(T)` for a temperature the caller fits on its own calibration data. The default is not changed on the strength of this one in-domain test set. The app rendering ("It is …", abstention last) scores 1.9 pt below the evaluation rendering on this data; that is the author's serving format and is not changed either.

## Laya: reconciliation with the existing evaluation (`Benchmarks.md`)

The existing run is FluidUse's closest match to an official suite, with limits that `Benchmarks.md` states only in part:

- **What matches the official protocol.** Nine application suites are transcribed from laya `research/scripts/bench_apps.py` @ `ee760389` (wording, criteria, slicing, seed 13; banking77 consumes no draws, so omitting it does not shift the others). MASSIVE-en follows the T4 notebook (fresh `Random(13)`, first 300 rows, 20 options). The reference column is the correct checkpoint (laya-multilingual). Laya's own `max_len` 1024 applies on both sides: 16 rows are truncated by laya's native rule and none are dropped.
- **Where it deviates.** banking77 excluded (77 options > 32 slots). The upstream datasets are unpinned, and the author's per-row `app_benchmark.json` was never published, so the upstream rows cannot be recovered (our `suites.jsonl` stores the rows we evaluated). The Python reference used laya 0.3.4 from PyPI, which predates the harness commit.
- **Not covered at all:** typed-decisions (the authors' headline), XNLI, MASSIVE in the other 13/50 languages and MASSIVE scenario, the T4 English suites (600-row AG News, Emotion, SST-5, BoolQ, prompt injections), option-order robustness, calibration.
- **Corrections to `Benchmarks.md` wording.** The "Upstream (T4)" column header is unverified: `BENCHMARKS.md` gives no hardware for the application run and `bench_apps.py` records `device: cpu`. The gaps against the published numbers (model routing 0.441 vs 0.123, guardrails 0.805 vs 0.755, support triage 0.540 vs 0.522) are unexplained; "upstream run artefact" is a hypothesis, not a finding.
- **Conversion evidence stands:** Core ML matches our PyTorch reference on every suite (100% row argmax agreement on eight, 99.8% on two), which isolates conversion from the unexplained upstream gaps.

No new Laya run is planned in this pass.

## Request counts versus scored examples

- **LFM (RLCD):** 60 requests = 2 warmup passes × 12 cases (24, discarded as the official protocol does) + 3 measured repeats × 12 cases (36). Accuracy is over the 36 measured rows (108 fields), i.e. 12 distinct cases scored three times; outputs are deterministic, so each repeat scores the same. Latency statistics use the 36 measured rows.
- **Verdict:** 2,500 requests = 500 calibration (used only to fit the temperature, never scored) + 1,000 held-out test (the headline metrics) + 1,000 challenge-slice records (7 slices, each scored separately with the fitted temperature). The row-agreement check covers the 1,000 test rows.
