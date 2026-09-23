# Decision-model runtime support

FluidUse serves the weighted sub-1B models on the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) through two paths. Native Swift managers run Core ML directly. A Python-assisted bridge runs a model's published Core ML packages with the model's own released tokenizer and decoding code. Neither path loads an original PyTorch checkpoint. The smoke results below check that each serving path works. They are not Decision Index scores or 2048 quality comparisons.

| Model | FluidUse API | Artifacts | Path |
| --- | --- | --- | --- |
| Laya | `LayaManager` | [laya-coreml](https://huggingface.co/FluidInference/laya-coreml) | Native Swift |
| GLiNER 2.5 small / base / multilingual | `GLiNER2Manager` (`.small`, `.base`, `.multilingual`) | [small](https://huggingface.co/FluidInference/gliner2-5-small-coreml), [base](https://huggingface.co/FluidInference/gliner2-5-base-coreml), [multi](https://huggingface.co/FluidInference/gliner2-5-multi-coreml) | Native Swift |
| Verdict | `VerdictManager` | [verdict-coreml](https://huggingface.co/FluidInference/verdict-coreml) | Native Swift, calibrated, with trained abstention |
| GLiClass Edge Apps v2 | `GLiClassManager` | [gliclass-edge-apps-coreml](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml) | Native Swift |
| Kev 0.5B / 0.6B | `PublishedCoreMLManager` + `evaluate(SystemOneRequest)` | [0.5B](https://huggingface.co/FluidInference/kev-0-5b-coreml), [0.6B](https://huggingface.co/FluidInference/kev-0.6b-coreml) | Bridge |
| Decision 1.0 Kai / Lex | `PublishedCoreMLManager` + `evaluate(SystemOneRequest)` | [Kai](https://huggingface.co/FluidInference/decision-1.0-kai-coreml), [Lex](https://huggingface.co/FluidInference/decision-1.0-lex-coreml) | Bridge |
| LFM2.5-350M-RLCD | `PublishedCoreMLManager` + `constrained(context:fields:)` | [lfm2-5-350m-rlcd-coreml](https://huggingface.co/FluidInference/lfm2-5-350m-rlcd-coreml) | Bridge |
| Jeff | `PublishedCoreMLManager` + `classify(text:labels:)` | [jeff-coreml](https://huggingface.co/FluidInference/jeff-coreml) | Bridge |
| NanoJev | `PublishedCoreMLManager.start` + `decide(state:question:)` | [source only](https://huggingface.co/FluidInference/nanojev-coreml) | Bridge on a local conversion; weights not redistributed |

## Bridge usage

```swift
let kai = try await PublishedCoreMLManager.load(model: .kai, configuration: .init(precision: "w8"))
let response = try await kai.evaluate(SystemOneRequest(
    state: "Please refund the duplicate charge. I need this fixed today.",
    questions: [
        .noul("refund_requested", "Does the customer explicitly request a refund?"),
        .choice("team", "Which team should handle this request?",
                options: [DecisionOption("Billing", "Charges and refunds"),
                          DecisionOption("Support", "Technical problems")]),
    ]))
if case .choice(let team, _, _) = response["team"] { print(team) }
```

`load` downloads the pinned revision into `~/Library/Application Support/FluidUse/Models/<repo>`. Every file is checked against the size and SHA-256 in `Resources/published-coreml-manifest.json`, and only the packages for the requested precision are fetched. It then runs `uv sync --frozen --no-dev` on the repository's own `pyproject.toml`/`uv.lock`, and `uv` installs Python 3.12 if needed. The first environment build downloads PyTorch; later builds finish in under a second. To serve from an existing snapshot and interpreter, use `start(model:from:python:)`. `evaluate(_ request: Data)` remains for raw requests in each runtime's JSON format.

Behavior to know:

- **Order is preserved.** Request bytes reach Python unchanged apart from line breaks, and replies come back as written. Choice criteria order is the option order the model sees.
- **One request at a time, in call order.** Concurrent callers queue.
- **Timeouts and cancellation end the session.** They are set by `startupTimeout` (default 600 s) and `requestTimeout` (default 300 s). A timed-out or cancelled request terminates the worker, and later calls throw `.closed`. Start a new session afterwards.
- **Model errors keep the session.** A model-side failure, such as a Kev request with two questions, throws `.runtime` for that request only.
- **Worker output is separated.** The worker writes replies to a private copy of stdout, and anything else a library prints goes to stderr. `standardErrorTail` and startup failures include the last 16 KB of stderr.
- **Per-model quirks:**
  - Kev packages take one question per call and round reported values to two decimals, as upstream does.
  - Kai and Lex release one typed package before loading the next, because holding two crashed Core ML during conversion. A request that mixes question types therefore reloads packages.
  - LFM must stay on `.all`. Jeff's scores are independent sigmoids.

Verdict score levels are `Double` and render like the author's engine, which validates them as floats: `0` renders as `0.0`. `VerdictManager.Configuration.calibration` chooses between the released per-K calibrator (`.shipped`, the default and the author's serving behavior), `.uncalibrated`, and `.temperature(T)`; see [OfficialBenchmarks.md](OfficialBenchmarks.md) for why the shipped calibrator is under-confident on the author's in-domain test set. Verdict caches its compiled model beside the package, and a re-downloaded package invalidates that cache.

## Validation (Apple M5 Pro, macOS 27.0)

Every bridged model was run from freshly materialized snapshots of the pinned Hub revisions, through `ensure` → `prepareEnvironment` → `start` → typed request. Times include first-call Core ML compilation.

| Model | Precision | Start | First / next request | Result |
| --- | --- | ---: | ---: | --- |
| Kev 0.5B | fp16 / e8 | 34.1 s / 13.5 s | 0.31 s / 7.3 s first | Billing (0.93 / 0.92); reversed options reported in request order |
| Kev 0.6B | fp16 / w8 | 27.7 s / 12.1 s | 1.13 → 0.021 s / 10.4 → 0.024 s | Billing |
| Kai | fp16 / w8 | 3.4 s / 2.4 s | 13.9 s / 29.3 s (three types) | refund 0.87, Billing, urgency 0.99 |
| Lex | fp16 / w8 | 2.4 s / 2.6 s | 14.8 s / 24.5 s (three types) | refund 0.25, Billing, urgency 1.90 |
| LFM2.5-350M-RLCD | fp16 | 4.2 s | 0.31 s | route billing, urgent true |
| Jeff | fp16 / w8 | 18.1 s / 11.5 s | 1.48 s / 0.91 s | billing (0.99996) |
| NanoJev (local) | fp16 | 18.7 s | — | boolean `true`; Core ML matches native argmax on the conversion fixtures |
| Verdict | fp16 L128+L512 | 84.9 s fresh download + compile; 0.68 s cached | — | `card_lost` 0.62838 (published reference value) |

Failure paths were exercised against the real LFM worker: startup timeout, startup crash (stderr surfaced), request timeout then `.closed`, cancellation then `.closed`, a per-request `KeyError` with the session still usable, concurrent callers served in order, and shutdown. `PublishedCoreMLTests` covers request encoding against the published Kai example, response decoding, the manifest, and the line channel. `PublishedCoreMLIntegrationTests` repeats the real-model checks when `FLUIDUSE_PUBLISHED_COREML_CACHE` is set.

## Upstream notes

- Before Hub revisions `bdb0bc4e` (Kai) and `6ca0547a` (Lex), both repositories shipped a `conversion/typed_coreml.py` older than their `run_coreml.py`, so the published runtime failed at import (`marker_tensor`). Those revisions upload the matching file from Mobius `models/computer-use/decision-vela/coreml/`. On the published `marker_positions` packages it only adds the helper. The manifest pins the fixed revisions, and `ensure` replaces a stale local copy.
- NanoJev's converted weights stay local until the checkpoint owner clarifies redistribution rights. Its checkpoint also differs from the Decision Index-era one.
- To move a bridged model to a new Hub revision, edit `Tools/pin_published_coreml.py` and regenerate the manifest.

Conversion and parity sources live under [`models/computer-use`](https://github.com/FluidInference/mobius/tree/main/models/computer-use) in Mobius. The bucket limits and license notes there remain authoritative.
