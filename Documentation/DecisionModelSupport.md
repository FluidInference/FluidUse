# Decision-model runtime support (work in progress)

This branch extends FluidUse toward the weighted sub-1B models on the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index). The table separates **published model availability** from **FluidUse runtime implementation**. A model being on the Hub or in Mobius does not imply it has a native Swift serving path, and none of these small smoke checks establishes the historical leaderboard score or comparative 2048 quality.

| Model | FluidUse path | Artifact status | This branch |
| --- | --- | --- | --- |
| Laya | `LayaManager` | [Published](https://huggingface.co/FluidInference/laya-coreml) | Existing native Swift runtime |
| GLiNER 2.5 small | `GLiNER2Manager.load(variant: .small)` | [Published](https://huggingface.co/FluidInference/gliner2-5-small-coreml) | New native Swift variant; real Core ML smoke passed |
| GLiNER 2.5 base / multilingual | `GLiNER2Manager` | [Base](https://huggingface.co/FluidInference/gliner2-5-base-coreml), [multilingual](https://huggingface.co/FluidInference/gliner2-5-multi-coreml) | Existing native Swift runtime |
| Verdict | `VerdictManager` | [Published](https://huggingface.co/FluidInference/verdict-coreml) | New native Swift typed path; real choice and abstention smoke passed |
| Kev 0.5B | `PublishedCoreMLManager(model: .kev05, …)` | [Published](https://huggingface.co/FluidInference/kev-0-5b-coreml) | Python-assisted Core ML bridge; real typed choice smoke passed |
| Kev 0.6B | `PublishedCoreMLManager(model: .kev06, …)` | [Published](https://huggingface.co/FluidInference/kev-0.6b-coreml) | Python-assisted Core ML bridge; two consecutive W8 requests passed |
| Decision 1.0 Kai / Lex | `PublishedCoreMLManager(model: .kai/.lex, …)` | [Kai](https://huggingface.co/FluidInference/decision-1.0-kai-coreml), [Lex](https://huggingface.co/FluidInference/decision-1.0-lex-coreml) | Python-assisted Core ML bridge; each passed one choice+noul+score request |
| LFM2.5-350M-RLCD | `PublishedCoreMLManager(model: .lfm350, …)` | [Published](https://huggingface.co/FluidInference/lfm2-5-350m-rlcd-coreml) | Python-assisted Core ML bridge; real constrained schema request passed through Swift |
| NanoJev | `PublishedCoreMLManager(model: .nanojev, …)` | [Source-only](https://huggingface.co/FluidInference/nanojev-coreml) | Local FP16 encoder+head supported; real four-choice smoke passed. Converted weights must stay local pending redistribution permission. |
| Jeff (extra converted model) | `PublishedCoreMLManager(model: .jeff, …)` | [Published](https://huggingface.co/FluidInference/jeff-coreml) | Python-assisted Core ML bridge; real two-label classification smoke passed |
| GLiClass Edge Apps v2 (extra comparator) | `GLiClassManager` | [Published](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml) | Existing native Swift runtime |

Every **published weighted** repository above has a root `config.json`; NanoJev's Hub repository is source-only. The Python-assisted bridge runs published Core ML packages locally. It does not load the original trained PyTorch checkpoint. It keeps each model's own tokenizer, rendering, and decoding contract, which differs across model families. The Python interpreter supplied to the bridge must have that model's published toolkit dependencies installed. It currently accepts and returns JSON `Data` because Kev/Kai/Lex use typed System One requests, LFM uses constrained JSON schemas, and Jeff uses dynamic-label classification. For NanoJev, point to the local Mobius conversion directory containing `build/` packages and `assets.lock.json`; source/tokenizer files may be fetched from its pinned upstream revision, but trained weights are not downloaded by this runtime.

For example, with a materialized Kev 0.5B Hub snapshot and its toolkit's Python environment:

```swift
let manager = try PublishedCoreMLManager(
    model: .kev05,
    from: URL(fileURLWithPath: "/path/to/kev-0-5b-coreml"),
    python: URL(fileURLWithPath: "/path/to/python3"))
let answerJSON = try await manager.evaluate(Data(requestJSON.utf8))
```

`requestJSON` is one **unlabelled** Kev System One request, such as:

```json
{"model":"kev-0.5b","state":"The piece leaves one hole beneath it.","questions":{"q":{"type":"choice","instructions":"Classify the placement.","criteria":{"clean":"No buried holes","risky":"Creates a cavity"}}}}
```

The bridge loads a model process once and serializes subsequent requests. Kev 0.6B now retains the loaded package between calls. Kai/Lex deliberately release one typed package before loading another because retaining two caused a Core ML crash in the conversion work. LFM must use its validated `.all` compute placement; the published CPU+ANE run failed numerical parity. Jeff's outputs are independent sigmoid label scores, not a softmax distribution. Verdict always appends its trained abstention option and calibrates **raw logits** with the released `calibrator.json`; its Core ML `probabilities` output is uncalibrated. Verdict and GLiNER small run directly in Swift without Python.

The bridged model code was inspected at these Hub revisions: Kev 0.5B `06b6bad2c1209d96d1f87600ccfcaee01959a788`, Kev 0.6B `f2a24a115626e2be75f0a8816448ff0f34520c16`, Kai `977330be74e49d2b32421941972b455943c1d2b1`, Lex `92149c311eabf50a8490fe744623614a09b88057`, LFM `78cd6a54f3704c9cd4fc8bce904455da8786f3b0`, and Jeff `37e70eee651a3de62d93de229c459e850afed539`. Materialize a snapshot at the intended revision (for example, `hf download FluidInference/kev-0-5b-coreml --revision 06b6bad2c1209d96d1f87600ccfcaee01959a788 --local-dir /path/to/kev-0-5b-coreml`) before passing its directory to FluidUse. The bridge currently checks package presence, not hashes; content pinning is follow-up work.

## Validation so far

- `swift build -c release` passed after the new managers and bridge were added.
- A temporary Swift executable (removed from the branch) loaded the real Verdict L128 package and the real GLiNER small W8 package. Verdict produced the exact published 53-token choice sequence, selected `card_lost`, and abstained on the no-information case. GLiNER small produced finite two-label scores.
- The same temporary Swift executable exercised the persistent bridge against real Kev 0.5B and LFM packages. Kev selected `risky`; LFM selected the billing route. The Python worker was separately exercised with real Kev 0.6B W8 (twice in one process), Kai, Lex, Jeff, and local NanoJev packages.
- `swift test` cannot run on this Mac's Command Line Tools installation: both `Testing` and `XCTest` modules are missing. The added XCTest integration tests are therefore **not yet CI-validated**. Run them with a full Xcode toolchain and the environment variables named in each test.

## Handoff work

1. Run the full test suite with Xcode. Run the optional real-package tests for GLiNER small and Verdict, then add comparable integration tests for each bridged model.
2. Check the bridge against **materialized snapshots of the current published Hub revisions**, especially Kai/Lex's optional W8 paths and Jeff W8. The local smoke checks used conversion-stage packages; some stage directories predate the root `config.json` addition.
3. Add a pinned, checksum-verified downloader for the Python-assisted packages. Today the caller provides a local snapshot and a Python environment; native Laya, GLiClass, GLiNER, and Verdict already have their own stores.
4. Review process lifetime, startup timeout, stderr handling, cancellation, and error reporting in `PublishedCoreMLManager`. Its current synchronous startup can wait indefinitely if a model load hangs.
5. Add typed Swift request/answer wrappers per family if these APIs are intended for app developers. The JSON bridge preserves correctness but is a lower-level interface.
6. Do not publish NanoJev's locally converted weights until the checkpoint owner clarifies redistribution rights. Its current checkpoint also differs from the Decision Index-era checkpoint.
7. Verify Verdict score rendering for fractional JSON values: the current Swift `Double` API preserves the common integer level labels, while Python's formatter retains whether a source JSON number was written as an integer or float. Add a representation that preserves that distinction if upstream parity for fractional score levels is required.

The Mobius conversion and parity sources for these models are under [`models/computer-use`](https://github.com/FluidInference/mobius/tree/main/models/computer-use). The fixed bucket limits and per-model license notes in those directories remain authoritative.
