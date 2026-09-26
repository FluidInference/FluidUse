# FluidUse

See [decision-model runtime support](Documentation/DecisionModelSupport.md) for the current Core ML model matrix and handoff notes.

Local computer use on Apple silicon. FluidUse reads a form in a running Mac
app or browser through the Accessibility API, asks a small on-device model
what belongs in each field, and types the answer into the real app. About
1 ms per decision on the Neural Engine, nothing leaves the machine.

The first model is [CUA-S1-FORMS](https://huggingface.co/FluidInference/cua-s1-forms-coreml),
a 706K-parameter form specialist from [Cua](https://github.com/trycua/cua),
converted to Core ML and served by [FluidAudio](https://github.com/FluidInference/FluidAudio).

## Demo video



https://github.com/user-attachments/assets/a0b31285-05be-4bcf-a645-4283eb327c35



## Use

```swift
.package(url: "https://github.com/FluidInference/FluidUse.git", from: "0.3.0")
```

```swift
import FluidAudio
import FluidUse

let model = try await CuaS1FormsManager.load()
let driver = AccessibilityFormDriver(application: safari)   // any NSRunningApplication
let page = try await driver.snapshot()                       // fields, labels, values
let options = FormSchema.renderOptions(entities: profile)    // "fill Email: …", check, click, skip

for field in page.elements where field.isActionable {
    let context = FormSchema.renderContext(formTitle: page.title, element: field)
    let decision = try await model.score(context: context, options: options)
    // decode with FormSchema.decode, then driver.type / driver.click
}
```

`WebFormDriver` does the same for an embedded `WKWebView`. `DocumentEntities`
turns a PDF or text file of `Label: value` lines into the profile; an optional
`PredeterminedAnswer` sheet covers question-style fields the model does not decide.

## laya typed decisions

[laya](https://github.com/NandhaKishorM/laya) (Convai Innovations, Apache-2.0) is an open
Jev-style decision model: a 322M mmBERT encoder plus decision head that answers typed
`choice` / `score` / `noul` questions about a text state with calibrated probabilities in one
pass, no generated tokens. `LayaManager` runs the Core ML buckets from
[FluidInference/laya-coreml](https://huggingface.co/FluidInference/laya-coreml) (128/256/512/1024
tokens, 32 options) on the Neural Engine: 3.7 ms per short question on an M5 Pro, about 7×
faster than the ~27 ms upstream reports on an M1 Max GPU.

```swift
let laya = try await LayaManager.load()  // downloads the 128 + 512 buckets and tokenizer.json
let answers = try await laya.answer(
    state: "Customer: I was charged twice for order #4471 and nobody replies. Refund me today.",
    questions: [
        .choice("What does the customer want?", options: ["refund", "order status", "technical help"]),
        .score("How frustrated is the customer?", levels: ["calm", "annoyed", "angry"]),
        .noul("Is the customer likely to churn?"),
    ])
print(answers[0].selectedLabel, answers[1].expectedScore!, answers[2].noul!)
```

`LayaManager.Configuration` picks the buckets to load, their compute units (128 → CPU+ANE,
longer → all units) and the weight precision (`fp16`, or `e8` with an int8 embedding table at 30%
less weight and the same accuracy); a prompt runs on the smallest loaded bucket that fits, and the largest one
truncates the state on the right like laya's `max_len`. The tokenizer is a Swift port of the
mmBERT/Gemma byte-fallback BPE and matches HuggingFace `tokenizers` on the conversion fixtures.

On laya's own published suites (3,899 questions rebuilt from upstream's scripts) the Core ML
buckets match the PyTorch reference's accuracy on every suite (AG News 0.935, Emotion 0.537,
MASSIVE-20 0.657, spam/phishing 0.993, guardrails 0.808, …) at 5.2 ms median per question versus
61.6 ms for PyTorch on the same Mac's CPU. Full tables in [Benchmarks.md](Benchmarks.md); the
questions, reference answers, reports and conversion live in
[mobius `models/computer-use/laya/coreml`](https://github.com/FluidInference/mobius/tree/main/models/computer-use/laya/coreml).

```bash
swift run -c release FluidUseLaya answer --state "…" --type choice \
    --instructions "What does the customer want?" --options "refund|order status|technical help"
swift run -c release FluidUseLaya tetris --shortlist --describe graded --pieces 200  # headless Tetris, P(clean) per landing
swift run -c release FluidUseLaya 2048 --precision lut8 --games 10
swift run -c release FluidUseLaya benchmark --suites <mobius>/benchmark/suites.jsonl --reference <mobius>/benchmark/reference-rows.jsonl
swift run -c release LayaTetrisDemo                              # SwiftUI: GLiClass/laya play Tetris
swift run -c release GLiClass2048Demo                            # SwiftUI: GLiClass plays 2048
swift run -c release Decision2048BenchDemo                       # SwiftUI: GLiClass vs laya at 2048
```

`LayaTetrisDemo` ([Sources/LayaTetrisDemo](Sources/LayaTetrisDemo), [25 s clip](Media/laya-tetris.mp4))
defaults to GLiClass comparing the heuristic's two strongest legal landings in one call. Its policy
control can instead run laya, which scores each surviving landing independently with *"Is this a
clean placement?"*, or the heuristic and random baselines. The window is a single column so it sits
next to a terminal; the app prints decisions and model latency to stdout for presentations:

```bash
tmux new-session -d -s laya -c . && tmux send-keys -t laya 'sudo asitop' C-m
tmux split-window -v -l 22 -c . && tmux send-keys -t laya:0.1 'LAYA_DEMO_AUTOLOAD=1 swift run -c release LayaTetrisDemo' C-m
tmux attach -t laya
```

The current ten-seed capped run averages 3,667 pieces for GLiClass LUT8 and 2,874 for the corrected
heuristic control. See [Benchmarks.md](Benchmarks.md) for the exact policy, per-seed results, and the
limits of comparison with the older laya measurements.

## GLiNER 2.5 classification

`GLiNER2Manager` runs the published base or multilingual classification head on device. Both
packages use an L128 bucket, support up to eight labels, and quantize only the embedding table
to eight bits. The manager downloads pinned, SHA-256 checked Core ML assets, source config files,
and the matching Unigram tokenizer from [base](https://huggingface.co/FluidInference/gliner2-5-base-coreml) or
[multilingual](https://huggingface.co/FluidInference/gliner2-5-multi-coreml) on first use.

```swift
let classifier = try await GLiNER2Manager.load(variant: .base)
let answer = try await classifier.classify(
    text: "The rocket launched successfully.",
    task: "topic",
    labels: ["science", "sports", "politics"])
print(answer.selectedLabel, answer.probabilities)
```

Use `.multilingual` for the multilingual checkpoint. The base package is 291 MB and the
multilingual package is 385 MB. Inputs that exceed 128 tokens raise an error. These packages
contain the classification path; the native entity, relation, and record extraction heads
are not exposed by this Swift manager. A ten-seed 2048 comparison with GLiClass is in
[Benchmarks.md](Benchmarks.md).

## Kev decisions

`KevFastManager` runs [Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) (Qwen3.5 backbone, Apache-2.0) on the
GPU through Core ML. One call reads the text and answers all of a request's questions (multiple choice, yes/no, or a
score, with calibrated probabilities); questions that do not fit fall back to one call per question. The pinned,
checksummed snapshot downloads from [FluidInference/kev-0.8b-coreml](https://huggingface.co/FluidInference/kev-0.8b-coreml)
on first use (~3.5 GB, macOS 15 / iOS 18).

```swift
let kev = try await KevFastManager.load(from: try await KevModelStore.ensure())
let answers = try await kev.answer(
    state: "Shoes arrived two weeks late and in the wrong size.",
    questions: [
        .choice("Which team should handle this?", options: [("returns", nil), ("shipping", nil), ("billing", nil)]),
        .noul("Does this need urgent human attention?"),
    ])
print(answers.map(\.best))
```

On an M5 Pro a short ticket with two questions takes 18 ms and a Wikipedia bio with twelve yes/no questions about 38 ms.
`swift run -c release KevGuessWhoDemo` plays Guess Who over 80 Wikipedia people with it
([Sources/KevGuessWhoDemo](Sources/KevGuessWhoDemo/README.md)).

## Demo

```bash
swift run -c release FluidUseDemo
```

Pick a running app, load a profile, press **Fill form** (or **9** from any
app). Every model call is logged with its input, ranked options, and time.
Requires Accessibility access for the launching terminal.

## Benchmarks

The on-device models, measured on the same Mac with checked-in reports: [Benchmarks.md](Benchmarks.md).
CUA-S1-FORMS: 0.9 ms per decision on the Neural Engine, accuracy identical to PyTorch on the
24,370-row synthetic test. laya: 3.6 ms per short question, identical to PyTorch on laya's ten
published suites, e8 buckets 30% smaller at the same accuracy. GLiClass Edge Apps v2: 1.61 ms FP16
or 1.81 ms LUT8 for a two-option L128 decision. Its
[Core ML packages and config](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml)
are on Hugging Face, with the conversion pipeline in
[mobius PR #101](https://github.com/FluidInference/mobius/pull/101).
GLiClass demos download the selected Core ML bucket and tokenizer on first use. The loader reads the
published `config.json`, checks each file against `checksums.json`, and caches the assets under
`~/Library/Application Support/FluidUse/Models/gliclass-edge-apps-coreml`. `GLICLASS_MODEL_DIR`
or CLI `--model-dir` still selects a local directory. The Hub publishes FP16 at L128/L256/L512 and
LUT8 at L128; other local precision variants require an explicit local directory.

## Scope

The model matches a field to a value it is given. It does not read résumés,
reason about dropdown options, or write free text. The harness handles
observation, typing, selection, and an answer sheet; uploads and essays are
left to the person. Submit is never clicked unless enabled.

## License

Apache 2.0. CUA-S1-FORMS is MIT, from Cua.
