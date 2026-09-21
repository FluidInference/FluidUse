# FluidUse

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
.package(url: "https://github.com/FluidInference/FluidUse.git", from: "0.1.0")
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

`LayaManager.Configuration` picks the buckets to load and their compute units (128 → CPU+ANE,
longer → all units); a prompt runs on the smallest loaded bucket that fits, and the largest one
truncates the state on the right like laya's `max_len`. The tokenizer is a Swift port of the
mmBERT/Gemma byte-fallback BPE and matches HuggingFace `tokenizers` on the conversion fixtures.

On laya's own published suites (3,899 questions rebuilt from upstream's scripts) the Core ML
buckets match the PyTorch reference's accuracy on every suite (AG News 0.935, Emotion 0.537,
MASSIVE-20 0.657, spam/phishing 0.993, guardrails 0.805, …) at 5.2 ms median per question versus
61.6 ms for PyTorch on the same Mac's CPU; reports and reproduction live in
[mobius `models/computer-use/laya/coreml`](https://github.com/FluidInference/mobius/tree/main/models/computer-use/laya/coreml).

```bash
swift run -c release FluidUseLaya answer --state "…" --type choice \
    --instructions "What does the customer want?" --options "refund|order status|technical help"
swift run -c release FluidUseLaya tetris --pieces 200            # headless Tetris, P(clean) per landing
swift run -c release FluidUseLaya benchmark --suites suites.jsonl --reference reference-rows.jsonl
swift run -c release LayaTetrisDemo                              # SwiftUI: laya plays Tetris, live decisions
```

`LayaTetrisDemo` ([Sources/LayaTetrisDemo](Sources/LayaTetrisDemo)) scores every legal landing
of the current piece with *"Is this a clean placement?"* and plays the best one, showing each
sentence, its probability, and per-call latency. Zero-shot laya clears a few dozen lines before
topping out; the built-in heuristic policy plays indefinitely. It is a latency demo, not a Tetris
player.

## Demo

```bash
swift run -c release FluidUseDemo
```

Pick a running app, load a profile, press **Fill form** (or **9** from any
app). Every model call is logged with its input, ranked options, and time.
Requires Accessibility access for the launching terminal.

## Scope

The model matches a field to a value it is given. It does not read résumés,
reason about dropdown options, or write free text. The harness handles
observation, typing, selection, and an answer sheet; uploads and essays are
left to the person. Submit is never clicked unless enabled.

## License

Apache 2.0. CUA-S1-FORMS is MIT, from Cua.
