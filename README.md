# FluidUse

Local computer use on Apple silicon. FluidUse reads a form in a running Mac
app or browser through the Accessibility API, asks a small on-device model
what belongs in each field, and types the answer into the real app. About
1 ms per decision on the Neural Engine, nothing leaves the machine.

The first model is [CUA-S1-FORMS](https://huggingface.co/FluidInference/cua-s1-forms-coreml),
a 706K-parameter form specialist from [Cua](https://github.com/trycua/cua),
converted to Core ML and served by [FluidAudio](https://github.com/FluidInference/FluidAudio).

## Demo video

[Media/demo.mp4](Media/demo.mp4) (41 s, 9.9 MB): a 706K-parameter model filling a live job
application in Safari, with every decision logged and asitop showing the
Neural Engine barely moving.

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
