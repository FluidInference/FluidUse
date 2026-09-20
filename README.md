# CUA-S1-FORMS Demo (macOS)

SwiftUI app that drives a web form with `CuaS1FormsManager` from the parent
package: a source document on the left, a live web page on the right, and one
on-device decision per form element.

```bash
cd Examples/CuaFormsDemo
swift run -c release CuaFormsDemo
```

No Xcode project is needed. **Load model** downloads the 1.5 MB FP16 Core ML
package from `FluidInference/cua-s1-forms-coreml` on first use and reads its
compute plan, so the Neural Engine badge reports measured op placement rather
than an assumption. Use a release build for latency numbers; debug builds add
several milliseconds of overhead per call.

## Flow

1. **Sample profile** (or **Open…**) extracts `Label: value` lines from a PDF or
   text file, the same way upstream's `cua_s1.pdf` does, and adds first/last
   name variants. Each entity becomes one `fill Label: value` option; the model
   scores at most 29 entities plus `check`, `click`, and `skip`.
2. **Sample form** loads a bundled job-board style application page. Any URL
   works in the address field; the observer runs in the page's main frame.
3. **Plan only** scores every actionable element and lists the decisions.
   **Fill form** scores and acts as it goes: text fields are typed in through
   the page's native value setter with `input`/`change` events, checkboxes are
   clicked and re-read, and file inputs receive the source document.
4. Button clicks are collected but never executed unless **Allow submit click**
   is on, and then only one control whose label is a recognized submit label.

The decision log shows the element role and label, the chosen action, the
model's confidence, and the wall time of that `score` call including the actor
hop. The overlay on the page repeats the last and median latency and the
compute-plan placement.

## What is and is not the model

The model sees exactly upstream's context string, for example:

```
TASK fill the form from the document, then submit
FORM Senior Software Engineer, Perception - Example Robotics - Job Application
ELEMENT Edit "Email address" value=""
```

Everything else is host code: DOM observation and label derivation
(`WebFormDriver.swift`), document parsing (`DocumentEntities.swift`), ordering
and authorization (`DemoModel.swift`). Attaching the document to a file input
is a host rule because the model's vocabulary has no attach action; upstream
trains "Upload file" buttons as `skip`. The log labels those rows `host rule`.

## Observed behavior worth knowing

- Placeholder text is passed as the `hint` field and shifts decisions. On the
  sample page, `USD / year`, `YYYY-MM-DD`, and `https://linkedin.com/in/…`
  placeholders made the model skip fields it fills without them, even though
  the labels are in upstream's training catalogue. The sample page ships
  without those placeholders; real sites with unusual hints will see the effect.
- The model can fill a free-text field with an unrelated entity. On the sample
  page it filled "Message to hiring manager" with the website entity at 88%,
  above the default 0.5 threshold.
- Re-running on a filled page skips every filled field; the current value is
  part of the context.
- Radio buttons, custom dropdowns, and inputs inside iframes are not observed.

Set `CUA_DEMO_AUTORUN=1` to load the model, the sample profile, and the sample
form and fill it without clicking; add `CUA_DEMO_QUIT=1` to exit afterwards,
`CUA_DEMO_DOCUMENT=/path/to/file.pdf` or `CUA_DEMO_URL=…` to substitute inputs.
