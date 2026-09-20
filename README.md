# CUA-S1-FORMS Demo (macOS)

SwiftUI app that drives a form with `CuaS1FormsManager` from the parent
package: a source document on the left, a target on the right, and one
on-device decision per form element. The target is either the embedded web
page or another running application's window, driven through the macOS
Accessibility API the way Cua's native driver and Codex computer use work.

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
2. Pick a **Target**. *Embedded web page* observes the page in the app's own
   web view (**Sample form** loads a bundled job-board style application; any
   URL works). Any other entry is a running app: **Observe window** reads its
   frontmost window's accessibility tree and draws the controls it found to
   scale, with the labels the model will see. This needs Accessibility access
   for the terminal that launched the demo (System Settings › Privacy &
   Security › Accessibility).
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

## Driving a real app

With an app target, controls come from `AXTextField`, `AXTextArea`,
`AXCheckBox`, `AXPopUpButton`/`AXComboBox`, and `AXButton` elements, mapped to
the `Edit`, `CheckBox`, `ComboBox`, and `Button` roles the checkpoint was
trained on. Labels use the element's own description or title when the app
provides one (Safari web content, AppKit forms). PDF viewers expose fields
without names, so the driver falls back to nearby page text: the caption
directly above the box, else the row text to the left, and for checkboxes the
text to their right; leading form numbering such as `(a)` is stripped. Values
are typed through `AXValue`, checkboxes and buttons through `AXPress`, and a
floating outline marks the element being acted on over the target window.

Verified on Preview with the IRS W-4 (`https://www.irs.gov/pub/irs-pdf/fw4.pdf`,
whose fields carry no tooltips): the driver observed 21 controls, labels came
out as "First name and middle initial", "Last name", "Address", "City or
town, state, and ZIP code", "Social security number", "Single or Married
filing separately", and so on, and the model's fills landed in the document,
which Preview marked edited and auto-saved. Safari web content is observed
with its own labels. Chrome needs `AXManualAccessibility` enabled and was not
tested.

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
- On the W-4 the model skipped "First name and middle initial" at 93%, skipped
  the combined "City or town, state, and ZIP code" field, and checked "Head of
  household" at 100%. Government forms are well outside its synthetic training
  distribution; treat app-target runs as a demonstration of the loop, and keep
  **Allow submit click** off.

Set `CUA_DEMO_AUTORUN=1` to load the model, the sample profile, and the sample
form and fill it without clicking; add `CUA_DEMO_QUIT=1` to exit afterwards,
`CUA_DEMO_DOCUMENT=/path/to/file.pdf` or `CUA_DEMO_URL=…` to substitute inputs,
`CUA_DEMO_TARGET=Preview` to drive a running app instead of the web view, and
`CUA_DEMO_PLAN_ONLY=1` to score without acting.
