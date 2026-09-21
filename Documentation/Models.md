# Models

What FluidUse runs today, what is converted but not wired in, and the open-weight
candidates worth converting next. Sizes are fp16 on disk. Traction was checked on
2026-09-21; licenses were read from each model card.

## Shipped

| Model | Params / fp16 | License | Role in FluidUse |
| --- | ---: | --- | --- |
| [cua-ai/cua-s1-forms](https://huggingface.co/cua-ai/cua-s1-forms) → [FluidInference/cua-s1-forms-coreml](https://huggingface.co/FluidInference/cua-s1-forms-coreml) | 706K / 1.4 MB | MIT | Matches each form field to a value from the document: `fill`, `check`, `click`, or `skip`. ~1 ms per decision on the Neural Engine. |

CUA-S1-FORMS is a one-pass option scorer (a "System One" model in TypeSafe's terms), not an
LLM. It is a byte-level 2-layer Transformer with the option-attention head from
[jevlike](https://github.com/vinnylarouge/jevlike), trained by Cua on 10,000 synthetic forms.
It does not read goals, write text, or reason about dropdown options; those are the gaps the
rest of this page is about.

## Converted, not yet wired in

| Model | Params / fp16 | License | Where |
| --- | ---: | --- | --- |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) → [FluidInference/laya-coreml](https://huggingface.co/FluidInference/laya-coreml) | 322M / 644 MB | Apache-2.0 | `LayaManager` on FluidAudio branch `feat/laya-coreml` (PR #946). Four token buckets, 32 option slots. Answers `choice` / `score` / `noul` questions over a text state in one pass. |

On macOS 26 and newer, the on-device Apple Intelligence model covers entity extraction,
question answers, and dropdown choice through `FoundationModelsAssistant` with no conversion
at all. The candidates below matter for macOS 14 and 15, and for the one thing Apple's model
does not do: choosing an element on a page.

## Candidates to convert

Under 1 GB, permissively licensed, untouched so far. Ordered by how much each adds.

| Model | Params / fp16 | License | Purpose | Evidence |
| --- | ---: | --- | --- | --- |
| [cklxx/laya-browser](https://huggingface.co/cklxx/laya-browser) v10s | 322M / 644 MB | Apache-2.0 | Picks which element to act on and which operation (`CLICK`, `TYPE_TEXT`, `SELECT`, `DONE`) from a goal and the page's element table. The role TypeSafe's Jev plays in browser agents. | Element top-1 0.63 over ~45 candidates, operation accuracy 0.88, 50% on 16 live browser tasks. Same architecture as the laya port above, so it is a weight bucket plus a sequence format, not a new conversion. |
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M / 843 MB | Apache-2.0 | Dropdown choice, yes/no verification, and ordered-scale questions over a profile. | 0.766 on the typed-decisions benchmark, above Jev 1.13's published 0.727. Loads through `LayaManager`. |
| [urchade/gliner_small-v2.1](https://huggingface.co/urchade/gliner_small-v2.1) | ~150M / ~300 MB | Apache-2.0 | Pulls name, phone, employer, and any other labeled span out of an unstructured résumé or letter, replacing the `Label: value` parser. | 19k downloads a month; GLiNER repo 3.9k stars. |
| [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) | 231M / 463 MB | MIT | Captions icons and regions in a screenshot, for apps with no accessibility tree. OmniParser's captioner. | 3M downloads a month. Only useful once a screenshot path exists. |

Fallbacks, also under 1 GB and untouched:

| Model | Params / fp16 | License | Purpose |
| --- | ---: | --- | --- |
| [urchade/gliner_multi_pii-v1](https://huggingface.co/urchade/gliner_multi_pii-v1) | ~300M / ~580 MB | Apache-2.0 | Same extractor, tuned for identity documents and personal-data fields. |
| [torontodeveloper/mind2web-candidate-ranker](https://huggingface.co/torontodeveloper/mind2web-candidate-ranker) | 184M / 369 MB | MIT | Ranks page elements by how likely each is the next click target; no operation choice. Recall@10 93% on unseen Mind2Web domains. |
| [shreyanbr/system-one-distilled](https://huggingface.co/shreyanbr/system-one-distilled) | 71M / 142 MB | Apache-2.0 | Smallest generic typed-decision model, for routing and gating where laya is overkill. |

### More S1-class specialists

Nothing else at CUA-S1-FORMS's scale exists with weights. The way to get more is to train
them: Cua's trainer, synthetic generator, and evaluator ship under MIT in
[trycua/cua `libs/cua-s1`](https://github.com/trycua/cua/tree/main/libs/cua-s1), and
[jevbetter](https://github.com/olanotolu/jevbetter) accepts the same JSONL with a stronger
encoder. Closed-vocabulary tasks that fit the shape, each 1 to 5 MB:

- Submit-or-advance: which button submits, continues, or dismisses a dialog.
- Field-role tagging: search box, login form, cookie-banner accept.
- Clutter-or-ad per element.

## Demo ideas

| Model | Demo | What the viewer sees |
| --- | --- | --- |
| laya-browser | Goal-driven browsing in Safari: "book the cheapest nonstop to New York in October" | Each pick and its probability logged while the agent steps through Google Flights, fully offline. |
| laya-typed-decisions | Dropdowns and consent fields in an HR onboarding flow | "Years of experience", "work authorization", and "I agree" resolved from the profile with a confidence next to each. |
| gliner_small-v2.1 | Drop a real résumé PDF instead of a `Label: value` profile | The entity list fills from prose in under a second, then S1 fills the form from it. |
| gliner_multi_pii-v1 | Photo of an ID or insurance card via Vision OCR | Name, date of birth, member ID pulled from the card into a patient-intake form. Nothing leaves the machine. |
| Florence-2-base | Screenshot of an app with no accessibility tree | Icons and buttons labeled in place, then a decision made on those labels. |
| mind2web-candidate-ranker | "Where would you click next" overlay | Top five elements highlighted with scores, no action taken. |
| system-one-distilled | Routing benchmark | Hundreds of tickets classified per second with latency per call. |

## Ruled out

- **TypeSafe Jev**: hosted API only, no weights.
- **Screenshot grounders at 2B and up** (GUI-Owl-1.5-2B, UI-TARS-2B, ShowUI-2B, GUI-Actor-2B, Holo2-4B): 4 to 9 GB fp16, no Core ML precedent for their vision towers.
- **Non-commercial licenses**: Holo1.5-3B (Qwen Research), pngwn/system-one-qwen3.5-4b-scorer (CC-BY-NC), laya-vision-smolvlm and laya-grounded (CC-BY-NC).
- **AGPL**: OmniParser `icon_detect`, macpaw-research/yolov11l-ui-elements-detection.
- **No weights**: Apple Ferret-UI Lite, Ferret-UI 2, UI-JEPA.
- **"1 MB" heads that need a 0.6B backbone**: samatv256/mini-Jev and similar.
- **Decoder-based Jev clones over 1 GB**: Mapika/decider-0.8b and 2b, AndeyTait/JevForge-0.8B, IamBusy/OpenJev-0.6B. decider-2b is the best supported of these (20k downloads a month) if a larger chooser is ever needed on Mac.
