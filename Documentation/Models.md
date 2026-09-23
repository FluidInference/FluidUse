# Models

What FluidUse runs today and the open-weight candidates worth converting next.
Sizes are fp16 on disk unless stated otherwise. Traction was checked on
2026-09-21; licenses were read from each model card. The [Decision Index 0.1](https://huggingface.co/spaces/multimodalart/jev-decision-index)
snapshot below was checked on 2026-09-22. Its scores measure a frozen set of static
decision tasks, not game play.

## Converted and integrated

| Model | Params / fp16 | License | Role in FluidUse |
| --- | ---: | --- | --- |
| [cua-ai/cua-s1-forms](https://huggingface.co/cua-ai/cua-s1-forms) → [FluidInference/cua-s1-forms-coreml](https://huggingface.co/FluidInference/cua-s1-forms-coreml) | 706K / 1.4 MB | MIT | Matches each form field to a value from the document: `fill`, `check`, `click`, or `skip`. ~1 ms per decision on the Neural Engine. |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) → [FluidInference/laya-coreml](https://huggingface.co/FluidInference/laya-coreml) | 322M / ~644 MB | Apache-2.0 | `LayaManager` answers text-state `choice`, `score`, and `noul` questions. Core ML buckets at 128/256/512/1024 tokens; Tetris and 2048 demos. |
| [knowledgator/gliclass-edge-v3.0](https://huggingface.co/knowledgator/gliclass-edge-v3.0) → [FluidInference/gliclass-edge-apps-coreml](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml) | 32.7M / 65.7 MB (33.0 MB LUT8 L128) | Apache-2.0 | Application-tuned `GLiClassManager` compares 2–25 supplied labels in one pass; Tetris, 2048, and GLiClass-vs-Laya demos. FP16 buckets at 128/256/512 tokens. |

CUA-S1-FORMS is a one-pass option scorer (a "System One" model in TypeSafe's terms), not an
LLM. It is a byte-level 2-layer Transformer with the option-attention head from
[jevlike](https://github.com/vinnylarouge/jevlike), trained by Cua on 10,000 synthetic forms.
It does not read goals, write text, or reason about dropdown options. Laya and
GLiClass handle more general decisions but the form-filling app still uses CUA-S1-FORMS.
Apple's built-in Foundation Models are not a FluidUse conversion and are not
included in this inventory.

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

### Game decisions

See [Decision game demos](GameDemos.md) for the game list, the Flappy Bird
comparison specification, and the matched-seed evaluation protocol. Flappy Bird
has been tried with every integrated model, including Kev, LFM, and Lex; none played it
usefully ([results](GameDemos.md#flappy-bird-trial-results)). Connect Four and a
local Snake check gave the same answer: every model lost 0–20 to a simple
heuristic at Connect Four, and Laya ate 2 food to the heuristic's 27 at Snake. The
recommended next demos are forgiving merge puzzles such as drop-merge and Threes;
Minesweeper is kept as a calibration benchmark
([details](GameDemos.md#recommended-build-order)).

### Promising Index models below 1B parameters

These are **parameter-count** candidates from the [Index data](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json),
not models already converted for FluidUse. The Index score is its balanced static
score out of 100; it is included to identify candidates, not to predict game
performance. License and input type come from the linked model cards. Approximate
fp16 sizes include only weights (2 bytes per parameter), before runtime overhead.

| Model | Params / fp16 | Index score | Input and best demo role | Status |
| --- | ---: | ---: | --- | --- |
| [Kev-0.6B](https://huggingface.co/jaredpalmer/kev-0.6b) | 596M / ~1.19 GB | 31.30 | Text state, typed `choice` / `score` / `noul`; first generic Flappy Bird decision candidate. | Apache-2.0; LoRA and pointer head require its Qwen3 base and a new runtime port. |
| [LFM2.5-350M-RLCD](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD) | 354M / ~709 MB | 25.79 | Text decision baseline for Flappy Bird or routing. | `lfm1.0` custom license: review terms before conversion or redistribution. |
| [GLiNER2.5-base](https://huggingface.co/fastino/gliner2.5-base-v1) | 194M / ~387 MB | 24.70 | Text extraction and classification; best fit for form/document decisions. | Apache-2.0; needs a separate GLiNER runtime, not a direct game controller. |
| [GLiNER2.5-small](https://huggingface.co/fastino/gliner2.5-small-v1) | 74M / ~148 MB | 23.93 | Smaller text extraction/classification baseline. | Apache-2.0; different from the GLiNER small v2.1 candidate above. |
| [GLiNER2.5-multi](https://huggingface.co/fastino/gliner2.5-multi-v1) | 287M / ~575 MB | 22.42 | Multilingual form/document decisions. | Apache-2.0; separate GLiNER runtime. |
| [Decision-1.0-Lex](https://huggingface.co/llm-semantic-router/decision-1.0-lex) | 308M served in Index / ~616 MB | 19.57 | Text choices and scores; operational tasks and a Flappy Bird text-state trial. | Apache-2.0; card describes a 572M full checkpoint, so verify the artifact before sizing a port. |
| [Decision-1.0-Kai](https://huggingface.co/llm-semantic-router/decision-1.0-kai) | 308M served in Index / ~616 MB | 18.37 | General text decision comparator for Lex. | Apache-2.0; card calls it 0.6B, so confirm the exact artifact before conversion. |
| [Laya](https://huggingface.co/convaiinnovations/laya) | 421M / ~843 MB | 16.39 | Text typed decisions; comparison with the existing multilingual Core ML port. | Apache-2.0; the Index tests the English checkpoint, not `laya-multilingual`. |

The Index also includes [NanoJev](https://huggingface.co/C-Tianyu/NanoJev)
(596M, 26.19), but its Hub card does not declare a license; keep it out of the
conversion shortlist until terms are clear. [Kev-0.5B](https://huggingface.co/jaredpalmer/kev-0.5b)
(494M, 30.34) is explicitly described by its author as a superseded prototype.
Neither is a first-choice port. All Index models in this table take text input;
there is no evidence in the Index that they can read Flappy Bird screenshots.

[PlayJev-0.8B](https://huggingface.co/OmniJev/PlayJev-0.8B) is a separate,
Apache-2.0 **vision** candidate under 1B parameters, not a scored Index entrant.
It has a published Flappy Bird run and can take rendered frames, making it the
most direct visual comparator. Its fp16 weights are roughly 1.6 GB, above the
1 GB fp16 limit used for the main FluidUse conversion shortlist. Its published
Flappy Bird score must not be compared directly with a new harness run; use the
same physics, seeds, action timing, and scoring for every model in the demo.

## Ruled out

- **TypeSafe Jev**: hosted API only, no weights.
- **Screenshot grounders at 2B and up** (GUI-Owl-1.5-2B, UI-TARS-2B, ShowUI-2B, GUI-Actor-2B, Holo2-4B): 4 to 9 GB fp16, no Core ML precedent for their vision towers.
- **Non-commercial licenses**: Holo1.5-3B (Qwen Research), pngwn/system-one-qwen3.5-4b-scorer (CC-BY-NC), laya-vision-smolvlm and laya-grounded (CC-BY-NC).
- **AGPL**: OmniParser `icon_detect`, macpaw-research/yolov11l-ui-elements-detection.
- **No weights**: Apple Ferret-UI Lite, Ferret-UI 2, UI-JEPA.
- **"1 MB" heads that need a 0.6B backbone**: samatv256/mini-Jev and similar.
- **Decoder-based Jev clones over 1 GB**: Mapika/decider-0.8b and 2b, AndeyTait/JevForge-0.8B, IamBusy/OpenJev-0.6B. decider-2b is the best supported of these (20k downloads a month) if a larger chooser is ever needed on Mac.
