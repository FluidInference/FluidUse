# GuardrailDemo — On-device Guardrail (Vela-2.0-0.3B on Core ML)

A native macOS SwiftUI port of the `vela2/guard` web demo. A chat app where every message and reply is checked on this
Mac by vLLM Semantic Router's **Vela 2.0 0.3B** (`Vela2Manager`, Core ML) — nothing is sent anywhere. The point of the
demo is to show **where each check runs**: encoder sequences of ≤ 128 tokens go to the **Neural Engine**, longer ones to
the **GPU**, and every result in the UI carries a badge such as `⚡ Neural Engine · 3.4 ms total 3.6` or
`GPU · 16.2 ms total 17.3` (encoder ms, then the full check incl. tokenization + heads).

Layout: source document (left, editable — replies are checked against it), chat thread + composer with live PII
highlighting (center), and "The model's last check" (right): big ms number with its compute unit, verdict, bars for
Prompt attack / Harmful / Needs fact-check, top-3 route, personal information found, and what leaves the device
(PII replaced by `[LABEL]`). Three scenarios (refund policy, medication leaflet, prompt attack) with suggestion chips and
scripted replies; replies are editable with ↻ re-check, unsupported claims are underlined red with a tooltip. The header
keeps a running tally `checks: N · on ANE: M`.

## Run

```bash
swift run -c release GuardrailDemo [--model <dir>]     # or VELA_DIR=<dir>; default: the pinned FluidInference/vela-2.0-0.3b-coreml snapshot (Vela2ModelStore)
swift run -c release GuardrailDemo --selftest           # no window: all scenarios through all lanes, prints, exits
```

The model directory is the Core ML release (`coreml_config.json`, `calibration.json`, `heads.*`, `tokenizer.json`,
`Vela2Encoder.mlpackage`/`.mlmodelc`). It is passed as a flag, not a bare path, so AppKit does not try to open it as a
document. Hidden `--autoplay [refund|medicine|attack]` sends a scenario's suggestions for screenshots.
One stdout line is printed per check.

## The three lanes

| Lane | Questions | Schema | Runs on |
| --- | --- | --- | --- |
| 1a. While typing (120 ms debounce) | `attack` + `p_harm` | ≈92 tokens | ANE for short messages (up to ~30 message tokens), GPU above |
| 1b. While typing, live highlight | `pii` span (17 labels) | ≈192 tokens | GPU (256 bucket) |
| 2. On send, one call | `attack`, `p_harm`, `domain`, `factcheck`, `pii` | ≈535 tokens | GPU (1024 bucket) |
| 3. Reply check | `halu` span over the answer, with user + source context | small | ANE when user + source + reply ≤ 128 tokens |

A message is blocked when P(jailbreak) ≥ 0.8 or P(unsafe) ≥ 0.8 (`server.py` FLAG); otherwise the masked text is what
appears in the thread and what the reply check sees as the request. Live results carry a sequence number and the text
they were computed for, so stale results are dropped. Question wording and option order are Vela 2.0's trained ones
(`v03/USAGE.md` `TRAINED`, `calibration.json` `pii_schema`).

## Measured (M5 Pro, `--selftest`, steady state)

```
   3.4 ms ANE  live    attack 0.00 harm 0.06   (total 3.6 ms, tokens→bucket 106→128)
   5.3 ms GPU  pii     pii []                  (total 5.6 ms, tokens→bucket 206→256)
  16.2 ms GPU  send    attack 0.06 harm 0.11  route business  pii []   (total 17.1 ms, tokens→bucket 549→1024)
   3.6 ms ANE  reply   unsupported ["can", "items within 90 days", "also", "your original shipping costs", ...]  (116→128)
   3.4 ms ANE  live    attack 0.99 harm 0.91   (DAN message, 115→128)
  16.1 ms GPU  send    BLOCKED (prompt injection / jailbreak)

live   on ANE 3/5, encoder p50 3.4 ms   (the Maria / Tom messages are 133–145 tokens → GPU 256, ~4.9 ms)
pii    on ANE 0/5, encoder p50 5.3 ms
send   on ANE 0/5, encoder p50 16.2 ms, total p50 17.4 ms
reply  on ANE 2/3, encoder p50 3.6 ms
```

In the interactive app, isolated checks after idle gaps measure higher (ANE ~4–9 ms, GPU send ~28 ms) than the
back-to-back selftest, as the compute units clock down between keystrokes.
