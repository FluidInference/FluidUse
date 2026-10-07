# FrontDoorDemo — Chatbot Front Door

A support chatbot's incoming traffic (40 made-up messages) screened and sorted on this Mac by
Vela-2.0-0.3B (Core ML). Each message gets two calls:

1. **guard** — `attack` + `p_harm` (trained wording; ≈92-token schema → 128-token Neural Engine bucket).
   jailbreak ≥ 0.8 → blocked (jailbreak); else unsafe ≥ 0.8 → blocked (harmful).
2. **route** — only if not blocked: `topic` (billing / technical / account / shipping / general) + `pii` spans (GPU).
   Personal info is shown redacted as `[LABEL]` chips.

```
swift run -c release FrontDoorDemo [--model <dir>]   # or VELA_DIR=<dir>; default /Users/hanweng/Documents/vela2/release03
swift run -c release FrontDoorDemo --demo            # start the run automatically
swift run -c release FrontDoorDemo --selftest        # no window: all 40 + model-vs-expected confusion
```

The run is strictly sequential, stops after the last message with a summary banner, and prints one stdout line per message. The inbox reads like a chat: oldest at the top, newest arriving at the bottom.
Click any card for the original text (PII highlighted), attack / harm / topic probabilities and both timings.
