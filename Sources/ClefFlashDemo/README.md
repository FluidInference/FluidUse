# ClefFlashDemo

Support-ticket triage with a Clef decision model on this Mac. Each ticket gets three typed decisions in one forward
pass — team (choice), urgency (score), refund (noul) — while the board works through a fixed backlog of 1,000
fictional tickets (`MockTickets.swift`, deterministic); typed tickets jump the queue.

```bash
CLEF_MODEL=text Sources/ClefFlashDemo/demo.sh   # clef-text-0.6b, GPU: ~48 ms / ticket, 1,000 tickets in ~57 s
Sources/ClefFlashDemo/demo.sh                   # Cloudflare clef-flash 9B, GPU: ~0.5 s / ticket
```

`demo.sh` downloads the pinned bundle on first run (`CLEF_TEXT_BUNDLE` / `CLEF_FLASH_BUNDLE` for a local one) and
opens Ghostty with live CPU / GPU bar charts (`bars.py` over `macmon pipe`, no sudo) above the decision log.

- **clef-text-0.6b** — [FluidInference/clef-text-0.6b-coreml](https://huggingface.co/FluidInference/clef-text-0.6b-coreml),
  `ClefTextManager`: Qwen3-0.6B + Clef joint head distilled from clef-flash; runs on the GPU or 100% on the Neural
  Engine (`.cpuAndNeuralEngine`, ~77 ms / ticket).
- **clef-flash 9B** — [FluidInference/clef-flash-coreml](https://huggingface.co/FluidInference/clef-flash-coreml),
  `ClefFlashManager`: 8 decoder packages (8-bit) chained on the GPU; ~7 GB of RAM.

`ClefCompareDemo` runs both side by side on the same tickets (9B on the GPU, 0.6B on the Neural Engine).
