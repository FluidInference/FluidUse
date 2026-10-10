# ClefFlashDemo

Cloudflare [clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Qwen3.5-9B decision model, Apache-2.0) triaging
support tickets on this Mac through Core ML. Each ticket gets three typed decisions in one forward pass: team (choice),
urgency (score), refund (noul). Tickets are fictional mock data; typed tickets jump the queue.

```bash
Sources/ClefFlashDemo/demo.sh            # app + Ghostty (macmon on top, decision log below)
```

- Runtime: `ClefFlashManager` — 8 decoder packages (4 layers each, 8-bit weights, 6.5 GB) chained on the GPU + the
  joint schema head (fp32), 512-token bucket. ~0.50 s per ticket on an M5 Pro; ~40 s to load (first launch also
  compiles the packages into `<bundle>/compiled/`).
- Bundle: built in model-lab (`models/clef-flash-coreml`, `assemble_bundle.py`); not on Hugging Face.
- GPU only: the Qwen3.5 decoder puts 0 ops on the Neural Engine (every op falls back to CPU under `CPU_AND_NE`).
- Same answers as the Python Core ML pipeline on the mock tickets; 8-bit vs fp32 reference: 1 flip / 611 ARC questions.
