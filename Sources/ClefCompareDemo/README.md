# ClefCompareDemo

The same support tickets through two Core ML decision models at once:

- **clef-flash 9B** ([FluidInference/clef-flash-coreml](https://huggingface.co/FluidInference/clef-flash-coreml),
  Cloudflare's model, 8-bit) on the GPU
- **clef-text-0.6b** ([FluidInference/clef-text-0.6b-coreml](https://huggingface.co/FluidInference/clef-text-0.6b-coreml),
  Qwen3-0.6B distilled from clef-flash) on the Neural Engine

Each ticket gets three typed decisions (team / urgency / refund) from each model; mismatches are outlined, the header
shows the share of decisions that match and both models' median time per ticket. Tickets are fictional mock data.

```bash
Sources/ClefCompareDemo/demo.sh     # downloads both bundles on first run (~12 GB), or set CLEF_FLASH_BUNDLE / CLEF_TEXT_BUNDLE
```
