# UnlimitedOcrDemo

Drop a PDF: its pages (up to 8) are read by Baidu Unlimited-OCR in **one generation**, on this Mac, offline. The
page thumbnails highlight the layout block being written while the markdown streams in.

```bash
Sources/UnlimitedOcrDemo/demo.sh paper.pdf   # app + Ghostty window: CPU / GPU bars on top, block-by-block log below
swift run -c release UnlimitedOcrDemo        # app only (MLX builds need Xcode's Metal toolchain: DEVELOPER_DIR=…/Xcode.app)
```

The pinned bundle ([FluidInference/unlimited-ocr-coreml](https://huggingface.co/FluidInference/unlimited-ocr-coreml),
~3.8 GB) downloads on first launch; `UNLIMITED_OCR_BUNDLE=<dir>` uses a local one. The first launch also compiles
the packages (about a minute), later launches load in ~12 s.

On an M5 Pro the first 8 pages of Baidu's paper (6,727 tokens) take 160 s at 43 tokens/s, with a 1.9 GB footprint
plus 2.7 GB of memory-mapped decoder weights.

`UnlimitedOcrCheck <bundle dir> <page.png … | doc.pdf> [--ref output.md]` runs the same pipeline from the command
line; with `--npy pixels.npy` (reference pixels) it reproduces upstream's fp32 output token for token.
