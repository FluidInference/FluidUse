# Sort photos

Sorts Oxford-IIIT Pets photos into 37 breeds with SigLIP 2 (base, 256 px) on Core ML. The only hint the model
gets is each breed's name, in the prompt `a photo of a {breed}, a type of pet.`; nothing is trained on these photos.

```bash
swift run -c release ImageSortDemo
```

The first run downloads the Core ML packages (about 750 MB, checksum-verified) from
[FluidInference/siglip2-base-patch16-256-coreml](https://huggingface.co/FluidInference/siglip2-base-patch16-256-coreml)
and 1,000 test photos from the Hugging Face dataset viewer; both are cached. `SIGLIP2_MODEL_DIR` loads a local
conversion instead.

- Each sorted photo flies from the current-photo panel into its breed's row, so the rows grow into a bar chart made
  of photos; a red frame marks a photo whose true breed differs. Orange rows are cat breeds, blue are dog breeds.
  The panel shows the latest photo with its five most likely breeds.
- Four photos are in flight at once, so decoding and resizing on the CPU overlap the encoder on the Neural Engine.
- `IMAGE_SORT_COUNT` sets the sample size (default 1,000; up to 3,669 test photos, or 7,349 when the train split is
  also cached). `IMAGE_SORT_WAIT=1` waits for Start (Space); `IMAGE_SORT_LOG=1` prints decisions to the terminal.

Headless: `swift run -c release ImageSortCheck` (M5 Pro, macOS 27: 94.85% on the 3,669 test photos, 202 photos/s;
`--split=all` 94.26% on 7,349). The same 7,349 photos with the PyTorch model (transformers, fp32, MPS, batch 32)
take 102 s at 72 photos/s and 4.2 GB peak memory, against 36 s, 202 photos/s, and 262 MB here.

Data: Oxford-IIIT Pets (Parkhi et al., 2012), CC BY-SA 4.0. Model: google/siglip2-base-patch16-256, Apache-2.0.
