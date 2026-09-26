# Sort photos

Sorts Oxford-IIIT Pets photos into 37 breeds with SigLIP 2 (base, 256 px) on Core ML. The only hint the model
gets is each breed's name, in the prompt `a photo of a {breed}, a type of pet.`; nothing is trained on these photos.

```bash
SIGLIP2_MODEL_DIR=/path/to/siglip2-base-patch16-256 IMAGE_SORT_AUTOPLAY=12 swift run -c release ImageSortDemo
```

- `IMAGE_SORT_AUTOPLAY=N` flies N photos in Show mode, then switches to Turbo; `IMAGE_SORT_AUTOSTART=show|turbo`
  starts a run in that mode. Without either, press Start.
- `IMAGE_SORT_COUNT` sets the sample size (default 1,000; the test split has 3,669, and a cache that also holds
  the train split allows up to 7,349). `IMAGE_SORT_WAIT=1` waits for Start (Space); `IMAGE_SORT_LOG=1` prints
  each decision.
- Each sorted photo becomes a tile in its breed's row, so the rows grow into a bar chart made of photos; a red
  frame marks a photo whose gold breed differs. Orange rows are cat breeds, blue are dog breeds. The left panel
  shows the latest photo with its five most likely breeds.

Headless: `swift run -c release ImageSortCheck --inflight=4` (M5 Pro, macOS 27: 94.85% on all 3,669 test photos,
193 photos/s).

Data: Oxford-IIIT Pets test split (Parkhi et al., 2012), CC BY-SA 4.0. Model: google/siglip2-base-patch16-256,
Apache-2.0.
