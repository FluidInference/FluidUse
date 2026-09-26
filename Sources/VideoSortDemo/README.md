# SigLIP 2 on video

Plays a video in real time and labels the newest frame with SigLIP 2 (base, 256 px) on Core ML, as fast as the
model allows.

```bash
SIGLIP2_MODEL_DIR=/path/to/siglip2-base-patch16-256 VIDEO_SORT_WAIT=1 swift run -c release VideoSortDemo
```

- Default scene, **Name the animal**: `animals.mp4`, 26 real clips of different animals from Wikimedia Commons,
  about 4 s each. Every frame is scored against the 26 animal names (`a photo of a {animal}.`); the caption shows
  the top label, the side panel the top five, and the strip below adds a card each time a new animal holds for six
  labeled frames. "Correct frames" compares each frame with the animal its clip shows (`animals-credits.json`).
  M5 Pro: every frame at 31 fps, 9.5 ms per frame, 94.2% of frames correct, 26 of 26 species spotted.
- `VIDEO_SORT_SCENE=potatoes`: a USDA potato-sorting video (public domain) with a 6 × 4 grid; each cell is labeled
  from ten phrases (potatoes, a gloved hand, a truck, the sky, …), about 8 grids per second.
- `VIDEO_SORT_WAIT=1` waits on the first frame for Start (Space); `VIDEO_SORT_LOG=1` prints each labeled frame;
  `VIDEO_SORT_FILE` / `VIDEO_SORT_START` play another file.

Videos are read from `~/Library/Caches/FluidUse/video-sort/`; they are not bundled. Clip sources, licenses, and
authors are listed in `animals-credits.json`.
