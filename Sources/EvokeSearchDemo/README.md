# EvokeSearchDemo

Search-as-you-type over a mock social timeline with
[Granite-Embedding-30M-Sparse](https://huggingface.co/FluidInference/granite-embedding-30m-sparse-coreml), the
learned sparse encoder behind Intelligent Internet's [Evoke](https://github.com/Intelligent-Internet/Evoke), on the
Neural Engine. Each query becomes weighted vocabulary terms, including related terms you never typed (✨ in the UI),
scored against the posts with a sparse dot product, the same shape as a BM25 inverted index.

```bash
Sources/EvokeSearchDemo/demo.sh [posts.json]               # app + Ghostty/tmux: macmon on top, live model log below
swift run -c release EvokeSearchDemo                       # built-in 40 posts
uv run --with pandas --with pyarrow --with huggingface_hub python Sources/EvokeSearchDemo/fetch-posts.py posts.json
swift run -c release EvokeSearchDemo --posts posts.json    # 1000 public tweets across 19 topics
```

The first launch downloads the 64-token package (58 MB, checksum-pinned) into the FluidUse cache; `EVOKE_MODEL_DIR`
points at a local copy instead.

- **Autoplay** types each suggested query with no pauses: every keystroke is encoded, searched and drawn before the
  next. It runs for one take (`--seconds`, default 55) and stops on a fully typed query. Typing or clicking pauses
  it (resuming after 20 s idle). The banner shows live searches per second.
- **Restart** drops the index, re-encodes every post (progress and time in the app and the log), and starts a new take.
- **Keyword / Evoke tabs** compare BM25 over literal words with Evoke terms. Chips under each post show the terms that
  matched it.
- **Speed** (M5 Pro): ~0.8–1.1 ms per query on the Neural Engine, 1000 posts indexed in ~1.2 s, 60–100 searches/s
  with the UI redrawing every keystroke. Posts and queries share one 64-token package; longer text is truncated.
