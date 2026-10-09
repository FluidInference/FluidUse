# Sort by topic

Streams 10,080 posts into a timeline-style feed, files each one into a broad topic as it arrives, then splits any
broad topic into specific subtopics on request. Everything runs on the Mac: EmbeddingGemma 2 (text encoder, Core ML,
Neural Engine) embeds the posts; topics are spherical k-means clusters of those embeddings; names are phrases from
the posts themselves, picked by how close EmbeddingGemma 2 puts them to the topic's centre and how far from its
siblings.

```bash
swift run -c release TopicSortDemo
Sources/TopicSortDemo/demo.sh          # same app, plus one terminal: macmon (ANE / GPU / power) above the live log
```

The log (`$TMPDIR/topic-sort-demo.log`, colour-coded) shows the model load, every batch (posts, Neural Engine calls, ms,
posts/s, one post and where it was filed), each re-sort with its topics, and each split with its subtopics.

The first run downloads the model (about 530 MB, checksum-verified) from
[FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml); the first
load on a Mac then compiles it for the Neural Engine once (a few minutes). `EMBEDDINGGEMMA2_MODEL_DIR` loads a local
copy instead. Needs macOS 15+.

- **Start** (Space) streams the posts; **Pause** / Resume (Space); **Reset** (⌘R) clears everything for a replay.
- Broad topics are found after the first batch and re-found at 160, 400, 1,000, 4,000 posts and at the end; colours
  carry over between re-sorts. In between, each post joins the nearest topic (one embedding plus a dot product per
  topic).
- **Split** sorts a topic's posts into 2–6 subtopics; **Merge** undoes it; right-click renames.
- Posts are embedded eight to a Neural Engine call (`pack_256`) with 64 per batch in flight: about 540 posts/s on an
  M5 Pro, all 10,080 sorted about 30 s after launch.

Data: `Resources/mock-posts.jsonl` is fictional: six themes (AI, food, fitness, travel, money, space) × four subtopics
× 420 posts, written for this demo (`Tools/mock-posts/`, assembled by `Tools/make_mock_posts.py`). Scored against
those labels, the broad topics match the themes at 92.8% purity (NMI 0.82) and splitting every topic matches the 24
subtopics at 81.3% (NMI 0.76).

Options: `--posts=file.jsonl` (one `Bookmark` JSON per line), `--rate=N` posts/s, `--topics=N`, `--limit=N`,
`--autostart`, `--auto-split`, `--split-all`, `--dump=out.jsonl`, `--snapshot=out.png`.
