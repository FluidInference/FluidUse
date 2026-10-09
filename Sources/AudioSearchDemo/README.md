# Search audio

Indexes hours of audio in seconds, then finds moments in it from a text query: a phrase someone said, a topic, a
sound. Everything runs on the Mac with EmbeddingGemma 2: its audio encoder turns each 10 s window into tokens on the
GPU, and its text model (Neural Engine) embeds those tokens into the same space as text queries.

```bash
swift run -c release AudioSearchDemo                 # the public datasets already on this Mac (see below)
swift run -c release AudioSearchDemo --audio=~/Podcasts --audio=meeting.m4a
Sources/AudioSearchDemo/demo.sh                      # same app, plus one terminal: macmon above the live log
```

The first run downloads the models (about 1.1 GB, checksum-verified) from
[FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml) and compiles
them once (a few minutes). Needs macOS 15+.

- **Play** (↩) runs the show once, in order: **📥 index** every file (16 kHz mono, 10 s windows across all files with
  four in flight, so the GPU and the Neural Engine work at the same time), then **🔎 listen** to eight example searches (each
  typed, then its top three windows play, 3 s each), then **⚡ search speed** for 30 s (~100 different queries
  back to back, 64 at a time, embedded eight per Neural Engine call, ranked against every window with one matrix
  multiply). **Pause** (Space) holds it anywhere, timers included; **Replay** (⌘R) starts over from an empty index.
- When it is done, type a query (or pick a suggestion); ▶ plays a window. `--segment=` and `--clip=` change the step
  and clip lengths; `--autostart` presses Play on launch

Default audio, when present: the 1-hour Earnings-22 sample and FLEURS English/French used by FluidAudio's
benchmarks (`~/Library/Application Support/FluidAudio`), and ESC-50 sound clips
(`~/Library/Application Support/FluidUse/Datasets/esc50/wav`), renamed to random ids (`clip_3f9a1c.wav`) so the file
name cannot give the answer away; the classes live in `esc50/labels.json`. Nothing is bundled.

Numbers (M5 Pro, macOS 27):

| | |
|---|---|
| 1 hour earnings call, 360 windows | 4.6–5.7 s, 626–777× real time |
| 4 h 36 min (2,301 files incl. 2,000 five-second clips) | 34.7 s, 478× real time |
| Search speed over those 2,786 windows | ~700 searches/s, 1.4 ms each (query embedding 1.4 ms on the ANE, ranking 0.02 ms) |
| Swift vs sentence-transformers (fp32) | cosine 0.9995 on a LibriSpeech clip |

Quality: speech search works well when the query names something said ("forward-looking statements" finds the
safe-harbor sentence; "operator opens the line" finds the call-in prompt). Sound events are weaker: zero-shot ESC-50
accuracy is 24% over 50 classes (PyTorch fp32 gets the same, so that is the model, not the conversion); distinctive
sounds such as crying or laughing work, similar ones (rain vs waves vs helicopter) get mixed up.
