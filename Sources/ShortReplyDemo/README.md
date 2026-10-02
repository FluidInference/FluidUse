# Short Reply demo

Open a post's reply box in any app (X or Reddit in Chrome or Safari, Slack, Mail) and press **9**: the reply is drafted
on device and pasted into the box, no UI. Press **0** (or 9 again on the same post) to regenerate: the first draft is the
model's greedy reply (the benchmarked one); later ones are sampled (temperature 0.7, top-p 0.9, seeded), never
repeating an earlier draft, and they replace the box's contents. The panel has an **Again** button for the same. With text selected instead of a box focused, a floating panel shows the
draft with Copy / Insert (the menu-bar item can turn auto-insert off so the panel always shows). The model is the FluidUse short-reply Qwen3-0.6B fine-tune as one 724 MB Core ML package: the
prompt runs on the Neural Engine, the reply tokens on the GPU. The panel shows the timing split; **Copy** puts the
draft on the clipboard, **Insert** pastes it back into the app the post came from.

```bash
chmod +x Sources/ShortReplyDemo/demo.sh
Sources/ShortReplyDemo/demo.sh --x              # app + terminal (macmon on top, model log below) + x.com in Chrome
Sources/ShortReplyDemo/demo.sh --mock           # same, with the local mock feed instead of x.com
```

The model directory holds `short_reply_0_6b.mlpackage` (functions `prefill` and `decode`), `tokenizer.json` and
`config.json`. Get it from Hugging Face, then point the launcher at it:

```bash
hf download FluidInference/short-reply-0.6b-coreml --local-dir ~/Models/short-reply-0.6b-coreml
Sources/ShortReplyDemo/demo.sh --x ~/Models/short-reply-0.6b-coreml      # or set SHORT_REPLY_MODEL_DIR
```

The first launch compiles the package once (kept as `.mlmodelc` beside it) and runs one warm-up prompt; the log
prints `model ready`.

**Keys.** The bare **9** and **0** keys are observed system-wide while the app runs (a global monitor can't swallow
them, so a 9 typed into a focused field is deleted again by the app). That is convenient for a demo and wrong for
daily use: turn off "Bare 9 / 0 keys (demo mode)" in the menu-bar item and use ⌃⌥R instead.

**Mock feed for recording.** `mock-feed/index.html` is a local feed of fictional posts with a reply box under each;
nothing on it is posted anywhere. Select a post, press 9, then **Insert**: the page routes the paste into that post's reply box, and "Reply" only
appends the text on the page. On the real site, click into the post's reply box before pressing Insert.

Browsers: Safari exposes the selection through Accessibility directly; for Chrome the app switches on Chrome's
accessibility tree (`AXEnhancedUserInterface`) and otherwise falls back to a ⌘C round trip that restores your clipboard.

The terminal uses macmon (`brew install macmon`, no sudo) for the GPU / ANE / power rows; asitop is the fallback but
its 0.0.24 release crashes on M5 chips. Permissions: the app reads the selection through Accessibility and listens for the global hotkey, so macOS asks for
**Accessibility** access on first use (System Settings › Privacy & Security). Without it, use the menu-bar item
"Draft reply from clipboard" after copying a post.

What to expect: replies are short drafts for a person to review, not auto-posted. On the project's 100-post benchmark
about 1 in 8 drafts is generic or slightly off; the demo does not hide those.

Measured on an M5 Pro (macOS 27): 89 of 100 benchmark replies identical to the PyTorch checkpoint; prefill 28 ms on the
Neural Engine, decode 24 ms per token on the GPU, about 190 ms per reply.

Parity and latency of the Swift host against the Python export check:

```bash
swift run -c release ShortReplyCheck <model dir> <responses.jsonl> tuned 100
```
