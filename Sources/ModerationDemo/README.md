# ModerationDemo — comment moderation firehose (d1-omni-600M on Core ML)

5,000 real comments from the Civil Comments test set (CC0) stream through Liquid AI's **d1-omni-600M** on this Mac
(`D1OmniManager`, Core ML), nothing sent anywhere. Each comment gets one yes/no question, "Is this comment toxic?", with
the Civil Comments annotators' definition of toxic; it is flagged at P(toxic) ≥ 0.8. Passed comments scroll in the live
feed; flagged ones fly into the toxic box. Every row shows the model's confidence, the share of the ~10 human raters who
called the comment toxic, and the chip that answered it.

The **Neural Engine + GPU** mode runs both chips at once from one queue: the Neural Engine takes one comment per call
(short comments first), the GPU eight per call (long ones first). On an M5 Pro: ~240–300 comments/s together, ~100 on the
Neural Engine alone, ~180–215 on the GPU alone. The side panel shows comments/s (the clock includes tokenization),
flagged count, agreement with the human labels and the per-chip split.

## Run

```bash
swift run -c release ModerationDemo [--model <dir>]     # or D1_MODERATION_DIR=<dir>; default: the pinned FluidInference/d1-omni-600m-coreml snapshot (D1OmniModelStore)
swift run -c release ModerationCheck [ane|gpu|both]      # headless: throughput, parity, agreement with humans
Tools/moderation/terminals.sh                            # Ghostty/tmux pane: macmon on top, the per-comment log below
```

`MODERATION_MODE=ane|gpu|both` preselects the chips; `MODERATION_AUTOSTART=0` loads but waits for Start (space bar).
Every comment is also appended, ANSI-colored, to `MODERATION_DEMO_LOG` (default `/tmp/moderation-demo.log`).

## Data and accuracy

The bundled sample (`Sources/Moderation/Resources/civil-comments-5000.json`) keeps comments with clear labels (rater
toxicity 0 or ≥ 0.5, 78% of the test set) whose prompt fits 256 tokens, in random order, with the PyTorch model's
P(toxic) for each. Against the human labels: 95.4% accuracy, toxic recall 67.8%, precision 82.4%. Core ML makes the
same flag as PyTorch on all 5,000 (`ModerationCheck`). On unfiltered test comments accuracy is lower (~93% at the same
threshold): most extra flags are borderline comments some raters did call toxic.
