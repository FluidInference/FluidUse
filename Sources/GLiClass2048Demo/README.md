# GLiClass plays 2048

SwiftUI demo and deterministic 4×4 engine for GLiClass Edge Apps v2. A conventional 2048 safety
heuristic shortlists the strongest legal swipes; GLiClass compares their resulting-board descriptions
in one L128 encoder pass.

```bash
swift run -c release GLiClass2048Demo
```

The default L128 LUT8 package and tokenizer download from
[FluidInference/gliclass-edge-apps-coreml](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml)
on first use and are cached locally. Set `GLICLASS_MODEL_DIR=/path/to/assets` to use local assets,
or `GLICLASS_PRECISION=fp16` for the published FP16 bucket.

Set `GAME2048_AUTORUN=1` to load and play immediately, `GAME2048_SEED=<n>` to choose the deterministic
tile sequence, `GAME2048_CANDIDATES=2|3|4` to choose the comparison width, and
`GAME2048_MARGIN=0...1` to require a confidence margin before GLiClass overrides the heuristic leader.

The measured default is LUT8, top two, one-move expectimax, margin 0.50, seed 46. It reached tile 4096
with a score of 70,864 in 3,230 moves, agreeing with the expectimax leader on 97.7% of comparisons.
The SwiftUI run completes in 33.6 seconds with no artificial move delay and 2.10 ms of model work per move.
Across seeds 1–10 this policy averaged 18,970 points and 1,012 moves versus 25,382 points and 1,318
moves for expectimax alone. The selected visual-demo seed is disclosed rather than presented as a
representative mean.
