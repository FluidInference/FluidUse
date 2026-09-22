# GLiClass plays 2048

SwiftUI demo and deterministic 4×4 engine for GLiClass Edge Apps v2. A conventional 2048 safety
heuristic shortlists the strongest legal swipes; GLiClass compares their resulting-board descriptions
in one L128 encoder pass.

```bash
GLICLASS_MODEL_DIR=/path/to/assets GLICLASS_PRECISION=lut8 swift run -c release GLiClass2048Demo
```

Set `GAME2048_AUTORUN=1` to load and play immediately, `GAME2048_SEED=<n>` to choose the deterministic
tile sequence, `GAME2048_CANDIDATES=2|3|4` to choose the comparison width, and
`GAME2048_MARGIN=0...1` to require a confidence margin before GLiClass overrides the heuristic leader.

The measured default is LUT8, top two, margin 0.40, seed 9. It reached tile 2048 with a score of
33,812 in 1,702 moves. Across seeds 1–10 it averaged 13,419 points versus 14,104 for the heuristic and
1,187 for random play; this is a demo of selective model intervention, not a claim that the model
beats the heuristic on average.
