# On-device models play Tetris (macOS)

SwiftUI app for `GLiClassManager` and `LayaManager`. GLiClass compares the two strongest legal
landings as natural-language labels in one encoder pass; laya scores every offered landing with
*"Is this a clean placement?"*. The scoreboard shows elapsed time, pieces, lines, latency, and calls.

```bash
swift run -c release LayaTetrisDemo
```

No Xcode project is needed. Set `GLICLASS_MODEL_DIR` to a directory holding `tokenizer.json` and
`gliclass_edge_apps_fp16_L128_options25.mlpackage`, or select laya and let it download from
`FluidInference/laya-coreml` (`LAYA_MODEL_DIR` skips that download). **Play** scores on the Neural Engine
with no artificial delay by default. The *delay per scored landing* slider slows the scoring down
so each candidate can be watched being evaluated on the board (orange outline),
and the chosen landing is drawn in green.

**Policy** switches among GLiClass, laya, a feature-weighted heuristic, and random play. The harness
toggle applies to every policy. See [Benchmarks.md](../../Benchmarks.md) for the reported runs.

Headless smoke test (used by CI-less verification):

```bash
GLICLASS_MODEL_DIR=/path/to/model LAYA_DEMO_AUTORUN=1 LAYA_DEMO_QUIT_AFTER=20 swift run -c release LayaTetrisDemo
LAYA_DEMO_AUTORUN=1 LAYA_DEMO_STRESS=1 LAYA_DEMO_QUIT_AFTER=30 swift run -c release LayaTetrisDemo  # pause/reset/play every second
```

`LAYA_DEMO_AUTOLOAD=1` loads the model on launch without playing, `LAYA_DEMO_SEED=<n>` fixes the
piece sequence.

The simulation (`Sources/LayaTetris`) is shared with `swift run FluidUseLaya tetris`.

## Controls

- **Harness** (on): withholds landings that bury a cell when a clean one exists, and uses wording
  that stays discriminative on a tall board. Together they raise the reported mean from 76 to 568
  pieces. Wording alone regresses to 48; filtering alone improves modestly to 87.
- **Marathon** (off): starts a new board after each top-out. The scoreboard carries pieces, lines,
  calls and elapsed time across games; the header shows the game number. Useful
  for a long recording; it does not make the model survive longer.
- **Lookahead** (off): also scores the board each of the top N landings leaves for the next piece.
  Worse on average, ~4x the calls per piece, so a single game runs much longer in wall time.
- **Policy**: `gliclass` uses one call over a heuristic top-two shortlist; `laya` scores each offered
  landing separately; `heuristic` and `random` are controls that never call a model.
- **Seed**: read when a fresh game starts, so set it before pressing Play.

The current ten-seed GLiClass run averaged 3,482.3 pieces with four seeds still alive at the
5,000-piece cap, compared with 2,874.4 for the corrected heuristic control. GLiClass agreed with the
heuristic leader 98.2% of the time; its remaining choices raised mean survival by 21.1%. See the
benchmark document for the per-seed values and the comparison limits around the older laya results.
