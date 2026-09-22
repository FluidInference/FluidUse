# laya plays Tetris (macOS)

SwiftUI app that exercises `LayaManager` from FluidUse: every legal landing of the
current piece is described in one sentence, laya answers *"Is this a clean placement?"* on the
Neural Engine, and the landing with the highest P(true) is played. The console lists each scored
sentence with its probability and per-call latency; the tiles show milliseconds per decision,
decisions per minute, lines, and the bucket in use.

```bash
swift run -c release LayaTetrisDemo
```

No Xcode project is needed. **Load model** downloads the 128-token bucket and tokenizer from
`FluidInference/laya-coreml` (614 MB + 34 MB) on first use; set `LAYA_MODEL_DIR` to a directory
holding the bundles to skip the download. **Play** scores every landing on the Neural Engine (about 5 ms per decision inside the app on an M5 Pro,
3.8 ms from `FluidUseLaya tetris` without UI updates) and pauses briefly after each placed piece; the *delay per scored landing* slider slows the
scoring down so each candidate can be watched being evaluated on the board (orange outline),
and the chosen landing is drawn in green.

**Policy** switches to a feature-weighted heuristic or random play for comparison. Zero-shot laya
clears a few dozen lines before topping out; the heuristic plays indefinitely. The demo is about
decision latency on device, not Tetris skill.

Headless smoke test (used by CI-less verification):

```bash
LAYA_DEMO_AUTORUN=1 LAYA_DEMO_QUIT_AFTER=20 swift run -c release LayaTetrisDemo
LAYA_DEMO_AUTORUN=1 LAYA_DEMO_STRESS=1 LAYA_DEMO_QUIT_AFTER=30 swift run -c release LayaTetrisDemo  # pause/reset/play every second
```

`LAYA_DEMO_AUTOLOAD=1` loads the model on launch without playing, `LAYA_DEMO_SEED=<n>` fixes the
piece sequence.

The simulation (`Sources/LayaTetris`) is shared with `swift run FluidUseLaya tetris`.

## Controls

- **Harness** (on): withholds landings that bury a cell when a clean one exists, and uses wording
  that stays discriminative on a tall board. Worth 76 -> 568 pieces, but only as a pair; each half
  alone is a regression.
- **Marathon** (off): starts a new board after each top-out and carries the totals forward. Useful
  for a long recording; it does not make the model survive longer.
- **Lookahead** (off): also scores the board each of the top N landings leaves for the next piece.
  Worse on average, ~4x the calls per piece, so a single game runs much longer in wall time.
- **Policy**: `heuristic` and `random` are controls that never call the model.
- **Seed**: read when a fresh game starts, so set it before pressing Play.
