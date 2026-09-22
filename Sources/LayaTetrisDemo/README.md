# On-device models play Tetris (macOS)

SwiftUI app for `GLiClassManager` and `LayaManager`. GLiClass compares the two strongest legal
landings as natural-language labels in one encoder pass; laya scores every offered landing with
*"Is this a clean placement?"*. The scoreboard shows elapsed time, pieces, lines, cumulative model
milliseconds per move, and calls.

```bash
swift run -c release LayaTetrisDemo
```

No Xcode project is needed. GLiClass downloads its L128 FP16 package and tokenizer from
`FluidInference/gliclass-edge-apps-coreml` on first use. Set `GLICLASS_MODEL_DIR` to a local asset
directory to skip that download. Laya downloads from `FluidInference/laya-coreml` unless
`LAYA_MODEL_DIR` selects a local directory. **Play** scores flat out with
Core ML on CPU + ANE and continues on the next seed after a top-out. SwiftUI presents the evolving board at
the display refresh rate while the model stays hot. The *pause per piece* slider can expose individual
moves, but long pauses let the Neural Engine idle and raise measured latency. The *delay per scored
landing* slider slows laya's candidate-by-candidate scoring so it can be watched on the board (orange
outline); the chosen landing is drawn in green.

The GLiClass training, Core ML conversion, compression, and parity reports are in
[mobius PR #101](https://github.com/FluidInference/mobius/pull/101). The published GLiClass packages
and runtime config are at [Hugging Face](https://huggingface.co/FluidInference/gliclass-edge-apps-coreml).

**Policy** switches among GLiClass, laya, a feature-weighted heuristic, and random play. The harness
toggle applies to every policy. See [Benchmarks.md](../../Benchmarks.md) for the reported runs.

Headless smoke test (used by CI-less verification):

```bash
GLICLASS_MODEL_DIR=/path/to/model LAYA_DEMO_AUTORUN=1 LAYA_DEMO_QUIT_AFTER=20 swift run -c release LayaTetrisDemo
LAYA_DEMO_AUTORUN=1 LAYA_DEMO_STRESS=1 LAYA_DEMO_QUIT_AFTER=30 swift run -c release LayaTetrisDemo  # pause/reset/play every second
```

`LAYA_DEMO_AUTOLOAD=1` loads the model on launch without playing, `LAYA_DEMO_SEED=<n>` fixes the
piece sequence.

Set `GLICLASS_PRECISION=lut8` or `lut6` to load the matching palettized package from the same model
directory. FP16 remains the default. LUT8 is the recommended compact package: 33.0 MB, 97.1% FP16
choice agreement on the L128 application-suite rows, and 1.81 ms median complete-call latency in the
optimized 1,000-piece seed-24 run. FP16 measured 1.61 ms in the same run. LUT6 is an aggressive
24.8 MB option; LUT4 is intentionally not documented as deployable because it lost 15.75 accuracy
points.

`GLICLASS_PRECISION=fp16-mask` selects the experimental float-mask FP16 package. It moves one more
operation from CPU to ANE with exact application-suite parity, but improved a paired three-seed
Tetris run by only 0.62%; FP16 remains the production default pending broader device testing.

The simulation (`Sources/LayaTetris`) is shared with `swift run FluidUseLaya tetris`.

## Controls

- **Harness** (on): withholds landings that bury a cell when a clean one exists, and uses wording
  that stays discriminative on a tall board. Together they raise the reported mean from 76 to 568
  pieces. Wording alone regresses to 48; filtering alone improves modestly to 87.
- **Marathon** (on): starts a new board after each top-out. The scoreboard carries pieces, lines,
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
