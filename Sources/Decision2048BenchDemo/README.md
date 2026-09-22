# 2048 Bench for decision models

Side-by-side macOS benchmark for GLiClass Edge Apps v2 LUT8 and Laya Multilingual E8. Both models play
independent 2048 boards initialized with the same seed. The engine supplies the same top-two one-move
expectimax shortlist to each model. GLiClass compares the candidates in one classifier pass; Laya applies
its established `noul` question to each candidate, requiring two passes per move.

```bash
swift run -c release Decision2048BenchDemo
```

Both models download their published Core ML assets on first use. To use existing assets, set
`GLICLASS_MODEL_DIR` and/or `LAYA_MODEL_DIR` to their respective model directories.

Set `GAME2048_AUTOLOAD=1 GAME2048_AUTORUN=1` to launch immediately, `GAME2048_SEED=<n>` to select the
deterministic starting seed, and `GAME2048_DELAY_MS=<n>` to control visual pacing. Model inference runs
alternately so the models do not contend for the Neural Engine; the visual delay is excluded from latency.
