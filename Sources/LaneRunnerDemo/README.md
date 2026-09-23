# Lane runner · local model comparison

```bash
swift run -c release LaneRunnerDemo
swift run -c release LaneRunnerCheck gliner2Base --seeds=1,2,3,4
```

To start a demo with no clicks, set the model and seed, for example GLiNER 2.5 base's 43-row run:

```bash
LANE_RUNNER_MODEL=gliner2Base LANE_RUNNER_SEED=2 swift run -c release LaneRunnerDemo
```

`LANE_RUNNER_ROW_MS` sets the row interval. The first load checks and compiles the model, which takes about 20 s.

This is a Subway Surfers-style endless runner with three lanes. Each step enters the next row, and the runner picks one of five moves: **left**, **right**, **jump** (clears a low bar), **slide** (passes under a high bar), or **stay**. Trains block a lane, and coins are a bonus.

Like Subway Surfers and Temple Run, it gets harder the longer you survive:

- **Denser track.** The first 30 rows are easy: obstacle rows alternate with open rows, and each obstacle row leaves one lane open. Over the next 240 rows, the open rows disappear and more lanes are blocked. Two obstacle rows appear back to back only when every lane passable in the first has a legal move into the second, so every track can still be survived.
- **Faster app runs.** Each row is 0.5% faster than the one before, down to 40% of the starting interval. At the default 400 ms, the interval reaches 160 ms after about 180 rows. The headless checker has no clock, so only density applies there.

The runner chooses once per row, not once per frame, and each option states its consequence. The first Flappy Bird trial showed that frame-by-frame control does not suit these models, so this setup follows the Tetris harness pattern. The model reads a short state (`Lane 2 of 3. Next row: train, open, low bar. Then: …`) plus one label per legal move, such as `jump: crash, hit a train`, `left: safe, coin`, or `stay: safe, then trapped`. The labels and the heuristic both look only at the three visible rows. Option order rotates every row, so a fixed position preference does not look like skill.

The app renders a 3D third-person track in SceneKit with CC0 models from Kenney's [Train Kit](https://kenney.nl/assets/train-kit) and [Platformer Kit](https://kenney.nl/assets/platformer-kit). The models and their license files are in `Resources/Kenney/`. It also shows the exact text the model reads, with unsafe options in red. Rows advance on a timer (150–800 ms). If a reply misses its row, the runner stays in its lane and the row counts as late. `LaneRunnerCheck` runs the same tracks headless with no clock, runs each seed up to 200 rows by default (`--max-rows=` changes it), and prints per-seed JSON. Pass a model name, `heuristic`, or `random`.

## Results

Seeds 1–4, up to 300 rows each (1,200 maximum), Apple M5 Pro:

| Policy | Rows (per seed) | Total | Median call |
| --- | --- | ---: | ---: |
| Heuristic | 300, 300, 300, 300 | 1,200 | — |
| GLiNER 2.5 base W8 | 30, 43, 36, 20 | 129 | 7.5–10.1 ms |
| GLiNER 2.5 multilingual W8 | 20, 12, 20, 10 | 62 | 7.2–11.2 ms |
| GLiNER 2.5 small W8 | 10, 18, 10, 10 | 48 | 6.8–11.4 ms |
| Laya multilingual E8 | 6, 2, 4, 16 | 28 | 5.0–5.2 ms |
| GLiClass Edge Apps v2 LUT8 | 4, 2, 2, 12 | 20 | 2.0–2.2 ms |
| LFM2.5-350M-RLCD FP16 (bridge) | 4, 2, 2, 12 | 20 | 73–138 ms |
| Random legal move | 2, 2, 8, 2 | 14 | — |
| Kev 0.6B FP16 (bridge) | 4, 2, 4, 2 | 12 | 12.4–16.3 ms |
| Verdict FP16 | 2, 2, 2, 4 | 10 | 10.1–10.8 ms |
| Jeff FP16 (bridge) | 2, 2, 2, 4 | 10 | 9.8–10.3 ms |
| Kev 0.5B FP16 (bridge) | 2, 2, 4, 2 | 10 | 10.0–10.4 ms |

Decision 1.0 Kai and Lex publish Choice packages with three candidate slots, and NanoJev's head takes four options, so they cannot choose among the runner's four or five moves.

Every model crashed on every seed, each time by choosing an option labeled `crash` although a safe option was offered. Only GLiNER 2.5 base regularly got past the 30-row easy opening; its best run reached row 43. The three GLiNER 2.5 models are the only ones clearly above random play. Kev 0.6B chose the first-listed option in 11 of 16 decisions. Before the difficulty ramp, the tracks differed and the ranking was similar: GLiNER base, then GLiNER multilingual, led the models. Four seeds make a small check, not a model ranking. The raw JSON is in FluidAudio's ignored `.mobius/lane-runner/ramp/` directory.
