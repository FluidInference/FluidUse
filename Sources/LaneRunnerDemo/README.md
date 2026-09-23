# Lane runner · local model comparison

```bash
swift run -c release LaneRunnerDemo
swift run -c release LaneRunnerCheck gliner2Base --seeds=1,2,3,4
```

This is a Subway Surfers-style endless runner with three lanes. Each step enters the next row, and the runner picks one of five moves: **left**, **right**, **jump** (clears a low bar), **slide** (passes under a high bar), or **stay**. Trains block a lane, and coins are a bonus. Obstacle rows alternate with open rows, and each obstacle row leaves one lane open, so every track can be survived.

The runner chooses once per row, not once per frame, and each option states its consequence. The first Flappy Bird trial showed that frame-by-frame control does not suit these models, so this setup follows the Tetris harness pattern. The model reads a short state (`Lane 2 of 3. Next row: train, open, low bar. Then: …`) plus one label per legal move, such as `jump: crash, hit a train`, `left: safe, coin`, or `stay: safe, then trapped`. The labels and the heuristic both look only at the three visible rows. Option order rotates every row, so a fixed position preference does not look like skill.

The app draws the track and shows the exact text the model reads, with unsafe options in red. Rows advance on a timer (150–800 ms). If a reply misses its row, the runner stays in its lane and the row counts as late. `LaneRunnerCheck` runs the same tracks headless with no clock, runs each seed up to 200 rows, and prints per-seed JSON. Pass a model name, `heuristic`, or `random`.

## Results

Seeds 1–4, up to 200 rows each (800 maximum), Apple M5 Pro:

| Policy | Rows (per seed) | Total | Median call |
| --- | --- | ---: | ---: |
| Heuristic | 200, 200, 200, 200 | 800 | — |
| GLiNER 2.5 base W8 | 12, 68, 96, 8 | 184 | 10.6–11.3 ms |
| GLiNER 2.5 multilingual W8 | 16, 122, 12, 10 | 160 | 7.7–9.4 ms |
| Kev 0.6B FP16 (bridge) | 84, 18, 28, 2 | 132 | 19.2–22.8 ms |
| GLiClass Edge Apps v2 LUT8 | 28, 2, 38, 2 | 70 | 2.0–2.2 ms |
| GLiNER 2.5 small W8 | 12, 2, 32, 10 | 56 | 8.5–11.4 ms |
| Jeff FP16 (bridge) | 10, 4, 10, 6 | 30 | 15.0–20.7 ms |
| Random legal move | 12, 2, 8, 8 | 30 | — |
| Laya multilingual E8 | 10, 2, 4, 6 | 22 | 5.1 ms |
| Verdict FP16 | 2, 2, 6, 6 | 16 | 10.3–10.7 ms |
| LFM2.5-350M-RLCD FP16 (bridge) | 2, 6, 2, 2 | 12 | 56–158 ms |
| Kev 0.5B FP16 (bridge) | 2, 2, 4, 2 | 10 | 10.1–10.9 ms |

Decision 1.0 Kai and Lex publish Choice packages with three candidate slots, and NanoJev's head takes four options, so they cannot choose among the runner's four or five moves.

Every model crashed on every seed, and each run ended on its first unsafe choice: an option labeled `crash` although a safe option was offered. The best models, GLiNER base and multilingual and Kev 0.6B, went about 4–6 times as far as random play. Flappy Bird's best model passed one pipe in total. Most models were at or below random. Kev 0.6B chose the first-listed option in 90 of 136 decisions, so part of its distance comes from position preference. GLiClass chose `stay` in 53 of 74 decisions. Four seeds make a small check, not a model ranking. The raw JSON is in FluidAudio's ignored `.mobius/lane-runner/` directory.
