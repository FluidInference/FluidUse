# GLiClass plays ViZDoom `defend_the_center`

Headless check in the Jev style: the model never sees pixels. Code reads ViZDoom's labels buffer
and writes the state as text (health, ammo, each visible monster's bearing, distance, and whether it
is on the crosshair). GLiClass picks attack / turn left / turn right through `GLiClassServe`, and the
engine holds the action for 3 tics (about 12 decisions per second at 35 Hz).

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv vizdoom numpy
swift build -c release --product GLiClassServe
.venv/bin/python Tools/doom/defend_the_center.py --seeds 20
```

Seeds 1–20, M5 Pro, GLiClass Edge Apps v2 L128:

| Policy | Kills | Survived | Agrees with aimer | ms per call |
| --- | ---: | ---: | ---: | ---: |
| Hand-coded aimer | 14.80 | 26.9 s | — | — |
| Random | 1.00 | 9.0 s | 32% | — |
| GLiClass LUT8, bare labels | 0.00 | 8.3 s | 71% | 3.6 |
| GLiClass LUT8, consequence labels | 14.00 | 25.3 s | 91% | 4.0 (p95 ≤ 5.7) |
| GLiClass fp16, consequence labels | 13.75 | 24.6 s | 88% | 3.3 |

With bare labels GLiClass turns the right way but never fires. With labels that state each action's
consequence ("attack: shoots the monster on the crosshair", "turn left: moves the crosshair off the
monster"), as the Tetris and 2048 shortlists do, it comes within one kill of the aimer (6 wins, 5 ties,
9 losses per seed). The consequence labels carry most of the decision, so this shows the loop is fast
enough for a 35 Hz shooter, not that GLiClass plays Doom better than 20 lines of code.
