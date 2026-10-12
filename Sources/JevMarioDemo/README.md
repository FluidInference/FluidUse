# JevMarioDemo

[JEV-9B](https://huggingface.co/autotrust/JEV-9B) with vision plays Super Mario Bros. 1-1 on Core ML
([FluidInference/jev-9b-coreml](https://huggingface.co/FluidInference/jev-9b-coreml), `JevManager`).

While Mario is on the ground, the emulator pauses and the model answers three yes/no questions, each about a close-up
just ahead of him:

| close-up | question |
|---|---|
| 4 blocks ahead, 3 above his feet to 2 below | Is there an enemy (a brown Goomba or a Koopa) in this picture? |
| the same, only above his feet | Is there a solid obstacle (a pipe, a block or a step) in this picture? |
| 6 blocks of floor | Is the brick floor in this picture interrupted by a gap? |

If any probability is above 0.5, Mario takes a long running jump; otherwise he runs right for 6 frames. Then the
move plays at 60 fps.

**The harness, not the model, provides:**

- Mario's on-screen position (read from emulator RAM), which places the close-ups.
- Whether he is on the ground (RAM).
- The fixed button sequences for each move.

The model only answers the three questions. It reaches the flag in 70 decisions (deterministic), at about 1.2 s
per decision on an M5 Pro.

## Student brain

The picker's second option is a small pixel model from model-lab `mario-tiny`: a MobileNetV3 with RAM tile / enemy
auxiliary targets, ~9M parameters, 17 MB Core ML, ~0.5 ms per decision on the Neural Engine. The current model (r10)
is trained on a beam-search solver's routes for all 32 levels plus DAgger, and picks one of 11 eight-frame moves from
the screen around Mario (now and 4 frames earlier).

Lookahead (on by default, toggle in the app, `MARIO_LOOKAHEAD=0` to start off): before each move the app saves the
emulator state and simulates candidates in the student's order (the move, then the student's next 6 moves, each a
Core ML decision); it plays the first that survives and gains ground, else the one that lasts longest. A stalled
continuation also tries the move repeated (a swim stroke only rises if repeated). Over 32 levels × 3 start offsets:

| | Flags /96 | Levels cleared at least once | Every start |
|---|---|---|---|
| Student alone | 53 | 23 | 12 |
| Student + lookahead | 78 | 30 | 21 |

The maze castles 4-4 and 8-4 stay unsolved. With lookahead it is the student plus emulator simulation, not the network
alone. Speed (release, uncapped): 1-1 5.5x real time (532 simulated moves), 7-1 2.0x, 5-2 1.9x (4,269).

```bash
JEV_MARIO_BRAIN=student MARIO_STUDENT_MODEL=/path/to/MarioStudentR10.mlpackage swift run -c release JevMarioDemo
MARIO_LEVEL=5-2 JEV_MARIO_BRAIN=student MARIO_STUDENT_MODEL=... swift run -c release JevMarioDemo   # another level
```

## Run

```bash
swift run -c release JevMarioDemo                               # downloads the model (~10 GB) on first run
JEV_MODEL_DIR=/path/to/jev-9b-coreml swift run -c release JevMarioDemo
```

The emulator is `nes_bridge.py`, run through `uv` with `gym-super-mario-bros` 7.4.0, `nes-py`, `gym` 0.26.2 and
`numpy<2`. **`gym-super-mario-bros` bundles the Super Mario Bros. ROM.** Nothing in this repository contains it, and
installing that package is your choice; keep recordings local. To use your own Python environment, set
`JEV_MARIO_BRIDGE="/path/to/python /path/to/nes_bridge.py"`.

Close other GPU-heavy apps: the decoder needs ~7 GB of memory to itself.
