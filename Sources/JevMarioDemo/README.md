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
auxiliary targets, ~9M parameters, 17 MB Core ML, ~0.4 ms per decision on the Neural Engine. It imitates a search
expert and picks run / jump / hop / wait from the screen around Mario (now and 4 frames earlier). Over 32 levels × 3
start offsets it reaches the flag in 12/96 runs (8 levels at least once; 1-1 and 6-1 every time).

```bash
JEV_MARIO_BRAIN=student MARIO_STUDENT_MODEL=/path/to/MarioStudent.mlpackage swift run -c release JevMarioDemo
MARIO_LEVEL=6-1 JEV_MARIO_BRAIN=student MARIO_STUDENT_MODEL=... swift run -c release JevMarioDemo   # another level
```

In the app it reaches the 1-1 flag in 32 decisions at ~0.65 ms each.

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
