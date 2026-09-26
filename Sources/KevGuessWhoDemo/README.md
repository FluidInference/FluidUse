# Guess Who — Kev-0.8B on Core ML

A self-playing Guess Who over 80 real Wikipedia people (DBpedia-14 test split: artists, athletes, politicians).

1. **Scan.** Kev reads each card's abstract once and answers all 12 yes/no questions in a single fused Core ML
   call (state + packed questions), so the whole wall — 960 decisions — fills in a few seconds.
2. **Play.** A hidden person is picked; each turn asks the unused question that splits the cards still up closest to
   half, and every card whose answer differs from the hidden person's flips down, until one is left.

The header shows the last and median call time, decisions per second, and the running decision count. Everything
runs on the GPU through Core ML; there is no input.

## Run

```bash
Sources/KevGuessWhoDemo/demo.sh [model dir]            # app + Terminal: asitop on top, live model log below
KEV_MODEL_DIR=<model dir> swift run -c release KevGuessWhoDemo   # app only
```

The app opens idle; Play (Space) starts it, Pause / Play toggles, Reset (⌘R) deals a fresh wall. The launcher's asitop pane asks for the sudo
password (powermetrics); the tmux session is reused across launches.

`<model dir>` holds `tokenizer.json`, one row bucket (`L512_K16/` with `embeddings.f16`), and `fused/` with the
multifunction `KevFused.mlmodelc` + `config.json` built by mobius `models/computer-use/kev-0.8b/coreml`
(`convert-stages.py --stage fused`, `combine-stages.py`). The demo uses `fused_S{32,64,128,192,256,384}_P192_B16_K16`
only: every function in a package adds to load time, so a package with just these starts in ~30–45 s (the first launch
of a new package also compiles the GPU programs, which takes a few minutes).

## Numbers (M5 Pro)

~38 ms median per card (12 answers per call); a wall of 960 decisions in 3.0–3.5 s (270–320 decisions/s wall clock,
including host-side tokenization and input building). A Core ML function left idle pays a 0.3–0.8 s re-setup on its
next call, so every game re-warms its functions while dealing, outside the timed scan.

Data: DBpedia-14 (Zhang et al., 2015), Wikipedia text via DBpedia, CC BY-SA 3.0, fetched from the Hugging Face dataset
viewer and cached; nothing is bundled. Model: Kev-0.8B (jaredpalmer/kev, Apache-2.0). The yes/no questions are the
demo's own and have no gold labels; answers are Kev's.
