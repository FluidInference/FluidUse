# SauerkrautLM-Doom-MultiVec on Core ML

[SauerkrautLM-Doom-MultiVec-1.3M](https://huggingface.co/VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M)
(Apache 2.0, VAGO solutions) converted to Core ML and playing ViZDoom `defend_the_center`.

```bash
Tools/doom/sauerkraut/demo.sh                                  # game window + asitop + decision log
Tools/doom/sauerkraut/demo.sh --record doom.mp4 --episodes 1   # also writes an mp4 of the window
.venv/bin/python Tools/doom/sauerkraut/play.py --check 100     # headless score check, seeds 10000-10099
```

`demo.sh` also opens one Terminal window split into two rows with tmux: `sudo asitop` on top (type
your password there) and `tail -f /tmp/doom-demo.log` below, one colored line per decision with the
action, the four probabilities, and the Core ML call time. Without tmux it opens two windows. Keys in the game window: space pause, n next episode, q quit.

## What the model reads

1026 tokens per frame: a character per cell of a 40×25 grid (plus newlines, `[CLS]`, `[SEP]`) and a
learned embedding of each cell's depth, 16 bins. Upstream's ASCII path overflows `uint8` and writes
`@` for every cell, in training and at inference, so the depth bins carry all of the information. The
demo panel draws those bins. It is not reading pixels or game state.

## Results, M5 Pro, seeds 10000–10099, 4 tics per decision, 2100-tic episodes

| Model | Mean kills | Mean survival | Full 60 s | ms per decision |
| --- | ---: | ---: | ---: | ---: |
| PyTorch (upstream) | 20.42 (sd 5.32) | 50.5 s | 31 / 100 | 57.7 (1 thread), 26.6 (8), 19.0 (MPS) |
| Core ML fp32, GPU | 20.42 (sd 5.32), identical on 100/100 seeds | 50.5 s | 31 / 100 | 3.3–4.9 |
| Core ML fp16, 1026 tokens, GPU (demo default) | 20.54 (sd 5.27) | 50.6 s | 35 / 100 | 1.5–3.4 |

For reference on the same seeds: our hand-coded aimer 13.05 kills and GLiClass with consequence
labels 11.98 (both in `Tools/doom/defend_the_center.py`, three buttons, 320×240), random 1.26. An
independent 1000-episode evaluation of the PyTorch model
([tiny-doom-defender](https://huggingface.co/spaces/anakin87/tiny-doom-defender)) reports 20.38.

The Neural Engine is slower than the GPU for this model (7.8–8.4 ms); `--units CPU_AND_NE` to compare.

The window renders every game tic, and ViZDoom's rendering consumes game randomness, so a seed does
not replay the headless path exactly. The play is equally strong: seeds 10000–10029 average 19.70
kills / 50.5 s rendered against 20.37 / 50.7 s headless. Rendered seeds 10016 and 10005 reach 25 kills
and survive the full 60 s.

## Conversion

`convert.py` re-implements the ModernBERT forward with static masks (Hugging Face's mask construction
does not trace through coremltools) and exports fp32/fp16 with an attention mask at 1100 tokens.
`convert_ane.py` exports the demo model: fixed 1026 tokens (no frame has padding) and one-hot matmul
embeddings. Both check parity against upstream PyTorch on real frames and need the upstream package:

```bash
uv venv -p 3.12 convenv && uv pip install -p convenv vizdoom==1.3.0 torch==2.7.* transformers==4.56.2 \
    coremltools git+https://github.com/VAGOsolutions/SauerkrautLM-Doom-MultiVec
cd Tools/doom/sauerkraut && ../../../convenv/bin/python convert.py && ../../../convenv/bin/python convert_ane.py
mv *.mlpackage models/
```

`sauer_eval.py` is the PyTorch/Core ML episode evaluator used for the table.
