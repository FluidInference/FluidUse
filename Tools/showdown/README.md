# Showdown harness (optional tooling)

Python tooling that plays Pokémon Showdown with a typed-decision model and produced
[FluidInference/intern-decision-0.8b-showdown-coreml](https://huggingface.co/FluidInference/intern-decision-0.8b-showdown-coreml).
The Swift package does not depend on anything here; `InternDecisionManager` and `InternDecisionModelStore.ensure(.showdown)`
work on their own. This folder is for reproducing the numbers, training a new student, or running the live demo.

## What it is

Every turn, the harness reads the battle from [poke-env](https://github.com/hsahovic/poke-env), renders a compact
JSON state (both teams, HP, status, boosts, field) and one `choice` question whose options are the legal moves and
switches with their facts (type, category, power, STAB, effectiveness, accuracy, PP, switch matchups), asks the model
for a probability per option, plays the top one, and logs the request and probabilities. That is Intern-Decision's own
wire format, so any Intern-Decision checkpoint (or the Core ML export) plays unchanged.

| File | Role |
| --- | --- |
| `decision_player.py` | poke-env player; deciders for a Core ML directory, a PyTorch checkpoint, and a text-only heuristic |
| `run_battles.py` | a decider vs poke-env's random / max-base-power / simple-heuristics players, with latency |
| `collect_teacher.py`, `collect_student.py` | log a teacher's (or the student's) decisions over many battles |
| `label_with_teacher.py` | relabel logged states with a teacher (on-policy distillation data) |
| `train_student.py` | LoRA distillation: KL(teacher ‖ student) on the restricted softmax at the `<decision>` marker |
| `export_student.sh` | merged student → Core ML buckets (via `coreml/`) → battles |
| `demo_battle.py`, `demo.sh`, `go.sh` | one battle in the browser with a live decision log (Ghostty + macmon) |
| `measure_footprint.py` | footprint / latency / wins of a Core ML variant |
| `coreml/` | the Core ML export pipeline shared with the stock model |

## Setup

```bash
git clone https://github.com/smogon/pokemon-showdown && cd pokemon-showdown && npm install && node pokemon-showdown start --no-security
# in this folder (uv installs poke-env, torch, transformers, coremltools into a throwaway environment)
ENV=(--with poke-env --with torch==2.9.1 --with torchvision==0.24.1 --with transformers==5.14.1 --with Pillow --with safetensors --with coremltools==9.0 --with 'numpy<2.3')
uv run --no-project --python 3.12 "${ENV[@]}" python run_battles.py --model-dir <coreml dir> --checkpoint <Intern-Decision-0.8B snapshot> --battles 30
```

`<coreml dir>` is a local copy of the Hub repo (buckets, `embeddings.f16`, `tokenizer.json`); `<snapshot>` is the
Hugging Face snapshot of `internlm/Intern-Decision-0.8B` (its `inference.py` renders the prompt). Spectate a local
battle at `https://localhost.psim.us/<room>`.

## Results (gen9randombattle, our side first)

| Player | vs random | vs max-base-power | vs simple heuristics | ms/decision, M5 Pro |
| --- | ---: | ---: | ---: | ---: |
| stock Intern-Decision-0.8B, Core ML (10) | 2-3 | 1-9 | 1-9 | 89 |
| Intern-Decision-4B teacher, PyTorch (60) | 58-2 | 45-15 | 16-44 | 170 |
| fine-tuned 0.8B, Core ML (30) | 9-1 (10) | 24-6 | 9-21 | 89 (512 bucket), 125 (640) |

Teacher: the 4B played 220 battles (6,974 decisions); student: LoRA r32, 1.5 epochs, 200 min on the M5 Pro, held-out
agreement with the 4B 35% → 80%. Neither the 27B nor any GPU server was involved. The heuristic bot is poke-env's
standard baseline and both the 4B and its student sit around 30% against it.

## Gotchas

- `pokemon-showdown --version` starts the server. After a crashed poke-env process, restart the server: stale logins
  keep the default usernames and new players hang silently. Login names are capped at 18 characters.
- Do not run a PyTorch teacher and a training job at the same time on a 24 GB Mac.
- `play.pokemonshowdown.com/?~~localhost:8000` connects to the public server; use `https://localhost.psim.us/`.
- asitop crashes on the M5 Pro; the demo uses `macmon`.
