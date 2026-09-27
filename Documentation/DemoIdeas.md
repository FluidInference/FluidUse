# Demo ideas and models to convert

A working list of what to build and post next, based on what the game trials
showed. Model runtimes that already ship are in
[DecisionModelSupport.md](DecisionModelSupport.md).

## What the trials showed

Stock decision models do well when each move is a short-horizon choice among a
few described options and a bad move costs points, not the game. They fail
when one wrong move is fatal or a good move needs lookahead. A small model
trained on search or heuristic labels for the game fixes much of that, and it
runs in 1–3 ms per move. All numbers are from an Apple M5 Pro.

| Game | Stock models | Trained for the game | Status |
| --- | --- | --- | --- |
| Tetris (shortlist + heuristic top-2) | GLiClass LUT8 3,666.7 lines vs heuristic 2,874.4 (seeds 1–10, cap 5,000) | 706K tiny scorer 4,712.7 at 1.29 ms; fine-tuned GLiClass 4,600.5 | Shipped (`LayaTetris`); trained models local |
| 2048 | Model comparison shipped | — | Shipped (`GLiClass2048Demo`, `Decision2048BenchDemo`) |
| Snake (10×10, labeled options) | Best: Lex 112 food vs heuristic 305 | Lex fine-tuned on heuristic play: 293 food, 0 avoidable crashes; 345 vs 344 with no step cap | Public Core ML: [decision-1.0-lex-snake-coreml](https://huggingface.co/FluidInference/decision-1.0-lex-snake-coreml) |
| Connect Four vs heuristic | Every model 0 of 20 | Fine-tuned GLiClass 7 wins; 706K tiny 5 wins, 3 draws; tiny trained on raw boards 6 wins | Local branch |
| Flappy Bird, lane runner | 0–1 pipes; every runner game ends on a crash-labeled pick | — | [#10](https://github.com/FluidInference/FluidUse/pull/10), [#11](https://github.com/FluidInference/FluidUse/pull/11) |
| Wordle (GLiNER2.5-Decide picks a guess) | Same as random over the same shortlist | — | Dropped |

Two cautions for posting results:

- On Tetris, a 5-weight linear scorer trained on the same labels scored 14,518
  lines against 13,281 for the 706K model (cap 20,000, seeds 1–20). When the
  options are already numeric features, model size adds nothing. Say so, or
  give the model information the heuristic lacks (next piece, raw board).
- Cap-5,000, 10-seed Tetris runs are noisy. Use 20 seeds and a higher cap for
  any number that gets compared.

## Demos to post

Ordered by how well they fit a short clip: one visible number, a hook in the
first two seconds, and a result that holds up when someone checks it.

| Rank | Demo | Hook | Built? |
| ---: | --- | --- | --- |
| 1 | **Train a game brain in 4 minutes** | Record a Mac training a 706K model on Connect Four search labels, then beating the heuristic that every stock model lost to 20–0. | Pipeline exists (`ConnectFourCheck dump`, Cua S1 trainer); needs a recorded run and a replay app |
| 2 | **Fine-tuned Lex vs the heuristic, Snake** | Stock Lex 112 food, fine-tuned 293; side by side on the same seed. | Model public; `SnakeDemo` app on local branch |
| 3 | **Sort Anything** | 1,000 Wikipedia abstracts into 14 buckets in under 6 s, 89.0%, about 4× faster than PyTorch on MPS with a sixth of the peak memory. | Shipped (`SortAnythingDemo`) |
| 4 | **Model race** | Split screen, same seed, several models playing at once, live ms-per-move counter. The speed gap is the story. | Needs a multi-pane app; policies already shared in `DecisionPolicy` |
| 5 | **Forms in real apps** | CUA-S1-FORMS filling a real job application in Safari and Chrome at about 1 ms per field. | Local demo app; needs a clean recording |
| 6 | **Describe a decision, get a model** | Type "is this email urgent?", label examples with a larger local model, train a tiny classifier, and run it on the Neural Engine, all offline. Local take on [ProgramAsWeights](https://programasweights.com/). | Not built; reuses the Connect Four and Tetris training loop |
| 7 | **Drop-merge or Threes** | Forgiving merge puzzles, closest to 2048, which already works. | Not built; reuses `Game2048` |
| 8 | **Codenames guesser** | Semantic association, which is how these models are trained; no lookahead. | Not built |
| 9 | **Minesweeper calibration** | "80% safe" checked against exact mine probabilities over thousands of positions. A benchmark chart, not a live game. | Not built |

### Different decision shapes

The games above are mostly board geometry. These test other kinds of decision,
so each shows something the Tetris, Snake, and Connect Four trials do not.

| Demo | Why it is different |
| --- | --- |
| Wordle / Connections | Language and constraints, not board geometry. The Wordle guess-picker trial matched random, so Connections grouping is the one to try. |
| Minesweeper | Uncertainty and null scores: the model must say when no cell is safe. Same idea as rank 9. |
| Chess puzzles (mate in 1–2) | "Is this the only winning move?" A single correct answer, checkable against an engine. |
| Customer-support desk | The actual Laya use case, not a game: route or answer tickets from a fixed set of actions. |
| Model router | Given a prompt, pick ASR vs TTS vs vision vs skip. Maps directly to choosing a FluidAudio model. |

Ruled out as live demos: real-time control games (Flappy Bird, runners,
platformers, Pac-Man, Breakout), stock models on Connect Four or Snake, and
multiplayer Snake. Each one ends on the first fatal mistake, which stock models
make within a few moves.

### Next up

Record before building anything new:

1. **Connect Four train-then-beat** (rank 1). Show seed count and the result
   as it is: 0 of 20 stock, 5 wins and 3 draws for the 706K model. It does not
   beat the heuristic outright, so don't caption it that way.
2. **Lex Snake, same seed** (rank 2). Link the Hugging Face package.
3. **Safari job-application form fill** (rank 5) with a ms-per-field overlay.
4. **Sort Anything** (rank 3): a polished 20 s cut if not posted yet, then a
   follow-up that edits categories mid-run ("add musician") so it doesn't look
   like a fixed benchmark.

Then start the laya-browser conversion, and build the model race (rank 4) once
those clips are out.

A combo that fits the same rules: **voice → typed decision → Accessibility
action**. Parakeet EOU hears the task, laya-browser or CUA-S1-FORMS picks the
control, FluidUse types it. Each step is still a short choice among described
options, and it puts the audio stack and the forms demo in one video.

### Doom (tried) and Minecraft

TypeSafe's Jev Doom demo never sees pixels: ViZDoom state goes in as text, one
typed choice comes out about 10 times a second, and code holds the button
between calls
([Register](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711),
[doom-jev](https://github.com/mikespins/doom-jev)). We ran both a stock model
in that style and a small model trained on the game, on ViZDoom
`defend_the_center` (spin in place and shoot, 60 s episodes). Code is on local
branch `feat/vizdoom-gliclass` (`Tools/doom`).

| Player | Input | Kills | Survived | ms per decision |
| --- | --- | ---: | ---: | ---: |
| [SauerkrautLM-Doom-MultiVec 1.3M](https://huggingface.co/VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M), Core ML fp16, GPU | 40×25 depth grid | 20.54 | 50.6 s | 1.5–3.4 |
| Same model, PyTorch | same | 20.42 | 50.5 s | 57.7 (1 thread), 19.0 (MPS) |
| Hand-coded aimer | exact monster bearings | 13.05 | 24.8 s | — |
| GLiClass LUT8, consequence labels | text state + labels | 11.98 | 23.1 s | 4.5 |
| GLiClass LUT8, bare labels ("attack", "turn left") | text state | 0.00 | 8.3 s | 3.6 |
| Random | — | 1.26 | 9.7 s | — |

Seeds 10000–10099 at 4 tics per decision (GLiClass bare labels: seeds 1–20 at 3 tics).
Aimer and GLiClass use three buttons and 320×240, so they are close to, not
identical with, the Sauerkraut setup.

- Stock GLiClass turns the right way but never fires on bare labels. With
  labels that state each action's consequence it reaches the aimer's level,
  but the labels do most of the deciding. Same lesson as Connect Four and
  Snake: stock models don't play games where one miss is fatal.
- SauerkrautLM-Doom (Apache 2.0) converts cleanly: Core ML fp32 matches
  PyTorch kill for kill on 100 of 100 seeds, and fp16 on the GPU is 17–38×
  faster than one-thread PyTorch. The Neural Engine is slower than the GPU for
  this model (about 8 ms). An independent 1,000-episode evaluation of the
  PyTorch model reports 20.38
  ([tiny-doom-defender](https://huggingface.co/spaces/anakin87/tiny-doom-defender)).
- It reads depth, not a text screen. The upstream text channel overflows and
  is `@` in every cell, in training too, so all information comes from 16-level
  depth bins. Caption it "plays Doom from a 40×25 depth grid".
- Demo is built: split screen with the game, the depth grid, action
  probabilities, and ms per decision, plus Terminal windows for `sudo asitop`
  and a decision log. Seeds 10016 and 10005 reach 25 kills and survive the full
  60 s. Next step is a Hugging Face upload of the Core ML model.
- Stronger follow-up:
  [tiny-doom-defender](https://huggingface.co/anakin87/tiny-doom-defender)
  (1.1M, supervised then PPO) reports 23.12 kills.

Minecraft has one real Core ML candidate: [VPT](https://github.com/openai/Video-Pre-Training)
(OpenAI, MIT), or [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1) on top
of it for text instructions ("chop a tree"). Pixels in, keyboard and mouse out
at 20 Hz; smaller 1x/2x/3x widths exist, checkpoints on Hugging Face via
[MineStudio](https://github.com/CraftJarvis/MineStudio). Harder than Doom:
MineRL needs Java Minecraft on Apple Silicon, the transformer memory has to be
carried between calls, and there is no single score, so the clip is "gets
logs in N seconds". Start with a half-day check (MineRL on this Mac, VPT 1x
size and speed in PyTorch) before converting.

Not worth converting for Minecraft: OpenHA (Qwen2-VL 7B) is too large for real
time, and the Jev Ender Dragon runs
([hermes-and-jev](https://github.com/teknium1/hermes-and-jev-play-minecraft),
[minecraft-agent](https://github.com/rmalde/minecraft-agent)) are a planner
plus Mineflayer scripts with the decision model picking from valid moves.
Swapping in a local chooser is easy but the harness does the playing.

## Models to convert or use

From the open Jev-style models tracked on
[madewithjev.com](https://madewithjev.com/open-source-jev) and the GLiNER
family, excluding what already ships.

| Model | Why | Conversion notes |
| --- | --- | --- |
| [laya-browser](https://huggingface.co/cklxx/laya-browser) | Laya fine-tuned on Mind2Web as a web-element chooser; the closest open match to FluidUse's job. | Same architecture as the shipped `LayaManager`; reuse the laya export. |
| Kev 0.8B / 4B ([kev](https://github.com/jaredpalmer/kev)) | Newest Kev checkpoints on Qwen3.5. | 0.8B port matches Kev fp32 (3 flips in 2,232 questions); needs a state cache to beat Kev's MLX speed. |
| [KaLM-Jev](https://github.com/KaLM-Embedding/kalm-jev) | Nano, Small, and Large sizes; Nano may fit the 1 ms budget. | Repo has no license file; ask before converting. |
| Verdict 2.0 ([source](https://github.com/Heman10x-NGU/Verdict-open-jev)) | Listed as 151M ModernBERT, 77.1% with Brier 0.064. | Shipped Verdict is pinned at `heman10x/rlcd-modernbert-151m@8af2496`; re-convert if 2.0 is a newer checkpoint. |
| [QwenJev](https://github.com/RJMSWD/QwenJev) | Vision input, 0.169 s per frame. | No license file. Qwen3.5-4B: GPU only, same path as Cua-S1-4B ([#12](https://github.com/FluidInference/FluidUse/pull/12)). |
| [Simple Jev](https://github.com/featherless-ai/simple-jev) | Serves any Hugging Face model as a Jev-style endpoint. | Nothing to convert; use it as a reference for which base models answer typed questions well. |
| [ProgramAsWeights](https://github.com/programasweights/programasweights-python) | English spec compiled to a LoRA on Qwen3 0.6B or GPT-2. | Generates text, 30–500 ms per call; compare against, not ship. |
| GLiNER / GLiClass family | Author's family tree lists siblings not yet converted. | Same export as GLiNER 2.5 and GLiClass Edge. |

Conversion order, by closeness to FluidUse's job: laya-browser first (makes
form fill look like general computer use), then the Kev 0.8B state cache, then
GLiNER / GLiClass siblings (cheapest). KaLM-Jev Nano and Verdict 2.0 wait on
licenses, and Nano only if it stays near 1 ms. QwenJev is a GPU-only 4B model,
so keep it out of the Neural Engine race clips.
