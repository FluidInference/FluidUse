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

Ruled out as live demos: real-time control games (Flappy Bird, runners,
platformers, Pac-Man, Breakout), stock models on Connect Four or Snake, and
multiplayer Snake. Each one ends on the first fatal mistake, which stock models
make within a few moves.

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
