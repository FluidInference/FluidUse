# Decision game demos

Ideas for comparing small decision models in observable, repeatable environments.
These are proposed demos, not games currently shipped in FluidUse. Tetris, 2048,
and Snake are the starting examples from the discussion. Flappy Bird has been
tried ([results](#flappy-bird-trial-results)): no text-state model played it
usefully, so real-time control games are [ruled out](#ruled-out-real-time-games).
The [model inventory](Models.md) identifies possible sub-1B models
and distinguishes text-state models from vision models.

## Recommended build order

This is a judgment about **solo and side-by-side demo value for FluidUse**, not
a measured model ranking. It favors turn-based games where each legal move can be
listed with its consequence in words, which is the setup that took the Tetris
harness from 76 to 568 pieces. It also favors a clear score and visible
differences between models. Tetris, 2048, and Snake rank last only because they
are already the starting examples.

| Rank | Game | Main reason to demo it |
| ---: | --- | --- |
| 1 | Connect Four | At most seven legal columns, each describable ("wins now", "blocks their win"); easy model-vs-model duels. |
| 2 | Chess (position challenge) | Recognizable tactics; tagged candidate moves scored against an engine ranking. |
| 3 | Codenames (guesser) | Semantic association, the closest match to how these models are trained. |
| 4 | Minesweeper | Risk-sensitive choices; safe cells and constraints can be spelled out. |
| 5 | Wordle-style word game | Information gathering with a compact state and a fixed candidate list. |
| 6 | MiniWoB-style web tasks | Closest to FluidUse's computer-use purpose, with objective task success. |
| 7 | Battleship | Hidden-information search, but slower to watch. |
| 8 | Sokoban / Boxoban | Clear planning failures; reusable PlayJev game, but needs multi-step lookahead. |
| 9 | MiniGrid DoorKey | Multi-step planning with an existing environment. |
| 10 | Multiplayer Snake arena | Direct competition; run in lockstep with no clock. |
| 11 | Hanabi | Cooperative partial information, but harder to explain at a glance. |
| 12 | Tower defense | Resource allocation, but a large action and state space. |
| 13 | FrozenLake | Useful stochastic baseline, but visually less compelling. |
| 14 | Tetris | Already a starting example. |
| 15 | 2048 | Already a starting example. |
| 16 | Snake | Already a starting example. |

## Demo list

| Status | Game | Model's decision | What it reveals | Suggested state |
| --- | --- | --- | --- | --- |
| Candidate | **Sokoban / Boxoban** | Pick a legal move or push. | Planning and irreversible traps. | Grid, player, crates, goals, and legal moves. |
| Candidate | **Chess** | Choose or rank legal moves. | Tactical judgment, position evaluation, and planning across turns. | FEN, side to move, legal moves, and remaining time; a board image only for vision models. |
| Candidate | **Minesweeper** | Open or flag a cell. | Decision-making under uncertainty. | Revealed grid and legal cells; show mine probabilities only if actually computed. |
| Candidate | **Wordle-style word game** | Choose the next valid guess. | Information gathering versus an immediate attempt to solve. | Previous guesses, color feedback, and a fixed candidate-word list. |
| Candidate | **Connect Four** | Choose a non-full column. | Short tactical lookahead with only a few legal actions. | Board, player to move, and legal columns. |
| Candidate | **Multiplayer Snake arena** | Choose direction while several snakes move simultaneously. | Direct competition, collisions, and survival under pressure. | Board, all visible snakes, food, and legal directions. |
| Candidate | **Codenames** | Choose a clue or a guess from a fixed set. | Semantic association under constraints. | Visible words, team, prior clues, and legal choices. |
| Candidate | **Hanabi** | Play, discard, or give a legal hint. | Cooperation with incomplete information. | Only what the acting player may observe. |
| Candidate | **MiniGrid DoorKey** | Turn, move, pick up, or open. | Multi-step planning when a key must be found before reaching the goal. | Partial grid observation, carried item, and door state. |
| Candidate | **FrozenLake** | Choose one of four directions. | Planning under uncertain movement when the lake is slippery. | Map, position, goal, holes, and known movement rules. |
| Candidate | **Battleship** | Choose an untried target square. | Search under hidden information and using feedback from earlier turns. | Hits, misses, sunk ships, and remaining legal squares. |
| Candidate | **Tower defense** | Place, upgrade, sell, or wait. | Resource allocation with delayed effects. | Map, waves, towers, budget, and legal placements. |
| Candidate | **MiniWoB-style web tasks** | Choose an element and operation. | Practical computer use with a clear success condition. | Accessibility element table and task goal; screenshots for vision models. |
| Existing example | **Tetris** | Choose a placement or movement. | Long-term board management. | Board, current piece, next piece if allowed, and legal placements. |
| Existing example | **2048** | Choose a legal slide. | Repeated choices with random future tiles. | Board and each legal resulting board before the random spawn. |
| Existing example | **Snake** | Choose direction. | Path planning while avoiding self-traps. | Grid, body, food, and current direction. |

Connect Four, chess positions, and Codenames each test a different strength
from the existing Tetris, 2048, and Snake examples: tactical lookahead,
position judgment, and semantic association. Minesweeper adds uncertainty, and
Sokoban adds planning.

Chess has two useful modes. A **position challenge** gives every model the same
FEN and legal move list, then compares its ranked moves with published reference
values. The Decision Index already includes this kind of static test as
[ChessBench](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json).
A **full-game demo** pairs models against each other at the same clock setting
and matched starting positions. Record legal-move rate, move time, game result,
and engine evaluation loss per move. Supply legal moves from a chess rules engine
so the model is judged on choosing among them, not on formatting notation.

## Competitive games

Make **model vs model** the main viewer mode. The following modes use the same
decision interface, with fixed-opponent runs retained as a diagnostic:

| Mode | Purpose | How it works |
| --- | --- | --- |
| Model vs model | Main spectator match. | Run a round robin with paired colors and matched opening positions. Show wins, draws, losses, illegal or timed-out decisions, and move time. |
| Same model vs itself | Self-play showcase. | Launch two independent instances of the same checkpoint. Swap colors across paired games and vary openings. If choices are sampled, give each instance an independent random seed and show the sampling setting. |
| Multi-model arena | Actual battle royale. | Put three or more models in one simultaneous-action game, such as multiplayer Snake. Give every model the same board snapshot each tick, collect actions independently, then resolve them together at the deadline. Rotate spawn positions across matches. |
| Human vs model | Interactive side mode. | The person picks a side and a legal move; show the model's top choices, probabilities when available, and response time. Do not count these self-selected games in model rankings. |
| Model vs fixed opponent | Diagnostic. | Each candidate faces the same version and settings of a reference policy. Run paired games from the same starting positions with colors or first turn swapped. |

For competitive demos, start with **Connect Four** for two-player self-play and
cross-model matches, then **multiplayer Snake** for a true arena, then **chess**.
An existing [MIT-licensed multiplayer Snake game](https://github.com/simondiep/node-multiplayer-snake)
has spectator mode, bots, and adjustable speed; evaluate it as a reusable base
instead of expanding PlayJev's single-player Snake from scratch. Connect Four
has a small action set and short games. Use a rules engine to provide legal
columns and a fixed search policy only for diagnostics. [chess.js](https://github.com/jhlywa/chess.js)
can provide legal moves, validation, and game-end detection in a browser demo;
[Stockfish](https://github.com/official-stockfish/Stockfish) can be the fixed
reference opponent for diagnostics at a stated strength and time budget. Models see the same FEN,
move history limit, legal move list, and clock information. Rotate colors and
use a fixed set of opening positions. Keep any engine evaluation out of the
model's input; calculate it afterward for the viewer.

In self-play, identical deterministic policies can repeat the same line or draw
often. Vary the opening positions and show that behavior honestly; optional
sampling makes matches more varied, but its temperature must be declared and
held fixed across models. For rankings, use many paired matches and record the
model version, side or spawn, opening or map seed, action deadline, and full
move log. A single spectacular match is a demo clip, not a performance estimate.

For hidden-information games, keep roles and observations honest. In Battleship,
each player sees only its own ships and prior shots. In Codenames, score clue
giver and guesser separately, with a fixed partner for comparable runs. Hanabi
is cooperative rather than adversarial: use fixed teammate policies and report
team score, not win rate against another model. Record opponent version, seeds,
starting positions, rules, action deadline, and every move so a result can be
replayed.

Wordle and Battleship add information-gathering decisions; Connect Four is a
compact turn-based duel. For reusable planning
environments, [MiniGrid DoorKey](https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/)
and [Gymnasium FrozenLake](https://gymnasium.farama.org/main/environments/toy_text/frozen_lake/)
provide configurable tasks with small discrete action spaces. Start each model
from the same puzzle or seeded map, and keep its observations limited to what
the game rules permit.

## Reuse existing games

The goal is to adapt playable games, not rebuild their physics, visuals, and
controls. A demo adapter should expose `reset(seed)`, `observe()`, `legalActions()`,
`step(action)`, `score()`, and `done()`. It can call an existing game's input
functions or simulate its keyboard controls. Keep gameplay and assets upstream
where possible; record the upstream revision and changes to the adapter.

| Source | Good for | Reuse notes |
| --- | --- | --- |
| [PlayJev game harness](https://github.com/OmniJev/PlayJev) | Sokoban, Tetris, Snake, and 2048; its other six games are real-time and ruled out. | Already exposes seeded `start`, `step`, `frame`, `score`, `done`, and actions. Each vendored game has its own license; check art and levels separately before publishing. |
| [Node Multiplayer Snake](https://github.com/simondiep/node-multiplayer-snake) | Multi-model survival arena. | MIT-licensed browser game with spectator mode, bots, and adjustable speed. Adapt its player controllers to model actions and add seeded resets. |
| [MiniGrid](https://minigrid.farama.org/environments/minigrid/) and [Gymnasium](https://gymnasium.farama.org/main/environments/) | DoorKey, FrozenLake, and other compact decision tasks. | Existing reset/step environments; add a viewer and a model input adapter. |

### Start with PlayJev

Use [OmniJev/PlayJev](https://github.com/OmniJev/PlayJev) as the harness for
Sokoban and the existing examples, rather than implementing them again. Its browser
hook provides `window.pj.start(seed)`, `step(action)`, `frame()`, `score()`,
`done()`, and an action list. The repository also includes random and teacher
policies, recorded runs, and a [0.8B vision model](https://huggingface.co/OmniJev/PlayJev-0.8B)
that can serve as a visual reference. Keep the game, seed, tick rate, and action
list fixed while swapping the decision provider. For text-only models, add a
state extractor for each game and report text-state and pixel-input results in
separate tracks. Re-run all compared models in the same harness; PlayJev's
published scores are background context, not directly comparable results.

Before vendoring or publishing a game, review its own license and bundled art.
PlayJev documents the licenses of its ten vendored games and notes that some
sprites are owned by third parties.

## Flappy Bird comparison specification

1. The game engine advances at a fixed rate. Models receive a state at the same
   decision interval and choose `FLAP` or `COAST`. The engine owns physics and
   collision rules.
2. Run each model separately on the same seeded pipe sequences. Fix game speed,
   decision interval, state schema, action semantics, and deadline. Give vision
   models the same rendered frame; give text models the same structured facts.
   Report vision and text-state results as different input tracks.
3. Show each action and its probability when available, plus the time it took.
   An action probability expresses the model's preference between `FLAP` and
   `COAST`; it is not a survival probability.
4. Keep the clock running during inference. If a result misses the deadline,
   apply a declared fallback action and count a missed deadline.
5. Report pipes passed, survival ticks, median and p95 decision latency, missed
   deadlines, and number of calls for every seed. Add a random policy and a
   simple physics controller as reference runs. Publish the seed list and
   per-run results before reporting aggregates.

The [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
measures a frozen panel of static decision tasks. Its current score excludes
interactive environments, including Boxoban, Hanabi, Codenames, and MiniWoB++.
Use matched-seed game results to establish game performance; Index rank is only
a way to find candidate models. [PlayJev](https://github.com/OmniJev/PlayJev)
is a separate 0.8B vision model with published results on several of these games,
including Flappy Bird. Its published numbers use its own harness and should not
be placed on the same chart as new runs until the harness matches.

## Flappy Bird trial results

Flappy Bird was built and run as
[FluidUse #10](https://github.com/FluidInference/FluidUse/pull/10) (`GLiClassFlappyDemo` app and
`FlappyBirdCheck` headless runner). Each model got a text state plus a 300 ms
physics forecast for each action, and chose `FLAP` or `COAST` every six frames (10 Hz).
Without the safety guard, on seeds 1–4 (Apple M5 Pro):

| Policy | Pipes passed | Combined survival |
| --- | ---: | ---: |
| GLiNER 2.5 multilingual W8 | 1 | 10.98 s |
| Kev 0.6B, Decision 1.0 Lex | 0 | 9.03 s each |
| Kev 0.5B, Decision 1.0 Kai, GLiNER 2.5 small | 0 | 8.5–8.7 s |
| LFM2.5-350M-RLCD, Jeff | 0 | 6.7–7.1 s |
| Laya, GLiClass, NanoJev, Verdict, GLiNER 2.5 base | 0 | 5.4–5.9 s |
| Physics heuristic | 40 | 80 s (four 20 s caps) |

None of the 13 models is a usable controller. Most ignored the forecast. GLiClass,
for example, flapped on almost every call, and LFM always chose the option listed second.
The safety guard reached the cap but overrode 106–131 of 200 choices, so that
result measures the guard. LFM (~150 ms) and NanoJev (~78 ms) also miss the
100 ms decision period. Per-tick timing control does not suit
label-choosing models. A fairer follow-up would ask a yes/no safety question or
query only at hard decision points.

## Ruled out: real-time games

The Flappy Bird result applies to every game that needs control on each frame
or tick. These games have more actions than Flappy Bird, need inputs held
for exact frame counts, or have many fatal mistakes per run:

| Game | Why it is ruled out |
| --- | --- |
| Mario-style platformer | Left/right/jump/run combinations; jump height depends on hold duration; many fatal mistakes per level. |
| Subway Surfers-style runner | Lane, jump, and slide timing against obstacles at speed. |
| Frogger-style crossing | Timing windows around moving hazards. |
| Pac-Man | Per-tick direction under chasing ghosts. |
| Space Invaders | Continuous aiming and dodging. |
| Breakout / Pong | Paddle control against ball speed. |
| Racer | Per-frame steering corrections. |

A "decision point" variant could reuse some of them. It pauses at each obstacle and
offers consequence-labeled options while scripted movement handles the rest. In
that setup the game logic does most of the work, so it would say little about
the models.
