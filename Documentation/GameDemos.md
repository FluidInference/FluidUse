# Decision game demos

Ideas for comparing small decision models in observable, repeatable environments.
These are proposed demos, not games currently shipped in FluidUse. Tetris, 2048,
and Snake are the starting examples from the discussion. Flappy Bird has been
tried ([results](#flappy-bird-trial-results)): no text-state model played it
usefully. The [model inventory](Models.md) identifies possible sub-1B models
and distinguishes text-state models from vision models.

## Recommended build order

This is a judgment about **solo and side-by-side demo value for FluidUse**, not
a measured model ranking. It favors visible differences between models, a clear
score, distinct decision types, and reusable game code. Tetris, 2048, and Snake rank last only
because they are already the starting examples. [PlayJev](https://github.com/OmniJev/PlayJev)
reduces the work for ten browser games; the three-lane
[Neon Cyberpunk Runner](https://github.com/markstent/runner) is the reuse path
for a Subway Surfers-style demo.

| Rank | Game | Main reason to demo it |
| ---: | --- | --- |
| 1 | Flappy Bird | **Tried:** no model passed more than one pipe; see [results](#flappy-bird-trial-results). |
| 2 | Sokoban / Boxoban | Clear planning failures; reusable PlayJev game. |
| 3 | Subway Surfers-style runner | High visual appeal and distinct lane, jump, and slide choices; reusable runner exists. |
| 4 | Mario-style platformer | Familiar, varied actions and long action sequences; PlayJev has a game harness. |
| 5 | Chess | Recognizable tactics and a static-position mode that is easy to compare. |
| 6 | Wordle-style word game | Adds information-gathering decisions with a compact state. |
| 7 | Minesweeper | Shows risk-sensitive choices and uncertainty. |
| 8 | Frogger-style crossing | Shows timing around moving hazards with simple actions. |
| 9 | Connect Four | Small, legal move set and easy human-versus-model play. |
| 10 | Pac-Man | Visually engaging risk/reward decisions; PlayJev has a game harness. |
| 11 | Space Invaders | Fast aiming and dodging; PlayJev has a game harness. |
| 12 | MiniWoB-style web tasks | Closest to FluidUse's computer-use purpose, with objective task success. |
| 13 | Breakout / Pong | Simple control and visible prediction errors; PlayJev includes Breakout. |
| 14 | MiniGrid DoorKey | Multi-step planning with an existing environment. |
| 15 | Codenames | Semantic association, distinct from movement games. |
| 16 | Racer | Fast steering decisions; PlayJev has a game harness. |
| 17 | Battleship | Hidden-information search, but slower to watch. |
| 18 | Hanabi | Cooperative partial information, but harder to explain at a glance. |
| 19 | Tower defense | Resource allocation, but a large action and state space. |
| 20 | FrozenLake | Useful stochastic baseline, but visually less compelling. |
| 21 | Tetris | Already a starting example. |
| 22 | 2048 | Already a starting example. |
| 23 | Snake | Already a starting example. |

## Demo list

| Status | Game | Model's decision | What it reveals | Suggested state |
| --- | --- | --- | --- | --- |
| Tried | **Flappy Bird** | `FLAP` or `COAST` each decision tick. | Reaction timing, latency, and costly one-step mistakes. | Bird height and velocity; next pipe distance and gap, or a rendered frame for a vision model. |
| Candidate | **Sokoban / Boxoban** | Pick a legal move or push. | Planning and irreversible traps. | Grid, player, crates, goals, and legal moves. |
| Candidate | **Mario-style platformer** | Choose left, right, jump, or a legal combination at each tick. | Jump timing, momentum, obstacle anticipation, and longer action sequences. | Player position and velocity, nearby platforms, enemies, and camera offset; or a rendered frame. |
| Candidate | **Chess** | Choose or rank legal moves. | Tactical judgment, position evaluation, and planning across turns. | FEN, side to move, legal moves, and remaining time; a board image only for vision models. |
| Candidate | **Minesweeper** | Open or flag a cell. | Decision-making under uncertainty. | Revealed grid and legal cells; show mine probabilities only if actually computed. |
| Candidate | **Wordle-style word game** | Choose the next valid guess. | Information gathering versus an immediate attempt to solve. | Previous guesses, color feedback, and a fixed candidate-word list. |
| Candidate | **Frogger-style crossing** | Move up, down, left, right, or wait. | Timing around moving hazards and choosing safe windows. | Player, lane hazards, speeds, and goal positions; or a rendered frame. |
| Candidate | **Subway Surfers-style runner** | Switch lane, jump, slide, or hold. | Fast obstacle recognition, action timing, and choosing between survival and coins. | Current lane, speed, nearby obstacles and distances; or a rendered frame. |
| Candidate | **Connect Four** | Choose a non-full column. | Short tactical lookahead with only a few legal actions. | Board, player to move, and legal columns. |
| Candidate | **Multiplayer Snake arena** | Choose direction while several snakes move simultaneously. | Direct competition, collisions, and survival under pressure. | Board, all visible snakes, food, and legal directions. |
| Candidate | **Pac-Man** | Direction at each junction. | Reward versus moving hazards. | Map, player, ghosts, pellets, and power timer. |
| Candidate | **Codenames** | Choose a clue or a guess from a fixed set. | Semantic association under constraints. | Visible words, team, prior clues, and legal choices. |
| Candidate | **Hanabi** | Play, discard, or give a legal hint. | Cooperation with incomplete information. | Only what the acting player may observe. |
| Candidate | **MiniGrid DoorKey** | Turn, move, pick up, or open. | Multi-step planning when a key must be found before reaching the goal. | Partial grid observation, carried item, and door state. |
| Candidate | **FrozenLake** | Choose one of four directions. | Planning under uncertain movement when the lake is slippery. | Map, position, goal, holes, and known movement rules. |
| Candidate | **Battleship** | Choose an untried target square. | Search under hidden information and using feedback from earlier turns. | Hits, misses, sunk ships, and remaining legal squares. |
| Candidate | **Tower defense** | Place, upgrade, sell, or wait. | Resource allocation with delayed effects. | Map, waves, towers, budget, and legal placements. |
| Candidate | **MiniWoB-style web tasks** | Choose an element and operation. | Practical computer use with a clear success condition. | Accessibility element table and task goal; screenshots for vision models. |
| Candidate | **Breakout / Pong** | Move paddle left, right, or stay. | Ball prediction and control latency. | Ball/paddle positions and velocities. |
| Candidate | **Space Invaders** | Move, fire, or combine actions. | Avoidance and aiming under continuous pressure. | Player, projectiles, enemies, and cooldowns. |
| Candidate | **Racer** | Steer left, right, or straight. | Lookahead and fast corrections. | Track geometry, position, speed, and obstacles. |
| Existing example | **Tetris** | Choose a placement or movement. | Long-term board management. | Board, current piece, next piece if allowed, and legal placements. |
| Existing example | **2048** | Choose a legal slide. | Repeated choices with random future tiles. | Board and each legal resulting board before the random spawn. |
| Existing example | **Snake** | Choose direction. | Path planning while avoiding self-traps. | Grid, body, food, and current direction. |

Flappy Bird looked like the clearest first comparison: two actions, immediate
outcomes, and a visible cost when a decision arrives late. The trial showed that
per-tick control is a poor fit for these text-state choosers. Sokoban adds planning, and
Minesweeper adds uncertainty. Together, those three test different strengths
than the existing Tetris, 2048, and Snake examples.

Mario is a strong follow-up to Flappy Bird: both expose late decisions, while a
platformer also requires direction and jump choices across several frames.
Track distance reached, obstacles cleared, deaths, completed levels, and decision
latency on the same seeded levels. Define how many frames each action lasts so a
slower model is not silently given more game time. PlayJev already includes an
[Infinite Mario](https://github.com/OmniJev/PlayJev) environment, which is useful
as a reference. For a new public demo, use original or clearly licensed art;
the PlayJev repository notes that the Mario sprites in its vendored game belong
to Nintendo, despite the game's code being under the Unlicense.

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

Wordle and Battleship add information-gathering decisions; Frogger adds moving
hazards; Connect Four is a compact turn-based duel. For reusable planning
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
| [PlayJev game harness](https://github.com/OmniJev/PlayJev) | Tetris, Snake, Pac-Man, Racer, Space Invaders, Sokoban, Infinite Mario, Floppy Bird, Breakout, 2048. | Already exposes seeded `start`, `step`, `frame`, `score`, `done`, and actions. Each vendored game has its own license; check art and levels separately before publishing. |
| [Neon Cyberpunk Runner](https://github.com/markstent/runner) | First Subway Surfers-style demo. | MIT-licensed Three.js browser game with three lanes, jump, slide, seedable track generation, and game logic separated from rendering. Add a thin adapter for observations and model actions. |
| [Node Multiplayer Snake](https://github.com/simondiep/node-multiplayer-snake) | Multi-model survival arena. | MIT-licensed browser game with spectator mode, bots, and adjustable speed. Adapt its player controllers to model actions and add seeded resets. |
| [Cave Runner](https://github.com/tope-olajide/cave-runner) | Alternate 3D runner. | MIT-licensed, but its online score path uses Netlify and PlanetScale; assess whether a local-only demo can bypass that path. |
| [MiniGrid](https://minigrid.farama.org/environments/minigrid/) and [Gymnasium](https://gymnasium.farama.org/main/environments/) | DoorKey, FrozenLake, and other compact decision tasks. | Existing reset/step environments; add a viewer and a model input adapter. |

### Start with PlayJev

Use [OmniJev/PlayJev](https://github.com/OmniJev/PlayJev) as the first game
harness to evaluate, rather than implementing its ten games again. Its browser
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
sprites are owned by third parties. The runner above fills a gap in its current
game list; it is not one of PlayJev's ten games.

For the runner, use the same generated track seeds, action interval, and speed
curve for every model. Report distance, obstacles cleared, coins, collisions,
and missed decision deadlines. An unlicensed [Subway Surfers clone](https://github.com/eeshadutta/Subway-Surfers)
exists, but its repository does not declare a reuse license, so it is not the
recommended source for a published demo. Call the result an endless runner and
use the upstream game's own art and name unless rights to the Subway Surfers
branding and assets are available.

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
