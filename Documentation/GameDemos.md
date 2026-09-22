# Decision game demos

Ideas for comparing small decision models in observable, repeatable environments.
These are proposed demos, not games currently shipped in FluidUse. Tetris, 2048,
and Snake are the starting examples from the discussion; Flappy Bird is the next
demo to build. The [model inventory](Models.md) identifies possible sub-1B models
and distinguishes text-state models from vision models.

## Demo list

| Priority | Game | Model's decision | What it reveals | Suggested state |
| --- | --- | --- | --- | --- |
| Next | **Flappy Bird** | `FLAP` or `COAST` each decision tick. | Reaction timing, latency, and costly one-step mistakes. | Bird height and velocity; next pipe distance and gap, or a rendered frame for a vision model. |
| Next | **Sokoban / Boxoban** | Pick a legal move or push. | Planning and irreversible traps. | Grid, player, crates, goals, and legal moves. |
| Next | **Mario-style platformer** | Choose left, right, jump, or a legal combination at each tick. | Jump timing, momentum, obstacle anticipation, and longer action sequences. | Player position and velocity, nearby platforms, enemies, and camera offset; or a rendered frame. |
| Next | **Chess** | Choose or rank legal moves. | Tactical judgment, position evaluation, and planning across turns. | FEN, side to move, legal moves, and remaining time; a board image only for vision models. |
| Next | **Minesweeper** | Open or flag a cell. | Decision-making under uncertainty. | Revealed grid and legal cells; show mine probabilities only if actually computed. |
| Next | **Wordle-style word game** | Choose the next valid guess. | Information gathering versus an immediate attempt to solve. | Previous guesses, color feedback, and a fixed candidate-word list. |
| Next | **Frogger-style crossing** | Move up, down, left, right, or wait. | Timing around moving hazards and choosing safe windows. | Player, lane hazards, speeds, and goal positions; or a rendered frame. |
| Next | **Subway Surfers-style runner** | Switch lane, jump, slide, or hold. | Fast obstacle recognition, action timing, and choosing between survival and coins. | Current lane, speed, nearby obstacles and distances; or a rendered frame. |
| Next | **Connect Four** | Choose a non-full column. | Short tactical lookahead with only a few legal actions. | Board, player to move, and legal columns. |
| Later | **Pac-Man** | Direction at each junction. | Reward versus moving hazards. | Map, player, ghosts, pellets, and power timer. |
| Later | **Codenames** | Choose a clue or a guess from a fixed set. | Semantic association under constraints. | Visible words, team, prior clues, and legal choices. |
| Later | **Hanabi** | Play, discard, or give a legal hint. | Cooperation with incomplete information. | Only what the acting player may observe. |
| Later | **MiniGrid DoorKey** | Turn, move, pick up, or open. | Multi-step planning when a key must be found before reaching the goal. | Partial grid observation, carried item, and door state. |
| Later | **FrozenLake** | Choose one of four directions. | Planning under uncertain movement when the lake is slippery. | Map, position, goal, holes, and known movement rules. |
| Later | **Battleship** | Choose an untried target square. | Search under hidden information and using feedback from earlier turns. | Hits, misses, sunk ships, and remaining legal squares. |
| Later | **Tower defense** | Place, upgrade, sell, or wait. | Resource allocation with delayed effects. | Map, waves, towers, budget, and legal placements. |
| Later | **MiniWoB-style web tasks** | Choose an element and operation. | Practical computer use with a clear success condition. | Accessibility element table and task goal; screenshots for vision models. |
| Later | **Breakout / Pong** | Move paddle left, right, or stay. | Ball prediction and control latency. | Ball/paddle positions and velocities. |
| Later | **Space Invaders** | Move, fire, or combine actions. | Avoidance and aiming under continuous pressure. | Player, projectiles, enemies, and cooldowns. |
| Later | **Racer** | Steer left, right, or straight. | Lookahead and fast corrections. | Track geometry, position, speed, and obstacles. |
| Existing example | **Tetris** | Choose a placement or movement. | Long-term board management. | Board, current piece, next piece if allowed, and legal placements. |
| Existing example | **2048** | Choose a legal slide. | Repeated choices with random future tiles. | Board and each legal resulting board before the random spawn. |
| Existing example | **Snake** | Choose direction. | Path planning while avoiding self-traps. | Grid, body, food, and current direction. |

Flappy Bird is the clearest first comparison: two actions, immediate outcomes,
and a visible cost when a decision arrives late. Sokoban adds planning, and
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
A **full-game demo** pairs models against the same opponent at the same clock
setting and starting positions. Record legal-move rate, move time, game result,
and engine evaluation loss per move. Supply legal moves from a chess rules engine
so the model is judged on choosing among them, not on formatting notation.

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
