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
| Next | **Minesweeper** | Open or flag a cell. | Decision-making under uncertainty. | Revealed grid and legal cells; show mine probabilities only if actually computed. |
| Later | **Pac-Man** | Direction at each junction. | Reward versus moving hazards. | Map, player, ghosts, pellets, and power timer. |
| Later | **Codenames** | Choose a clue or a guess from a fixed set. | Semantic association under constraints. | Visible words, team, prior clues, and legal choices. |
| Later | **Hanabi** | Play, discard, or give a legal hint. | Cooperation with incomplete information. | Only what the acting player may observe. |
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
