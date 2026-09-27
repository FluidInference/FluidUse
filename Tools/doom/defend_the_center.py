"""Headless ViZDoom defend_the_center check: GLiClass vs a hand-coded aimer on the same seeds.

The model never sees pixels. Code turns the labels buffer into text (health, ammo, each visible
monster's bearing and distance, whether it is on the crosshair); the policy picks attack / turn
left / turn right, and the engine holds that action for --tics tics.

    uv pip install vizdoom numpy
    swift build -c release --product GLiClassServe
    python Tools/doom/defend_the_center.py --policies aimer,random,gliclass,gliclass-described --seeds 20
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

import vizdoom as vzd

ACTIONS = ["attack", "turn left", "turn right"]
BUTTONS = {"attack": [0, 0, 1], "turn left": [1, 0, 0], "turn right": [0, 1, 0]}
PROMPT = (
    "You are playing Doom, standing in the middle of a round room. Monsters walk toward you. "
    "Shoot a monster only when it is on the crosshair; otherwise turn toward the nearest one. "
    "Choose the next action."
)
SCREEN_CENTER = 160


class GLiClassProcess:
    def __init__(self, binary, precision):
        self.proc = subprocess.Popen(
            [binary, "--precision", precision], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        ready = json.loads(self.proc.stdout.readline())
        if not ready.get("ready"):
            raise RuntimeError(f"GLiClassServe failed to start: {ready}")
        self.load_s = ready["load_s"]

    def classify(self, text, labels):
        self.proc.stdin.write(json.dumps({"text": text, "labels": labels, "prompt": PROMPT}) + "\n")
        response = json.loads(self.proc.stdout.readline())
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response["index"], response["probabilities"], response["ms"]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def observe(state):
    ammo, health, angle, px, py = state.game_variables[:5]
    monsters = []
    for label in state.labels:
        if label.object_name == "DoomPlayer":
            continue
        dx, dy = label.object_position_x - px, label.object_position_y - py
        bearing = (math.degrees(math.atan2(dy, dx)) - angle + 180) % 360 - 180  # + = left
        monsters.append(
            {
                "bearing": bearing,
                "distance": math.hypot(dx, dy),
                "on_crosshair": label.x <= SCREEN_CENTER <= label.x + label.width,
            }
        )
    monsters.sort(key=lambda m: m["distance"])
    return {"ammo": int(ammo), "health": int(health), "monsters": monsters}


def describe(obs):
    parts = [f"Health {obs['health']}, ammo {obs['ammo']}."]
    if not obs["monsters"]:
        parts.append("No monsters in view.")
    else:
        parts.append(f"{len(obs['monsters'])} monsters in view.")
        for i, m in enumerate(obs["monsters"][:3]):
            side = "left" if m["bearing"] > 0 else "right"
            aim = "on the crosshair" if m["on_crosshair"] else "not on the crosshair"
            name = "Nearest" if i == 0 else "Next"
            parts.append(f"{name}: {abs(m['bearing']):.0f} degrees {side}, {m['distance']:.0f} units away, {aim}.")
    return " ".join(parts)


def described_labels(obs):
    """Labels that state each action's consequence, like the Tetris and 2048 shortlists."""
    target = next((m for m in obs["monsters"] if m["on_crosshair"]), None)
    if obs["ammo"] == 0:
        attack = "attack: out of ammo, does nothing"
    elif target:
        attack = f"attack: shoots the monster on the crosshair {target['distance']:.0f} units away"
    else:
        attack = "attack: no monster on the crosshair, wastes a bullet"
    nearest = obs["monsters"][0] if obs["monsters"] else None
    if target:
        left = "turn left: moves the crosshair off the monster"
        right = "turn right: moves the crosshair off the monster"
    elif nearest is None:
        left, right = "turn left: search for monsters", "turn right: search for monsters"
    elif nearest["bearing"] > 0:
        left = f"turn left: toward the nearest monster {abs(nearest['bearing']):.0f} degrees left"
        right = "turn right: away from the nearest monster"
    else:
        left = "turn left: away from the nearest monster"
        right = f"turn right: toward the nearest monster {abs(nearest['bearing']):.0f} degrees right"
    return [attack, left, right]


def aimer(obs):
    if obs["ammo"] > 0 and any(m["on_crosshair"] for m in obs["monsters"]):
        return "attack"
    if obs["monsters"]:
        return "turn left" if obs["monsters"][0]["bearing"] > 0 else "turn right"
    return "turn left"


def make_game():
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)
    for variable in ("ANGLE", "POSITION_X", "POSITION_Y", "KILLCOUNT"):
        game.add_available_game_variable(getattr(vzd.GameVariable, variable))
    game.init()
    return game


def run_episode(game, seed, policy, tics, model, rng):
    game.set_seed(seed)
    game.new_episode()
    decisions, shots, wasted, disagreements, ms = 0, 0, 0, 0, []
    kills = 0
    while not game.is_episode_finished():
        state = game.get_state()
        obs = observe(state)
        kills = int(state.game_variables[5])
        if policy == "aimer":
            action = aimer(obs)
        elif policy == "random":
            action = rng.choice(ACTIONS)
        else:
            labels = described_labels(obs) if policy == "gliclass-described" else ACTIONS
            index, _, elapsed = model.classify(describe(obs), labels)
            action = ACTIONS[index]
            ms.append(elapsed)
        if action != aimer(obs):
            disagreements += 1
        if action == "attack" and obs["ammo"] > 0:
            shots += 1
            if not any(m["on_crosshair"] for m in obs["monsters"]):
                wasted += 1
        decisions += 1
        game.make_action(BUTTONS[action], tics)
    tics_alive = game.get_episode_time()
    died = game.is_player_dead()
    return {
        "seed": seed,
        "kills": kills,
        "tics": tics_alive,
        "died": died,
        "decisions": decisions,
        "shots": shots,
        "wasted_shots": wasted,
        "aimer_disagreement": disagreements / max(decisions, 1),
        "ms_median": statistics.median(ms) if ms else None,
        "ms_p95": sorted(ms)[int(0.95 * (len(ms) - 1))] if ms else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="aimer,random,gliclass,gliclass-described")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--tics", type=int, default=3, help="tics each decision is held (35 tics = 1 s)")
    parser.add_argument("--precision", default="lut8")
    parser.add_argument("--binary", default=".build/release/GLiClassServe")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    policies = args.policies.split(",")
    model = GLiClassProcess(args.binary, args.precision) if any(p.startswith("gliclass") for p in policies) else None
    if model:
        print(f"GLiClass {args.precision} loaded in {model.load_s:.1f} s", file=sys.stderr)
    game = make_game()
    results = {}
    for policy in policies:
        rng = random.Random(0)
        episodes = []
        started = time.time()
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            episodes.append(run_episode(game, seed, policy, args.tics, model, rng))
        results[policy] = episodes
        kills = [e["kills"] for e in episodes]
        line = (
            f"{policy:20s} kills {statistics.mean(kills):5.2f} (min {min(kills)}, max {max(kills)})"
            f"  survived {statistics.mean(e['tics'] for e in episodes) / 35:5.1f} s"
            f"  wasted {sum(e['wasted_shots'] for e in episodes) / max(sum(e['shots'] for e in episodes), 1):5.1%}"
            f"  vs aimer {statistics.mean(e['aimer_disagreement'] for e in episodes):5.1%}"
        )
        if episodes[0]["ms_median"] is not None:
            line += f"  {statistics.median(e['ms_median'] for e in episodes):.2f} ms/call"
        print(line + f"  ({time.time() - started:.0f} s wall)")
    game.close()
    if model:
        model.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tics": args.tics, "precision": args.precision, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
