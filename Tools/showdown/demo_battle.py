"""Demo: the distilled Intern-Decision-0.8B (Core ML) plays Pokémon Showdown, one battle at a time, with a live
decision log meant for a terminal next to the battle in the browser.

    uv run ... python demo_battle.py --model-dir <student coreml dir> --checkpoint <merged student> \
        --opponent heuristics --battles 3 --delay 1.5

Spectate at  https://localhost.psim.us/<battle room>  (the official client served for a local server)  (the room is printed per battle).
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import random
import subprocess

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from decision_player import CoreMLDecider, HFDecider, InternDecisionPlayer, render_options, render_state, QUESTION

BASELINES = {"random": RandomPlayer, "max_power": MaxBasePowerPlayer, "heuristics": SimpleHeuristicsPlayer}
RED, DIM, BOLD, GREEN, YELLOW, RESET = "\033[31m", "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[0m"


class DemoPlayer(InternDecisionPlayer):
    def __init__(self, decider, delay: float, top: int, open_rooms: bool, label: str = "Intern-Decision-0.8B",
                 color: str = RED, **kwargs):
        super().__init__(decider, log_path=None, **kwargs)
        self.delay = delay
        self.top = top
        self.open_rooms = open_rooms
        self.label = label
        self.color = color
        self.announced = set()

    def choose_move(self, battle):
        if battle.battle_tag not in self.announced:
            self.announced.add(battle.battle_tag)
            url = f"https://localhost.psim.us/{battle.battle_tag}"
            print(f"\n{BOLD}▶ {battle.battle_tag}{RESET}  {DIM}watch: {url}{RESET}", flush=True)
            if self.open_rooms:
                # Chrome incognito when available (clean window for recording), else the default browser
                command = (["open", "-na", "Google Chrome", "--args", "--incognito", url]
                           if Path("/Applications/Google Chrome.app").exists() else ["open", url])
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(3)  # let the client join before the first move
        options = render_options(battle)
        if len(options) <= 1:
            return super().choose_move(battle)
        state = render_state(battle)
        request = {"state": state, "questions": {"action": {"type": "choice", "instructions": QUESTION,
                                                             "criteria": {label: text for label, text, _ in options}}}}
        try:
            result = self.decider.decide(request)
        except ValueError:
            return self.choose_random_move(battle)
        probabilities = result["answers"]["action"]
        ranked = sorted(options, key=lambda option: -probabilities[option[0]])
        best = ranked[0]
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        print(f"{BOLD}turn {battle.turn:>2}{RESET} {self.color}[{self.label}]{RESET} {me.species} {round(me.current_hp_fraction * 100)}%  vs  "
              f"{opp.species if opp else '?'} {round(opp.current_hp_fraction * 100) if opp else '?'}%", flush=True)
        for label, text, _ in ranked[: self.top]:
            p = probabilities[label]
            marker = f"{GREEN}◀{RESET}" if label == best[0] else " "
            bar = "█" * int(p * 20)
            print(f"   {marker} {p:5.1%} {YELLOW}{bar:<20}{RESET} {label:<24} {DIM}{text[:70]}{RESET}", flush=True)
        print(f"   {self.color}{self.label} · {result['ms']:.0f} ms · {result['tokens']} tokens · "
              f"bucket {result['bucket'] or 'torch'}{RESET}", flush=True)
        self.latencies.append(result["ms"])
        if self.delay:
            time.sleep(self.delay)
        return self.create_order(best[2])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--opponent", default="heuristics",
                    help="random | max_power | heuristics | base (untrained 0.8B Core ML dir) | teacher (4B via PyTorch)")
    ap.add_argument("--base-dir", type=Path, help="untrained 0.8B Core ML dir, for --opponent base")
    ap.add_argument("--base-checkpoint", type=Path, help="untrained 0.8B snapshot, for --opponent base")
    ap.add_argument("--teacher-checkpoint", type=Path, help="Intern-Decision-4B snapshot, for --opponent teacher")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--battles", type=int, default=10)
    ap.add_argument("--delay", type=float, default=4.0, help="seconds per turn so a viewer can follow")
    ap.add_argument("--no-open", action="store_true", help="do not open each battle room in the browser")
    ap.add_argument("--wait-for", type=Path, help="load, then wait until this file appears before the first battle")
    ap.add_argument("--top", type=int, default=5, help="options shown per turn")
    args = ap.parse_args()
    print(f"{DIM}loading {args.model_dir}{RESET}", flush=True)
    decider = CoreMLDecider(args.model_dir, args.checkpoint)
    decider.warm()
    suffix = random.randint(10, 99)  # a fresh login name per launch: stale names hang on the local server
    names = {"random": "Random bot", "max_power": "MaxPower bot", "heuristics": "Heuristic bot",
             "base": "Stock 0.8B", "teacher": "Teacher 4B"}
    player = DemoPlayer(decider, args.delay, args.top, not args.no_open, label="fine-tuned 0.8B · Core ML",
                        battle_format=args.format, server_configuration=LocalhostServerConfiguration,
                        account_configuration=AccountConfiguration(f"Finetuned 0.8B {suffix}", None))
    common = dict(battle_format=args.format, server_configuration=LocalhostServerConfiguration,
                  account_configuration=AccountConfiguration(f"{names[args.opponent]} {suffix}", None))
    if args.opponent == "base":
        base = CoreMLDecider(args.base_dir, args.base_checkpoint)
        base.warm()
        opponent = DemoPlayer(base, 0, args.top, False, label="stock 0.8B · Core ML", color=DIM, **common)
    elif args.opponent == "teacher":
        opponent = DemoPlayer(HFDecider(args.teacher_checkpoint), 0, args.top, False, label="teacher 4B · PyTorch",
                              color=DIM, **common)
    else:
        opponent = BASELINES[args.opponent](**common)
    print(f"{BOLD}fine-tuned Intern-Decision-0.8B vs {names[args.opponent]} · {args.format} · "
          f"{args.battles} battle{'s' if args.battles != 1 else ''}{RESET}", flush=True)
    if args.wait_for:
        args.wait_for.unlink(missing_ok=True)
        print(f"{GREEN}ready.{RESET} start with:  {BOLD}./go.sh{RESET}", flush=True)
        while not args.wait_for.exists():
            time.sleep(0.5)
        args.wait_for.unlink(missing_ok=True)
    for i in range(args.battles):
        won_before = player.n_won_battles
        await player.battle_against(opponent, n_battles=1)
        outcome = f"{GREEN}WON{RESET}" if player.n_won_battles > won_before else f"{RED}LOST{RESET}"
        latencies = sorted(player.latencies)
        print(f"\n{BOLD}battle {i + 1}: {outcome}{RESET}  record {player.n_won_battles}-{player.n_lost_battles}  ·  "
              f"decisions {len(latencies)}  ·  p50 {latencies[len(latencies) // 2]:.0f} ms", flush=True)
        if i + 1 < args.battles:
            time.sleep(6)
    print(json.dumps({"won": player.n_won_battles, "lost": player.n_lost_battles, "opponent": args.opponent}))


if __name__ == "__main__":
    asyncio.run(main())
