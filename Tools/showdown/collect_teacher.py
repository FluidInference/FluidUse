"""Collect teacher decisions: an Intern-Decision checkpoint (the 4B) plays Showdown battles through the harness and
every request is logged with its probability vector. One shared engine serves both sides of self-play.

    uv run ... python collect_teacher.py --checkpoint <Intern-Decision-4B snapshot> --out dataset \
        --battles 60 --self-play 40
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from poke_env import LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from decision_player import HFDecider, InternDecisionPlayer

BASELINES = {"random": RandomPlayer, "max_power": MaxBasePowerPlayer, "heuristics": SimpleHeuristicsPlayer}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--battles", type=int, default=60, help="per scripted opponent")
    ap.add_argument("--self-play", type=int, default=40)
    ap.add_argument("--opponents", nargs="+", choices=list(BASELINES), default=list(BASELINES))
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    decider = HFDecider(args.checkpoint)
    summary = {}
    for name in args.opponents:
        player = InternDecisionPlayer(decider, log_path=args.out / f"teacher-vs-{name}{args.tag}.jsonl",
                                      battle_format=args.format, server_configuration=LocalhostServerConfiguration)
        opponent = BASELINES[name](battle_format=args.format, server_configuration=LocalhostServerConfiguration)
        start = time.time()
        await player.battle_against(opponent, n_battles=args.battles)
        summary[name] = {"won": player.n_won_battles, "lost": player.n_lost_battles, "decisions": len(player.latencies),
                         "wall_s": round(time.time() - start)}
        print(name, json.dumps(summary[name]), flush=True)
    if args.self_play:
        a = InternDecisionPlayer(decider, log_path=args.out / f"teacher-self-a{args.tag}.jsonl",
                                 battle_format=args.format, server_configuration=LocalhostServerConfiguration)
        b = InternDecisionPlayer(decider, log_path=args.out / f"teacher-self-b{args.tag}.jsonl",
                                 battle_format=args.format, server_configuration=LocalhostServerConfiguration)
        start = time.time()
        await a.battle_against(b, n_battles=args.self_play)
        summary["self"] = {"a_won": a.n_won_battles, "b_won": b.n_won_battles,
                           "decisions": len(a.latencies) + len(b.latencies), "wall_s": round(time.time() - start)}
        print("self", json.dumps(summary["self"]), flush=True)
    (args.out / f"summary{args.tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
