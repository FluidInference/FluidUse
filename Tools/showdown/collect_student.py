"""On-policy collection: the Core ML student plays the scripted players and itself; every request is logged so a
teacher can label the states the student actually reaches (`label_with_teacher.py`).

    uv run ... python collect_student.py --model-dir <student coreml dir> --checkpoint <merged student> \
        --battles 70 --self-play 40 --out student-runs
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from poke_env import LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from decision_player import CoreMLDecider, InternDecisionPlayer

BASELINES = {"random": RandomPlayer, "max_power": MaxBasePowerPlayer, "heuristics": SimpleHeuristicsPlayer}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--battles", type=int, default=70)
    ap.add_argument("--self-play", type=int, default=40)
    ap.add_argument("--opponents", nargs="+", choices=list(BASELINES), default=list(BASELINES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    decider = CoreMLDecider(args.model_dir, args.checkpoint)
    decider.warm()
    summary = {}
    for name in args.opponents:
        player = InternDecisionPlayer(decider, log_path=args.out / f"student-vs-{name}.jsonl",
                                      battle_format=args.format, server_configuration=LocalhostServerConfiguration)
        opponent = BASELINES[name](battle_format=args.format, server_configuration=LocalhostServerConfiguration)
        start = time.time()
        await player.battle_against(opponent, n_battles=args.battles)
        summary[name] = {"won": player.n_won_battles, "lost": player.n_lost_battles, "decisions": len(player.latencies),
                         "wall_s": round(time.time() - start)}
        print(name, json.dumps(summary[name]), flush=True)
    if args.self_play:
        a = InternDecisionPlayer(decider, log_path=args.out / "student-self-a.jsonl", battle_format=args.format,
                                 server_configuration=LocalhostServerConfiguration)
        b = InternDecisionPlayer(decider, log_path=args.out / "student-self-b.jsonl", battle_format=args.format,
                                 server_configuration=LocalhostServerConfiguration)
        start = time.time()
        await a.battle_against(b, n_battles=args.self_play)
        summary["self"] = {"a_won": a.n_won_battles, "b_won": b.n_won_battles,
                           "decisions": len(a.latencies) + len(b.latencies), "wall_s": round(time.time() - start)}
        print("self", json.dumps(summary["self"]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
