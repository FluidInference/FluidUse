"""Intern-Decision-0.8B (Core ML) vs poke-env's scripted players on a local Pokémon Showdown server.

    node pokemon-showdown start --no-security          # in a smogon/pokemon-showdown checkout
    uv run --no-project --python 3.12 --with poke-env --with torch==2.9.1 --with transformers==5.14.1 --with Pillow \
        --with safetensors --with coremltools==9.0 --with "numpy<2.3" python run_battles.py \
        --model-dir <hub snapshot dir> --checkpoint <Intern-Decision-0.8B snapshot> --battles 20
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from poke_env import LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from decision_player import CoreMLDecider, HeuristicDecider, HFDecider, InternDecisionPlayer

BASELINES = {"random": RandomPlayer, "max_power": MaxBasePowerPlayer, "heuristics": SimpleHeuristicsPlayer}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--battles", type=int, default=10)
    ap.add_argument("--opponents", nargs="+", choices=list(BASELINES), default=list(BASELINES))
    ap.add_argument("--units", default="cpu_gpu")
    ap.add_argument("--decider", choices=["coreml", "hf", "heuristic"], default="coreml")
    ap.add_argument("--tag", default="", help="name for the log files (e.g. 4b)")
    ap.add_argument("--permutations", type=int, default=1, help="average the model over K option orders")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.decider == "coreml":
        decider = CoreMLDecider(args.model_dir, args.checkpoint, args.units)
        decider.warm()
    elif args.decider == "hf":
        decider = HFDecider(args.checkpoint)
    else:
        decider = HeuristicDecider()
    report = {"format": args.format, "battles_per_opponent": args.battles, "decider": args.decider,
              "permutations": args.permutations, "results": {}}
    for name in args.opponents:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        player = InternDecisionPlayer(decider, log_path=args.out / f"decisions-{args.decider}{args.tag}-{name}-{stamp}.jsonl",
                                      permutations=args.permutations,
                                      battle_format=args.format, server_configuration=LocalhostServerConfiguration,
                                      max_concurrent_battles=1)
        opponent = BASELINES[name](battle_format=args.format, server_configuration=LocalhostServerConfiguration,
                                   max_concurrent_battles=1)
        start = time.time()
        await player.battle_against(opponent, n_battles=args.battles)
        latencies = sorted(player.latencies)
        result = {"won": player.n_won_battles, "lost": player.n_lost_battles, "tied": player.n_tied_battles,
                  "decisions": len(latencies), "p50_ms": latencies[len(latencies) // 2] if latencies else None,
                  "p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else None,
                  "mean_tokens": sum(player.tokens) / max(1, len(player.tokens)),
                  "max_tokens": max(player.tokens, default=0), "wall_s": time.time() - start}
        report["results"][name] = result
        print(name, json.dumps(result), flush=True)
    (args.out / f"report-{args.format}-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
