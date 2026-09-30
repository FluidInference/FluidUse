"""Memory footprint, latency and win rate of one Core ML variant playing Showdown battles.

Samples macOS `footprint` (phys_footprint) once a second while the harness plays, and reports the peak and the steady
state at the end. Note: phys_footprint excludes file-backed pages, and Core ML maps its weights from disk, so this
number is dominated by the harness itself (its fp32 embedding copy, Python, activations) and barely moves between fp16
and int8 packages; add the weight files that were touched for the whole picture, or measure the Swift runtime.

    uv run ... python measure_footprint.py --model-dir <variant dir> --checkpoint <merged student> --battles 15
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from poke_env import LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer

from decision_player import CoreMLDecider, InternDecisionPlayer


def footprint_bytes(pid: int) -> int | None:
    out = subprocess.run(["footprint", "--pid", str(pid), "-f", "bytes"], capture_output=True, text=True).stdout
    match = re.search(r"Footprint:\s+(\d+)", out)
    return int(match.group(1)) if match else None


class Sampler(threading.Thread):
    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid = pid
        self.samples: list[tuple[float, int]] = []
        self.stop = threading.Event()

    def run(self):
        start = time.time()
        while not self.stop.is_set():
            value = footprint_bytes(self.pid)
            if value:
                self.samples.append((time.time() - start, value))
            self.stop.wait(1.0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--battles", type=int, default=15)
    ap.add_argument("--opponent", default="max_power")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    sampler = Sampler(os.getpid())
    sampler.start()
    baseline = footprint_bytes(os.getpid())
    t0 = time.time()
    decider = CoreMLDecider(args.model_dir, args.checkpoint)
    decider.warm()
    loaded = footprint_bytes(os.getpid())
    load_s = time.time() - t0
    player = InternDecisionPlayer(decider, log_path=None, battle_format="gen9randombattle",
                                  server_configuration=LocalhostServerConfiguration)
    opponent = MaxBasePowerPlayer(battle_format="gen9randombattle", server_configuration=LocalhostServerConfiguration)
    t1 = time.time()
    await player.battle_against(opponent, n_battles=args.battles)
    battle_s = time.time() - t1
    sampler.stop.set()
    sampler.join()
    final = footprint_bytes(os.getpid())
    latencies = sorted(player.latencies)
    peak = max(v for _, v in sampler.samples)
    weights = sum(f.resolve().stat().st_size for f in Path(args.model_dir).resolve().rglob("weight.bin"))
    report = {
        "variant": args.model_dir.name, "weights_on_disk_gb": round(weights / 1e9, 2),
        "footprint_gb": {"before_load": round(baseline / 1e9, 2), "after_load_and_warm": round(loaded / 1e9, 2),
                         "peak": round(peak / 1e9, 2), "end": round(final / 1e9, 2)},
        "load_and_warm_s": round(load_s, 1), "battles": args.battles, "won": player.n_won_battles,
        "lost": player.n_lost_battles, "decisions": len(latencies),
        "p50_ms": round(latencies[len(latencies) // 2], 1), "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "mean_tokens": round(sum(player.tokens) / len(player.tokens)), "battle_s": round(battle_s, 1),
    }
    print(json.dumps(report), flush=True)
    if args.out:
        args.out.write_text(json.dumps({**report, "samples": sampler.samples}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
