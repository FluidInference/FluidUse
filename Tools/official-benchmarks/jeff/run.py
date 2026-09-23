"""Jeff's own accuracy benchmark through FluidUse (Jeff Core ML, bridge path), plus a native arm.

Official source: logan-markewich/jeff @ 34b32f99a727c47b679adde33f4702a001e02979, `bench/eval_accuracy.py` over the
committed `bench/data/*.jsonl` (8 tasks × 200). Published numbers use `--prompt-profile original`, variant `default`,
temperature 3.2 (`DEFAULT_TEMPERATURE`).

The unchanged `jeff.core.Engine` renders each System One request into GLiFormer classification groups and calls
`Backend.score(texts, groups)`, which returns raw per-label sigmoid scores. FluidUse stands in for that backend in
three steps: record the exact (text, group) units the engine sends, score them with FluidUse from Swift, then replay
them through the same engine, whose temperature decoding produces the answers. Rows follow `run_jeff` and are scored
with the unchanged `task_metrics`. Units FluidUse cannot take (L128, 1–8 labels, one group per pass) are kept as
unsupported rows and reported as coverage.

    <jeff env>/python Tools/official-benchmarks/jeff/run.py --jeff <checkout> --arm fluiduse --bench <FluidUseOfficialBench> --out <dir>
    <jeff env>/python Tools/official-benchmarks/jeff/run.py --jeff <checkout> --arm native --model <gliformer dir> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jeff", required=True, type=Path, help="logan-markewich/jeff checkout at the pinned commit")
    parser.add_argument("--arm", required=True, choices=["fluiduse", "native"])
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--precision")
    parser.add_argument("--model", help="gliformer-large-v1 directory for the native arm")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.jeff / "src"))
    spec = importlib.util.spec_from_file_location("jeff_eval_accuracy", args.jeff / "bench" / "eval_accuracy.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    from jeff.core import Engine, PromptOptions, SystemOneRequest
    from jeff.core.backend import ScoredText

    items = official.load_items(args.tasks, args.limit)
    opts = PromptOptions(**official.VARIANTS["default"])
    args.out.mkdir(parents=True, exist_ok=True)
    unsupported: dict[str, str] = {}
    meta: dict = {}

    def unit_key(text, groups):
        return json.dumps([text, [[g.key, list(g.labels), g.name, g.description] for g in groups]], ensure_ascii=False)

    if args.arm == "fluiduse":
        class Recorder:
            name = "recorder"

            def __init__(self):
                self.units = {}

            def score(self, texts, groups):
                out = []
                for text, gs in zip(texts, groups):
                    self.units.setdefault(unit_key(text, gs), (text, gs))
                    out.append(ScoredText(scores={g.key: [0.5] * len(g.labels) for g in gs}, input_tokens=0))
                return out

        recorder = Recorder()
        Engine(recorder, "fluiduse-jeff", opts).run_batch(
            [SystemOneRequest.model_validate(official.body_for(it, "fluiduse-jeff")) for it in items])
        plan = list(recorder.units.items())
        with (args.out / "requests.jsonl").open("w", encoding="utf-8") as handle:
            for _, (text, gs) in plan:
                if len(gs) != 1:
                    raise SystemExit("A unit with several groups needs a joint encoder pass FluidUse does not serve")
                (g,) = gs
                handle.write(json.dumps({"text": text, "labels": list(g.labels), "name": g.name or "",
                                         "description": g.description or ""}, ensure_ascii=False) + "\n")
        command = [str(args.bench), "run", "--model", "jeff", "--in", str(args.out / "requests.jsonl"),
                   "--out", str(args.out / "results.jsonl"), "--meta", str(args.out / "meta.json")]
        if args.precision:
            command += ["--precision", args.precision]
        subprocess.run(command, check=True)
        results = [json.loads(line) for line in (args.out / "results.jsonl").read_text().splitlines()]
        scores, latency = {}, {}
        for (key, (text, gs)), result in zip(plan, results):
            if result["status"] == "ok":
                scores[key] = result["answer"]["probabilities"]
                latency[key] = result["latency_ms"]
            else:
                unsupported[key] = result["error"]

        class Replay:
            name = "fluiduse"

            def score(self, texts, groups):
                return [ScoredText(scores={gs[0].key: scores[unit_key(text, gs)]}, input_tokens=0)
                        for text, gs in zip(texts, groups)]

        backend = Replay()
        meta = json.loads((args.out / "meta.json").read_text())
    else:
        from jeff.backends.torch_backend import TorchBackend

        backend = TorchBackend(args.model, device=args.device, batch_size=args.batch)
        meta = {"execution": "native PyTorch TorchBackend", **backend.info()}

    engine = Engine(backend, "jeff", opts)
    rows, skipped = [], []
    for item in items:
        request = SystemOneRequest.model_validate(official.body_for(item, "jeff"))
        if args.arm == "fluiduse":
            keys = [unit_key(t, gs) for t, gs in _units(request, opts)]
            failed = [unsupported[k] for k in keys if k in unsupported]
            if failed:
                skipped.append({"task": item["task"], "id": item["id"], "error": failed[0]})
                continue
            item_latency = sum(latency[k] for k in keys)
        start = time.perf_counter()
        (response,) = engine.run_batch([request])
        if args.arm == "native":
            item_latency = (time.perf_counter() - start) * 1000
        rows.append({"run": f"{args.arm}/original/default", "system": args.arm, "variant": "default",
                     "prompt_profile": "original", "temperature": engine.temperature, "task": item["task"],
                     "kind": item["kind"], "id": item["id"], "gold": item["gold"],
                     "answer": response.model_dump()["answers"]["q"], "latency_ms": round(item_latency, 2),
                     "batch": 1, "input_tokens": response.usage.input_tokens})

    with (args.out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    requested: dict[str, int] = {}
    for item in items:
        requested[item["task"]] = requested.get(item["task"], 0) + 1
    report = {"arm": args.arm, "source": "logan-markewich/jeff@34b32f99a727c47b679adde33f4702a001e02979",
              "prompt_profile": "original", "temperature": engine.temperature, "execution": meta,
              "tasks": {task: {"requested": requested[task], "answered": len(by_task.get(task, [])),
                               **(official.task_metrics(by_task[task]) if task in by_task else {})}
                        for task in sorted(requested)},
              "unsupported": skipped}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "unsupported"}, indent=2))


def _units(request, opts):
    """The (text, groups) units Engine.run_batch sends for one request, in its own order."""
    from jeff.core.groups import build_groups
    from jeff.core.state import serialize_state

    text = serialize_state(request.state, opts.state_format)
    units, shared = [], []
    for g in build_groups(request.questions, opts):
        if opts.isolated(request.questions[g.key]):
            units.append((text, [g]))
        else:
            shared.append(g)
    if shared:
        units.append((text, shared))
    return units


if __name__ == "__main__":
    main()
