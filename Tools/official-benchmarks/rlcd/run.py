"""RLCD diagnostic suite through FluidUse (LFM2.5-350M-RLCD Core ML, bridge path).

Official source: notnotsamuel/LFM2.5-350M-RLCD @ deb589d803d141cabd158ef55f6617b128529f36 (`rlcd/tasks.py`,
`rlcd/benchmark.py`). Cases, schemas, warmups (2), repeats (3), and the seeded per-repeat shuffle follow
`rlcd.benchmark.run`; every output is scored with the unchanged `rlcd.benchmark.evaluate`. Only the constrained
method is converted, so the summary reports the constrained block computed as in `rlcd.benchmark.summarize`.

    python Tools/official-benchmarks/rlcd/run.py --rlcd <snapshot> --bench .build/release/FluidUseOfficialBench --out <dir>
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlcd", required=True, type=Path, help="snapshot containing rlcd/ at the pinned revision")
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.rlcd))
    from rlcd.benchmark import evaluate
    from rlcd.tasks import CASES

    cases = CASES[: args.limit] if args.limit else CASES
    plan = [("warmup", w, case) for w in range(args.warmups) for case in cases]
    for rep in range(args.repeats):
        order = list(cases)
        random.Random(42 + rep).shuffle(order)
        plan += [("measured", rep, case) for case in order]

    args.out.mkdir(parents=True, exist_ok=True)
    requests = args.out / "requests.jsonl"
    with requests.open("w", encoding="utf-8") as handle:
        for _, _, (_, schema, context, _) in plan:
            handle.write(json.dumps({"context": context, "schema": schema}, ensure_ascii=False) + "\n")
    command = [str(args.bench), "run", "--model", "lfm2-5-350m-rlcd", "--in", str(requests),
               "--out", str(args.out / "results.jsonl"), "--meta", str(args.out / "meta.json")]
    if args.cache:
        command += ["--cache", str(args.cache)]
    subprocess.run(command, check=True)

    results = [json.loads(line) for line in (args.out / "results.jsonl").read_text().splitlines()]
    if len(results) != len(plan):
        raise SystemExit(f"{len(results)} results for {len(plan)} requests")
    rows = []
    for (phase, rep, (case_id, schema, _, expected)), result in zip(plan, results):
        if phase != "measured":
            continue
        row = {"case": case_id, "repeat": rep, "method": "constrained", "status": result["status"],
               "latency_ms": result["latency_ms"]}
        if result["status"] == "ok":
            answer = result["answer"]
            row.update(text=answer["text"], branches=answer["branches"], model_calls=answer["model_calls"],
                       **evaluate(answer["text"], schema, expected))
        else:
            # A failed request has no output; score it as invalid JSON, as the official scorer would.
            row.update(error=result["error"], **evaluate("", schema, expected))
        rows.append(row)

    times = [r["latency_ms"] for r in rows]
    summary = {"n": len(rows), "failures": sum(r["status"] != "ok" for r in rows),
               "latency_mean_ms": statistics.mean(times), "latency_median_ms": statistics.median(times),
               "latency_p95_ms": float(np.percentile(times, 95)),
               "syntax_valid_rate": statistics.mean(r["syntax_valid"] for r in rows),
               "schema_compliant_rate": statistics.mean(r["schema_compliant"] for r in rows),
               "field_accuracy": sum(r["correct_fields"] for r in rows) / sum(r["field_count"] for r in rows),
               "exact_match_rate": statistics.mean(r["exact_match"] for r in rows)}
    # Row-level conversion check against the author's published native outputs (same cases, same schemas).
    agreement = {}
    for name in ("m2-max", "l40s", "h100"):
        published = args.rlcd / "results" / f"{name}.json"
        if not published.exists():
            continue
        native = {}
        for r in json.loads(published.read_text())["rows"]:
            if r["method"] == "constrained":
                native.setdefault(r["case"], json.loads(r["text"]))
        ours = {r["case"]: json.loads(r["text"]) for r in rows if r["status"] == "ok"}
        shared = sorted(set(native) & set(ours))
        agreement[name] = {"cases": len(shared), "identical_objects": sum(native[c] == ours[c] for c in shared),
                           "fields": sum(len(native[c]) for c in shared),
                           "identical_fields": sum(native[c].get(k) == v for c in shared for k, v in ours[c].items()),
                           "differing_cases": [c for c in shared if native[c] != ours[c]]}
    summary["agreement_with_published_native"] = agreement
    meta = json.loads((args.out / "meta.json").read_text())
    report = {"benchmark": "RLCD diagnostic suite (constrained method)",
              "source": "notnotsamuel/LFM2.5-350M-RLCD@deb589d803d141cabd158ef55f6617b128529f36",
              "reference": {"field_accuracy": 0.778, "exact_match_rate": 0.500,
                            "where": "results/REPORT.md (M2 Max, L40S, H100 identical)"},
              "protocol": {"warmups": args.warmups, "repeats": args.repeats, "seed": 42, "cases": len(cases)},
              "fluiduse": meta, "summary": summary, "rows": rows}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "startup_ms": meta["startup_ms"]}, indent=2))


if __name__ == "__main__":
    main()
