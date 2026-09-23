"""Kev's frozen benchmark suites through FluidUse (Kev 0.5B / 0.6B Core ML, bridge path), plus native arms.

Official source: jaredpalmer/kev @ 90990a5fac2995b9faa3190f7d437e84f2067768 (the revision the Core ML runtime pins;
suite files are identical to 37cb2f61, which produced the published scores). Records come from `kev.suite.load_split`
and are scored by the unchanged `kev.benchmark.evaluate_records` (`prediction_rows`, `summarize`, `kev.metrics`).

Arms (`--arm`):
  fluiduse-strict   FluidUse Core ML; one request per question (the packages accept one); a request that would need
                    state shortening or more than 32 options is rejected and counted in coverage, never truncated.
  fluiduse-runtime  FluidUse Core ML with the published runtime's behavior (shortens state to fit L128).
  native            The pinned PyTorch checkpoint through the official `LocalPredictor` (packed questions).
  native-split      The same checkpoint, one question per call, to measure the request adaptation alone.
`--only-ids` restricts any arm to records another run evaluated (matched-subset comparisons).

Rejected records are listed by the official scorer; its report states they are excluded from `clean` and must be
counted as wrong in any full-denominator headline, which `summary.json` adds explicitly.

    <kev env>/python Tools/official-benchmarks/kev/run.py --kev <checkout> --suite evals/v7/decision-v7 --model kev-0.6b \
        --arm fluiduse-strict --bench .build/release/FluidUseOfficialBench --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODELS = {"kev-0-5b": ("kev-0.5b", "jaredpalmer/kev-0.5b", "9ce2fd39db3a397c89733f94af948e3d1fdfffcd"),
          "kev-0.6b": ("kev-0.6b", "jaredpalmer/kev-0.6b", "dece6dba")}
# Exact rejection messages from the worker's strict check and the published runtime's packer.
CAPACITY = ("without truncation", "does not fit length", "; capacity is ")


def question_requests(record, api_request, model_name):
    request = api_request(record)
    return [{"model": model_name, "state": request["state"], "questions": {qid: q}}
            for qid, q in request["questions"].items()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kev", required=True, type=Path, help="jaredpalmer/kev checkout at the pinned commit")
    parser.add_argument("--suite", required=True, help="suite directory relative to the checkout")
    parser.add_argument("--split", default="development", choices=["development", "test"])
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--precision")
    parser.add_argument("--arm", required=True, choices=["fluiduse-strict", "fluiduse-runtime", "native", "native-split"])
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--native-run", help="checkpoint path or Hub id for native arms")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--only-ids", type=Path, help="predictions.jsonl of another run; evaluate those records only")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.kev))
    from kev.benchmark import evaluate_records
    from kev.data import api_request
    from kev.suite import load_split, read_manifest

    suite = args.kev / args.suite
    records = load_split(suite, args.split, allow_test=args.split == "test")
    if args.only_ids:
        keep = {json.loads(line)["id"] for line in args.only_ids.read_text().splitlines()}
        records = [r for r in records if r["_meta"]["id"] in keep]
    if args.limit:
        records = records[: args.limit]
    heldout = tuple(read_manifest(suite)["holdout_sources"])
    model_name = MODELS[args.model][0]
    work = args.out.with_name(args.out.name + "-work")
    work.mkdir(parents=True, exist_ok=True)

    if args.arm.startswith("fluiduse"):
        plan = [(r["_meta"]["id"], q) for r in records for q in question_requests(r, api_request, model_name)]
        with (work / "requests.jsonl").open("w", encoding="utf-8") as handle:
            for _, request in plan:
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
        command = [str(args.bench), "run", "--model", args.model, "--in", str(work / "requests.jsonl"),
                   "--out", str(work / "results.jsonl"), "--meta", str(work / "meta.json")]
        if args.precision:
            command += ["--precision", args.precision]
        if args.arm == "fluiduse-strict":
            command.append("--strict-context")
        subprocess.run(command, check=True)
        results = [json.loads(line) for line in (work / "results.jsonl").read_text().splitlines()]
        answers = {}
        for (record_id, request), result in zip(plan, results):
            answers.setdefault(record_id, []).append((request, result))

        def predictor(record):
            probabilities, latency, tokens = {}, 0.0, 0
            for request, result in answers[record["_meta"]["id"]]:
                if result["status"] != "ok":
                    error = result["error"]
                    if any(marker in error for marker in CAPACITY):
                        # evaluate_records(skip_overlong=True) counts these as rejected records.
                        raise ValueError(f"FluidUse package capacity: request exceeds it ({error})")
                    raise RuntimeError(error)
                answer = result["answer"]
                (qid,) = request["questions"]
                probabilities[qid] = dict(zip(answer["option_keys"], answer["probabilities"]))
                latency += result["latency_ms"]
                tokens += answer["usage"]["input_tokens"]
            return {"probabilities": probabilities, "latency_ms": latency, "input_tokens": tokens}

        predictor.temperature = 1.0
    else:
        from kev.checkpoint import LoadOptions
        from kev.predictors import LocalPredictor

        local = LocalPredictor(args.native_run or MODELS[args.model][1], args.device, LoadOptions.from_env())
        if args.arm == "native":
            predictor = local
        else:
            def predictor(record):
                merged = {"probabilities": {}, "logits": {}, "latency_ms": 0.0, "input_tokens": 0}
                for qid, question in record["questions"].items():
                    part = local({**record, "questions": {qid: question}})
                    for key in ("probabilities", "logits"):
                        merged[key].update(part[key])
                    merged["latency_ms"] += part["latency_ms"]
                    merged["input_tokens"] += part["input_tokens"]
                merged["inference_temperature"] = local.temperature
                return merged

            predictor.temperature = local.temperature

    report, rows = evaluate_records(records, predictor, args.out, heldout_sources=heldout, skip_overlong=True)
    coverage = report["coverage"]
    knowable = [(r, q) for r in records for q in r["questions"]
                if r["_meta"]["variant"] == "clean" and r["_meta"]["source"] != "unknowable"]
    correct = sum(1 for row in rows if row["variant"] == "clean" and row["source"] != "unknowable"
                  and max(range(len(row["p"])), key=row["p"].__getitem__) == row["label"])
    summary = {"arm": args.arm, "model": args.model, "precision": args.precision, "suite": args.suite,
               "split": args.split, "coverage": coverage,
               "official_clean_answered": {k: report["clean"][k] for k in ("n", "acc", "nll", "ece", "brier")},
               "clean_knowable_requested": len(knowable), "clean_knowable_correct": correct,
               "accuracy_full_denominator": correct / len(knowable) if knowable else None,
               "latency_ms": report["latency_ms"]}
    if (work / "meta.json").exists():
        summary["fluiduse"] = json.loads((work / "meta.json").read_text())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
