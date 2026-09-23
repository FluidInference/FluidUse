"""Verdict's held-out data through the app path, `VerdictManager.answer` (not the author's evaluation protocol).

`answer` renders each option as "It is {description}", always appends the trained abstention option last, and
calibrates with the per-K `calibrator.json` it ships. The author's evaluation instead passes raw descriptions with
`insufficient evidence` at the dataset's position and fits one temperature on the calibration split. This script
measures the app path on the same records and separates three effects:

  shipped      answer()'s own calibrated probabilities (what an app receives)
  uncalibrated the same logits at T = 1
  refit        one temperature fitted on the calibration split with the author's `TemperatureCalibrator`

Metrics are the author's unchanged `compute_metrics`. Targets: the gold candidate id, or the abstention id for
out-of-scope records.

    <verdict env>/python Tools/official-benchmarks/verdict/answer_path.py --verdict <checkout> --bench <runner> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

ABSTAIN = "__insufficient_evidence__"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdict", required=True, type=Path)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(args.verdict))
    spec = importlib.util.spec_from_file_location("verdict_evaluate", args.verdict / "scripts" / "evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    import torch
    from core.calibration import TemperatureCalibrator

    records = {name: official.load_jsonl(args.verdict / f"data/real_banking_{name}.jsonl")[: args.limit]
               for name in ("cal", "test")}
    plan = [(name, r) for name, rs in records.items() for r in rs]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for _, r in plan:
            options = [{"id": c["id"], "description": c["description"]} for c in r["candidates"] if c["id"] != ABSTAIN]
            handle.write(json.dumps({"context": r["text"], "question": {
                "type": "choice", "question": r["question"], "options": options}}, ensure_ascii=False) + "\n")
    subprocess.run([str(args.bench), "run", "--model", "verdict", "--lengths", "128,512",
                    "--in", str(args.out / "requests.jsonl"), "--out", str(args.out / "results.jsonl"),
                    "--meta", str(args.out / "meta.json")], check=True)
    results = [json.loads(line) for line in (args.out / "results.jsonl").read_text().splitlines()]

    def tensors(name):
        logits, calibrated, targets, meta = [], [], [], []
        for (split, r), result in zip(plan, results):
            if split != name or result["status"] != "ok":
                continue
            answer = result["answer"]
            ids = answer["candidate_ids"]
            target = r["target_id"] if r["target_id"] in ids else ABSTAIN
            logits.append(answer["logits"])
            calibrated.append([math.log(max(p, 1e-12)) for p in answer["probabilities"]])
            targets.append(ids.index(target))
            meta.append({"id": r["id"], "text": r["text"], "target_id": target, "candidate_ids": ids,
                         "is_abstention": r.get("is_abstention", False),
                         "abstention_subtype": r.get("abstention_subtype", "unknown")})
        width = max(len(row) for row in logits)
        pad = lambda rows: torch.tensor([row + [float("-inf")] * (width - len(row)) for row in rows])
        return pad(logits), pad(calibrated), torch.tensor(targets), meta

    cal_logits, _, cal_targets, _ = tensors("cal")
    test_logits, test_shipped, test_targets, test_meta = tensors("test")
    calibrator = TemperatureCalibrator(model_id="verdict-answer-path", scope="answer-path", artifact_hash="fluiduse")
    calibrator.fit(cal_logits, cal_targets, valid_mask=torch.isfinite(cal_logits))
    with torch.no_grad():
        refit = calibrator(test_logits)
    keys = ("accuracy", "negative_log_likelihood", "brier_score", "ece_equal_width")
    report = {"path": "VerdictManager.answer (It is … rendering, abstention appended last, shipped per-K calibrator)",
              "refit_temperature": calibrator.temperature}
    for label, tensor in (("shipped", test_shipped), ("uncalibrated", test_logits), ("refit", refit)):
        metrics = official.compute_metrics(tensor, test_targets, test_meta)
        report[label] = {**{k: metrics[k] for k in keys},
                         "abstention": {k: metrics["abstention"][k] for k in ("recall", "precision")}}
    report["unsupported"] = sum(r["status"] != "ok" for r in results)
    report["fluiduse"] = json.loads((args.out / "meta.json").read_text())
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
