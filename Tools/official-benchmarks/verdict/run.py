"""Verdict's own held-out evaluation through FluidUse (Verdict Core ML, in-process Swift).

Official source: Heman10x-NGU/Verdict-open-jev @ 30f15564821626ca5c1ad5b2638c4eb7078787dd, `scripts/evaluate.py`,
data `data/real_banking_{cal,test}.jsonl` and `data/slice_*.jsonl`. The evaluated checkpoint
(`artifacts/v2/model.safetensors`, sha256 d2528239…) is the one FluidInference/verdict-coreml converts.

The unchanged `extract_logits_and_targets` builds every prompt with `build_model_input`; a stand-in model returns
the FluidUse logits for that exact prompt, so target indexing, padding, temperature fitting (`TemperatureCalibrator`
on the calibration split) and `compute_metrics` are all the author's code. Steps follow `evaluate_pipeline` 1–3.
Also reported: the per-K `calibrator.json` FluidUse ships, and row agreement with the author's `predictions_v2.jsonl`.

    <verdict venv>/python Tools/official-benchmarks/verdict/run.py --verdict <checkout> --bench <FluidUseOfficialBench> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SLICES = ["missing_option", "distant_oos", "cardinality_k3", "cardinality_k5", "cardinality_k9",
          "cardinality_k17", "cardinality_k25"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdict", required=True, type=Path, help="Verdict-open-jev checkout at the pinned commit")
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="records per file (smoke runs)")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.verdict))
    spec = importlib.util.spec_from_file_location("verdict_evaluate", args.verdict / "scripts" / "evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    import torch
    from core.calibration import TemperatureCalibrator
    from core.formatting import build_model_input

    files = {"cal": "data/real_banking_cal.jsonl", "test": "data/real_banking_test.jsonl"}
    files.update({name: f"data/slice_{name}.jsonl" for name in SLICES})
    records = {name: official.load_jsonl(args.verdict / path)[: args.limit] for name, path in files.items()}

    # One FluidUse request per record: the components extract_logits_and_targets renders.
    args.out.mkdir(parents=True, exist_ok=True)
    plan = [(name, r) for name, rs in records.items() for r in rs]
    with (args.out / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for _, r in plan:
            handle.write(json.dumps({"question": r["question"], "context": r["text"],
                                     "labels": [c["description"] for c in r["candidates"]]}, ensure_ascii=False) + "\n")
    command = [str(args.bench), "run", "--model", "verdict", "--lengths", "128,512",
               "--in", str(args.out / "requests.jsonl"), "--out", str(args.out / "results.jsonl"),
               "--meta", str(args.out / "meta.json")]
    if args.cache:
        command += ["--cache", str(args.cache)]
    subprocess.run(command, check=True)
    results = [json.loads(line) for line in (args.out / "results.jsonl").read_text().splitlines()]
    if len(results) != len(plan):
        raise SystemExit(f"{len(results)} results for {len(plan)} requests")

    by_prompt, unsupported = {}, {name: [] for name in records}
    for (name, r), result in zip(plan, results):
        prompt = build_model_input(r["question"], r["text"], [c["description"] for c in r["candidates"]])
        if result["status"] == "ok":
            by_prompt[prompt] = result["answer"]["logits"]
        else:
            unsupported[name].append({"id": r["id"], "error": result["error"]})

    class Batch(dict):
        def to(self, device):
            return self

    def tokenizer(prompts, **kwargs):
        return Batch(prompts=prompts)

    class Model:
        def eval(self):
            return self

        def __call__(self, prompts):
            rows = [by_prompt[p] + [0.0] * (25 - len(by_prompt[p])) for p in prompts]
            return type("Output", (), {"logits": torch.tensor(rows, dtype=torch.float32)})()

    def extract(name):
        answered = [r for r in records[name] if r["id"] not in {u["id"] for u in unsupported[name]}]
        return official.extract_logits_and_targets(Model(), tokenizer, answered, torch.device("cpu"))

    # evaluate_pipeline step 1: temperature on the calibration split.
    cal_logits, cal_targets, _ = extract("cal")
    calibrator = TemperatureCalibrator(model_id="verdict-open-jev-modernbert-151.4m",
                                       scope="restricted_5_candidate_selection", artifact_hash="fluiduse-coreml")
    calibrator.fit(cal_logits, cal_targets, valid_mask=torch.isfinite(cal_logits))
    # Step 2: held-out test, uncalibrated and calibrated.
    test_logits, test_targets, test_meta = extract("test")
    with torch.no_grad():
        calibrated_test = calibrator(test_logits)
    report = {"temperature": calibrator.temperature,
              "uncalibrated": official.compute_metrics(test_logits, test_targets, test_meta),
              "calibrated": official.compute_metrics(calibrated_test, test_targets, test_meta), "slices": {}}
    # Step 3: challenge slices with the fitted temperature.
    for name in SLICES:
        logits, targets, meta = extract(name)
        with torch.no_grad():
            metrics = official.compute_metrics(calibrator(logits), targets, meta)
        report["slices"][name] = {"samples": len(records[name]), "scored": len(meta), "accuracy": metrics["accuracy"],
                                  "brier": metrics["brier_score"], "ece": metrics["ece_equal_width"],
                                  "abstention_recall": metrics["abstention"]["recall"],
                                  "abstention_precision": metrics["abstention"]["precision"],
                                  "abstention_f1": metrics["abstention"]["f1_score"],
                                  "false_abstentions": metrics["abstention"]["false_abstentions"]}

    # FluidUse's shipped per-K calibrator on the same test logits.
    shipped = json.loads((args.verdict / "artifacts" / "calibrator.json").read_text())
    counts = [len(m["candidate_ids"]) for m in test_meta]
    scales = torch.tensor([shipped["per_k"].get(str(k), shipped["temperature"]) for k in counts]).unsqueeze(1)
    report["shipped_per_k_calibrated"] = official.compute_metrics(test_logits / scales, test_targets, test_meta)

    # Row agreement with the author's published calibrated test predictions.
    published = {json.loads(l)["id"]: json.loads(l) for l in
                 (args.verdict / "reports" / "v2" / "predictions_v2.jsonl").read_text().splitlines()}
    probs = torch.softmax(calibrated_test, dim=-1)
    ours = {m["id"]: m["candidate_ids"][int(probs[i].argmax())] for i, m in enumerate(test_meta)}
    shared = [i for i in ours if i in published]
    report["agreement_with_published_predictions"] = {
        "rows": len(shared), "same_prediction": sum(ours[i] == published[i]["predicted_id"] for i in shared),
        "differing_ids": [i for i in shared if ours[i] != published[i]["predicted_id"]][:50]}
    report["coverage"] = {name: {"records": len(records[name]), "unsupported": len(unsupported[name]),
                                 "examples": unsupported[name][:5]} for name in records}
    reference = json.loads((args.verdict / "reports" / "v2" / "evaluation_report_v2.json").read_text())
    report["reference"] = {"temperature": reference["temperature"],
                           "calibrated": {k: reference["calibrated"][k] for k in
                                          ("accuracy", "negative_log_likelihood", "brier_score", "ece_equal_width")},
                           "calibrated_abstention": {k: reference["calibrated"]["abstention"][k]
                                                     for k in ("recall", "precision")},
                           "slices": reference["slices"]}
    report["fluiduse"] = json.loads((args.out / "meta.json").read_text())
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    brief = {"temperature": report["temperature"], "reference_temperature": reference["temperature"]}
    for key in ("accuracy", "negative_log_likelihood", "brier_score", "ece_equal_width"):
        brief[key] = {"fluiduse": report["calibrated"][key], "reference": reference["calibrated"][key]}
    brief["abstention"] = {k: {"fluiduse": report["calibrated"]["abstention"][k],
                               "reference": reference["calibrated"]["abstention"][k]} for k in ("recall", "precision")}
    brief["agreement"] = {k: v for k, v in report["agreement_with_published_predictions"].items() if k != "differing_ids"}
    brief["unsupported"] = {k: v["unsupported"] for k, v in report["coverage"].items()}
    brief["startup_ms"] = report["fluiduse"]["startup_ms"]
    print(json.dumps(brief, indent=2, default=float))


if __name__ == "__main__":
    main()
