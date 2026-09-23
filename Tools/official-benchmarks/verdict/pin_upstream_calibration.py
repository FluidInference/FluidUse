"""Pin the author's serving-engine outputs for Verdict's calibration regression test.

Runs `core.engine_encoder.DecisionEngine` from Heman10x-NGU/Verdict-open-jev @ 30f15564 on the pinned checkpoint
directory (heman10x/rlcd-modernbert-151m @ 8af2496e: model.onnx, model.safetensors, calibrator.json). On CPU the
engine serves through its bundled FP32 ONNX session and auto-loads the shipped per-K `calibrator.json`. For each
query the fixture records the exact prompt, the raw candidate logits, the temperature the engine applied, and its
calibrated probabilities. Contexts come from the author's held-out test file; nothing is fitted.

    <verdict env>/python Tools/official-benchmarks/verdict/pin_upstream_calibration.py --verdict <checkout> \
        --checkpoint <snapshot dir> --out Tests/FluidUseTests/Fixtures/verdict-upstream-calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ABSTAIN = "__insufficient_evidence__"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdict", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.verdict))
    from core.engine_encoder import DecisionEngine
    from core.formatting import build_model_input, format_query
    from core.primitives import Choice, Level, Noul, Option, Score

    engine = DecisionEngine(str(args.checkpoint), device="cpu")
    if engine.ort_session is None:
        raise SystemExit("expected the engine's default CPU ONNX session")
    captured = []
    run = engine.ort_session.run

    def capture(outputs, inputs):
        result = run(outputs, inputs)
        captured.append(result[0])
        return result

    engine.ort_session.run = capture

    test = [json.loads(line) for line in (args.verdict / "data" / "real_banking_test.jsonl").read_text().splitlines()]
    k25 = [json.loads(line) for line in (args.verdict / "data" / "slice_cardinality_k25.jsonl").read_text().splitlines()]

    def choice(record, count):
        options = [c for c in record["candidates"] if c["id"] != ABSTAIN][:count]
        return Choice(id="q", question=record["question"],
                      options=[Option(id=c["id"], description=c["description"]) for c in options])

    queries = [
        ("choice-k3", test[0]["text"], choice(test[0], 2)),
        ("choice-k5", test[1]["text"], choice(test[1], 4)),
        ("choice-k8-fallback", k25[0]["text"], choice(k25[0], 7)),
        ("choice-k25", k25[1]["text"], choice(k25[1], 24)),
        ("noul-k3", test[2]["text"], Noul(id="q", proposition="The customer is asking about a card payment", semantics="conditional_on_sufficient_evidence_v2")),
        ("score-k4", test[3]["text"], Score(id="q", question="How urgent is the request?", levels=[
            Level(id="low", description="Not urgent", value=0), Level(id="mid", description="Somewhat urgent", value=1),
            Level(id="high", description="Very urgent", value=2.5)])),
    ]
    per_k = engine.calibrator.per_k
    cases = []
    for name, context, query in queries:
        captured.clear()
        result = engine.evaluate(context, [query]).results[0]
        text, labels, ids = format_query(context, query)
        prompt = build_model_input(query.question if query.kind in ("choice", "score") else "",
                                   context if query.kind in ("choice", "score") else text, labels)
        logits = [float(x) for x in captured[0][0][: len(ids)]]
        temperature = float(per_k[str(len(ids))]) if str(len(ids)) in per_k else engine.calibrator.temperature
        question = {"type": query.kind}
        if query.kind == "choice":
            question.update(question=query.question, options=[{"id": o.id, "description": o.description} for o in query.options])
        elif query.kind == "score":
            question.update(question=query.question,
                            levels=[{"id": l.id, "description": l.description, "value": l.value} for l in query.levels])
        else:
            question.update(proposition=query.proposition)
        cases.append({"name": name, "context": context, "query": question, "prompt": prompt, "candidate_ids": ids,
                      "logits": logits, "temperature": temperature,
                      "probabilities": [float(result.probabilities[i]) for i in ids]})
    calibrator = args.checkpoint / "calibrator.json"
    fixture = {"source": "Heman10x-NGU/Verdict-open-jev@30f15564821626ca5c1ad5b2638c4eb7078787dd core/engine_encoder.py",
               "checkpoint": "heman10x/rlcd-modernbert-151m@8af2496eb63c7fa66d7d234e1f62629380030eb4 (model.onnx, CPU)",
               "calibrator_sha256": hashlib.sha256(calibrator.read_bytes()).hexdigest(),
               "calibrator": json.loads(calibrator.read_text()), "cases": cases}
    args.out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    for case in cases:
        print(case["name"], len(case["candidate_ids"]), "T=", case["temperature"], [round(p, 4) for p in case["probabilities"]])


if __name__ == "__main__":
    main()
