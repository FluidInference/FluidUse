"""Collect Kev and Jeff runs into the report sections.

Sections: official full-suite results (answered-only as the official scorer computes; the Kev full-denominator figure,
which Kev's scorer says rejected records must count as, labelled separately), coverage and rejection reasons,
native vs Core ML on identical answered rows, and Kev's shortened-input runs labelled as an adapted protocol.

    python3 Tools/official-benchmarks/summarize.py <runs dir>
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def kev_predictions(run):
    out = {}
    path = run / "predictions.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        record = json.loads(line)
        for row in record["rows"]:
            out[(row["id"], row["question"])] = row
    return out


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def kev_compare(a, b):
    shared = sorted(set(a) & set(b))
    clean = [k for k in shared if a[k]["variant"] == "clean" and a[k]["source"] != "unknowable"]
    same = sum(argmax(a[k]["p"]) == argmax(b[k]["p"]) for k in shared)
    acc = lambda side: sum(argmax(side[k]["p"]) == side[k]["label"] for k in clean) / len(clean) if clean else None
    delta = max((max(abs(x - y) for x, y in zip(a[k]["p"], b[k]["p"])) for k in shared), default=None)
    return {"shared_questions": len(shared), "argmax_agreement": same / len(shared) if shared else None,
            "clean_knowable": len(clean), "accuracy_a": acc(a), "accuracy_b": acc(b), "max_abs_probability_delta": delta}


def reasons(run):
    rejected = load(run / "rejected.json") or []
    counter = collections.Counter()
    for entry in rejected:
        match = re.search(r"needs (\d+) tokens", entry["error"])
        if match:
            counter["request longer than 128 tokens (no truncation)"] += 1
        elif "options" in entry["error"] or "capacity is" in entry["error"]:
            counter["more than 32 options"] += 1
        else:
            counter[entry["error"][:80]] += 1
    return dict(counter)


def main() -> None:
    runs = Path(sys.argv[1])
    report = {"kev": {}, "jeff": {}}
    for model, suites in (("kev06", ("decision-v7", "transfer-v4")), ("kev05", ("transfer-v4",))):
        precisions = ("fp16", "w8") if model == "kev06" else ("fp16", "e8")
        for suite in suites:
            base = f"{model}-{suite}"
            native = kev_predictions(runs / f"{base}-native")
            split = kev_predictions(runs / f"{base}-native-split-matched")
            entry = {"native_full": load(runs / f"{base}-native" / "summary.json")}
            for precision in precisions:
                for arm in ("strict", "runtime"):
                    run = runs / f"{base}-{arm}-{precision}"
                    summary = load(run / "summary.json")
                    if summary is None:
                        continue
                    ours = kev_predictions(run)
                    label = f"{arm}-{precision}" if arm == "strict" else f"ADAPTED shortened-state {precision}"
                    entry[label] = {"summary": summary, "rejections": reasons(run),
                                    "vs_native_packed_same_rows": kev_compare(ours, native),
                                    "vs_native_one_question_per_call_same_rows": kev_compare(ours, split)}
            entry["native_split_vs_packed_same_rows"] = kev_compare(split, native)
            report["kev"][base] = entry
    for precision in ("fp16", "w8"):
        run = load(runs / f"jeff-{precision}" / "report.json")
        native = load(runs / "jeff-native" / "report.json")
        if run is None or native is None:
            continue
        ours = {json.loads(l)["id"]: json.loads(l) for l in (runs / f"jeff-{precision}" / "rows.jsonl").read_text().splitlines()}
        theirs = {json.loads(l)["id"]: json.loads(l) for l in (runs / "jeff-native" / "rows.jsonl").read_text().splitlines()}
        shared = sorted(set(ours) & set(theirs))

        def decision(row):
            answer = row["answer"]
            if "choice" in answer:
                return answer["choice"]
            if "noul" in answer:
                return answer["noul"] >= 0.5
            return max(answer["probabilities"], key=answer["probabilities"].get)

        report["jeff"][precision] = {
            "tasks": run["tasks"], "native_tasks": native["tasks"], "unsupported": collections.Counter(
                u["task"] for u in run["unsupported"]),
            "same_rows": len(shared), "decision_agreement": sum(decision(ours[i]) == decision(theirs[i]) for i in shared)}
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
