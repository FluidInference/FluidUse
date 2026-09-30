"""Re-label logged decisions with a teacher checkpoint (on-policy distillation data).

Reads decision logs written by the harness (any decider), sends each logged request to the teacher and writes the
same record shape with the teacher's probabilities, so `train_student.py` can consume it unchanged.

    uv run ... python label_with_teacher.py --checkpoint <Intern-Decision-4B snapshot> \
        --logs "student-runs/decisions-*.jsonl" --out dataset-r2/teacher-labels.jsonl
"""
import argparse
import glob
import json
import time
from pathlib import Path

from decision_player import HFDecider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    teacher = HFDecider(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in open(args.out):
            row = json.loads(line)
            done.add((row["battle"], row["turn"], row["chosen_by_student"]))
    rows = [json.loads(line) for pattern in args.logs for path in sorted(glob.glob(pattern)) for line in open(path)]
    if args.limit:
        rows = rows[: args.limit]
    start = time.time()
    written = 0
    with open(args.out, "a") as out:
        for i, row in enumerate(rows):
            key = (row["battle"], row["turn"], row["chosen"])
            if key in done:
                continue
            result = teacher.decide(row["request"])
            probabilities = result["answers"]["action"]
            out.write(json.dumps({"battle": row["battle"], "turn": row["turn"], "request": row["request"],
                                  "probabilities": probabilities,
                                  "chosen": max(probabilities, key=probabilities.get), "chosen_by_student": row["chosen"],
                                  "student_probabilities": row["probabilities"], "tokens": result["tokens"],
                                  "bucket": 0, "ms": result["ms"]}) + "\n")
            written += 1
            if written % 200 == 0:
                out.flush()
                print(f"{written} labelled, {i + 1}/{len(rows)} read, {(time.time() - start) / 60:.1f} min", flush=True)
    agree = 0
    total = 0
    for line in open(args.out):
        row = json.loads(line)
        total += 1
        agree += row["chosen"] == row["chosen_by_student"]
    print(json.dumps({"labelled": total, "new": written, "student_agrees_with_teacher": agree / max(1, total),
                      "minutes": (time.time() - start) / 60}), flush=True)


if __name__ == "__main__":
    main()
