"""Bounded CLI regression checks: python3 Tools/test_laya_cli.py <binary>."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


BINARY = sys.argv.pop(1)


class TetrisCommandTests(unittest.TestCase):
    def run_game(self, policy, shortlist):
        args = [BINARY, "tetris", "--policy", policy, "--seed", "7", "--pieces", "30", "--json"]
        if shortlist:
            args.append("--shortlist")
        result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=10)
        return json.loads(result.stdout)

    def test_filter_applies_to_both_control_policies(self):
        for policy in ("random", "heuristic"):
            with self.subTest(policy=policy):
                plain = self.run_game(policy, False)
                filtered = self.run_game(policy, True)
                self.assertFalse(plain["shortlist"])
                self.assertTrue(filtered["shortlist"])
                self.assertEqual(filtered["seed"], 7)
                self.assertEqual(filtered["decisions"], 0)
                self.assertEqual(plain["landings_after_filter_mean"], plain["landings_before_filter_mean"])
                self.assertLess(filtered["landings_after_filter_mean"], filtered["landings_before_filter_mean"])
                self.assertEqual(filtered["landings_offered_to_model_mean"], 0)
                self.assertNotEqual((plain["pieces"], plain["lines"]), (filtered["pieces"], filtered["lines"]))

    def test_invalid_lookahead_options_fail(self):
        for flag, value in (("--combine", "bad"), ("--lookahead", "-1"), ("--lookahead", "oops")):
            with self.subTest(flag=flag, value=value):
                result = subprocess.run(
                    [BINARY, "tetris", "--policy", "random", flag, value],
                    capture_output=True, text=True, timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)


class LayaBenchmarkTests(unittest.TestCase):
    def test_invalid_limit_is_rejected_before_loading_assets(self):
        for value in ("oops", "-1", "0"):
            result = subprocess.run(
                [BINARY, "benchmark", "--suites", "/missing-suite.jsonl", "--limit", value],
                capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--limit must be a positive integer", result.stdout + result.stderr)

    def test_empty_selection_is_rejected_before_loading_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "empty.jsonl"
            suite.write_text("")
            result = subprocess.run(
                [BINARY, "benchmark", "--suites", str(suite)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No benchmark rows selected", result.stdout + result.stderr)

    @unittest.skipUnless(os.environ.get("FLUIDUSE_LAYA_MODEL_DIR"), "requires a real local laya model")
    def test_failed_question_writes_incomplete_report_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "questions.jsonl"
            report = Path(directory) / "report.json"
            rows = [
                {"suite": "release-smoke", "index": index, "state": "The board is empty.",
                 "type": "noul", "instructions": instructions, "options": [], "gold": 1}
                for index, instructions in enumerate(("Is the board empty?", ""))
            ]
            suite.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = subprocess.run(
                [BINARY, "benchmark", "--suites", str(suite), "--report", str(report),
                 "--lengths", "128", "--model-dir", os.environ["FLUIDUSE_LAYA_MODEL_DIR"]],
                capture_output=True, text=True, timeout=60,
            )
            self.assertNotEqual(result.returncode, 0)
            data = json.loads(report.read_text())
            self.assertFalse(data["complete"])
            self.assertEqual(data["completed"], 1)
            self.assertEqual(data["dropped"], 1)
            self.assertEqual(data["questions"], 2)


if __name__ == "__main__":
    unittest.main()
