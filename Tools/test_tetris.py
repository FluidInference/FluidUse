"""Bounded, model-free CLI regression checks: python3 Tools/test_tetris.py <binary>."""

import json
import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()
