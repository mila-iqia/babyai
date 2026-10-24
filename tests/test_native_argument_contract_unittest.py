"""Native CLI argument validation tests; no RL environment or agent is started."""

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "babyai" / "arguments.py"
SPEC = importlib.util.spec_from_file_location("native_babyai_arguments", SOURCE)
ARGS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARGS)


class NativeArgumentContractTests(unittest.TestCase):
    def parse(self, *args):
        with patch.object(sys, "argv", ["babyai-train", *args]):
            return ARGS.ArgumentParser().parse_args()

    def assert_invalid(self, name, value):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                self.parse(name, str(value))
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(name, stderr.getvalue())

    def test_positive_defaults_accepted(self):
        args = self.parse()
        self.assertGreater(args.procs, 0)
        self.assertGreater(args.frames_per_proc, 0)
        self.assertGreater(args.val_episodes, 0)

    def test_positive_explicit_counts_accepted(self):
        args = self.parse("--procs", "2", "--frames-per-proc", "8", "--val-episodes", "7")
        self.assertEqual((args.procs, args.frames_per_proc, args.val_episodes), (2, 8, 7))

    def test_zero_process_count_rejected(self):
        self.assert_invalid("--procs", 0)

    def test_negative_frame_count_rejected(self):
        self.assert_invalid("--frames-per-proc", -1)

    def test_zero_validation_episodes_rejected(self):
        self.assert_invalid("--val-episodes", 0)


if __name__ == "__main__":
    unittest.main()
