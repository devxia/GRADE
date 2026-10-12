import sys
import unittest
from unittest.mock import patch

from grade.inference import parse_args


class InferenceCLITests(unittest.TestCase):
    def test_target_cache_is_enabled_by_default(self) -> None:
        argv = [
            "grade-inference",
            "--target-seq",
            "ACDE",
            "--peptides",
            "peptides.fa",
            "--output-csv",
            "scores.csv",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertFalse(args.no_target_cache)

    def test_target_cache_can_be_disabled(self) -> None:
        argv = [
            "grade-inference",
            "--target-seq",
            "ACDE",
            "--peptides",
            "peptides.fa",
            "--output-csv",
            "scores.csv",
            "--no-target-cache",
            "--model-dir",
            "custom",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.no_target_cache)


if __name__ == "__main__":
    unittest.main()
