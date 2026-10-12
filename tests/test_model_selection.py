from contextlib import redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from grade.cli import parse_args
from grade import inference


BASE = ["--target-seq", "ACDE", "--peptides", "peptides.fa", "--output-csv", "scores.csv"]
ROOT = Path(__file__).resolve().parents[1]


class ModelSelectionTests(unittest.TestCase):
    def parse_error(self, extra):
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as caught:
            parse_args(BASE + extra)
        self.assertEqual(caught.exception.code, 2)
        return error.getvalue()

    def test_default_paper(self):
        args = parse_args(BASE)
        self.assertEqual(args.model, "GRADE-paper")
        self.assertIsNone(args.model_dir)
        self.assertEqual(args.mode, "cloud")

    def test_explicit_paper(self):
        self.assertEqual(vars(parse_args(BASE)), vars(parse_args(BASE + ["--model", "GRADE-paper"])))

    def test_custom_directory_is_not_labelled_paper(self):
        for directory in ("custom/checkpoint", "models/grade-paper"):
            args = parse_args(BASE + ["--model-dir", directory])
            self.assertIsNone(args.model)
            self.assertEqual(args.model_dir, directory)

    def test_explicit_options_are_mutually_exclusive(self):
        for extra in (
            ["--model", "GRADE-paper", "--model-dir", "custom"],
            ["--model-dir", "custom", "--model", "GRADE-paper"],
        ):
            self.assertIn("not allowed with argument", self.parse_error(extra))

    def test_unknown_and_future_models_rejected(self):
        for model in ("GRADE-base", "GRADE-medium", "", "unknown"):
            self.assertIn("invalid choice", self.parse_error(["--model", model]))

    def test_existing_options_and_validation(self):
        args = parse_args(BASE + ["--model-dir", "custom", "--no-resume", "--no-target-cache", "--top-k", "2", "--output-format", "json"])
        self.assertTrue(args.no_resume)
        self.assertTrue(args.no_target_cache)
        self.assertEqual(args.top_k, 2)
        self.assertEqual(args.output_format, "json")
        self.assertFalse(parse_args(BASE).no_target_cache)
        for extra in (["--checkpoint-interval", "0"], ["--top-k", "0"], ["--fp16", "--bf16"], ["--target-name", "x"]):
            self.parse_error(extra)
        self.assertEqual(parse_args(BASE[:-1] + ["scores.tsv"]).output_format, "tsv")

    def test_reexport_and_sys_argv_compatibility(self):
        self.assertIs(inference.parse_args, parse_args)
        with patch.object(sys, "argv", ["grade-inference"] + BASE):
            self.assertEqual(inference.parse_args().model, "GRADE-paper")

    def test_actual_module_help_without_runtime_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-S", "-B", "-m", "grade.inference", "--help"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model {GRADE-paper}", result.stdout)
        self.assertIn("--model-dir", result.stdout)
        self.assertNotIn("--num-layers", result.stdout)
        self.assertNotIn("--num-hidden", result.stdout)

    def test_real_imports_do_not_load_heavy_modules(self):
        result = subprocess.run(
            [sys.executable, "-S", "-B", "-c",
             "import sys; import grade.model_registry, grade.cli, grade.resume, grade.inference; "
             "assert not any(n in sys.modules for n in ('torch', 'transformers', 'tqdm'))"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
