"""Unified user-facing inference contracts; no GPU, credentials or network required."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from grade import cloud, inference, runner
from grade.cli import parse_args
from grade.inputs import load_peptides
from scripts.sync_cloud_skill import check_sync

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "synthetic-client-token-" + "x" * 24


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.target = self.root / "target.fa"
        self.target.write_text(">target with full header\n a c d \n")
        self.peptides = self.root / "peptides.fa"
        self.peptides.write_text(">low\n e f g\n>high\nKLM\n")
        self.output = self.root / "scores.csv"
        self.config = self.root / "client.json"
        self.stack.enter_context(patch.object(cloud, "config_path", return_value=self.config))
        self.stack.enter_context(patch.dict(os.environ, {"GRADE_EMAIL": "", "WORLD_SIZE": "1"}))
        self.stderr, self.stdout = io.StringIO(), io.StringIO()
        self.stack.enter_context(contextlib.redirect_stderr(self.stderr))
        self.stack.enter_context(contextlib.redirect_stdout(self.stdout))
        self.identity = self.stack.enter_context(patch.object(cloud, "client_identity", return_value={"client_token": TOKEN}))
        self.api = self.stack.enter_context(patch.object(cloud, "call_api", return_value=[
            {"peptide_name": "low", "sequence": "EFG", "score": 0.1},
            {"peptide_name": "high", "sequence": "KLM", "score": 0.9},
        ]))
        self.local = Mock()

    def args(self, *extra):
        return parse_args(["--target-fasta", str(self.target), "--peptides", str(self.peptides),
                           "--output", str(self.output), *extra])

    def run_cloud(self, *extra):
        args = self.args("--email", "user@example.org", *extra)
        runner.run(args, self.local)
        self.local.assert_not_called()
        return args

    def test_cloud_default_preserves_target_name_and_six_column_output(self):
        self.run_cloud()
        self.api.assert_called_once_with("ACD", [
            {"name": "low", "sequence": "EFG"}, {"name": "high", "sequence": "KLM"},
        ], email="user@example.org", client_token=TOKEN, model="GRADE-paper")
        rows = list(csv.DictReader(io.StringIO(self.output.read_text())))
        self.assertEqual(list(rows[0]), ["rank", "target", "peptide_name", "src", "sequence", "score"])
        self.assertEqual([row["peptide_name"] for row in rows], ["high", "low"])
        self.assertEqual(rows[0]["target"], "target with full header")
        self.assertEqual(rows[0]["src"], "grade-api")
        self.assertIn("Using Cloud inference: GRADE-paper", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")

    def test_cloud_formats_source_and_topk_keep_full_submission(self):
        for fmt in ("csv", "tsv", "json"):
            self.run_cloud("--output-format", fmt, "--top-k", "1", "--source-label", "experiment")
            text = self.output.read_text()
            rows = json.loads(text) if fmt == "json" else list(csv.DictReader(io.StringIO(text), delimiter="\t" if fmt == "tsv" else ","))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["peptide_name"], "high")
            self.assertEqual(rows[0]["src"], "experiment")
            self.assertEqual(len(self.api.call_args.args[1]), 2)

    def test_target_selection_and_input_failures_never_register(self):
        for text in (">a\nACD\n>b\nEFG\n", ">a\nACD\n>a\nEFG\n", ">a\n \n"):
            self.target.write_text(text)
            with self.assertRaises(SystemExit):
                self.run_cloud()
        self.identity.assert_not_called()
        self.api.assert_not_called()

    def test_explicit_target_header_resolved_for_cloud(self):
        self.target.write_text(">first\nACD\n>second full header\nYTV\n")
        self.run_cloud("--target-name", "second full header")
        self.assertEqual(self.api.call_args.args[0], "YTV")
        self.assertEqual(list(csv.DictReader(io.StringIO(self.output.read_text())))[0]["target"], "second full header")

    def test_bundled_target_available_without_torch(self):
        args = parse_args(["--target-name", "CCR5_HUMAN", "--peptides", str(self.peptides), "--output", str(self.output), "--email", "user@example.org"])
        runner.run(args, self.local)
        self.assertGreater(len(self.api.call_args.args[0]), 100)
        self.assertEqual(list(csv.DictReader(io.StringIO(self.output.read_text())))[0]["target"], "CCR5_HUMAN")

    def test_bad_output_and_empty_candidates_fail_before_identity(self):
        with self.assertRaises(SystemExit):
            self.run_cloud("--output", str(self.root / "missing/scores.csv"))
        self.peptides.write_text("")
        with self.assertRaises(SystemExit):
            self.run_cloud()
        self.identity.assert_not_called()
        self.api.assert_not_called()

    def test_failed_response_or_write_preserves_existing_output(self):
        self.output.write_text("original")
        for results in ([], [{"peptide_name": "not-submitted", "sequence": "ACD", "score": 0.2}]):
            self.api.return_value = results
            with self.assertRaises(SystemExit):
                self.run_cloud()
            self.assertEqual(self.output.read_text(), "original")
        self.api.return_value = [{"peptide_name": "low", "sequence": "EFG", "score": 0.2}, {"peptide_name": "high", "sequence": "KLM", "score": 0.5}]
        with patch("grade.writers.os.replace", side_effect=OSError("write failed")), self.assertRaises(SystemExit):
            self.run_cloud()
        self.assertEqual(self.output.read_text(), "original")
        self.assertEqual(list(self.root.glob(".scores.csv.*.tmp")), [])
        self.assertIn("quota may have been used", self.stderr.getvalue())

    def test_usage_queries_without_target_or_output(self):
        args = parse_args(["--usage", "--email", "user@example.org"])
        with patch.object(cloud, "query_usage", return_value={"remaining": 12, "limit": 50, "reset_at": "tomorrow"}) as usage:
            runner.run(args, self.local)
        usage.assert_called_once_with("user@example.org", TOKEN)
        self.api.assert_not_called()
        self.local.assert_not_called()
        self.assertIn("12/50", self.stdout.getvalue())

    def test_local_selection_does_not_read_email_or_credentials(self):
        model_dir = self.root / "model"
        model_dir.mkdir()
        args = self.args("--model-dir", str(model_dir), "--fp16")
        with patch.object(runner.importlib.util, "find_spec", return_value=object()), patch.object(cloud, "load_config", side_effect=AssertionError("read credentials")):
            runner.run(args, self.local)
        self.local.assert_called_once_with(args)
        self.identity.assert_not_called()
        self.api.assert_not_called()
        self.assertIsNone(args.model)

    def test_local_bad_path_missing_dependency_and_failure_never_fallback(self):
        with self.assertRaises(SystemExit):
            runner.run(self.args("--model-dir", str(self.root / "missing")), self.local)
        with patch.object(runner.importlib.util, "find_spec", return_value=None), self.assertRaises(SystemExit):
            runner.run(self.args("--model-dir", str(self.root)), self.local)
        self.assertIn('grade-client[local]', self.stderr.getvalue())
        self.local.side_effect = ValueError("bad checkpoint")
        with patch.object(runner.importlib.util, "find_spec", return_value=object()), self.assertRaisesRegex(ValueError, "bad checkpoint"):
            runner.run(self.args("--model-dir", str(self.root)), self.local)
        self.identity.assert_not_called()
        self.api.assert_not_called()

    def test_local_flags_and_ddp_without_path_fail_before_network(self):
        for options in (["--fp16"], ["--bf16"], ["--batch-size", "8"], ["--num-workers", "0"], ["--max-length", "1024"], ["--attn-implementation", "sdpa"], ["--checkpoint-interval", "100"], ["--no-resume"], ["--no-target-cache"]):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.args(*options)
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}), self.assertRaises(SystemExit):
            self.args()
        self.identity.assert_not_called()
        self.api.assert_not_called()

    def test_usage_and_local_email_conflicts(self):
        for options in (["--usage", "--peptides", "p.fa"], ["--usage", "--model-dir", "."], ["--usage", "--top-k", "2"]):
            with self.assertRaises(SystemExit):
                parse_args(options)
        with self.assertRaises(SystemExit):
            self.args("--model-dir", str(self.root), "--email", "user@example.org")
        with self.assertRaises(SystemExit):
            self.args("--model-dir", "")

    def test_output_alias_and_main_dispatch(self):
        modern = self.args()
        old = parse_args(["--target-fasta", str(self.target), "--peptides", str(self.peptides), "--output-csv", str(self.output)])
        self.assertEqual(vars(modern), vars(old))
        with patch.object(inference, "parse_args", return_value=modern), patch.object(runner, "run") as dispatch:
            inference.main()
        dispatch.assert_called_once_with(modern, inference.run_inference)

    def test_input_parser_preserves_local_contract(self):
        cases = [("csv", "name,sequence\np, a c d\n"), ("txt", "p acd\n"), ("fa", ">p\nACD\n")]
        for extension, text in cases:
            path = self.root / f"candidates.{extension}"
            path.write_text(text)
            self.assertEqual(load_peptides(str(path)), {"p": "ACD"})
        self.peptides.write_text(">same\nACD\n>same\nEFG\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_peptides(str(self.peptides))

    def test_imports_help_and_version_do_not_need_heavy_dependencies(self):
        for command in (["-c", "import sys; import grade.runner, grade.inputs; assert not any(m in sys.modules for m in ('torch','transformers','requests'))"], ["-m", "grade", "--help"], ["-m", "grade.inference", "--help"], [str(ROOT / "run_inference.py"), "--help"], ["-m", "grade", "--version"]):
            result = subprocess.run([sys.executable, "-S", "-B", *command], cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "grade-client 1.2.0")

    def test_generated_skill_is_synchronized_and_independent(self):
        check_sync(ROOT)
        skill = self.root / "grade-cloud"
        shutil.copytree(ROOT / "skills/grade-cloud", skill, ignore=shutil.ignore_patterns("__pycache__"))
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(skill / "scripts/grade_predict.py"), "--help"], cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--usage", result.stdout)
        (self.root / "grade").mkdir()
        (self.root / "grade/cloud.py").write_text("# deliberately stale source\n")
        (self.root / "skills/grade-cloud/scripts").mkdir(parents=True)
        shutil.copyfile(skill / "scripts/grade_predict.py", self.root / "skills/grade-cloud/scripts/grade_predict.py")
        with self.assertRaisesRegex(RuntimeError, "out of date"):
            check_sync(self.root)


if __name__ == "__main__":
    unittest.main()
