import csv
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from grade.cli import parse_args
from grade import inference, resume


@contextmanager
def pam_without_torch():
    """Load the real peptide parser with only its import-time torch types stubbed."""
    torch = ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    utils = ModuleType("torch.utils")
    data = ModuleType("torch.utils.data")
    data.Dataset = type("Dataset", (), {})

    class Sampler:
        def __class_getitem__(cls, item):
            return cls

    data.Sampler = Sampler
    with patch.dict(sys.modules, {"torch": torch, "torch.utils": utils, "torch.utils.data": data}):
        yield


class InferenceResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "scores.csv"
        self.checkpoint = resume.checkpoint_path(self.output, 0)
        self.shard = self.root / "scores_rank0.csv"
        self.identity = resume.model_identity("GRADE-paper", self.root / "paper")
        self.peptides = self.root / "peptides.fasta"
        self.peptides.write_text(">pep1\nACDE\n>pep2\nFGHI\n", encoding="utf-8")
        self.fingerprint = resume.input_fingerprint(
            "custom_target", "ACDE", {"pep1": "ACDE", "pep2": "FGHI"}, 1024, 1,
            source_label="peptides", fp16=False, bf16=False, attn_implementation="sdpa",
        )
        self.checkpoint_identity = {**self.identity, "input_fingerprint": self.fingerprint}

    def save(self, count):
        resume.write_checkpoint(self.checkpoint, count, **self.checkpoint_identity)

    def resolve(self, fresh=False, **identity):
        return resume.resolve_resume_point(
            self.output, 0, fresh, **{**self.checkpoint_identity, **identity}
        )

    def data(self):
        return json.loads(self.checkpoint.read_text())

    def rows(self, count):
        with self.shard.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["number", "Target", "Peptide_name", "src", "sequence", "score"])
            for i in range(count):
                writer.writerow(["", "target", f"peptide{i}", "source", "ACDE", 0.5])

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.root.iterdir() if p.is_file()}

    def assert_rejected_unchanged(self, **identity):
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "--no-resume.*new output path"):
            self.resolve(**identity)
        self.assertEqual(before, self.snapshot())

    def args(self, *extra):
        args = parse_args(["--target-seq", "ACDE", "--peptides", str(self.peptides),
                           "--output-csv", str(self.output), "--model-dir", self.identity["model_dir"], *extra])
        # Direct pipeline callers can retain a known registry identity; the CLI
        # correctly treats an arbitrary explicit directory as custom.
        args.model = self.identity["model_id"]
        return args

    def assert_run_rejected_unchanged(self, args, world_size=1):
        before = self.snapshot()
        with (pam_without_torch(),
              patch.dict("os.environ", {"WORLD_SIZE": str(world_size), "RANK": "0"}),
              patch.object(inference, "setup_distributed", side_effect=AssertionError("initialized")),
              self.assertRaisesRegex(ValueError, "input fingerprint.*--no-resume.*new output path")):
            inference.run_inference(args)
        self.assertEqual(before, self.snapshot())

    @contextmanager
    def stubbed_run(self, batches):
        """Exercise the real CSV/checkpoint loop without loading model dependencies."""
        handles = []
        original_open = Path.open

        def track_open(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            if path == self.shard and handle.writable():
                handles.append(handle)
            return handle

        with pam_without_torch():
            torch = sys.modules["torch"]
            torch.no_grad = nullcontext
            sys.modules["torch.utils.data"].DataLoader = lambda *args, **kwargs: batches
            functional = ModuleType("torch.nn.functional")
            functional.softmax = Mock(return_value=MagicMock())
            modeling = ModuleType("grade.modeling")
            modeling.load_grade_tokenizer = Mock()
            tqdm = ModuleType("tqdm")
            tqdm.tqdm = lambda iterable, **kwargs: iterable
            with (patch.dict(sys.modules, {
                      "torch.distributed": ModuleType("torch.distributed"),
                      "torch.nn": ModuleType("torch.nn"),
                      "torch.nn.functional": functional,
                      "grade.modeling": modeling,
                      "tqdm": tqdm,
                  }),
                  patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0"}),
                  patch.object(Path, "open", new=track_open),
                  patch.object(inference, "setup_distributed", return_value=(SimpleNamespace(type="cpu"), 0)),
                  patch.object(inference, "build_model", return_value=object()),
                  patch.object(inference, "get_autocast_dtype", return_value=None),
                  patch.object(inference, "score_batch", return_value=(object(), None)) as score,
                  patch.object(inference, "tensor_to_list", return_value=[0.5]),
                  patch.object(inference, "cleanup_distributed") as cleanup,
                  patch.object(inference, "finalize_output") as merge):
                yield handles, score, cleanup, merge

    def test_fresh_run_writes_zero_identity_before_csv(self):
        self.assertEqual(self.resolve(), (0, "w", True))
        self.assertEqual(self.data(), {"processed": 0, **self.checkpoint_identity})
        self.assertTrue(Path(self.data()["model_dir"]).is_absolute())
        self.assertFalse(self.shard.exists())

    def test_same_identity_resumes_without_rewriting_checkpoint(self):
        self.save(2)
        self.rows(2)
        before = self.snapshot()
        self.assertEqual(self.resolve(), (2, "a", False))
        self.assertEqual(before, self.snapshot())

    def test_csv_reconciliation_preserves_identity(self):
        for checkpoint_count in (0, 1, 5):
            with self.subTest(count=checkpoint_count):
                self.save(checkpoint_count)
                self.rows(2)
                self.assertEqual(self.resolve(), (2, "a", False))
                self.assertEqual(self.data(), {"processed": 2, **self.checkpoint_identity})

    def test_missing_or_empty_csv_restarts_with_zero_identity(self):
        self.save(4)
        self.assertEqual(self.resolve(), (0, "w", True))
        self.assertEqual(self.data(), {"processed": 0, **self.checkpoint_identity})
        self.save(4)
        self.rows(0)
        self.assertEqual(self.resolve(), (0, "w", True))
        self.assertEqual(self.data(), {"processed": 0, **self.checkpoint_identity})

    def test_zero_checkpoint_with_header_only_csv_is_safe(self):
        self.save(0)
        self.rows(0)
        self.assertEqual(self.resolve(), (0, "w", True))

    def test_model_id_mismatch_does_not_modify_results(self):
        self.save(0)
        self.rows(2)
        self.output.write_text("existing final output")
        self.assert_rejected_unchanged(model_id=None, model_dir=self.identity["model_dir"])

    def test_model_directory_mismatch_does_not_modify_results(self):
        self.save(0)
        self.rows(2)
        self.assert_rejected_unchanged(model_id="GRADE-paper", model_dir=self.root / "other")

    def test_unknown_stored_model_is_rejected(self):
        self.checkpoint.write_text(json.dumps({"processed": 1, **self.checkpoint_identity, "model_id": "GRADE-base"}))
        self.rows(1)
        self.assert_rejected_unchanged()

    def test_legacy_checkpoint_rejected_with_or_without_csv(self):
        self.checkpoint.write_text('{"processed": 0}')
        self.assert_rejected_unchanged()
        self.rows(2)
        self.assert_rejected_unchanged()

    def test_model_identity_only_checkpoint_cannot_resume(self):
        self.checkpoint.write_text(json.dumps({"processed": 1, **self.identity}))
        self.rows(1)
        self.output.write_bytes(b"existing final output")
        self.assert_rejected_unchanged()
        self.assert_run_rejected_unchanged(self.args())
        self.assertEqual(self.resolve(True), (0, "w", True))
        self.assertEqual(self.data(), {"processed": 0, **self.checkpoint_identity})
        self.assertFalse(self.shard.exists())
        self.assertEqual(self.output.read_bytes(), b"existing final output")

    def test_orphan_csv_rejected_even_when_empty(self):
        for rows in (0, 2):
            self.rows(rows)
            self.assert_rejected_unchanged()
        self.shard.write_text("")
        self.assert_rejected_unchanged()

    def test_malformed_checkpoint_and_progress_rejected(self):
        for text in ("{", "[]", "null", json.dumps({"processed": -1, **self.checkpoint_identity}),
                     json.dumps({"processed": True, **self.checkpoint_identity}),
                     json.dumps({"processed": "1", **self.checkpoint_identity})):
            self.checkpoint.write_text(text)
            self.assert_rejected_unchanged()

    def test_force_fresh_replaces_incompatible_or_legacy_artifacts(self):
        for previous in ('{"processed": 99}', '{"model_id": "other"}', "broken"):
            self.checkpoint.write_text(previous)
            self.rows(2)
            self.output.write_text("previous final output")
            self.assertEqual(self.resolve(True), (0, "w", True))
            self.assertFalse(self.shard.exists())
            self.assertEqual(self.data(), {"processed": 0, **self.checkpoint_identity})
            self.assertEqual(self.output.read_text(), "previous final output")

    def test_force_fresh_handles_orphan_csv(self):
        self.rows(2)
        self.assertEqual(self.resolve(True), (0, "w", True))
        self.assertEqual(self.data(), {"processed": 0, **self.checkpoint_identity})

    def test_custom_directory_identity_and_successive_saves(self):
        custom = {**resume.model_identity(None, self.root / "custom"),
                  "input_fingerprint": self.fingerprint}
        self.assertEqual(self.resolve(**custom), (0, "w", True))
        for count in (1, 3):
            resume.write_checkpoint(self.checkpoint, count, **custom)
            self.assertEqual(self.data(), {"processed": count, **custom})
        self.rows(3)
        self.assertEqual(self.resolve(**custom), (3, "a", False))
        self.assert_rejected_unchanged()

    def test_equivalent_paths_resolve_to_same_identity(self):
        self.save(2)
        self.rows(2)
        alternate = self.root / "unused" / ".." / "paper"
        self.assertEqual(self.resolve(model_id="GRADE-paper", model_dir=alternate), (2, "a", False))

    def test_different_working_directories_have_different_identity(self):
        from contextlib import chdir

        other = self.root / "other"
        other.mkdir()
        with chdir(self.root):
            first = resume.model_identity("GRADE-paper")
        with chdir(other):
            second = resume.model_identity("GRADE-paper")
        self.assertNotEqual(first, second)

    def test_atomic_write_has_no_leftover_temp_file(self):
        self.save(4)
        self.assertEqual(resume.read_checkpoint(self.checkpoint), 4)
        self.assertFalse(self.checkpoint.with_suffix(".tmp").exists())

    def test_reexports_preserve_public_helpers(self):
        for name in ("checkpoint_path", "read_checkpoint", "count_csv_data_rows", "write_checkpoint", "resolve_resume_point"):
            self.assertIs(getattr(inference, name), getattr(resume, name))

    def test_equivalent_normalized_inputs_resume(self):
        self.save(2)
        self.rows(2)
        before = self.snapshot()
        self.peptides.write_text(">pep1\n a c d e \n>pep2\n f g h i\n", encoding="utf-8")
        with pam_without_torch():
            from grade.pam import load_peptides
            from grade.tcm import resolve_target

            target_name, target_seq = resolve_target(None, " a c d e ", None)
            fingerprint = resume.input_fingerprint(
                target_name, target_seq, load_peptides(str(self.peptides)), 1024, 1,
                source_label="peptides", fp16=False, bf16=False, attn_implementation="sdpa",
            )
        self.assertEqual(fingerprint, self.fingerprint)
        self.assertEqual(self.resolve(input_fingerprint=fingerprint), (2, "a", False))
        self.assertEqual(before[self.checkpoint.name], self.checkpoint.read_bytes())
        self.assertEqual(before[self.shard.name], self.shard.read_bytes())

    def test_fingerprint_distinguishes_structured_fields(self):
        settings = {"source_label": "peptides", "fp16": False, "bf16": False,
                    "attn_implementation": "sdpa"}
        base = resume.input_fingerprint("ab", "CD", {"p": "EF"}, 1024, 1, **settings)
        self.assertEqual("ab" + "CD", "a" + "bCD")
        self.assertNotEqual(base, resume.input_fingerprint("a", "bCD", {"p": "EF"}, 1024, 1, **settings))
        self.assertEqual("p" + "EF", "pE" + "F")
        self.assertNotEqual(base, resume.input_fingerprint("ab", "CD", {"pE": "F"}, 1024, 1, **settings))
        self.assertNotEqual(base, resume.input_fingerprint("ab", "CD", {"p": "EF"}, 1024, 1,
                                                            **{**settings, "source_label": "other"}))
        self.assertEqual(len(base), 64)

    def test_renamed_candidates_with_default_label_rejected_without_writes(self):
        self.save(1)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        renamed = self.root / "renamed.fasta"
        renamed.write_bytes(self.peptides.read_bytes())
        args = self.args()
        args.peptides = str(renamed)
        self.assert_run_rejected_unchanged(args)
        args.source_label = ""
        self.assert_run_rejected_unchanged(args)

    def test_renamed_candidates_with_stable_explicit_label_resume(self):
        self.save(1)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        renamed = self.root / "renamed.fasta"
        renamed.write_bytes(self.peptides.read_bytes())
        args = self.args("--source-label", "peptides")
        args.peptides = str(renamed)
        before = self.snapshot()
        with self.stubbed_run([]) as (_, _, _, merge):
            inference.run_inference(args)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(merge.call_args.kwargs["source_label"], "peptides")

    def test_empty_explicit_label_uses_filename_stem(self):
        self.save(1)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        before = self.snapshot()
        with self.stubbed_run([]) as (_, _, _, merge):
            inference.run_inference(self.args("--source-label", ""))
        self.assertEqual(before, self.snapshot())
        self.assertEqual(merge.call_args.kwargs["source_label"], "peptides")

    def test_changed_explicit_label_rejected_without_writes(self):
        fingerprint = resume.input_fingerprint(
            "custom_target", "ACDE", {"pep1": "ACDE", "pep2": "FGHI"}, 1024, 1,
            source_label="original", fp16=False, bf16=False, attn_implementation="sdpa",
        )
        resume.write_checkpoint(self.checkpoint, 1, **self.identity,
                                input_fingerprint=fingerprint)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        self.assert_run_rejected_unchanged(self.args("--source-label", "changed"))

    def test_scoring_settings_change_rejected_without_writes(self):
        self.save(1)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        for flags in (("--fp16",), ("--bf16",),
                      ("--attn-implementation", "eager")):
            with self.subTest(flags=flags):
                self.assert_run_rejected_unchanged(self.args(*flags))

    def test_matching_scoring_settings_resume_and_presentation_options_can_change(self):
        for flags in (("--fp16",), ("--bf16",),
                      ("--attn-implementation", "eager")):
            with self.subTest(flags=flags):
                args = self.args(*flags, "--top-k", "1", "--output-format", "json")
                fingerprint = resume.input_fingerprint(
                    "custom_target", "ACDE", {"pep1": "ACDE", "pep2": "FGHI"}, 1024, 1,
                    source_label="peptides", fp16=args.fp16, bf16=args.bf16,
                    attn_implementation=args.attn_implementation,
                )
                resume.write_checkpoint(self.checkpoint, 1, **self.identity,
                                        input_fingerprint=fingerprint)
                self.rows(1)
                self.output.write_bytes(b"prior output")
                before = self.snapshot()
                with self.stubbed_run([]) as (_, _, _, merge):
                    inference.run_inference(args)
                self.assertEqual(before, self.snapshot())
                merge.assert_called_once()

    def test_v3_fingerprint_checkpoint_rejected_without_writes(self):
        import hashlib

        digest = hashlib.sha256(b"GRADE-input-fingerprint-v3\0")
        for value in ("custom_target", "ACDE", "2", "pep1", "ACDE", "pep2", "FGHI",
                      "1024", "1", "0", "0", "sdpa"):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        old_fingerprint = digest.hexdigest()
        self.assertNotEqual(old_fingerprint, self.fingerprint)
        resume.write_checkpoint(self.checkpoint, 1, **self.identity,
                                input_fingerprint=old_fingerprint)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        self.assert_run_rejected_unchanged(self.args())
        self.assert_run_rejected_unchanged(self.args("--source-label", "peptides"))

    def test_v2_fingerprint_checkpoint_rejected_without_writes(self):
        import hashlib

        digest = hashlib.sha256(b"GRADE-input-fingerprint-v2\0")
        for value in ("custom_target", "ACDE", "2", "pep1", "ACDE", "pep2", "FGHI", "1024", "1"):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        old_fingerprint = digest.hexdigest()
        self.assertNotEqual(old_fingerprint, self.fingerprint)
        resume.write_checkpoint(self.checkpoint, 1, **self.identity,
                                input_fingerprint=old_fingerprint)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        self.assert_run_rejected_unchanged(self.args())

    def test_parsed_input_changes_rejected_without_writes(self):
        self.save(1)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        changes = (
            ("target sequence", {"target_seq": "FGHI"}, None),
            ("target identifier", {"target_fasta": True}, None),
            ("peptide sequence", {}, ">pep1\nACDE\n>pep2\nFGHK\n"),
            ("peptide name", {}, ">pep1\nACDE\n>other\nFGHI\n"),
            ("peptide order", {}, ">pep2\nFGHI\n>pep1\nACDE\n"),
            ("max length", {"max_length": 512}, None),
        )
        for label, options, peptide_text in changes:
            with self.subTest(change=label):
                self.peptides.write_text(peptide_text or ">pep1\nACDE\n>pep2\nFGHI\n", encoding="utf-8")
                args = self.args()
                if options.get("target_fasta"):
                    fasta = self.root / "target.fasta"
                    fasta.write_text(">other_target\nACDE\n", encoding="utf-8")
                    args.target_seq = None
                    args.target_fasta = str(fasta)
                else:
                    for key, value in options.items():
                        setattr(args, key, value)
                self.assert_run_rejected_unchanged(args)

    def test_world_size_change_rejected_without_writes(self):
        self.save(1)
        self.rows(1)
        self.output.write_bytes(b"prior output")
        self.assert_run_rejected_unchanged(self.args(), world_size=2)

    def test_other_rank_input_mismatch_rejected_before_any_write(self):
        world_fingerprint = resume.input_fingerprint(
            "custom_target", "ACDE", {"pep1": "ACDE", "pep2": "FGHI"}, 1024, 2,
            source_label="peptides", fp16=False, bf16=False, attn_implementation="sdpa",
        )
        resume.write_checkpoint(self.checkpoint, 1, **self.identity,
                                input_fingerprint=world_fingerprint)
        rank1 = resume.checkpoint_path(self.output, 1)
        resume.write_checkpoint(rank1, 1, **self.checkpoint_identity)
        self.rows(1)
        (self.root / "scores_rank1.csv").write_bytes(b"prior shard")
        self.output.write_bytes(b"prior output")
        self.assert_run_rejected_unchanged(self.args(), world_size=2)

    def test_other_rank_scoring_mismatch_rejected_by_each_rank_before_writes(self):
        peptides = {"pep1": "ACDE", "pep2": "FGHI"}
        for rank, fp16 in ((0, False), (1, True)):
            fingerprint = resume.input_fingerprint(
                "custom_target", "ACDE", peptides, 1024, 2,
                source_label="peptides", fp16=fp16, bf16=False, attn_implementation="sdpa",
            )
            resume.write_checkpoint(resume.checkpoint_path(self.output, rank), 1,
                                    **self.identity, input_fingerprint=fingerprint)
            (self.root / f"scores_rank{rank}.csv").write_bytes(b"prior shard")
        self.output.write_bytes(b"prior output")
        before = self.snapshot()
        for rank in (0, 1):
            with (self.subTest(rank=rank), pam_without_torch(),
                  patch.dict("os.environ", {"WORLD_SIZE": "2", "RANK": str(rank)}),
                  patch.object(inference, "setup_distributed", side_effect=AssertionError("initialized")),
                  self.assertRaisesRegex(ValueError, "input fingerprint.*--no-resume.*new output path")):
                inference.run_inference(self.args())
            self.assertEqual(before, self.snapshot())

    def test_other_rank_source_label_mismatch_rejected_by_each_rank_before_writes(self):
        peptides = {"pep1": "ACDE", "pep2": "FGHI"}
        for rank, label in ((0, "peptides"), (1, "other")):
            fingerprint = resume.input_fingerprint(
                "custom_target", "ACDE", peptides, 1024, 2,
                source_label=label, fp16=False, bf16=False, attn_implementation="sdpa",
            )
            resume.write_checkpoint(resume.checkpoint_path(self.output, rank), 1,
                                    **self.identity, input_fingerprint=fingerprint)
            (self.root / f"scores_rank{rank}.csv").write_bytes(b"prior shard")
        self.output.write_bytes(b"prior output")
        before = self.snapshot()
        for rank in (0, 1):
            with (self.subTest(rank=rank), pam_without_torch(),
                  patch.dict("os.environ", {"WORLD_SIZE": "2", "RANK": str(rank)}),
                  patch.object(inference, "setup_distributed", side_effect=AssertionError("initialized")),
                  self.assertRaisesRegex(ValueError, "input fingerprint.*--no-resume.*new output path")):
                inference.run_inference(self.args())
            self.assertEqual(before, self.snapshot())

    def test_actual_inference_rejects_before_loading_dependencies(self):
        self.save(1)
        self.rows(1)
        before = self.snapshot()
        args = parse_args(["--target-seq", "ACDE", "--peptides", "unused.fa",
                           "--output-csv", str(self.output), "--model-dir", str(self.root / "other")])
        with (patch.dict("os.environ", {"WORLD_SIZE": "1"}),
              patch.dict(sys.modules, {"torch": None, "grade.pam": None}),
              self.assertRaisesRegex(ValueError, "model identity")):
            inference.run_inference(args)
        self.assertEqual(before, self.snapshot())

    def test_other_rank_mismatch_is_checked_before_any_write(self):
        self.save(1)
        self.rows(1)
        resume.write_checkpoint(resume.checkpoint_path(self.output, 1), 0,
                                input_fingerprint=self.fingerprint,
                                model_id=None, model_dir=self.root / "different")
        before = self.snapshot()
        args = parse_args(["--target-seq", "ACDE", "--peptides", "unused.fa", "--output-csv", str(self.output),
                           "--model-dir", self.identity["model_dir"]])
        args.model = self.identity["model_id"]
        with patch.dict("os.environ", {"WORLD_SIZE": "2", "RANK": "0"}), self.assertRaisesRegex(ValueError, "model identity"):
            inference.run_inference(args)
        self.assertEqual(before, self.snapshot())

    def test_header_failure_closes_shard_without_merging(self):
        self.output.write_bytes(b"prior final output")
        writer = Mock()
        writer.writerow.side_effect = OSError("header failed")
        with (self.stubbed_run([]) as (handles, _, cleanup, merge),
              patch.object(inference.csv, "writer", return_value=writer),
              self.assertRaisesRegex(OSError, "header failed")):
            inference.run_inference(self.args())
        self.assertEqual(len(handles), 1)
        self.assertTrue(handles[0].closed)
        self.assertEqual(self.data(), {"processed": 0, **self.checkpoint_identity})
        self.assertTrue(self.shard.exists())
        self.assertEqual(self.output.read_bytes(), b"prior final output")
        cleanup.assert_not_called()
        merge.assert_not_called()

    def test_scoring_failure_closes_shard_without_flushing_pending_rows(self):
        self.save(1)
        self.rows(1)
        self.output.write_bytes(b"prior final output")
        before = self.snapshot()
        batch = {"peptide_names": ["pep2"], "peptide_seqs": ["FGHI"]}
        with self.stubbed_run([batch.copy(), batch.copy()]) as (handles, score, cleanup, merge):
            score.side_effect = [(object(), None), RuntimeError("scoring failed")]
            args = self.args()
            args.checkpoint_interval = 2
            with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                inference.run_inference(args)
        self.assertEqual(len(handles), 1)
        self.assertTrue(handles[0].closed)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.resolve(), (1, "a", False))
        cleanup.assert_not_called()
        merge.assert_not_called()

    def test_checkpoint_failure_closes_shard_and_keeps_recoverable_rows(self):
        self.save(1)
        self.rows(1)
        self.output.write_bytes(b"prior final output")
        batch = {"peptide_names": ["pep2"], "peptide_seqs": ["FGHI"]}
        with (self.stubbed_run([batch]) as (handles, _, cleanup, merge),
              patch.object(inference, "write_checkpoint", side_effect=OSError("checkpoint failed")),
              self.assertRaisesRegex(OSError, "checkpoint failed")):
            inference.run_inference(self.args())
        self.assertEqual(len(handles), 1)
        self.assertTrue(handles[0].closed)
        self.assertEqual(self.data()["processed"], 1)
        self.assertEqual(resume.count_csv_data_rows(self.shard), 2)
        self.assertEqual(self.output.read_bytes(), b"prior final output")
        cleanup.assert_not_called()
        merge.assert_not_called()
        self.assertEqual(self.resolve(), (2, "a", False))

    def test_success_closes_shard_before_cleanup_and_merge(self):
        batch = {"peptide_names": ["pep1"], "peptide_seqs": ["ACDE"]}
        with self.stubbed_run([batch]) as (handles, _, cleanup, merge):
            def check_cleanup():
                self.assertTrue(handles[0].closed)

            def check_merge(**kwargs):
                self.assertTrue(cleanup.called)
                self.assertTrue(handles[0].closed)
                return 1

            cleanup.side_effect = check_cleanup
            merge.side_effect = check_merge
            inference.run_inference(self.args())
        self.assertEqual(resume.read_checkpoint(self.checkpoint), 1)
        self.assertEqual(resume.count_csv_data_rows(self.shard), 1)
        cleanup.assert_called_once_with()
        merge.assert_called_once()

    def test_missing_rank_shard_does_not_replace_output_or_clean_progress(self):
        self.save(1)
        self.rows(1)
        other_checkpoint = resume.checkpoint_path(self.output, 1)
        resume.write_checkpoint(other_checkpoint, 0, **self.checkpoint_identity)
        self.output.write_bytes(b"prior final output")
        before = self.snapshot()
        missing = self.root / "scores_rank1.csv"

        with self.assertRaisesRegex(FileNotFoundError, "missing rank shard") as raised:
            inference.finalize_output(
                self.output, [self.shard, missing], [self.checkpoint, other_checkpoint],
                "source", None, "csv",
            )

        self.assertIn(str(missing), str(raised.exception))
        self.assertEqual(before, self.snapshot())
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
