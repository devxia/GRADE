import csv
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grade import inference, writers


class AtomicOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shard = self.root / "scores_rank0.csv"
        self.checkpoint = self.root / "scores_rank0.checkpoint.json"
        self.write_progress()

    def write_progress(self):
        with self.shard.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["number", "Target", "Peptide_name", "src", "sequence", "score"])
            writer.writerow(["", "target", "low", "source", "ACDE", "0.25"])
            writer.writerow(["", "target", "high", "source", "FGHI", "0.75"])
        self.checkpoint.write_bytes(b'{"processed": 2}\n')

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.root.iterdir()}

    def finalize(self, output, format_name):
        return inference.finalize_output(
            output, [self.shard], [self.checkpoint], "source", None, format_name,
        )

    def test_formats_replace_output_and_remove_progress(self):
        rows = [
            {"rank": 1, "target": "target", "peptide_name": "high", "src": "source", "sequence": "FGHI", "score": 0.75},
            {"rank": 2, "target": "target", "peptide_name": "low", "src": "source", "sequence": "ACDE", "score": 0.25},
        ]
        output = self.root / "scores.out"
        for format_name in ("csv", "tsv", "json"):
            with self.subTest(format=format_name):
                output.write_bytes(b"previous output")
                output.chmod(0o640)
                with patch("grade.writers.os.fsync", wraps=os.fsync) as sync:
                    self.assertEqual(self.finalize(output, format_name), 2)
                sync.assert_called_once()
                if format_name == "json":
                    self.assertEqual(output.read_bytes(), (json.dumps(rows, indent=2) + "\n").encode("utf-8"))
                else:
                    delimiter = "," if format_name == "csv" else "\t"
                    expected = (
                        delimiter.join(writers.COLUMNS) + "\r\n"
                        + delimiter.join(map(str, rows[0].values())) + "\r\n"
                        + delimiter.join(map(str, rows[1].values())) + "\r\n"
                    )
                    self.assertEqual(output.read_bytes(), expected.encode("utf-8"))
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o640)
                self.assertEqual(self.snapshot(), {output.name: output.read_bytes()})
                self.write_progress()

    def test_creates_new_utf8_output(self):
        output = self.root / "new.csv"
        rows = [{"rank": 1, "target": "靶点", "peptide_name": "high", "src": "source",
                 "sequence": "FGHI", "score": 0.75}]
        writers.CSVWriter().write(output, rows)
        self.assertEqual(output.read_bytes(), writers.CSVWriter().write_to_string(rows).encode("utf-8"))
        self.assertEqual(set(self.snapshot()), {output.name, self.shard.name, self.checkpoint.name})

    def test_write_and_flush_failures_preserve_output_and_progress(self):
        output = self.root / "scores.csv"
        output.write_bytes(b"previous output")
        before = self.snapshot()
        create_temp = tempfile.NamedTemporaryFile

        class FailingIO:
            def __init__(self, handle, stage):
                self.handle = handle
                self.stage = stage

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def write(self, content):
                self.handle.write(content[:4] if self.stage == "write" else content)
                if self.stage == "write":
                    raise OSError("write failed")

            def flush(self):
                self.handle.flush()
                if self.stage == "flush":
                    raise OSError("flush failed")

        for stage in ("write", "flush"):
            with (self.subTest(stage=stage),
                  patch("grade.writers.tempfile.NamedTemporaryFile",
                        side_effect=lambda **kw: FailingIO(create_temp(**kw), stage)),
                  self.assertRaisesRegex(OSError, f"{stage} failed")):
                self.finalize(output, "csv")
            self.assertEqual(self.snapshot(), before)

    def test_fsync_failure_preserves_output_and_progress(self):
        output = self.root / "scores.tsv"
        output.write_bytes(b"previous output")
        before = self.snapshot()
        with (patch("grade.writers.os.fsync", side_effect=OSError("sync failed")),
              self.assertRaisesRegex(OSError, "sync failed")):
            self.finalize(output, "tsv")
        self.assertEqual(self.snapshot(), before)

    def test_replace_failure_preserves_output_and_progress(self):
        output = self.root / "scores.json"
        output.write_bytes(b"previous output")
        before = self.snapshot()

        def reject_replace(source, destination):
            self.assertEqual(Path(source).parent, output.parent)
            self.assertEqual(destination, output)
            self.assertTrue(Path(source).is_file())
            raise OSError("replace failed")

        with (patch("grade.writers.os.replace", side_effect=reject_replace),
              self.assertRaisesRegex(OSError, "replace failed")):
            self.finalize(output, "json")
        self.assertEqual(self.snapshot(), before)

    def test_serialization_failure_preserves_output_and_progress(self):
        output = self.root / "scores.csv"
        output.write_bytes(b"previous output")
        before = self.snapshot()
        with (patch.object(writers.CSVWriter, "write_to_string", side_effect=ValueError("serialize failed")),
              self.assertRaisesRegex(ValueError, "serialize failed")):
            self.finalize(output, "csv")
        self.assertEqual(self.snapshot(), before)

    def test_nonfinite_score_preserves_output_and_progress(self):
        for score in ("nan", "inf", "-inf"):
            for format_name in ("csv", "tsv", "json"):
                with self.subTest(score=score, format=format_name):
                    self.write_progress()
                    with self.shard.open(encoding="utf-8", newline="") as handle:
                        rows = list(csv.reader(handle))
                    rows[2][-1] = score
                    with self.shard.open("w", encoding="utf-8", newline="") as handle:
                        csv.writer(handle).writerows(rows)
                    output = self.root / "scores.out"
                    output.write_bytes(b"previous output")
                    before = self.snapshot()
                    with self.assertRaisesRegex(ValueError, "Non-finite score.*row 3"):
                        inference.finalize_output(
                            output, [self.shard], [self.checkpoint], "source", 1, format_name,
                        )
                    self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
