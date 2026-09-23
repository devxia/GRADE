"""Output format writers for GRADE inference results.

Provides a small abstraction over CSV, TSV, and JSON so that the inference
pipeline and the cloud API can share the same serialization path.
"""

import csv
import io
import json
import os
import stat
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

# Column order shared by CSV/TSV/JSON outputs.
COLUMNS = ["rank", "target", "peptide_name", "src", "sequence", "score"]


class ResultWriter(ABC):
    """Base class for result serialization."""

    @abstractmethod
    def write_to_string(self, rows: List[Dict[str, object]]) -> str:
        """Serialize ``rows`` to a string."""
        raise NotImplementedError

    def write(self, output_path: Path, rows: List[Dict[str, object]]) -> None:
        """Serialize ``rows`` to ``output_path``."""
        content = self.write_to_string(rows)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=output_path.parent,
                prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                try:
                    mode = stat.S_IMODE(output_path.stat().st_mode)
                except FileNotFoundError:
                    pass
                else:
                    os.fchmod(handle.fileno(), mode)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, output_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @property
    @abstractmethod
    def extension(self) -> str:
        """Canonical file extension for this format (including the dot)."""
        raise NotImplementedError


class CSVWriter(ResultWriter):
    extension = ".csv"

    def write_to_string(self, rows: List[Dict[str, object]]) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()


class TSVWriter(ResultWriter):
    extension = ".tsv"

    def write_to_string(self, rows: List[Dict[str, object]]) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()


class JSONWriter(ResultWriter):
    extension = ".json"

    def write_to_string(self, rows: List[Dict[str, object]]) -> str:
        return json.dumps(rows, indent=2) + "\n"


_WRITERS: Dict[str, type[ResultWriter]] = {
    "csv": CSVWriter,
    "tsv": TSVWriter,
    "json": JSONWriter,
}


def supported_formats() -> List[str]:
    """Return the list of supported output format names."""
    return list(_WRITERS.keys())


def infer_format_from_path(path: str | Path) -> str:
    """Infer a format name from a file path extension.

    Falls back to ``csv`` if the extension is not recognized.
    """
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in _WRITERS:
        return suffix
    return "csv"


def get_writer(format_name: str) -> ResultWriter:
    """Return a writer instance for the requested format."""
    fmt = format_name.lower()
    if fmt not in _WRITERS:
        raise ValueError(
            f"Unsupported output format: {format_name!r}. "
            f"Supported: {', '.join(_WRITERS)}"
        )
    return _WRITERS[fmt]()
