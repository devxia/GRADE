"""Dependency-light peptide file loading for both local and Cloud inference."""

import csv
from pathlib import Path
from typing import Dict

from .tcm import read_fasta, validate_sequence


def add_unique_peptide(peptides: Dict[str, str], name: str, sequence: str, source: Path | str) -> None:
    """Add a peptide record, rejecting duplicate peptide names."""
    if name in peptides:
        raise ValueError(f"Duplicate peptide name {name!r} in {source}")
    peptides[name] = validate_sequence(sequence, f"peptide {name!r}")


def load_peptides(path_str: str) -> Dict[str, str]:
    """Load candidate peptides from FASTA, CSV or TXT without importing torch."""
    path = Path(path_str)
    suffix = path.suffix.lower()

    if suffix in {".fasta", ".faa", ".fa"}:
        records = read_fasta(path)
        peptides: Dict[str, str] = {}
        for name, seq in records.items():
            add_unique_peptide(peptides, name, seq, path)
        return peptides

    if suffix == ".txt":
        peptides: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            for idx, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 1:
                    name = f"peptide_{idx}"
                    seq = parts[0]
                else:
                    name = parts[0]
                    seq = parts[-1]
                add_unique_peptide(peptides, name, seq, path)
        return peptides

    if suffix == ".csv":
        peptides = {}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            name_key = next((key for key in fieldnames if key.lower() in {"name", "peptide_name", "id"}), None)
            seq_key = next((key for key in fieldnames if key.lower() in {"sequence", "peptide", "seq"}), None)
            if seq_key is None:
                raise ValueError(f"CSV file {path} must contain a sequence column")
            for idx, row in enumerate(reader, start=1):
                name = row.get(name_key) if name_key else None
                add_unique_peptide(peptides, name or f"peptide_{idx}", row[seq_key], path)
        return peptides

    raise ValueError(f"Unsupported peptide file format: {path}")
