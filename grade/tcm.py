"""Target Condition Module (TCM).

Responsible for target (GPCR) sequence input, parsing, and preprocessing.
Supports three input modes: by name (from a FASTA lookup), by raw sequence,
or by FASTA file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


def read_fasta(path: Path) -> Dict[str, str]:
    """Parse a FASTA file into a dict of {id: sequence}."""
    records: Dict[str, str] = {}
    current_id = None
    seq_parts = []

    def store_record(record_id: str, parts: list[str]) -> None:
        if record_id in records:
            raise ValueError(f"Duplicate FASTA record id {record_id!r} in {path}")
        records[record_id] = "".join(parts)

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    store_record(current_id, seq_parts)
                elif seq_parts:
                    raise ValueError(
                        f"FASTA file {path} has sequence data before the first '>' header"
                    )
                current_id = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        if current_id is not None:
            store_record(current_id, seq_parts)
        elif seq_parts:
            raise ValueError(
                f"FASTA file {path} contains sequence data before any header line "
                f"(missing '>' header at line 1)"
            )
    return records


def normalize_sequence(seq: str) -> str:
    """Strip whitespace and uppercase a sequence."""
    return "".join(seq.split()).upper()


def validate_sequence(seq: str, field_name: str = "sequence") -> str:
    """Normalize a sequence and reject values that become empty."""
    if not isinstance(seq, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = normalize_sequence(seq)
    if not normalized:
        raise ValueError(f"{field_name} is empty after normalization")
    return normalized


def resolve_target(
    target_name: str | None,
    target_seq: str | None,
    target_fasta: str | None,
) -> Tuple[str, str]:
    """Resolve the target GPCR sequence from the provided arguments.

    Args:
        target_name: Target name to look up in a FASTA file.
        target_seq: Raw amino acid sequence (mutually exclusive with the above).
        target_fasta: Path to a FASTA file containing target sequence(s).

    Returns:
        A (name, sequence) tuple.

    Raises:
        KeyError: If the target name is not found in the FASTA file.
        FileNotFoundError: If the default FASTA file does not exist.
    """
    if target_seq:
        return "custom_target", validate_sequence(target_seq, "target sequence")

    if target_fasta:
        records = read_fasta(Path(target_fasta))
        if target_name:
            if target_name not in records:
                raise KeyError(f"Target {target_name} not found in {target_fasta}")
            return target_name, validate_sequence(records[target_name], f"target {target_name!r}")
        if len(records) != 1:
            raise ValueError(
                "--target-fasta must contain exactly one sequence when "
                "--target-name is not provided"
            )
        name, seq = next(iter(records.items()))
        return name, validate_sequence(seq, f"target {name!r}")

    # Fallback: look up by name in the default GPCR FASTA
    default_fasta = Path(__file__).resolve().parent / "targets" / "targets_gpcr_v2.fa"
    if not default_fasta.exists():
        raise FileNotFoundError(f"Default target fasta not found: {default_fasta}")
    records = read_fasta(default_fasta)
    if target_name not in records:
        raise KeyError(f"Target {target_name} not found in {default_fasta}")
    return target_name, validate_sequence(records[target_name], f"target {target_name!r}")
