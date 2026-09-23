"""Checkpoint identity and CSV resume handling using only the standard library."""

import csv
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from .model_registry import DEFAULT_MODEL_ID, get_model_spec


_RECOVERY_HINT = "Use --no-resume to start fresh or choose a new output path."


def model_identity(
    model_id: str | None = DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
) -> dict[str, str | None]:
    """Resolve an identity without opening model files.

    A null identifier denotes an explicitly supplied custom model directory.
    """
    if model_id is not None:
        spec = get_model_spec(model_id)
        if model_dir is None:
            model_dir = spec.local_dir
    elif model_dir is None:
        raise ValueError("A custom model requires a model directory")
    return {"model_id": model_id, "model_dir": str(Path(model_dir).resolve())}


def input_fingerprint(
    target_name: str,
    target_seq: str,
    peptides: Mapping[str, str],
    max_length: int,
    world_size: int,
    *,
    source_label: str,
    fp16: bool,
    bf16: bool,
    attn_implementation: str,
) -> str:
    """Hash resolved, ordered inputs and settings that determine shard results."""
    digest = hashlib.sha256(b"GRADE-input-fingerprint-v4\0")

    def update_field(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    update_field(target_name)
    update_field(target_seq)
    update_field(str(len(peptides)))
    for name, seq in peptides.items():
        update_field(name)
        update_field(seq)
    update_field(str(max_length))
    update_field(str(world_size))
    update_field(source_label)
    update_field("1" if fp16 else "0")
    update_field("1" if bf16 else "0")
    update_field(attn_implementation)
    return digest.hexdigest()


def checkpoint_path(output_path: Path, rank: int) -> Path:
    """Return the checkpoint file path for a given output and rank."""
    return output_path.parent / f"{output_path.stem}.checkpoint_rank{rank}.json"


def _rank_csv_path(output_path: Path, rank: int) -> Path:
    return output_path.parent / f"{output_path.stem}_rank{rank}.csv"


def read_checkpoint(chkpt_path: Path) -> int:
    """Read a progress count, preserving the legacy count-only reader API.

    Resume authorization is performed separately by validate_resume_identity.
    """
    try:
        with chkpt_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return int(data.get("processed", 0))
    except (FileNotFoundError, ValueError, TypeError, AttributeError):
        return 0


def count_csv_data_rows(csv_path: Path) -> int:
    """Count CSV data rows, returning zero for a missing or empty file."""
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def validate_resume_identity(
    output_path: Path,
    rank: int,
    *,
    model_id: str | None = DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
    input_fingerprint: str | None = None,
) -> int:
    """Validate existing artifacts without modifying them; return saved progress.

    Passing no fingerprint performs the early, torch-free model identity check only.
    """
    identity = model_identity(model_id, model_dir)
    chkpt_path = checkpoint_path(output_path, rank)
    if not chkpt_path.exists():
        if _rank_csv_path(output_path, rank).exists():
            raise ValueError(f"Rank CSV has no model identity checkpoint. {_RECOVERY_HINT}")
        return 0
    try:
        with chkpt_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid checkpoint: {chkpt_path}. {_RECOVERY_HINT}") from exc
    if not isinstance(data, dict) or not all(key in data for key in identity):
        raise ValueError(f"Checkpoint has no model identity: {chkpt_path}. {_RECOVERY_HINT}")
    if any(data[key] != value for key, value in identity.items()):
        raise ValueError(f"Checkpoint model identity does not match this run: {chkpt_path}. {_RECOVERY_HINT}")
    if input_fingerprint is not None:
        if "input_fingerprint" not in data:
            raise ValueError(f"Checkpoint has no input fingerprint: {chkpt_path}. {_RECOVERY_HINT}")
        if data["input_fingerprint"] != input_fingerprint:
            raise ValueError(f"Checkpoint input fingerprint does not match this run: {chkpt_path}. {_RECOVERY_HINT}")
    processed = data.get("processed")
    if type(processed) is not int or processed < 0:
        raise ValueError(f"Invalid checkpoint progress: {chkpt_path}. {_RECOVERY_HINT}")
    return processed


def write_checkpoint(
    chkpt_path: Path,
    processed_count: int,
    *,
    input_fingerprint: str,
    model_id: str | None = DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
) -> None:
    """Atomically write progress and both identities (fsync, then rename)."""
    if type(processed_count) is not int or processed_count < 0:
        raise ValueError("Checkpoint progress must be a non-negative integer")
    data = {"processed": processed_count, **model_identity(model_id, model_dir),
            "input_fingerprint": input_fingerprint}
    tmp_path = chkpt_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, chkpt_path)


def resolve_resume_point(
    output_path: Path,
    rank: int,
    force_fresh: bool,
    *,
    input_fingerprint: str,
    model_id: str | None = DEFAULT_MODEL_ID,
    model_dir: str | Path | None = None,
) -> tuple[int, str, bool]:
    """Return (processed_count, csv_mode, write_header) after identity validation.

    CSV progress is authoritative only after its checkpoint identities match.
    Every fresh start persists a zero-count identity before any CSV is opened.
    """
    identity = {**model_identity(model_id, model_dir), "input_fingerprint": input_fingerprint}
    chkpt_path = checkpoint_path(output_path, rank)
    rank_csv = _rank_csv_path(output_path, rank)
    if force_fresh:
        for path in (chkpt_path, rank_csv):
            path.unlink(missing_ok=True)
    else:
        checkpoint_processed = validate_resume_identity(output_path, rank, **identity)
        csv_rows = count_csv_data_rows(rank_csv)
        if csv_rows > 0:
            if checkpoint_processed != csv_rows:
                print(
                    f"[checkpoint] Reconciling progress: checkpoint={checkpoint_processed}, "
                    f"CSV={csv_rows}; resuming from CSV."
                )
                write_checkpoint(chkpt_path, csv_rows, **identity)
            print(f"[checkpoint] Resuming rank {rank} from CSV: {csv_rows} candidates already done")
            return csv_rows, "a", False
        if checkpoint_processed > 0:
            print(
                f"[checkpoint] Checkpoint says {checkpoint_processed} done but rank CSV "
                "missing/empty; starting fresh"
            )
    write_checkpoint(chkpt_path, 0, **identity)
    return 0, "w", True
