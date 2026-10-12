"""Unified CLI entrypoint and local multi-GPU inference pipeline.

The CLI defaults to Cloud. Explicit --model-dir selects the local pipeline
with incremental checkpointing, target caching and automatic resume.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from .cli import DEFAULT_MODEL_DIR, parse_args
from .resume import (
    checkpoint_path,
    count_csv_data_rows,
    input_fingerprint,
    model_identity,
    read_checkpoint,
    resolve_resume_point,
    validate_resume_identity,
    write_checkpoint,
)
from .writers import get_writer, infer_format_from_path, supported_formats

if TYPE_CHECKING:
    import torch


# ── distributed helpers ──────────────────────────────────────────────


def is_dist_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> Tuple[torch.device, int]:
    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cpu")

    if is_dist_enabled() and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    return device, local_rank


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ── model construction ───────────────────────────────────────────────


def build_model(args: argparse.Namespace, device: torch.device):
    from torch.nn.parallel import DistributedDataParallel as DDP

    from .modeling import load_grade_model

    model = load_grade_model(
        args.model_dir,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    if is_main_process():
        print(f"Loaded checkpoint from {Path(args.model_dir)}")

    if is_dist_enabled():
        if device.type == "cuda":
            model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
        else:
            model = DDP(model)
    return model


def get_autocast_dtype(args: argparse.Namespace):
    import torch

    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def tensor_to_list(tensor: torch.Tensor) -> List:
    return tensor.detach().cpu().tolist()


def score_batch(
    model,
    batch: dict[str, torch.Tensor],
    *,
    target_encoding: torch.Tensor | None,
    use_target_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Score one batch and lazily create a run-local target encoding."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    if use_target_cache and target_encoding is None:
        base_model = model.module if isinstance(model, DDP) else model
        target_encoding = base_model.encode_target(
            batch["input_ids"][:1, 0],
            batch["attention_mask"][:1, 0],
        )

    if use_target_cache:
        logits = model(**batch, target_encoding=target_encoding)[0]
    else:
        logits = model(**batch)[0]
    return logits, target_encoding


# ── checkpoint / resume ──────────────────────────────────────────────


def flush_rank_csv_buffer(csv_writer, csv_fh, buffer: List[dict], source_label: str) -> int:
    """Write buffered results to the internal rank CSV shard and fsync.

    Returns the number of rows written.
    """
    if not buffer:
        return 0
    for item in buffer:
        csv_writer.writerow(
            [
                None,  # number — filled during merge
                item["Target"],
                item["Peptide_name"],
                source_label,
                item["sequence"],
                item["score"],
            ]
        )
    csv_fh.flush()
    os.fsync(csv_fh.fileno())
    return len(buffer)


def _build_result_rows(all_results: List[dict], source_label: str) -> List[dict]:
    """Convert internal result dicts to the canonical output row schema."""
    return [
        {
            "rank": idx,
            "target": item["Target"],
            "peptide_name": item["Peptide_name"],
            "src": source_label,
            "sequence": item["sequence"],
            "score": float(item["score"]),
        }
        for idx, item in enumerate(all_results, start=1)
    ]


def finalize_output(
    output_path: Path,
    rank_csv_paths: List[Path],
    checkpoint_paths: List[Path],
    source_label: str,
    top_k: int | None,
    output_format: str,
) -> int:
    """Merge rank CSV shards, sort, apply top-k, and write the final output.

    Returns the number of rows written.
    """
    missing = [path for path in rank_csv_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize: missing rank shard(s): {', '.join(map(str, missing))}")

    all_results = []
    for r_csv in rank_csv_paths:
        with r_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                try:
                    score = float(row["score"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Invalid score in {r_csv} at row {row_number}") from exc
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite score in {r_csv} at row {row_number}")
                row["score"] = score
                all_results.append(row)

    all_results.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        all_results = all_results[:top_k]

    writer = get_writer(output_format)
    writer.write(output_path, _build_result_rows(all_results, source_label))

    for r_csv in rank_csv_paths:
        if r_csv.exists():
            r_csv.unlink()
    for r_chkpt in checkpoint_paths:
        if r_chkpt.exists():
            r_chkpt.unlink()

    return len(all_results)


# ── main inference loop ──────────────────────────────────────────────


def run_inference(args: argparse.Namespace) -> None:
    # Validate every participating shard before loading models or changing files.
    # Legacy programmatic callers with only model_dir are treated as custom.
    identity = model_identity(getattr(args, "model", None), args.model_dir)
    output_path = Path(args.output_csv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not args.no_resume:
        for shard_rank in range(world_size):
            validate_resume_identity(output_path, shard_rank, **identity)

    from .tcm import resolve_target
    from .pam import (
        GPCRPeptideDataset,
        PairCollator,
        SequentialDistributedSampler,
        SkipSampler,
        load_peptides,
    )

    target_name, target_seq = resolve_target(
        args.target_name, args.target_seq, args.target_fasta
    )
    peptides = load_peptides(args.peptides)
    if not peptides:
        raise ValueError("No peptides found in the provided candidate file")
    source_label = args.source_label or Path(args.peptides).stem
    checkpoint_identity = {
        **identity,
        "input_fingerprint": input_fingerprint(
            target_name, target_seq, peptides, args.max_length, world_size,
            source_label=source_label, fp16=args.fp16, bf16=args.bf16,
            attn_implementation=args.attn_implementation,
        ),
    }
    if not args.no_resume:
        for shard_rank in range(world_size):
            validate_resume_identity(output_path, shard_rank, **checkpoint_identity)

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from .modeling import load_grade_tokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device, _ = setup_distributed()

    rank = get_rank()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── checkpoint / resume detection ──────────────────────────────────
    chkpt_path = checkpoint_path(output_path, rank)
    rank_csv = output_path.parent / f"{output_path.stem}_rank{rank}.csv"

    processed_count, csv_mode, write_header = resolve_resume_point(
        output_path, rank, force_fresh=args.no_resume, **checkpoint_identity
    )

    if is_main_process():
        print(f"Target: {target_name}")
        print(f"Candidates: {len(peptides)}")
        print(f"Distributed world size: {int(os.environ.get('WORLD_SIZE', '1'))}")
        if processed_count > 0:
            print(f"[checkpoint] Resuming — {processed_count} candidates already processed by rank {rank}")

    # ── dataset / sampler / dataloader ─────────────────────────────────
    tokenizer = load_grade_tokenizer(args.model_dir)
    dataset = GPCRPeptideDataset(target_name=target_name, target_seq=target_seq, peptides=peptides)

    rank_exhausted = False
    if is_dist_enabled() and processed_count > 0:
        per_rank = math.ceil(len(dataset) / dist.get_world_size())
        if processed_count >= per_rank:
            if is_main_process():
                print(f"[checkpoint] Rank {rank} shard exhausted ({processed_count}/{per_rank}) — skipping inference")
            rank_exhausted = True

    sampler = SequentialDistributedSampler(dataset, skip=processed_count) if is_dist_enabled() else None
    if not is_dist_enabled() and processed_count > 0:
        sampler = SkipSampler(dataset, skip=processed_count)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=PairCollator(tokenizer, target_seq=target_seq, max_length=args.max_length),
        drop_last=False,
    )

    # ── model ──────────────────────────────────────────────────────────
    model = build_model(args, device)
    autocast_dtype = get_autocast_dtype(args)

    # ── open rank CSV for incremental writing ──────────────────────────
    with rank_csv.open(csv_mode, newline="", encoding="utf-8") as csv_fh:
        csv_writer = csv.writer(csv_fh)
        if write_header:
            csv_writer.writerow(["number", "Target", "Peptide_name", "src", "sequence", "score"])
            csv_fh.flush()
            os.fsync(csv_fh.fileno())

        # ── inference loop with incremental checkpointing ──────────────────
        buffer: List[dict] = []
        total_processed = processed_count
        batch_count = 0
        target_encoding: torch.Tensor | None = None

        if not rank_exhausted:
            progress = tqdm(dataloader, disable=not is_main_process(), desc="Inference")

            for batch in progress:
                peptide_names = batch.pop("peptide_names")
                peptide_seqs = batch.pop("peptide_seqs")
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

                with torch.no_grad():
                    if autocast_dtype is not None and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            logits, target_encoding = score_batch(
                                model,
                                batch,
                                target_encoding=target_encoding,
                                use_target_cache=not args.no_target_cache,
                            )
                    else:
                        logits, target_encoding = score_batch(
                            model,
                            batch,
                            target_encoding=target_encoding,
                            use_target_cache=not args.no_target_cache,
                        )
                    scores = F.softmax(logits, dim=-1)[:, 1]

                for peptide_name, peptide_seq, score in zip(peptide_names, peptide_seqs, tensor_to_list(scores)):
                    buffer.append(
                        {
                            "Target": target_name,
                            "Peptide_name": peptide_name,
                            "sequence": peptide_seq,
                            "score": float(score),
                        }
                    )

                batch_count += 1
                total_processed += len(peptide_names)

                if batch_count % args.checkpoint_interval == 0 and buffer:
                    flush_rank_csv_buffer(csv_writer, csv_fh, buffer, source_label)
                    buffer.clear()
                    write_checkpoint(chkpt_path, total_processed, **checkpoint_identity)

        # ── final flush ────────────────────────────────────────────────────
        if buffer:
            flush_rank_csv_buffer(csv_writer, csv_fh, buffer, source_label)
            buffer.clear()
            write_checkpoint(chkpt_path, total_processed, **checkpoint_identity)

    if is_main_process():
        print(f"Rank {rank} finished: {total_processed - processed_count} new candidates processed "
              f"({total_processed} total)")

    cleanup_distributed()

    # ── merge: rank 0 collects all rank shards → sorted top-k output ─────
    if is_main_process() and is_dist_enabled():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank_outputs = [
            output_path.parent / f"{output_path.stem}_rank{r}.csv"
            for r in range(world_size)
        ]
        checkpoint_files = [checkpoint_path(output_path, r) for r in range(world_size)]

        written = finalize_output(
            output_path=output_path,
            rank_csv_paths=rank_outputs,
            checkpoint_paths=checkpoint_files,
            source_label=source_label,
            top_k=args.top_k,
            output_format=args.output_format,
        )
        print(f"Merged {written} rows to {output_path}")
    elif not is_dist_enabled():
        # Single-process: read rank CSV, sort, apply top-k, write final output
        if rank_csv.exists():
            written = finalize_output(
                output_path=output_path,
                rank_csv_paths=[rank_csv],
                checkpoint_paths=[chkpt_path],
                source_label=source_label,
                top_k=args.top_k,
                output_format=args.output_format,
            )
            print(f"Saved {written} sorted rows to {output_path}")
        else:
            chkpt_path.unlink(missing_ok=True)


def main() -> None:
    from .runner import run

    args = parse_args()
    run(args, run_inference)


if __name__ == "__main__":
    main()
