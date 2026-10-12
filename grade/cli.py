"""Unified Cloud/local command-line parsing without inference dependencies."""

import argparse
import os
from collections.abc import Sequence

from . import __version__
from .model_registry import DEFAULT_MODEL_ID, available_models, get_model_spec
from .writers import infer_format_from_path, supported_formats

# Retained for Python callers; not used as an implicit local model selection.
DEFAULT_MODEL_DIR = get_model_spec(DEFAULT_MODEL_ID).local_dir
LOCAL_DEFAULTS = {
    "batch_size": 8, "num_workers": 0, "max_length": 1024,
    "fp16": False, "bf16": False, "attn_implementation": "sdpa",
    "checkpoint_interval": 100, "no_resume": False, "no_target_cache": False,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRADE: use Cloud by default, or --model-dir for local GPCR-peptide inference.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"grade-client {__version__}")
    parser.add_argument("--target-name", help="Full target header in --target-fasta or the bundled GPCR FASTA.")
    parser.add_argument("--target-seq", help="Raw target amino acid sequence.")
    parser.add_argument("--target-fasta", help="FASTA file containing target sequence(s).")
    parser.add_argument("--peptides", help="Candidate file (.fasta/.faa/.fa, .csv, .txt).")
    parser.add_argument("--output", "--output-csv", "-o", dest="output_csv", help="Output file (CSV, TSV or JSON); --output-csv is a compatibility alias.")
    parser.add_argument("--output-format", choices=supported_formats(), help="Output format; otherwise inferred from the output extension.")
    parser.add_argument("--top-k", type=int, help="Keep the highest K results; Cloud quota counts all submitted peptides.")
    parser.add_argument("--source-label", help="Optional src label; default: grade-api on Cloud, candidate file stem locally.")
    parser.add_argument("--email", help="Cloud email; otherwise GRADE_EMAIL, saved email, or terminal input.")
    parser.add_argument("--usage", action="store_true", help="Query Cloud quota without prediction inputs.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=available_models(), help=f"Cloud model (default: {DEFAULT_MODEL_ID}).")
    selection.add_argument("--model-dir", help="Use local weights/tokenizer from this directory instead of Cloud.")

    local = parser.add_argument_group("local inference only (requires --model-dir)")
    local.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Per-process batch size (default: 8).")
    local.add_argument("--num-workers", type=int, default=argparse.SUPPRESS, help="DataLoader workers per process (default: 0).")
    local.add_argument("--max-length", type=int, default=argparse.SUPPRESS, help="Tokenizer length (default: 1024).")
    local.add_argument("--fp16", action="store_true", default=argparse.SUPPRESS, help="Float16 autocast on CUDA.")
    local.add_argument("--bf16", action="store_true", default=argparse.SUPPRESS, help="Bfloat16 autocast on CUDA.")
    local.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=argparse.SUPPRESS, help="Encoder attention (default: sdpa).")
    local.add_argument("--checkpoint-interval", type=int, default=argparse.SUPPRESS, help="Save progress every N batches (default: 100).")
    local.add_argument("--no-resume", action="store_true", default=argparse.SUPPRESS, help="Start fresh, replacing progress and output.")
    local.add_argument("--no-target-cache", action="store_true", default=argparse.SUPPRESS, help="Disable target encoding reuse.")
    args = parser.parse_args(argv)

    if args.model_dir is not None and not args.model_dir.strip():
        parser.error("--model-dir must be a non-empty path")
    args.mode = "local" if args.model_dir is not None else "cloud"
    explicit_local = [name for name in LOCAL_DEFAULTS if hasattr(args, name)]
    if args.mode == "cloud" and explicit_local:
        options = ", ".join("--" + name.replace("_", "-") for name in explicit_local)
        parser.error(f"{options} require --model-dir; Cloud computation settings are fixed")
    if args.mode == "cloud" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Distributed execution requires --model-dir; refusing duplicate Cloud submissions")
    if args.mode == "local" and (args.email is not None or args.usage):
        parser.error("--email and --usage are Cloud-only options")
    for name, default in LOCAL_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    if args.mode == "cloud":
        args.model = args.model or DEFAULT_MODEL_ID

    if args.usage:
        if any(value is not None for value in (
            args.target_name, args.target_seq, args.target_fasta, args.peptides,
            args.output_csv, args.output_format, args.top_k, args.source_label,
        )):
            parser.error("--usage cannot be combined with prediction or output arguments")
        return args

    if not args.peptides or not args.output_csv:
        parser.error("--peptides and --output are required for prediction")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be a positive integer")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be a positive integer")
    if args.fp16 and args.bf16:
        parser.error("--fp16 and --bf16 cannot be enabled at the same time")
    if not any((args.target_name, args.target_seq, args.target_fasta)):
        parser.error("one of --target-name, --target-seq, or --target-fasta is required")
    if args.target_seq and (args.target_name or args.target_fasta):
        parser.error("--target-seq cannot be used together with --target-name or --target-fasta")
    if args.output_format is None:
        args.output_format = infer_format_from_path(args.output_csv)
    return args
