"""Command-line parsing without inference runtime dependencies."""

import argparse
from collections.abc import Sequence

from .model_registry import DEFAULT_MODEL_ID, available_models, get_model_spec
from .writers import infer_format_from_path, supported_formats


DEFAULT_MODEL_DIR = get_model_spec(DEFAULT_MODEL_ID).local_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRADE: GPCR-Peptide interaction prediction with cross-attention."
    )
    parser.add_argument("--target-name", help="Target name in --target-fasta or the default GPCR FASTA.")
    parser.add_argument("--target-seq", help="Raw target amino acid sequence.")
    parser.add_argument("--target-fasta", help="FASTA file containing target sequence(s).")
    parser.add_argument(
        "--peptides", required=True,
        help="Candidate peptides file. Supported: .fasta/.faa/.fa, .csv, .txt",
    )
    parser.add_argument(
        "--output-csv", required=True,
        help="Output path. Extension and --output-format determine format (csv/tsv/json).",
    )
    parser.add_argument(
        "--output-format", default=None, choices=supported_formats(),
        help="Output format (default: inferred from --output-csv extension, otherwise csv).",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Keep top K results in output.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-process batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per process.")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 autocast on CUDA.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on CUDA.")
    parser.add_argument(
        "--attn-implementation", default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for protein sequence encoders (default: sdpa).",
    )
    parser.add_argument("--source-label", default=None, help="Optional source label to store in output.")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100,
        help="Save checkpoint every N batches (default: 100).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Force a fresh start and replace existing checkpoint and rank CSV files.",
    )
    parser.add_argument(
        "--no-target-cache", action="store_true",
        help="Disable per-run GPCR encoding reuse and use the reference model path.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model", choices=available_models(), default=None,
        help=f"Registered model to use (default: {DEFAULT_MODEL_ID}).",
    )
    selection.add_argument(
        "--model-dir", default=None,
        help="Custom directory containing model weights and tokenizer; excludes --model.",
    )
    args = parser.parse_args(argv)

    if args.model_dir is None:
        args.model = args.model or DEFAULT_MODEL_ID
        args.model_dir = str(get_model_spec(args.model).local_dir)

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
