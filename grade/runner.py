"""Dispatch the unified CLI to Cloud or explicitly selected local inference."""
import importlib.util
import sys
from pathlib import Path

from . import cloud
from .inputs import load_peptides
from .tcm import resolve_target
from .writers import get_writer


def run_cloud(args):
    token = ""
    try:
        if not args.usage:
            target_name, target_seq = resolve_target(args.target_name, args.target_seq, args.target_fasta)
            candidates = load_peptides(args.peptides)
            if not candidates:
                raise ValueError("No peptides found in the provided candidate file")
            cloud.check_output_path(Path(args.output_csv))
        path = cloud.config_path()
        saved = cloud.load_config(path)
        token = saved.get("client_token", "")
        email = cloud.resolve_email(args.email, saved.get("email"))
        identity = cloud.client_identity(email, path)
        token = identity["client_token"]
        if args.usage:
            data = cloud.query_usage(email, token)
            print(cloud._redact(f"Daily quota remaining: {data['remaining']}/{data['limit']}. Resets at {data['reset_at']}.", token))
            return
        print(f"Using Cloud inference: {args.model}", file=sys.stderr)
        peptides = [{"name": name, "sequence": sequence} for name, sequence in candidates.items()]
        results = cloud.call_api(target_seq, peptides, email=email, client_token=token, model=args.model)
        # Check correspondence before publishing a table labelled with this input.
        returned = {row["peptide_name"]: row["sequence"] for row in results}
        if len(returned) != len(results) or returned != candidates:
            raise ValueError("Cloud results do not match the submitted candidates; output was not replaced.")
        results.sort(key=lambda row: float(row["score"]), reverse=True)
        if args.top_k is not None:
            results = results[:args.top_k]
        rows = [{
            "rank": rank, "target": target_name, "peptide_name": row["peptide_name"],
            "src": args.source_label or "grade-api", "sequence": row["sequence"], "score": float(row["score"]),
        } for rank, row in enumerate(results, 1)]
        try:
            get_writer(args.output_format).write(Path(args.output_csv), rows)
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not write output: {error}. Prediction completed and quota may have been used.") from None
        print(f"Saved {len(rows)} results to {args.output_csv}", file=sys.stderr)
    except (ValueError, KeyError, OSError, UnicodeError) as error:
        cloud._abort(cloud._redact(str(error), token))


def run_local(args, local_inference):
    # The explicit path is authoritative: errors never fall back to Cloud.
    if not Path(args.model_dir).is_dir():
        cloud._abort(f"Local model directory does not exist: {args.model_dir}")
    missing = [name for name in ("torch", "transformers", "safetensors", "tqdm") if importlib.util.find_spec(name) is None]
    if missing:
        cloud._abort(
            "Local inference dependencies are missing: " + ", ".join(missing)
            + '. Install them with python -m pip install "grade-client[local]". Local inference uses Python 3.11.'
        )
    print(f"Using local inference: {Path(args.model_dir).resolve()}", file=sys.stderr)
    local_inference(args)


def run(args, local_inference):
    if args.mode == "local":
        run_local(args, local_inference)
    else:
        run_cloud(args)
