#!/usr/bin/env python3
"""GRADE-paper API client — predict GPCR-peptide interactions via the cloud API.

Usage:
    python grade_predict.py --target-seq MRYR... --peptides candidates.fasta
    python grade_predict.py --api-key KEY --target-seq MRYR... --peptides peptides.csv --top-k 20 -o top20.csv

The ``requests`` dependency is imported only when a request is submitted.
"""

import argparse
import csv
import getpass
import json
import math
import os
import sys
import tempfile
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

API_URL = "https://devxia--grade-web.modal.run/predict"
SUPPORTED_MODEL = "GRADE-paper"
API_KEY_ENV = "GRADE_API_KEY"
API_KEY_REQUEST_EMAIL = "xiabb@im.ac.cn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRADE API client — predict GPCR-peptide interactions."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="GRADE API key. If omitted, use GRADE_API_KEY or hidden terminal input.",
    )
    parser.add_argument("--target-seq", required=True, help="Target GPCR amino acid sequence.")
    parser.add_argument(
        "--peptides",
        required=True,
        help="Peptide file (.fasta/.faa/.fa/.csv/.txt).",
    )
    parser.add_argument(
        "--model",
        default=SUPPORTED_MODEL,
        choices=[SUPPORTED_MODEL],
        help=f"Model to use (currently only {SUPPORTED_MODEL}; default: {SUPPORTED_MODEL}).",
    )
    parser.add_argument("--output", "-o", default=None, help="Output path (default: stdout). Extension selects format.")
    parser.add_argument("--output-format", default=None, choices=["csv", "tsv", "json"], help="Output format (default: inferred from --output extension, otherwise csv).")
    parser.add_argument("--top-k", type=int, default=None, help="Keep top K results.")
    args = parser.parse_args()

    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be a positive integer")

    if args.output_format is None:
        if args.output:
            suffix = Path(args.output).suffix.lower().lstrip(".")
            args.output_format = suffix if suffix in {"csv", "tsv", "json"} else "csv"
        else:
            args.output_format = "csv"

    return args


def _nonempty_key(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _interactive_terminal(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def _missing_key_message() -> str:
    return (
        "No GRADE API key was provided. Request one from the authors at "
        f"{API_KEY_REQUEST_EMAIL}. Provide it with --api-key, set {API_KEY_ENV}, "
        "or run in an interactive terminal for hidden input."
    )


def resolve_api_key(explicit_key: Optional[str]) -> str:
    """Resolve a key without printing or persisting it.

    Precedence is explicit argument, environment variable, then hidden terminal input.
    Empty values are treated as absent. Plain-text input is never used as a fallback.
    """
    key = _nonempty_key(explicit_key) or _nonempty_key(os.environ.get(API_KEY_ENV))
    if key:
        return key

    if not _interactive_terminal(sys.stdin) or not _interactive_terminal(sys.stderr):
        raise RuntimeError(_missing_key_message())

    print(
        f"If you do not have an API key, request one from the authors at {API_KEY_REQUEST_EMAIL}.",
        file=sys.stderr,
    )
    try:
        with warnings.catch_warnings():
            # getpass warns immediately before its plain-text fallback. Raising here
            # prevents that fallback from reading any input.
            warnings.simplefilter("error", getpass.GetPassWarning)
            entered = getpass.getpass("GRADE API key (input hidden): ", stream=sys.stderr)
    except (getpass.GetPassWarning, OSError):
        raise RuntimeError(
            "Hidden API key input is unavailable; refusing plain-text input. "
            f"Provide --api-key or {API_KEY_ENV}."
        ) from None
    except (EOFError, KeyboardInterrupt):
        raise RuntimeError("No API key was entered; refusing to submit a request.") from None

    key = _nonempty_key(entered)
    if not key:
        raise RuntimeError("No API key was entered; refusing to submit a request.")
    return key


def normalize_sequence(seq: str) -> str:
    return "".join(seq.split()).upper()


def validate_sequence(seq: str, field_name: str = "sequence") -> str:
    if not isinstance(seq, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = normalize_sequence(seq)
    if not normalized:
        raise ValueError(f"{field_name} is empty after normalization")
    return normalized


def read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    current_id = None
    seq_parts: List[str] = []

    def store_record(record_id: str, parts: List[str]) -> None:
        if record_id in records:
            raise ValueError(f"Duplicate FASTA record id {record_id!r} in {path}")
        records[record_id] = "".join(parts)

    with path.open("r", encoding="utf-8-sig") as handle:
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
                f"FASTA file {path} has sequence data before the first '>' header"
            )
    return records


def add_unique_peptide(
    peptides: List[Dict[str, str]],
    seen_names: set[str],
    name: str,
    sequence: str,
    source: Path,
) -> None:
    if name in seen_names:
        raise ValueError(f"Duplicate peptide name {name!r} in {source}")
    seen_names.add(name)
    peptides.append({"name": name, "sequence": validate_sequence(sequence, f"peptide {name!r}")})


def load_peptides(path_str: str) -> List[Dict[str, str]]:
    """Load peptides from a file. Returns a list of {"name": ..., "sequence": ...}."""
    path = Path(path_str)
    suffix = path.suffix.lower()
    peptides: List[Dict[str, str]] = []
    seen_names: set[str] = set()

    if suffix in {".fasta", ".faa", ".fa"}:
        records = read_fasta(path)
        for name, seq in records.items():
            add_unique_peptide(peptides, seen_names, name, seq, path)

    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            name_key = next(
                (k for k in fieldnames if k.lower() in {"name", "peptide_name", "id"}), None
            )
            seq_key = next(
                (k for k in fieldnames if k.lower() in {"sequence", "peptide", "seq"}), None
            )
            if seq_key is None:
                raise ValueError(f"CSV file {path} must contain a sequence column")
            for idx, row in enumerate(reader, start=1):
                name = row.get(name_key) if name_key else None
                add_unique_peptide(
                    peptides,
                    seen_names,
                    name or f"peptide_{idx}",
                    row[seq_key],
                    path,
                )

    elif suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            for idx, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 1:
                    add_unique_peptide(peptides, seen_names, f"peptide_{idx}", parts[0], path)
                else:
                    add_unique_peptide(peptides, seen_names, parts[0], parts[-1], path)

    else:
        raise ValueError(
            f"Unsupported peptide file format: {suffix}. "
            f"Supported: .fasta/.faa/.fa, .csv, .txt"
        )

    if not peptides:
        raise ValueError(f"No valid peptides found in {path}")

    return peptides


def _redact(text: str, api_key: str) -> str:
    """Do not echo a key reflected in a server error or model identity."""
    for secret in (api_key, json.dumps(api_key)[1:-1], repr(api_key)[1:-1]):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def api_error_message(resp: Any, api_key: str = "") -> str:
    """Return the API-provided error text, with the submitted key redacted."""
    try:
        data = resp.json()
    except (ValueError, TypeError):
        message = getattr(resp, "text", "") or "Unknown error"
    else:
        if isinstance(data, dict):
            data = data.get("error") or data.get("detail") or data
        message = str(data or "Unknown error")
    return _redact(message, api_key)


def _response_model(resp: Any, data: Dict[str, Any]) -> Optional[str]:
    """Read and validate the model identity reported by JSON and/or a header."""
    identities: List[str] = []
    if "model" in data:
        model = data["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("The GRADE API returned an invalid model identity.")
        identities.append(model)

    headers = getattr(resp, "headers", None) or {}
    if "X-GRADE-Model" in headers:
        header_model = headers["X-GRADE-Model"]
        if not isinstance(header_model, str) or not header_model.strip():
            raise ValueError("The GRADE API returned an invalid X-GRADE-Model identity.")
        identities.append(header_model)

    if not identities:
        print(
            "Warning: server did not report model identity; cannot verify requested model.",
            file=sys.stderr,
        )
        return None

    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("The GRADE API returned inconsistent model identities.")
    return identities[0]


def _abort(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1) from None


def check_output_path(path: Path) -> None:
    """Probe output directory without creating or truncating the result file."""
    if path.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {path}")
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(f"Output file is not writable: {path}")
    with tempfile.TemporaryFile(dir=path.parent):
        pass


def call_api(
    api_key: str,
    target_seq: str,
    peptides: List[Dict[str, str]],
    model: str = SUPPORTED_MODEL,
) -> List[Dict[str, Any]]:
    """Submit peptides to the GRADE API. Returns the compatible results list."""
    if model != SUPPORTED_MODEL:
        raise ValueError(f"Unsupported model. Only {SUPPORTED_MODEL} is currently available.")
    api_key = _nonempty_key(api_key)
    if not api_key:
        raise ValueError("API key must be a non-empty string.")

    try:
        import requests
    except ImportError:
        _abort("The requests package is required to submit a request.")

    try:
        resp = requests.post(
            API_URL,
            json={
                "api_key": api_key,
                "target_seq": target_seq,
                "peptides": peptides,
                "model": model,
            },
            timeout=300,
        )
    except requests.Timeout:
        _abort(
            "The GRADE API request timed out. The service may be starting up; "
            "the request may have reached the server and used quota. Check before retrying."
        )
    except requests.ConnectionError:
        _abort("Could not connect to the GRADE API. Check connectivity and the endpoint before retrying.")
    except requests.RequestException:
        _abort("Unable to complete the GRADE API request. Check connectivity and retry.")

    if resp.status_code == 401:
        _abort(
            "Invalid API key. Check that your key is valid. If you need a key, "
            f"request one from the authors at {API_KEY_REQUEST_EMAIL}."
        )
    elif resp.status_code == 422:
        _abort(
            f"{api_error_message(resp, api_key)}\n"
            "The request was rejected by the GRADE API. Check the API message above; "
            "it may reflect current cloud-side validation or request-size policy."
        )
    elif resp.status_code == 429:
        _abort(
            f"{api_error_message(resp, api_key)}\n"
            "A cloud-side usage or rate limit was reached. Limits are enforced by the "
            "hosted service and may change; retry later, reduce the request, or contact "
            "the GRADE authors if you need higher throughput."
        )
    elif resp.status_code != 200:
        _abort(f"API returned status {resp.status_code}: {api_error_message(resp, api_key)}")

    try:
        data = resp.json()
    except (ValueError, TypeError):
        _abort("API returned invalid JSON.")
    if not isinstance(data, dict):
        _abort("API returned an invalid response format.")

    try:
        actual_model = _response_model(resp, data)
    except ValueError as exc:
        _abort(str(exc))
    if actual_model is not None:
        print(f"Server reported model: {_redact(actual_model, api_key)}", file=sys.stderr)
        if actual_model != model:
            _abort(_redact(
                f"API returned model {actual_model!r}, but {model!r} was requested.", api_key
            ))

    results = data.get("results")
    if not isinstance(results, list):
        _abort("API returned a missing or invalid results list.")
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict) or not all(
            isinstance(item.get(field), str) and item[field]
            for field in ("peptide_name", "sequence")
        ):
            _abort(f"API result {index} is missing a valid peptide_name or sequence.")
        score = item.get("score")
        try:
            valid_score = not isinstance(score, bool) and math.isfinite(float(score))
        except (TypeError, ValueError, OverflowError):
            valid_score = False
        if not valid_score:
            _abort(f"API result {index} has an invalid score.")
    return results


def main() -> None:
    args = parse_args()

    provided_key = _nonempty_key(args.api_key) or _nonempty_key(os.environ.get(API_KEY_ENV))
    if not provided_key and (
        not _interactive_terminal(sys.stdin) or not _interactive_terminal(sys.stderr)
    ):
        _abort(_missing_key_message())

    try:
        target_seq = validate_sequence(args.target_seq, "target sequence")
        peptides = load_peptides(args.peptides)
        if args.output:
            check_output_path(Path(args.output))
    except (ValueError, OSError, UnicodeError) as exc:
        _abort(_redact(str(exc), provided_key or ""))

    try:
        api_key = resolve_api_key(args.api_key)
    except RuntimeError as exc:
        _abort(str(exc))

    print(f"Loaded {len(peptides)} peptides from {args.peptides}", file=sys.stderr)
    print(f"Submitting {len(peptides)} peptides to the GRADE API...", file=sys.stderr)

    all_results = call_api(api_key, target_seq, peptides, model=args.model)

    # Sort by score descending
    all_results.sort(key=lambda x: float(x["score"]), reverse=True)

    # Apply top-k
    if args.top_k is not None:
        all_results = all_results[: args.top_k]
        print(f"Keeping top {args.top_k} results", file=sys.stderr)

    # Build a unified result representation with rank for all formats
    ranked_results = [
        {
            "rank": idx,
            "peptide_name": item["peptide_name"],
            "sequence": item["sequence"],
            "score": item["score"],
        }
        for idx, item in enumerate(all_results, start=1)
    ]

    # Output in requested format
    try:
        destination = open(args.output, "w", newline="", encoding="utf-8") if args.output else nullcontext(sys.stdout)
        with destination as output_fh:
            if args.output_format == "json":
                json.dump(ranked_results, output_fh, indent=2)
                output_fh.write("\n")
            else:
                delimiter = "\t" if args.output_format == "tsv" else ","
                writer = csv.DictWriter(
                    output_fh,
                    fieldnames=["rank", "peptide_name", "sequence", "score"],
                    delimiter=delimiter,
                )
                writer.writeheader()
                for item in ranked_results:
                    writer.writerow(item)
    except (OSError, UnicodeError) as exc:
        _abort(_redact(
            f"Could not write output: {exc}. The API request has already completed; quota may have been used.",
            api_key,
        ))

    if args.output:
        print(f"Saved {len(ranked_results)} results to {args.output}", file=sys.stderr)
    else:
        print(f"\n{len(ranked_results)} results.", file=sys.stderr)


if __name__ == "__main__":
    main()
