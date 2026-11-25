"""Standalone GRADE Cloud implementation and independent Skill CLI.

This is the maintained source for skills/grade-cloud/scripts/grade_predict.py.
Run python scripts/sync_cloud_skill.py after editing this module.

Usage:
    python grade_predict.py --target-seq MRYR... --peptides candidates.fasta
    python grade_predict.py login --no-browser

The ``requests`` dependency is imported only when a request is submitted.
"""

import argparse
import csv
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
from urllib.parse import urlsplit

API_URL = "https://devxia--grade-web.modal.run/predict"
SKILL_VERSION = "1.3.0"
SUPPORTED_MODEL = "GRADE-paper"
API_BASE = (os.environ.get("GRADE_CLOUD_URL") or API_URL.rsplit("/", 1)[0]).rstrip("/")
AUTH_BASE = (os.environ.get("GRADE_AUTH_URL") or "https://grade.devxia.cc").rstrip("/")
CONFIG_VERSION = 2
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


def parse_auth_args(argv):
    if not argv or argv[0] not in {"login", "logout", "whoami"}:
        return None
    command = argv[0]
    parser = argparse.ArgumentParser(prog=f"{Path(sys.argv[0]).name} {command}", allow_abbrev=False)
    if command == "login":
        parser.add_argument("--email", help="Expected verified email (optional).")
        parser.add_argument("--no-browser", action="store_true", help="Print the browser instructions without opening a tab.")
    return parser.parse_args(argv[1:], namespace=argparse.Namespace(auth_command=command))


def parse_args() -> argparse.Namespace:
    auth = parse_auth_args(sys.argv[1:])
    if auth is not None:
        return auth
    parser = argparse.ArgumentParser(description="GRADE Cloud inference. Account commands: login, logout, whoami.")
    parser.add_argument("--version", action="version", version=f"grade-cloud {SKILL_VERSION}")
    parser.add_argument("--email", help="Expected verified email (optional).")
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("--usage", action="store_true", help="Show remaining daily quota without submitting a prediction.")
    parser.add_argument("--target-seq", help="Target GPCR amino acid sequence.")
    parser.add_argument(
        "--peptides",
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

    if args.api_key is not None:
        parser.error("API keys are no longer supported. Run login.")
    if not args.usage and (not args.target_seq or not args.peptides):
        parser.error("--target-seq and --peptides are required unless --usage is used")
    if args.usage and any((args.target_seq, args.peptides, args.output, args.output_format, args.top_k is not None)):
        parser.error("--usage cannot be combined with prediction or output arguments")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be a positive integer")

    if args.output_format is None:
        if args.output:
            suffix = Path(args.output).suffix.lower().lstrip(".")
            args.output_format = suffix if suffix in {"csv", "tsv", "json"} else "csv"
        else:
            args.output_format = "csv"

    return args


def _nonempty(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _interactive_terminal(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def normalize_email(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Please enter a valid email address.")
    value = value.strip().lower()
    if (
        len(value) > 254
        or not re.fullmatch(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,63}", value)
        or len(value.split("@")[0]) > 64
        or any(not part or part.startswith("-") or part.endswith("-") or len(part) > 63
               for part in value.split("@")[1].split("."))
        or value.startswith(".") or ".@" in value or ".." in value
    ):
        raise ValueError("Please enter a valid email address.")
    return value


def config_path() -> Path:
    if override := os.environ.get("GRADE_CONFIG_DIR"):
        root = Path(override)
        if not root.is_absolute():
            raise ValueError("GRADE_CONFIG_DIR must be an absolute path.")
        return root / "client.json"
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
        return root / "GRADE" / "client.json"
    if os.name == "nt":
        root = Path(_nonempty(os.environ.get("LOCALAPPDATA")) or Path.home() / "AppData" / "Local")
        suffix = Path("GRADE/client.json")
    else:
        root = Path(_nonempty(os.environ.get("XDG_CONFIG_HOME")) or Path.home() / ".config")
        suffix = Path("grade/client.json")
    if not root.is_absolute():
        raise ValueError("Client configuration directory must be an absolute path.")
    return root / suffix


def load_config(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("Client configuration must not be a symbolic link.")
    if not path.exists():
        return {}
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("Client configuration permissions must be 600.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == 1 and data.get("api_base") == API_BASE:
            return {}
        if (not isinstance(data, dict) or data.get("version") != CONFIG_VERSION
                or data.get("api_base") != API_BASE or data.get("auth_base") != AUTH_BASE
                or not valid_token(data.get("access_token")) or not valid_token(data.get("refresh_token"))
                or type(data.get("access_end")) is not int or type(data.get("session_end")) is not int):
            raise ValueError
        normalize_email(data.get("email"))
        return data
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("Client configuration is invalid or belongs to a different service.") from None


def save_config(path: Path, data: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".client-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def config_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_suffix(".lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        if os.name == "nt":
            import msvcrt
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def valid_token(value) -> bool:
    return isinstance(value, str) and TOKEN_PATTERN.fullmatch(value) is not None


def _check_origin(value):
    url = urlsplit(value)
    local = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
    if (not url.hostname or url.username is not None or url.path or url.query or url.fragment
            or (url.scheme != "https" and not local)):
        raise ValueError("GRADE service origins require HTTPS except on loopback.")


def _auth_post(endpoint, fields):
    _check_origin(AUTH_BASE)
    try:
        import requests
    except ImportError:
        raise ValueError("The requests package is required for Cloud login.") from None
    try:
        response = requests.post(AUTH_BASE + endpoint, data={"client_id": "grade-cli", **fields},
                                 timeout=10, allow_redirects=False)
        data = response.json()
    except (requests.RequestException, ValueError):
        raise ValueError("Could not complete the request to the GRADE login service.") from None
    if not isinstance(data, dict):
        raise ValueError("Invalid response from the GRADE login service.")
    return response.status_code, data


def _session(result, saved=None):
    token, lifetime = result.get("access_token"), result.get("expires_in")
    if not valid_token(token) or type(lifetime) is not int or not 1 <= lifetime <= 1800 or result.get("token_type") != "Bearer":
        raise ValueError("Invalid access token response from GRADE.")
    now = int(time.time())
    email = normalize_email(result.get("email"))
    if saved:
        if email != saved["email"]:
            raise ValueError("GRADE login returned a different account.")
        data = saved.copy()
    else:
        refresh, duration = result.get("refresh_token"), result.get("session_expires_in")
        if not valid_token(refresh) or type(duration) is not int or not 1 <= duration <= 14 * 86400:
            raise ValueError("Invalid session response from GRADE.")
        data = {"version": CONFIG_VERSION, "auth_base": AUTH_BASE, "api_base": API_BASE,
                "refresh_token": refresh, "session_end": now + duration, "email": email}
    return {**data, "access_token": token, "access_end": min(now + lifetime, data["session_end"])}


def _revoke(token):
    status, _ = _auth_post("/oauth/revoke", {"token": token, "token_type_hint": "refresh_token"})
    if status != 200:
        raise ValueError("GRADE could not revoke the session. Try again later.")


def _device_login(expected_email, no_browser=False):
    _check_origin(API_BASE)
    status, grant = _auth_post("/oauth/device/code", {"scope": "predict usage"})
    if status != 200:
        raise ValueError("GRADE could not start device login. Try again later.")
    uri, code, device = (grant.get(key) for key in ("verification_uri", "user_code", "device_code"))
    duration, interval = grant.get("expires_in"), grant.get("interval", 5)
    if (uri != AUTH_BASE + "/device" or not isinstance(code, str) or not re.fullmatch(r"[A-HJ-NP-Z2-9]{4}-[A-HJ-NP-Z2-9]{4}", code)
            or not valid_token(device) or type(duration) is not int or not 1 <= duration <= 600
            or type(interval) is not int or not 1 <= interval <= 60):
        raise ValueError("Invalid device login response from GRADE.")
    print(f"Open {uri} and enter code {code}.", file=sys.stderr, flush=True)
    if expected_email:
        print(f"Authorize with {expected_email}.", file=sys.stderr, flush=True)
    if not no_browser and _interactive_terminal(sys.stderr):
        import webbrowser
        try:
            webbrowser.open(uri)
        except (OSError, webbrowser.Error):
            pass
    deadline = time.monotonic() + duration
    while deadline - time.monotonic() > interval:
        time.sleep(interval)
        status, result = _auth_post("/oauth/token", {"grant_type": DEVICE_GRANT, "device_code": device})
        if status == 200:
            session = _session(result)
            if expected_email and session["email"] != expected_email:
                _revoke(session["refresh_token"])
                raise ValueError("The verified email differs from the requested account. Run login again.")
            return session
        error = result.get("error") if status == 400 else None
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "access_denied":
            raise ValueError("Device login was declined.")
        if error == "expired_token":
            break
        raise ValueError("Device login could not be completed. Run login again.")
    raise ValueError("The device code expired. Run login again.")


def _email_hint(explicit):
    hint = _nonempty(explicit) or _nonempty(os.environ.get("GRADE_EMAIL"))
    return normalize_email(hint) if hint else None


def client_identity(email: Optional[str], path: Path, *, interactive=True) -> dict:
    email = _email_hint(email)
    with config_lock(path):
        saved = load_config(path)
        now = time.time()
        if saved and saved["session_end"] > now:
            if email and saved["email"] != email:
                raise ValueError("The requested email differs from the active GRADE account. Run logout, then login.")
            if saved["access_end"] > now + 30:
                return saved
            status, result = _auth_post("/oauth/token", {"grant_type": "refresh_token", "refresh_token": saved["refresh_token"]})
            if status == 200:
                data = _session(result, saved)
                save_config(path, data)
                return data
            if status != 400 or result.get("error") != "invalid_grant":
                raise ValueError("GRADE could not refresh your login. Try again later.")
        if not interactive or not (_interactive_terminal(sys.stdin) and _interactive_terminal(sys.stderr)):
            raise ValueError("No active GRADE login. Run login in a terminal first.")
        data = _device_login(email)
        save_config(path, data)
        return data


def run_auth_command(args) -> None:
    try:
        path = config_path()
        if args.auth_command == "whoami":
            data = client_identity(None, path, interactive=False)
            query_usage(data["access_token"])
            print(data["email"])
            return
        with config_lock(path):
            old = load_config(path)
            if args.auth_command == "logout":
                if old:
                    _revoke(old["refresh_token"])
                    path.unlink(missing_ok=True)
                print("Signed out of GRADE.")
                return
            data = _device_login(_email_hint(args.email), args.no_browser)
            if old:
                _revoke(old["refresh_token"])
            save_config(path, data)
            print(f"Signed in to GRADE as {data['email']}.")
    except ValueError as error:
        _abort(str(error))
    except OSError:
        _abort("Could not update the GRADE login configuration.")
    except KeyboardInterrupt:
        _abort("GRADE login was cancelled.")


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


def _redact(text: str, client_token: str) -> str:
    """Do not echo a credential reflected in a server error or model identity."""
    for secret in (client_token, json.dumps(client_token)[1:-1], repr(client_token)[1:-1]):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def api_error_message(resp: Any, client_token: str = "") -> str:
    """Return the API-provided error text, with the submitted credential redacted."""
    try:
        data = resp.json()
    except (ValueError, TypeError):
        message = getattr(resp, "text", "") or "Unknown error"
    else:
        if isinstance(data, dict):
            data = data.get("error") or data.get("detail") or data
        message = str(data or "Unknown error")
    return _redact(message, client_token)


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


def post_api(endpoint: str, body: dict, client_token: str = "") -> Any:
    """Submit each Cloud request once."""
    _check_origin(API_BASE)
    try:
        import requests
    except ImportError:
        _abort("The requests package is required to submit a request.")

    try:
        resp = requests.post(
            API_BASE + endpoint,
            json=body,
            headers={"Authorization": f"Bearer {client_token}"} if client_token else {},
            timeout=300 if endpoint == "/predict" else 30,
            allow_redirects=False,
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
        _abort("GRADE login is no longer active. Run login again.")
    elif resp.status_code == 422:
        _abort(
            f"{api_error_message(resp, client_token)}\n"
            "The request was rejected by the GRADE API. Check the API message above; "
            "it may reflect current cloud-side validation or request-size policy."
        )
    elif resp.status_code == 429:
        headers = getattr(resp, "headers", {}) or {}
        reset = headers.get("X-GRADE-Quota-Reset")
        retry = headers.get("Retry-After")
        advice = f" Daily quota resets at {reset}." if reset else (f" Retry after {retry} seconds." if retry else " Retry later.")
        _abort(_redact(f"{api_error_message(resp, client_token)}\nA cloud-side usage or rate limit was reached.{advice}", client_token))
    elif resp.status_code != 200:
        _abort(f"API returned status {resp.status_code}: {api_error_message(resp, client_token)}")
    return resp


def show_quota(resp: Any, client_token: str) -> None:
    headers = getattr(resp, "headers", {}) or {}
    remaining, limit, reset = (headers.get(f"X-GRADE-Quota-{name}") for name in ("Remaining", "Limit", "Reset"))
    if remaining is not None and limit is not None and reset is not None:
        print(_redact(f"Daily quota remaining: {remaining}/{limit}. Resets at {reset}.", client_token), file=sys.stderr)


def call_api(
    target_seq: str,
    peptides: List[Dict[str, str]],
    *,
    client_token: str,
    model: str = SUPPORTED_MODEL,
) -> List[Dict[str, Any]]:
    """Submit one prediction; credentials are keyword-only to reject old Key calls."""
    if model != SUPPORTED_MODEL:
        raise ValueError(f"Unsupported model. Only {SUPPORTED_MODEL} is currently available.")
    if not valid_token(client_token):
        raise ValueError("Client credential must be a non-empty string.")
    resp = post_api("/predict", {
        "target_seq": target_seq, "peptides": peptides, "model": model,
    }, client_token)
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
        print(f"Server reported model: {_redact(actual_model, client_token)}", file=sys.stderr)
        if actual_model != model:
            _abort(_redact(
                f"API returned model {actual_model!r}, but {model!r} was requested.", client_token
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
    show_quota(resp, client_token)
    return results


def query_usage(client_token: str) -> dict:
    resp = post_api("/usage", {}, client_token)
    try:
        data = resp.json()
        if (not isinstance(data, dict) or type(data.get("limit")) is not int
                or type(data.get("remaining")) is not int
                or not 0 <= data["remaining"] <= data["limit"]
                or not isinstance(data.get("reset_at"), str)):
            raise ValueError
    except (ValueError, TypeError):
        _abort("API returned invalid usage information.")
    return data


def main() -> None:
    args = parse_args()
    if getattr(args, "auth_command", None):
        run_auth_command(args)
        return

    client_token = ""
    try:
        if not args.usage:
            target_seq = validate_sequence(args.target_seq, "target sequence")
            peptides = load_peptides(args.peptides)
            if args.output:
                check_output_path(Path(args.output))
        identity = client_identity(args.email, config_path())
        client_token = identity["access_token"]
    except (ValueError, OSError, UnicodeError) as exc:
        _abort(_redact(str(exc), client_token))
    except KeyboardInterrupt:
        _abort("GRADE login was cancelled.")

    if args.usage:
        data = query_usage(client_token)
        print(_redact(f"Daily quota remaining: {data['remaining']}/{data['limit']}. Resets at {data['reset_at']}.", client_token))
        return

    print(f"Loaded {len(peptides)} peptides from {args.peptides}", file=sys.stderr)
    print(f"Submitting {len(peptides)} peptides to the GRADE API...", file=sys.stderr)
    all_results = call_api(target_seq, peptides, client_token=client_token, model=args.model)

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
            client_token,
        ))

    if args.output:
        print(f"Saved {len(ranked_results)} results to {args.output}", file=sys.stderr)
    else:
        print(f"\n{len(ranked_results)} results.", file=sys.stderr)


if __name__ == "__main__":
    main()
