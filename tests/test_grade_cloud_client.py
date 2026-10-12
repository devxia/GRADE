"""Offline client contracts. All credentials and configuration directories are synthetic."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import typing
import unittest
from unittest import mock

CLIENT_PATH = Path(__file__).resolve().parents[1] / "skills/grade-cloud/scripts/grade_predict.py"
with mock.patch.dict(sys.modules, {"requests": None}):
    spec = importlib.util.spec_from_file_location("grade_cloud_client", CLIENT_PATH)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)

MODEL = "GRADE-paper"
EMAIL = "user@example.org"
TOKEN = "test-only-client-credential-" + "x" * 20
CID = "a" * 32
RESULTS = [
    {"peptide_name": "low", "sequence": "ACD", "score": 0.1},
    {"peptide_name": "high", "sequence": "KLM", "score": 0.9},
]


class Terminal(io.StringIO):
    def __init__(self, tty=False, text=""):
        super().__init__(text)
        self.tty = tty
    def isatty(self):
        return self.tty


class FakeResponse:
    def __init__(self, data=None, status=200, headers=None, text="", json_error=None):
        self.data, self.status_code = data, status
        self.headers, self.text, self.json_error = headers or {}, text, json_error
    def json(self):
        if self.json_error:
            raise self.json_error
        return self.data


class FakeHTTP(types.ModuleType):
    class RequestException(Exception): pass
    class Timeout(RequestException): pass
    class ConnectionError(RequestException): pass

    def __init__(self):
        super().__init__("requests")
        self.calls = []
        self.response = FakeResponse({"model": MODEL, "results": copy.deepcopy(RESULTS)})
        self.registration = FakeResponse({"client_token": TOKEN, "client_id": CID, "email": EMAIL})
        self.usage = FakeResponse({"limit": 50, "remaining": 48, "reset_at": "2026-09-22T00:00:00Z"})
        self.failure = None

    def post(self, url, **kwargs):
        self.calls.append((url, copy.deepcopy(kwargs)))
        if self.failure:
            raise self.failure
        if url.endswith("/client/register"):
            return self.registration
        if url.endswith("/usage"):
            return self.usage
        return self.response


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.http = FakeHTTP()
        self.work = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config_root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = self.config_root / "client.json"
        self.peptides = self.work / "peptides.fa"
        self.peptides.write_text(">low\n a c d \n>high\nklm\n")
        self.output = self.work / "results.csv"
        self.stdin, self.stdout, self.stderr = Terminal(), Terminal(), Terminal()
        self.stack.enter_context(mock.patch.dict(sys.modules, {"requests": self.http}))
        self.stack.enter_context(mock.patch.dict(os.environ, {"GRADE_EMAIL": "", "GRADE_API_KEY": "unused-legacy-key"}))
        self.stack.enter_context(mock.patch.object(client, "config_path", return_value=self.config))
        for name in ("stdin", "stdout", "stderr"):
            self.stack.enter_context(mock.patch.object(sys, name, getattr(self, name)))
        self.saved = {"version": 1, "api_base": client.API_BASE, "email": EMAIL, "client_id": CID, "client_token": TOKEN}
        client.save_config(self.config, self.saved)

    def run_main(self, *extra, usage=False):
        args = [str(CLIENT_PATH)]
        if not usage:
            args += ["--target-seq", " m r y \n", "--peptides", str(self.peptides)]
        with mock.patch.object(sys, "argv", [*args, *extra]):
            client.main()

    def call(self, **changes):
        values = {"email": EMAIL, "client_token": TOKEN, "model": MODEL, **changes}
        return client.call_api("MRY", [{"name": "p", "sequence": "ACD"}], **values)

    def assert_private(self):
        self.assertNotIn(TOKEN, self.stdout.getvalue() + self.stderr.getvalue())
        if self.output.exists():
            self.assertNotIn(TOKEN, self.output.read_text())

    def test_email_precedence_and_exact_prompt(self):
        os.environ["GRADE_EMAIL"] = "env@example.org"
        self.assertEqual(client.resolve_email(" ARG@EXAMPLE.ORG ", EMAIL), "arg@example.org")
        self.assertEqual(client.resolve_email(None, EMAIL), "env@example.org")
        os.environ["GRADE_EMAIL"] = ""
        self.assertEqual(client.resolve_email(None, EMAIL), EMAIL)
        with mock.patch.object(sys, "stdin", Terminal(True, "Prompt@Example.org\n")):
            self.stderr.tty = True
            self.assertEqual(client.resolve_email(None), "prompt@example.org")
        self.assertEqual(self.stderr.getvalue(), "Please enter your email address:\n")
        self.assertEqual(self.stdout.getvalue(), "")

    def test_first_use_registers_once_and_reuses_credential_across_email_and_cwd(self):
        self.config.unlink()
        self.run_main("--email", EMAIL)
        self.assertEqual([url.rsplit("/", 1)[-1] for url, _ in self.http.calls], ["register", "predict"])
        self.assertEqual(self.http.calls[0][1]["headers"], {})
        self.assertEqual(client.load_config(self.config)["client_token"], TOKEN)
        self.run_main("--email", "another@example.org")
        self.assertEqual(len(self.http.calls), 3)
        request = self.http.calls[-1][1]
        self.assertEqual(request["json"]["email"], "another@example.org")
        self.assertEqual(request["headers"], {"Authorization": "Bearer " + TOKEN})
        self.assertFalse(request["allow_redirects"])
        self.assertEqual(client.load_config(self.config)["client_id"], CID)
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        self.assert_private()

    def test_noninteractive_missing_email_and_invalid_email_never_register(self):
        self.config.unlink()
        for args, message in (((), "Email address required"), (("--email", "bad"), "Please enter a valid email address")):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.run_main(*args)
            self.assertIn(message, self.stderr.getvalue())
            self.assertEqual(self.http.calls, [])
        self.assertFalse(self.config.exists())

    def test_empty_eof_and_interrupted_input_stop(self):
        self.stderr.tty = True
        for value in ("", " \n"):
            with mock.patch.object(sys, "stdin", Terminal(True, value)), self.assertRaises(ValueError):
                client.resolve_email(None)
        stream = Terminal(True)
        with mock.patch.object(stream, "readline", side_effect=KeyboardInterrupt), mock.patch.object(sys, "stdin", stream):
            with self.assertRaises(ValueError):
                client.resolve_email(None)
        self.assertEqual(self.http.calls, [])

    def test_invalid_local_inputs_precede_prompt_and_registration(self):
        self.config.unlink()
        self.stdin.tty = self.stderr.tty = True
        for args in (("--target-seq", " "), ("--peptides", str(self.work / "missing.fa")), ("--output", str(self.work / "missing/out.csv")), ("--output", str(self.work))):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.run_main(*args)
        self.assertNotIn(client.EMAIL_PROMPT, self.stderr.getvalue())
        self.assertEqual(self.http.calls, [])
        self.assertFalse(self.config.exists())

    def test_bad_config_and_foreign_origin_do_not_register_new_identity(self):
        cases = ["bad json", "[]", json.dumps({**self.saved, "api_base": "https://other.invalid"}), json.dumps({**self.saved, "version": 0})]
        for content in cases:
            self.config.write_text(content)
            with self.subTest(content=content), self.assertRaises(SystemExit):
                self.run_main("--email", EMAIL)
        self.assertEqual(self.http.calls, [])
        self.assert_private()

    @unittest.skipIf(os.name == "nt", "POSIX mode and symbolic link contract")
    def test_unsafe_file_permissions_and_symlink_are_rejected(self):
        self.config.chmod(0o644)
        with self.assertRaises(SystemExit):
            self.run_main()
        self.config.unlink()
        destination = self.config_root / "other.json"
        destination.write_text(json.dumps(self.saved))
        self.config.symlink_to(destination)
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertEqual(self.http.calls, [])
        self.assert_private()

    def test_invalid_registration_is_not_saved(self):
        self.config.unlink()
        for data in (None, [], {}, {"client_token": TOKEN}, {"client_token": "short", "client_id": CID}):
            self.http.registration = FakeResponse(data)
            with self.subTest(data=data), self.assertRaises(SystemExit):
                self.run_main("--email", EMAIL)
            self.assertFalse(self.config.exists())
        self.assertTrue(all(url.endswith("/register") for url, _ in self.http.calls))
        self.assert_private()

    def test_usage_without_prediction_inputs(self):
        self.run_main("--usage", usage=True)
        self.assertEqual(len(self.http.calls), 1)
        self.assertTrue(self.http.calls[0][0].endswith("/usage"))
        self.assertIn("48/50", self.stdout.getvalue())
        for args in (("--usage", "--target-seq", "ACD"), ("--usage", "--output", "x.csv")):
            with self.assertRaises(SystemExit):
                self.run_main(*args, usage=True)
        self.assertEqual(len(self.http.calls), 1)

    def test_usage_invalid_response_is_handled(self):
        for data in (None, [], {}, {"limit": 50, "remaining": True, "reset_at": "later"}):
            self.http.usage = FakeResponse(data)
            with self.assertRaises(SystemExit):
                self.run_main("--usage", usage=True)
        self.assertNotIn("Traceback", self.stderr.getvalue())

    def test_invalid_models_old_keys_and_invalid_topk_are_local_errors(self):
        for args in (("--api-key", "old-key"), ("--model", "GRADE-base"), ("--model", "GRADE-medium"), ("--model", "paper"), ("--top-k", "0"), ("--top-k", "-1"), ("--output-format", "xml")):
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                self.run_main(*args)
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.http.calls, [])
        for model in (None, "GRADE-base", "GRADE-medium", ""):
            with self.assertRaises(ValueError):
                self.call(model=model)
        with self.assertRaises(TypeError):
            client.call_api("old-key", "ACD", [])
        with self.assertRaises(ValueError):
            self.call(client_token="")
        self.assertEqual(self.http.calls, [])

    def test_python_api_request_shape_return_and_normalization(self):
        self.assertEqual(self.call(), RESULTS)
        self.run_main()
        payload = self.http.calls[-1][1]["json"]
        self.assertEqual(payload, {"email": EMAIL, "model": MODEL, "target_seq": "MRY", "peptides": [
            {"name": "low", "sequence": "ACD"}, {"name": "high", "sequence": "KLM"},
        ]})
        self.assertNotIn("api_key", payload)
        self.assert_private()

    def test_model_identity_legacy_and_matching_header(self):
        self.http.response = FakeResponse({"results": RESULTS})
        self.assertEqual(self.call(), RESULTS)
        self.assertEqual(self.stderr.getvalue(), "Warning: server did not report model identity; cannot verify requested model.\n")
        self.http.response = FakeResponse({"results": RESULTS}, headers={"X-GRADE-Model": MODEL})
        self.assertEqual(self.call(), RESULTS)
        self.assertIn("Server reported model: GRADE-paper", self.stderr.getvalue())

    def test_invalid_model_identity_preserves_existing_output(self):
        self.output.write_text("existing\n")
        cases = [({"model": val}, {}) for val in (None, "", " ", 123, False, [], {}, "GRADE-base", "GRADE-medium", " GRADE-paper ", TOKEN)]
        cases += [({}, {"X-GRADE-Model": val}) for val in (None, "", 1, "GRADE-base")]
        cases += [({"model": MODEL}, {"X-GRADE-Model": "GRADE-base"})]
        for data, headers in cases:
            self.http.response = FakeResponse({**data, "results": RESULTS}, headers=headers)
            with self.subTest(data=data, headers=headers), self.assertRaises(SystemExit):
                self.run_main("--output", str(self.output))
            self.assertEqual(self.output.read_text(), "existing\n")
        self.assert_private()

    def test_http_errors_redact_and_401_never_reregisters(self):
        for status in (401, 422, 429, 500, 503):
            self.http.response = FakeResponse({"error": "error " + TOKEN}, status=status)
            with self.assertRaises(SystemExit):
                self.run_main("--output", str(self.output))
            self.assertFalse(self.output.exists())
            self.assertEqual(client.load_config(self.config)["client_token"], TOKEN)
        self.assertTrue(all(url.endswith("/predict") for url, _ in self.http.calls))
        self.assert_private()

    def test_quota_headers_and_rate_retry_guidance(self):
        self.http.response.headers = {"X-GRADE-Quota-Limit": "50", "X-GRADE-Quota-Remaining": "48", "X-GRADE-Quota-Reset": "2026-09-22T00:00:00Z"}
        self.call()
        self.assertIn("48/50", self.stderr.getvalue())
        self.http.response = FakeResponse({"error": "Too many requests"}, status=429, headers={"Retry-After": "30"})
        with self.assertRaises(SystemExit):
            self.call()
        self.assertIn("Retry after 30 seconds", self.stderr.getvalue())

    def test_network_failures_never_retry_or_leak(self):
        for failure in (self.http.Timeout, self.http.ConnectionError, self.http.RequestException):
            self.http.calls.clear()
            self.http.failure = failure(TOKEN)
            with self.assertRaises(SystemExit):
                self.call()
            self.assertEqual(len(self.http.calls), 1)
        self.assert_private()

    def test_malformed_response_preserves_output(self):
        self.output.write_text("existing\n")
        responses = [FakeResponse(json_error=ValueError(TOKEN)), FakeResponse([]), FakeResponse(None), FakeResponse({"model": MODEL})]
        for rows in ({}, [None], [{"sequence": "ACD", "score": 0.5}], [{"peptide_name": "p", "score": 0.5}]):
            responses.append(FakeResponse({"model": MODEL, "results": rows}))
        for score in (None, TOKEN, "NaN", float("inf"), True):
            responses.append(FakeResponse({"model": MODEL, "results": [{"peptide_name": "p", "sequence": "ACD", "score": score}]}))
        for response in responses:
            self.http.response = response
            with self.assertRaises(SystemExit):
                self.run_main("--output", str(self.output))
            self.assertEqual(self.output.read_text(), "existing\n")
        self.assert_private()

    def test_output_formats_ranking_and_topk_do_not_reduce_submitted_count(self):
        for suffix in ("csv", "tsv", "json"):
            path = self.work / f"results.{suffix}"
            self.run_main("--top-k", "1", "--output", str(path))
            text = path.read_text()
            rows = json.loads(text) if suffix == "json" else list(csv.DictReader(io.StringIO(text), delimiter="\t" if suffix == "tsv" else ","))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["peptide_name"], "high")
            self.assertEqual(str(rows[0]["rank"]), "1")
            self.assertEqual(len(self.http.calls[-1][1]["json"]["peptides"]), 2)
        self.assert_private()

    def test_stdout_is_only_csv(self):
        self.run_main()
        rows = list(csv.DictReader(io.StringIO(self.stdout.getvalue())))
        self.assertEqual([row["peptide_name"] for row in rows], ["high", "low"])
        self.assertEqual(set(rows[0]), {"rank", "peptide_name", "sequence", "score"})
        self.assert_private()

    def test_output_write_failure_explains_completed_request(self):
        real_open = open
        def fail(path, mode="r", *args, **kwargs):
            if path == str(self.output) and mode == "w":
                raise PermissionError("output became unwritable")
            return real_open(path, mode, *args, **kwargs)
        with mock.patch("builtins.open", side_effect=fail), self.assertRaises(SystemExit):
            self.run_main("--output", str(self.output))
        self.assertIn("quota may have been used", self.stderr.getvalue())
        self.assertEqual(len(self.http.calls), 1)

    def test_file_formats_bom_normalization_and_invalid_inputs(self):
        for ext, text in (("fa", ">p\n a c d \n"), ("fasta", ">p\nACD\n"), ("faa", ">p\nACD\n"), ("csv", 'name,sequence\np," a c d "\n'), ("txt", "p acd\n")):
            path = self.work / f"input.{ext}"
            path.write_text(text, encoding="utf-8" if ext == "txt" else "utf-8-sig")
            self.assertEqual(client.load_peptides(str(path)), [{"name": "p", "sequence": "ACD"}])
        for ext, text in (("fa", ">p\nACD\n>p\nEFG\n"), ("fa", "ACD\n>p\nEFG\n"), ("fa", ">p\n \n"), ("csv", "name,sequence\np,ACD\np,EFG\n"), ("csv", "name,other\np,ACD\n"), ("txt", "p ACD\np EFG\n"), ("txt", "\n"), ("bin", "ACD")):
            path = self.work / f"bad.{ext}"
            path.write_text(text)
            with self.assertRaises(SystemExit):
                self.run_main("--peptides", str(path))
        self.assertEqual(self.http.calls, [])

    def test_requests_is_lazy(self):
        with mock.patch.dict(sys.modules, {"requests": None}):
            typing.get_type_hints(client.call_api)
            self.assertEqual(len(client.load_peptides(str(self.peptides))), 2)
            with self.assertRaises(SystemExit):
                self.run_main()
        self.assertIn("requests package is required", self.stderr.getvalue())

    def test_help_subprocess_without_requests(self):
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(CLIENT_PATH), "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in ("--email", "GRADE_EMAIL", "--usage", "--model", MODEL):
            self.assertIn(text, result.stdout)
        self.assertNotIn("--api-key", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_concurrent_first_use_processes_share_one_registration(self):
        self.config.unlink()
        registration_log = self.config_root / "registrations.txt"
        runner = '''
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("client", sys.argv[1])
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
class Response:
    def json(self):
        return {"client_token": "synthetic-credential-" + "x" * 30, "client_id": "b" * 32}
def post(*args):
    with open(sys.argv[3], "a") as f:
        f.write("registered\\n")
    return Response()
c.post_api = post
c.client_identity("user@example.org", pathlib.Path(sys.argv[2]))
'''
        command = [sys.executable, "-I", "-S", "-B", "-c", runner, str(CLIENT_PATH), str(self.config), str(registration_log)]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=self.work) for _ in range(4)]
        for process in processes:
            out, err = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, err)
            self.assertEqual(out, "")
        self.assertEqual(registration_log.read_text().splitlines(), ["registered"])
        self.assertEqual(client.load_config(self.config)["client_id"], "b" * 32)


if __name__ == "__main__":
    unittest.main()
