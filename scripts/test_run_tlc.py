"""Offline downloader regressions; no application imports or Java execution."""

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("run_tlc.sh")
PAYLOAD = b"verified test fixture, not executable code\n"


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        source = SCRIPT.read_text()
        # Only the copied script accepts this fixture; production pins stay fixed.
        source = source.replace(
            'SHA="936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"',
            f'SHA="{hashlib.sha256(PAYLOAD).hexdigest()}"',
        ).replace('SIZE="2274532"', f'SIZE="{len(PAYLOAD)}"')
        self.script = scripts / SCRIPT.name
        self.script.write_text(source)
        self.jar = self.root / "cache with spaces" / "tools.jar"
        self.fixture = self.root / "fixture"
        self.fixture.write_bytes(PAYLOAD)
        self.calls = self.root / "java-calls"
        self.curl_calls = self.root / "curl-calls"
        self.curl_headers = self.root / "curl-headers"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.stub("java", 'printf "%s\\n" "$*" >> "$JAVA_CALLS"\n')
        self.stub(
            "curl",
            'printf "%s\\n" "$*" >> "$CURL_CALLS"\n'
            "while [[ $# -gt 0 ]]; do\n"
            '  if [[ "$1" == "-o" ]]; then out="$2"; shift; fi\n'
            '  if [[ "$1" == -H && "$2" == @* ]]; then\n'
            '    stat -c %a "${2#@}" >> "$CURL_HEADERS"\n'
            '    cat "${2#@}" >> "$CURL_HEADERS"\n'
            "  fi\n"
            "  shift\n"
            "done\n"
            'cp "$FIXTURE" "$out"\n'
            'exit "${CURL_STATUS:-0}"\n',
        )
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}:{os.environ['PATH']}",
            TLA_TOOLS_JAR=str(self.jar),
            FIXTURE=str(self.fixture),
            JAVA_CALLS=str(self.calls),
            CURL_CALLS=str(self.curl_calls),
            CURL_HEADERS=str(self.curl_headers),
            TMPDIR=str(self.root),
        )
        self.env.pop("GITHUB_TOKEN", None)

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)

    def run_script(self):
        return subprocess.run(
            ["bash", str(self.script)], env=self.env, capture_output=True, text=True
        )

    def assert_clean_failure(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists(), result.stdout)
        self.assertEqual(list(self.root.rglob("*.download.*")), [])

    def test_verified_download_and_both_configs(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.jar.read_bytes(), PAYLOAD)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn("OrderLifecycle.cfg", calls[0])
        self.assertIn("OrderLifecycleLiveness.cfg", calls[1])
        request = self.curl_calls.read_text()
        self.assertIn("releases/assets/184694200", request)
        self.assertIn("Accept: application/octet-stream", request)
        self.assertIn("--proto =https --proto-redir =https", request)
        self.assertEqual(list(self.root.rglob("*.download.*")), [])
        self.assertEqual(list(self.root.glob("tmp.*")), [])

    def test_anonymous_download_sends_no_credentials(self):
        self.assertEqual(self.run_script().returncode, 0)
        self.assertNotIn("Authorization", self.curl_calls.read_text())
        self.assertFalse(self.curl_headers.exists())

    def test_token_authenticates_from_private_file_not_argv(self):
        token = "ghs_fixture-token-not-real"
        self.env["GITHUB_TOKEN"] = token
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(token, self.curl_calls.read_text())
        mode, header = self.curl_headers.read_text().splitlines()
        self.assertEqual(mode, "600")
        self.assertEqual(header, f"Authorization: Bearer {token}")
        self.assertEqual(list(self.root.glob("tmp.*")), [])

    def test_token_file_removed_when_download_fails(self):
        self.env["GITHUB_TOKEN"] = "ghs_fixture-token-not-real"
        self.env["CURL_STATUS"] = "22"
        self.assert_clean_failure(self.run_script())
        self.assertEqual(list(self.root.glob("tmp.*")), [])

    def test_directory_override_does_not_retain_download(self):
        self.jar.mkdir(parents=True)
        self.assert_clean_failure(self.run_script())
        self.assertEqual(list(self.jar.iterdir()), [])

    def test_valid_existing_cache_skips_download(self):
        self.jar.parent.mkdir()
        self.jar.write_bytes(PAYLOAD)
        self.assertEqual(self.run_script().returncode, 0)
        self.assertFalse(self.curl_calls.exists())

    def test_corrupt_cache_preserved_and_never_executed(self):
        self.jar.parent.mkdir()
        self.jar.write_bytes(b"corrupt")
        result = self.run_script()
        self.assert_clean_failure(result)
        self.assertEqual(self.jar.read_bytes(), b"corrupt")
        self.assertFalse(self.curl_calls.exists())
        self.assertIn("Expected SHA-256:", result.stderr)
        self.assertIn("Actual   SHA-256:", result.stderr)
        self.assertIn("bytes: 7", result.stderr)

    def test_same_size_wrong_digest_rejected(self):
        self.fixture.write_bytes(b"x" * len(PAYLOAD))
        self.assert_clean_failure(self.run_script())
        self.assertFalse(self.jar.exists())

    def test_correct_digest_wrong_size_rejected(self):
        source = self.script.read_text().replace(
            f'SIZE="{len(PAYLOAD)}"', f'SIZE="{len(PAYLOAD) + 1}"'
        )
        self.script.write_text(source)
        self.assert_clean_failure(self.run_script())
        self.assertFalse(self.jar.exists())

    def test_java_failure_is_propagated_and_metadata_cleaned(self):
        self.stub("java", "exit 42\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 42)
        self.assertEqual(list(self.root.glob("tmp.*")), [])
        self.assertEqual(self.jar.read_bytes(), PAYLOAD)

    def test_truncated_download_rejected(self):
        self.fixture.write_bytes(PAYLOAD[:5])
        self.assert_clean_failure(self.run_script())
        self.assertFalse(self.jar.exists())

    def test_download_failure_removes_partial_bytes(self):
        self.env["CURL_STATUS"] = "22"
        self.assert_clean_failure(self.run_script())
        self.assertFalse(self.jar.exists())

    def test_interruption_removes_partial_download(self):
        self.stub(
            "curl",
            "while [[ $# -gt 0 ]]; do\n"
            '  if [[ "$1" == "-o" ]]; then out="$2"; shift; fi\n'
            "  shift\n"
            "done\n"
            'cp "$FIXTURE" "$out"\n'
            'kill -TERM "$PPID"\n',
        )
        self.assert_clean_failure(self.run_script())
        self.assertFalse(self.jar.exists())

    def test_default_cache_is_versioned(self):
        del self.env["TLA_TOOLS_JAR"]
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual((self.root / ".tla" / "tla2tools-1.7.4.jar").read_bytes(), PAYLOAD)


if __name__ == "__main__":
    unittest.main()
