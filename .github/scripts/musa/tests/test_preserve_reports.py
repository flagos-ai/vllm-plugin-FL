#!/usr/bin/env python3
"""Real temporary-file positive and negative controls for evidence preservation."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "preserve_reports.py"
spec = importlib.util.spec_from_file_location("preserve_reports", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=os.environ.get("MUSA_PRESERVATION_TEST_TMP")
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo, self.suite, self.install, self.gate = [
            self.root / name for name in ("repo", "suite", "install", "gate")
        ]
        for path in (self.repo, self.suite, self.install, self.gate):
            path.mkdir()

    def write(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path

    def fingerprint(self, path):
        body = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }

    def report(self, path, body):
        self.write(path, json.dumps(body).encode())

    def run_preservation(self):
        report = module.preserve(
            self.repo, self.suite, self.install, self.gate, self.root / "output"
        )
        manifest = json.loads(
            (self.root / "output/diagnostics/preservation-manifest.json").read_bytes()
        )
        return report, manifest

    def test_actual_bound_source_records_and_raw_are_preserved_without_venv_tree(self):
        source = self.write(
            self.install / "venv/lib/site-packages/package/graph.py", b"x = 1\r\n"
        )
        record = self.write(
            self.install / "venv/lib/site-packages/package.dist-info/RECORD",
            b"graph.py,sha256=abc,7\n",
        )
        direct = self.write(
            record.parent / "direct_url.json", b'{"url":"file:///existing.whl"}'
        )
        metadata = self.write(
            record.parent / "METADATA", b"Name: package\nVersion: 1\n"
        )
        shell = self.write(self.repo / "tools/gate/run.sh", b"exit 0\n")
        readme = self.write(
            self.repo / "tools/gate/README.md", b"original oracle documentation\n"
        )
        ignored = self.write(
            self.install / "venv/lib/site-packages/unrelated/unused.py",
            b"must not scan",
        )
        self.write(self.install / "test-tmp/unused.cache", b"must not copy")
        refs = [
            self.fingerprint(path)
            for path in (source, record, direct, metadata, shell, readme)
        ]
        refs[0] = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256_LF": hashlib.sha256(b"x = 1\n").hexdigest(),
        }
        self.report(self.suite / "native-admission/worker.json", {"sources": refs})
        raw = self.write(
            self.gate / "uuid/observer-traces/rank0.jsonl", b'{"event":1}\n' * 8193
        )
        self.report(
            self.gate / "uuid/observer-summary.json",
            {"input_fingerprints": [self.fingerprint(raw)]},
        )
        self.write(
            self.install / "wheels/vllm_plugin_fl-1-py3-none-any.whl",
            b"complete-plugin-wheel",
        )
        self.write(
            self.install / "wheels/musa_gate_observer-1-py3-none-any.whl",
            b"complete-observer-wheel",
        )
        self.write(
            self.install / "wheels/vllm-1-py3-none-any.whl", b"complete-runtime-wheel"
        )
        result, manifest = self.run_preservation()
        self.assertTrue(result["preservation_passed"], result)
        self.assertFalse(result["full_musa_acceptance"])
        rows = {row["original_path"]: row for row in manifest["files"]}
        for path in (source, record, direct, metadata, shell, readme):
            row = rows[str(path)]
            copied = self.root / "output/diagnostics" / row["artifact_path"]
            self.assertEqual(copied.read_bytes(), path.read_bytes())
        self.assertNotIn(str(ignored), rows)
        self.assertFalse(
            any("unused" in str(path) for path in (self.root / "output").rglob("*"))
        )
        self.assertEqual(
            (
                self.root / "output/observer-raw/gate/uuid/observer-traces/rank0.jsonl"
            ).read_bytes(),
            raw.read_bytes(),
        )
        self.assertFalse(
            any("rank0.jsonl" in row["artifact_path"] for row in manifest["files"])
        )
        self.assertTrue(
            (
                self.root / "output/wheels/installation/wheels/vllm-1-py3-none-any.whl"
            ).is_file()
        )
        self.assertTrue(
            (
                self.root
                / "output/diagnostics/installation/wheels/vllm_plugin_fl-1-py3-none-any.whl"
            ).is_file()
        )

    def test_digest_mismatch_is_not_pass_and_original_report_is_preserved(self):
        source = self.write(self.install / "venv/site/module.py", b"new bytes")
        pin = self.fingerprint(source)
        pin["sha256"] = "0" * 64
        self.report(self.gate / "uuid-inputs/source-binding.json", {"module": pin})
        result, manifest = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertTrue(
            any("digest changed" in row["error"] for row in result["errors"])
        )
        self.assertTrue(
            (
                self.root / "output/diagnostics/gate/uuid-inputs/source-binding.json"
            ).is_file()
        )
        self.assertFalse(manifest["preservation_passed"])

    def test_missing_bound_record_is_not_pass(self):
        self.report(
            self.install / "dependency-profile-installed.json",
            {
                "record": {
                    "path": str(self.install / "venv/site/dist-info/RECORD"),
                    "bytes": 2,
                    "sha256": "0" * 64,
                }
            },
        )
        result, _ = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertTrue(result["errors"])

    def test_setup_metadata_hash_mapping_copies_actual_metadata_and_record(self):
        directory = self.root / "vendor/site/package.dist-info"
        metadata = self.write(
            directory / "METADATA", b"Name: protected-package\nVersion: 1\n"
        )
        record = self.write(directory / "RECORD", b"original protected RECORD\n")
        observed = {
            "version": "1",
            "metadata_directory": str(directory),
            "files": {
                path.name: self.fingerprint(path)["sha256"]
                for path in (metadata, record)
            },
        }
        self.report(
            self.install / "test-tools.json",
            {
                "protected_before": {"package": observed},
                "protected_after": {"package": observed},
            },
        )
        result, manifest = self.run_preservation()
        self.assertTrue(result["preservation_passed"], result)
        rows = {row["original_path"]: row for row in manifest["files"]}
        for path in (metadata, record):
            row = rows[str(path)]
            self.assertEqual(
                (self.root / "output/diagnostics" / row["artifact_path"]).read_bytes(),
                path.read_bytes(),
            )
            self.assertEqual(row["sha256"], observed["files"][path.name])

    def test_changed_setup_metadata_hash_mapping_is_not_pass(self):
        directory = self.root / "vendor/site/package.dist-info"
        metadata = self.write(directory / "METADATA", b"Name: protected-package\n")
        record = self.write(directory / "RECORD", b"before mutation\n")
        observed = {
            "metadata_directory": str(directory),
            "files": {
                path.name: self.fingerprint(path)["sha256"]
                for path in (metadata, record)
            },
        }
        self.report(
            self.install / "test-tools.json",
            {"protected_before": {"package": observed}},
        )
        record.write_bytes(b"after mutation\n")
        result, manifest = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertFalse(manifest["preservation_passed"])
        self.assertTrue(
            any("digest changed" in row["error"] for row in result["errors"])
        )
        self.assertTrue(
            (self.root / "output/diagnostics/installation/test-tools.json").is_file()
        )

    def test_invalid_setup_metadata_digest_format_is_not_pass(self):
        directory = self.root / "vendor/site/package.dist-info"
        metadata = self.write(directory / "METADATA", b"Name: protected-package\n")
        record = self.write(directory / "RECORD", b"record bytes\n")
        observed = {
            "metadata_directory": str(directory),
            "files": {
                "METADATA": self.fingerprint(metadata)["sha256"][:-1],
                "RECORD": self.fingerprint(record)["sha256"],
            },
        }
        self.report(
            self.install / "test-tools.json",
            {"protected_before": {"package": observed}},
        )
        result, _ = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertTrue(
            any(
                "Invalid setup metadata snapshot digest" in row["error"]
                for row in result["errors"]
            )
        )

    def test_native_metadata_fingerprint_variant_still_preserves_originals(self):
        directory = self.install / "venv/site/package.dist-info"
        metadata = self.write(directory / "METADATA", b"Name: package\n")
        record = self.write(directory / "RECORD", b"native variant RECORD\n")
        self.report(
            self.suite / "native-admission/worker.json",
            {
                "plugin_distribution": {
                    "metadata_directory": str(directory),
                    "files": {
                        path.name: self.fingerprint(path) for path in (metadata, record)
                    },
                }
            },
        )
        result, manifest = self.run_preservation()
        self.assertTrue(result["preservation_passed"], result)
        originals = {row["original_path"] for row in manifest["files"]}
        self.assertTrue({str(metadata), str(record)} <= originals)

    def test_invalid_bound_digest_format_is_not_silently_ignored(self):
        source = self.write(self.install / "venv/site/module.py", b"observed source\n")
        pin = self.fingerprint(source)
        pin["sha256"] = pin["sha256"][:-1]
        self.report(self.suite / "worker.json", {"source": pin})
        result, manifest = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertFalse(manifest["preservation_passed"])
        self.assertTrue(
            any("Invalid bound digest" in row["error"] for row in result["errors"])
        )

    def test_non_string_bound_path_is_not_silently_ignored(self):
        self.report(
            self.suite / "worker.json", {"source": {"path": 17, "sha256": "0" * 64}}
        )
        result, _ = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertTrue(
            any("path must be a string" in row["error"] for row in result["errors"])
        )

    def test_malformed_diagnostic_json_is_preserved_but_not_pass(self):
        self.write(self.gate / "uuid-inputs/entry-result.json", b"{invalid json")
        result, _ = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertEqual(
            (
                self.root / "output/diagnostics/gate/uuid-inputs/entry-result.json"
            ).read_bytes(),
            b"{invalid json",
        )

    def test_credential_reference_is_rejected_without_copying_its_bytes(self):
        source = self.write(self.root / ".ssh/id_rsa", b"fixture-secret-do-not-copy")
        self.report(self.suite / "worker.json", {"source": self.fingerprint(source)})
        result, manifest = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertTrue(
            any("Credential/configuration" in row["error"] for row in result["errors"])
        )
        self.assertFalse(manifest["external_non_source_references"])
        for path in (self.root / "output").rglob("*"):
            if path.is_file():
                self.assertNotIn(b"fixture-secret-do-not-copy", path.read_bytes())

    def test_credential_source_reference_is_rejected(self):
        source = self.write(
            self.root / ".ssh/private.py", b"fixture-secret-do-not-copy"
        )
        self.report(self.suite / "worker.json", {"source": self.fingerprint(source)})
        result, _ = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        self.assertTrue(
            any("Credential/configuration" in row["error"] for row in result["errors"])
        )

    def test_empty_gate_raw_has_an_explicit_empty_manifest(self):
        self.report(
            self.gate / "uuid-inputs/entry-result.json",
            {"passed": False, "errors": ["preflight stopped before models"]},
        )
        result, _ = self.run_preservation()
        raw = json.loads(
            (self.root / "output/observer-raw/preservation-manifest.json").read_bytes()
        )
        self.assertTrue(result["preservation_passed"])
        self.assertFalse(raw["full_musa_acceptance"])
        self.assertEqual(raw["files"], [])

    def test_external_whole_wheel_reference_reuses_already_preserved_exact_payload(
        self,
    ):
        wheel = self.write(
            self.install / "wheels/vllm-1-py3-none-any.whl", b"same-payload"
        )
        pin = self.fingerprint(wheel)
        pin["path"] = str(self.root / "old-source-location/vllm-1-py3-none-any.whl")
        self.report(self.install / "empty-wheel.json", {"source": pin})
        result, manifest = self.run_preservation()
        self.assertTrue(result["preservation_passed"], result)
        self.assertEqual(len(manifest["same_whole_wheel_references"]), 1)
        self.assertEqual(len(list((self.root / "output").rglob("*.whl"))), 1)

    def test_cli_returns_failure_and_writes_all_group_manifests(self):
        self.report(
            self.suite / "worker.json",
            {
                "source": {
                    "path": str(self.install / "venv/missing.py"),
                    "sha256": "0" * 64,
                }
            },
        )
        execution = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(SCRIPT),
                "--candidate-root",
                str(self.repo),
                "--suite-root",
                str(self.suite),
                "--install-root",
                str(self.install),
                "--gate-root",
                str(self.gate),
                "--output-root",
                str(self.root / "output"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(execution.returncode, 1, execution.stderr)
        for group in ("diagnostics", "wheels", "observer-raw"):
            report = json.loads(
                (
                    self.root / "output" / group / "preservation-manifest.json"
                ).read_bytes()
            )
            self.assertFalse(report["preservation_passed"])

    def test_output_reuse_is_refused(self):
        output = self.root / "output"
        output.mkdir()
        marker = self.write(output / "marker.txt", b"old-evidence")
        with self.assertRaises(FileExistsError):
            module.preserve(self.repo, self.suite, self.install, self.gate, output)
        self.assertEqual(marker.read_bytes(), b"old-evidence")

    def test_missing_explicit_root_is_failure_while_available_reports_survive(self):
        self.report(self.suite / "unit.json", {"passed": True})
        result = module.preserve(
            self.repo,
            self.suite,
            self.install,
            self.root / "missing-gate",
            self.root / "output",
        )
        self.assertFalse(result["preservation_passed"])
        self.assertTrue(result["errors"])
        self.assertTrue((self.root / "output/diagnostics/suite/unit.json").is_file())

    def test_symlink_input_is_not_followed(self):
        outside = self.write(self.root / "outside.txt", b"outside-payload")
        try:
            (self.suite / "linked.txt").symlink_to(outside)
        except OSError as error:
            self.skipTest("Symlink creation unavailable: " + str(error))
        result, _ = self.run_preservation()
        self.assertFalse(result["preservation_passed"])
        for path in (self.root / "output").rglob("*"):
            if path.is_file():
                self.assertNotIn(b"outside-payload", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
