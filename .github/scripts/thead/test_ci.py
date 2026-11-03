# Copyright 2026 FlagOS Contributors
"""CPU-only regressions for CI acceptance boundaries; no vendor imports."""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
# These are the actual standalone CI modules, not substitutes for runtime code.
sys.path.insert(0, str(HERE))
import run_tests
import stack
from common import session_members, validate_junit
from config import resolve_config
from run_gate import (
    CHECKS,
    SCENARIOS,
    local_models,
    loopback_environment,
    parse_graph_observations,
    validate_documents,
)


class IsolatedEntryTests(unittest.TestCase):
    def test_actual_cli_entrypoints_under_isolated_python(self):
        for filename in ("run_tests.py", "run_gate.py", "stack.py"):
            with self.subTest(filename=filename):
                result = subprocess.run(
                    [sys.executable, "-I", "-B", str(HERE / filename), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_stack_cli_keeps_import_info_off_json_stdout(self):
        facts = {"provider": "ordinary-wheel", "version": "0.28.0+empty"}

        def check():
            print("INFO normal import check")
            return facts

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", ["stack.py", "check"]),
            patch.object(stack, "check_stack", side_effect=check),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            stack.main()
        self.assertEqual(json.loads(stdout.getvalue()), facts)
        self.assertIn("INFO normal import check", stderr.getvalue())
        self.assertNotIn("INFO", stdout.getvalue())


class LoopbackProxyTests(unittest.TestCase):
    def test_readiness_reaches_local_server_despite_runner_proxy(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Direct requests use a path; proxy requests use an absolute URL.
                if self.path != "/v1/models":
                    self.send_error(503, "Proxy rejects loopback requests")
                    return
                body = b'{"data": [{"id": "thead-ci-model"}]}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_port
            proxy = f"http://127.0.0.1:{port}"
            proxy_env = {
                name: value
                for name, value in os.environ.items()
                if not name.lower().endswith("_proxy")
            }
            proxy_env["http_proxy"] = proxy
            try:
                with (
                    patch.dict(os.environ, proxy_env, clear=True),
                    patch.object(urllib.request, "_opener", None),
                ):
                    # This is the former readiness call under the failing CI env.
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(proxy + "/v1/models", timeout=3)
                    self.assertEqual(error.exception.code, 503)
                    error.exception.close()
                    self.assertEqual(
                        local_models(port), {"data": [{"id": "thead-ci-model"}]}
                    )
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_child_environment_preserves_proxy_and_bypass_entries(self):
        original = {
            "HTTP_PROXY": "http://proxy.example:3128",
            "https_proxy": "http://secure-proxy.example:3128",
            "NO_PROXY": "internal.example,127.0.0.1",
            "no_proxy": "localhost,.svc",
            "UNCHANGED": "value",
        }
        env = loopback_environment(original)
        expected = "internal.example,127.0.0.1,localhost,.svc"
        self.assertEqual(env["NO_PROXY"], expected)
        self.assertEqual(env["no_proxy"], expected)
        for key in ("HTTP_PROXY", "https_proxy", "UNCHANGED"):
            self.assertEqual(env[key], original[key])
        self.assertEqual(original["NO_PROXY"], "internal.example,127.0.0.1")
        self.assertEqual(loopback_environment({})["no_proxy"], "127.0.0.1,localhost")
        self.assertEqual(loopback_environment(env), env)


class VllmProviderTests(unittest.TestCase):
    def setUp(self):
        self.origin = str((HERE / "image-provider/vllm/__init__.py").resolve())

    def assert_provider_rejected(self, module_version, distribution_version, origin):
        with self.assertRaises(RuntimeError) as error:
            stack.validate_vllm_provider(
                module_version, distribution_version, self.origin, origin
            )
        for value in (module_version, distribution_version, self.origin, origin):
            self.assertIn(repr(value), str(error.exception))

    def test_actual_split_versions_and_resolved_origin_pass(self):
        distribution_origin = str(HERE / "image-provider/vllm/../vllm/__init__.py")
        self.assertEqual(
            stack.validate_vllm_provider(
                "0.28.0", "0.28.0+empty", self.origin, distribution_origin
            ),
            {
                "vllm_module_version": "0.28.0",
                "vllm_distribution_version": "0.28.0+empty",
                "vllm_origin": self.origin,
                "vllm_distribution_origin": distribution_origin,
            },
        )

    def test_wrong_module_version_rejected_with_actual_values(self):
        self.assert_provider_rejected("0.23.0", "0.28.0+empty", self.origin)

    def test_distribution_without_empty_rejected_with_actual_values(self):
        self.assert_provider_rejected("0.28.0", "0.28.0", self.origin)

    def test_wrong_provider_origin_rejected_with_actual_values(self):
        wrong_origin = str((HERE / "system-provider/vllm/__init__.py").resolve())
        self.assert_provider_rejected("0.28.0", "0.28.0+empty", wrong_origin)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.defaults = {"container_options": "--device /dev/alixpu"}
        self.env = {
            "THEAD_CI_IMAGE": "registry.example/ppu@sha256:" + "a" * 64,
            "THEAD_CI_VISIBLE_DEVICES": "0,1,2,3",
            "THEAD_CI_MODEL_27B": "/models/dense",
            "THEAD_CI_MODEL_35B": "/models/moe",
        }

    def versioned_resources(self):
        return {
            "ci_image": "registry.example/ppu:v0.28.0-thead-ci",
            "runner_labels": ["reserved-ppu"],
            "container_volumes": ["/model-store:/models:ro"],
            "container_options": "--device /dev/alixpu",
            "visible_devices": "12,13,14,15",
            "model_27b": "/models/dense",
            "model_35b": "/models/moe",
            "base_python": "/opt/thead/venv/bin/python",
        }

    def test_empty_vars_use_versioned_resources(self):
        defaults = self.versioned_resources()
        empty = {
            key: ""
            for key in (
                *self.env,
                "THEAD_CI_RUNNER_LABELS",
                "THEAD_CI_CONTAINER_OPTIONS",
                "THEAD_CI_CONTAINER_VOLUMES",
                "THEAD_CI_BASE_PYTHON",
            )
        }
        self.assertEqual(resolve_config(defaults, empty), defaults)

    def test_invalid_versioned_resources_fail(self):
        for key, value in (
            ("ci_image", "registry.example/ppu:latest"),
            ("visible_devices", "12,12,14,15"),
            ("runner_labels", "reserved-ppu"),
            ("container_volumes", {"host": "/models"}),
            ("base_python", "python"),
        ):
            defaults = dict(self.versioned_resources(), **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve_config(defaults, {})

    def test_invalid_override_does_not_fall_back(self):
        with self.assertRaises(ValueError):
            resolve_config(
                self.versioned_resources(),
                {"THEAD_CI_IMAGE": "registry.example/ppu:latest"},
            )
        with self.assertRaises(ValueError):
            resolve_config(
                self.versioned_resources(),
                {"THEAD_CI_CONTAINER_VOLUMES": "not-json"},
            )

    def test_valid_overrides_replace_versioned_resources(self):
        config = resolve_config(
            self.versioned_resources(),
            dict(
                self.env,
                THEAD_CI_CONTAINER_VOLUMES='["/new-model-store:/models:ro"]',
                THEAD_CI_BASE_PYTHON="/custom/runtime/bin/python",
            ),
        )
        self.assertEqual(config["ci_image"], self.env["THEAD_CI_IMAGE"])
        self.assertEqual(config["visible_devices"], "0,1,2,3")
        self.assertEqual(config["runner_labels"], ["reserved-ppu"])
        self.assertEqual(config["container_volumes"], ["/new-model-store:/models:ro"])
        self.assertEqual(config["base_python"], "/custom/runtime/bin/python")

    def test_main_runner_default_and_overrides(self):
        config = resolve_config(self.defaults, self.env)
        self.assertEqual(config["runner_labels"], ["flagcicd-810e"])
        self.env["THEAD_CI_RUNNER_LABELS"] = '["self-hosted","ppu-dedicated"]'
        self.assertEqual(
            resolve_config(self.defaults, self.env)["runner_labels"][-1],
            "ppu-dedicated",
        )

    def test_unconfigured_image_fails(self):
        self.env.pop("THEAD_CI_IMAGE")
        with self.assertRaisesRegex(ValueError, "THEAD_CI_IMAGE"):
            resolve_config(self.defaults, self.env)

    def test_invalid_image_reference_fails(self):
        for image in (
            "registry.example/ppu:latest",
            "registry.example/ppu",
            "registry.example/ppu:v0.28.0 thead-ci",
            "registry.example/ppu@sha256:abc",
        ):
            with self.subTest(image=image), self.assertRaises(ValueError):
                resolve_config(self.defaults, dict(self.env, THEAD_CI_IMAGE=image))

    def test_duplicate_devices_and_privileged_fail(self):
        self.env["THEAD_CI_VISIBLE_DEVICES"] = "0,1,1,3"
        with self.assertRaises(ValueError):
            resolve_config(self.defaults, self.env)
        self.env["THEAD_CI_VISIBLE_DEVICES"] = "0,1,2,3"
        self.env["THEAD_CI_CONTAINER_OPTIONS"] = "--privileged"
        with self.assertRaises(ValueError):
            resolve_config(self.defaults, self.env)

    def test_invalid_json_shapes_fail(self):
        for value in ('"ppu"', "[]", "[1]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_config(
                    self.defaults, dict(self.env, THEAD_CI_RUNNER_LABELS=value)
                )


class AcceptanceTests(unittest.TestCase):
    def test_cleanup_rejects_reused_pid_and_ignores_unrelated_or_zombie(self):
        identity = {"pid": 100, "pgid": 100, "sid": 100, "start_ticks": 200}
        own = dict(identity, state="S")
        child = dict(identity, pid=101, start_ticks=201, state="S")
        unrelated = dict(identity, pid=102, pgid=102, sid=102, state="S")
        zombie = dict(identity, pid=103, state="Z")
        self.assertEqual(
            session_members(identity, [own, child, unrelated, zombie]), [own, child]
        )
        self.assertEqual(session_members(identity, [child]), [child])
        self.assertEqual(session_members(identity, [zombie]), [])
        with self.assertRaisesRegex(RuntimeError, "reused"):
            session_members(identity, [dict(own, start_ticks=300), child])
        with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
            session_members(identity, [own, dict(child, pgid=101)])

    def test_unit_cleanup_failure_stops_following_suites(self):
        for cleaned, expected_attempts in ((False, 1), (True, 3)):
            with self.subTest(cleaned=cleaned), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "units"
                receipt = {"clean": False, "cleanup_complete": cleaned}
                with (
                    patch.object(
                        sys, "argv", ["run_tests.py", "--output-dir", str(out)]
                    ),
                    patch.object(run_tests, "run_command", return_value=receipt) as run,
                    patch.object(run_tests, "validate_junit", return_value={}),
                ):
                    self.assertEqual(run_tests.main(), 1)
                facts = json.loads((out / "summary.json").read_text())
                self.assertEqual(run.call_count, expected_attempts)
                self.assertFalse(facts["passed"])
                self.assertEqual(facts["only_owned_cleanup_complete"], cleaned)

    def test_zero_skip_unique_junit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.xml"
            path.write_text(
                '<testsuites><testsuite><testcase classname="c" name="a"/><testcase classname="c" name="b"/></testsuite></testsuites>'
            )
            self.assertEqual(validate_junit(path, 2)["passed"], 2)
            for xml in (
                "<testsuites/>",
                '<testsuite><testcase name="a"><skipped/></testcase></testsuite>',
                '<testsuite><testcase name="a"/><testcase name="a"/></testsuite>',
                '<testsuite><testcase name="a"><error/></testcase></testsuite>',
            ):
                path.write_text(xml)
                with self.subTest(xml=xml), self.assertRaises(RuntimeError):
                    validate_junit(path, 2)

    def documents(self, root):
        for scenario, count in SCENARIOS.items():
            doc = {
                "case": {"scenario": scenario},
                "input": [{}] * count,
                "output": [
                    {"passed": True, "checks": dict.fromkeys(CHECKS, True)}
                    for _ in range(count)
                ],
            }
            (root / (scenario + ".json")).write_text(json.dumps(doc))

    def test_original_five_scenarios_26_requests_260_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.documents(root)
            facts = validate_documents(root)
            self.assertEqual((facts["logical_requests"], facts["checks"]), (26, 260))
            self.assertTrue(facts["passed"])
            p = root / "text_single.json"
            doc = json.loads(p.read_text())
            self.assertNotIn("quality_oracle_version", doc)
            doc["output"][0]["checks"]["expected_semantics"] = False
            p.write_text(json.dumps(doc))
            self.assertFalse(validate_documents(root)["passed"])

    def test_missing_check_or_scenario_is_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.documents(root)
            p = root / "text_single.json"
            doc = json.loads(p.read_text())
            doc["output"][0]["checks"].pop("expected_semantics")
            p.write_text(json.dumps(doc))
            with self.assertRaises(RuntimeError):
                validate_documents(root)
            p.unlink()
            with self.assertRaises(RuntimeError):
                validate_documents(root)

    def test_actual_info_graph_evidence_requires_capture_and_dispatch(self):
        # Sanitized INFO lines from the verified 0.28 graph runs.
        capture = "Capturing CUDA graphs (decode, FULL): 100%|####| 4/4 [00:01]"
        stats = "**CUDAGraph Stats:**\\n| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |\\n| 1 | 1 | 0 | FULL | 39 |"
        observed = parse_graph_observations(stats, capture)
        self.assertTrue(observed["graph_capture_observed"])
        self.assertTrue(observed["graph_runtime_dispatch_observed"])
        for output, error in (
            ("cudagraph_mode=FULL", ""),
            (stats.replace("39", "0"), capture.replace("4/4", "3/4")),
            ("| 1 | 1 | 0 | FULL | 39 |", "Capturing CUDA graphs 3/4"),
        ):
            with self.subTest(output=output):
                facts = parse_graph_observations(output, error)
                self.assertFalse(
                    facts["graph_capture_observed"]
                    and facts["graph_runtime_dispatch_observed"]
                )


if __name__ == "__main__":
    unittest.main()
