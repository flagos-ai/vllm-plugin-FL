# Copyright (c) 2026 BAAI. All rights reserved.
"""Real local CPU import contracts; no accelerator runtime acceptance.

Run this file directly with unittest to avoid tests/conftest.py hardware probes.
Children load the real source in fresh processes.  The no-site children cover
stdlib configuration and genuine missing dependencies only; Torch/reference
contracts use normal site with the actual local CPU Torch installation.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_FILE = REPO_ROOT / "vllm_fl" / "utils.py"
MARKER = "VLLM_FL_LOCAL_CPU_CONTRACT="
CHILD_TIMEOUT_SECONDS = 20

_FILE_CONFIG = r'''
import json, os, pathlib, runpy, sys
utils_file = pathlib.Path(sys.argv[1]).resolve()
cases = json.loads(sys.argv[2])
attempted = []
def observe(event, arguments):
    if event == "import" and arguments:
        name = arguments[0]
        if isinstance(name, str) and (name == "flag_gems" or name.startswith("flag_gems.")):
            attempted.append(name)
sys.addaudithook(observe)
results = []
for case in cases:
    if case["environment"] is None:
        os.environ.pop("VLLM_FL_OP_CONFIG", None)
    else:
        os.environ["VLLM_FL_OP_CONFIG"] = case["environment"]
    try:
        namespace = runpy.run_path(str(utils_file))
    except ValueError as error:
        result = {"name": case["name"], "error_type": type(error).__name__, "message": str(error)}
    else:
        initial = namespace["get_op_config"]()
        # get_op_config is a cached accessor, not an environment reloader.
        os.environ["VLLM_FL_OP_CONFIG"] = str(utils_file.parent / "missing-not-reloaded.json")
        assert namespace["get_op_config"]() == initial
        result = {"name": case["name"], "value": initial, "cached_after_environment_change": True}
    assert not attempted, attempted
    assert not any(name == "flag_gems" or name.startswith("flag_gems.") for name in sys.modules)
    results.append(result)
print("VLLM_FL_LOCAL_CPU_CONTRACT=" + json.dumps({"kind": "stdlib_config_only", "results": results, "flag_gems_import_attempts": attempted, "accelerator_acceptance": False}))
'''

_FILE_MISSING = r'''
import json, pathlib, runpy, sys, traceback
utils_file = pathlib.Path(sys.argv[1]).resolve()
namespace = runpy.run_path(str(utils_file))
assert sys.flags.no_site == 1
assert not any(name == "flag_gems" or name.startswith("flag_gems.") for name in sys.modules)
results = []
for name in ("DeviceInfo", "get_flaggems_all_ops"):
    try:
        namespace[name]()
    except ModuleNotFoundError as error:
        assert error.name == "flag_gems", (type(error).__name__, error.name, str(error))
        results.append({"entrypoint": name, "error_type": type(error).__name__, "missing_module": error.name, "traceback": traceback.format_exc()})
    else:
        raise AssertionError(name + " swallowed the genuine missing dependency")
print("VLLM_FL_LOCAL_CPU_CONTRACT=" + json.dumps({"kind": "no_site_missing_dependency_only", "results": results, "accelerator_acceptance": False}))
'''

_NORMAL_CPU = r'''
import hashlib, json, math, pathlib, sys
from types import SimpleNamespace
repo = pathlib.Path(sys.argv[1]).resolve()
# This is the actual checkout import path for a local source contract only.
sys.path.insert(0, str(repo))
assert sys.flags.no_site == 0
attempted = []
def observe(event, arguments):
    if event == "import" and arguments:
        name = arguments[0]
        if isinstance(name, str) and (name == "flag_gems" or name.startswith("flag_gems.")):
            attempted.append(name)
sys.addaudithook(observe)
import vllm_fl
from vllm_fl import utils
assert pathlib.Path(vllm_fl.__file__).resolve() == repo / "vllm_fl" / "__init__.py"
assert pathlib.Path(utils.__file__).resolve() == repo / "vllm_fl" / "utils.py"
assert utils.get_op_config() is None
import torch
from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend
from vllm_fl.dispatch import io_common
assert pathlib.Path(sys.modules[ReferenceBackend.__module__].__file__).resolve() == repo / "vllm_fl" / "dispatch" / "backends" / "reference" / "reference.py"
backend = ReferenceBackend()
assert backend.name == "reference" and backend.is_available()
x = torch.tensor([[1., -1., .5, -2.], [0., 2., 1., 3.]], dtype=torch.float64, device="cpu")
y = backend.silu_and_mul(None, x)
expected_silu = [[row[j] / (1 + math.exp(-row[j])) * row[j + 2] for j in range(2)] for row in x.tolist()]
assert y.device.type == "cpu" and y.dtype == torch.float64
assert torch.allclose(y, torch.tensor(expected_silu, dtype=torch.float64), rtol=1e-12, atol=1e-12)
values = [[1., 2.], [-3., 4.]]
residual_values = [[.5, -.5], [1., -1.]]
weights = [1.25, .75]
epsilon = 1e-5
obj = SimpleNamespace(weight=torch.tensor(weights, dtype=torch.float64, device="cpu"), variance_epsilon=epsilon)
norm_x = torch.tensor(values, dtype=torch.float64, device="cpu")
residual = torch.tensor(residual_values, dtype=torch.float64, device="cpu")
def expected_rms(rows):
    return [[value * weights[j] / math.sqrt(sum(v * v for v in row) / len(row) + epsilon) for j, value in enumerate(row)] for row in rows]
normalized = backend.rms_norm(obj, norm_x)
assert normalized.device.type == "cpu" and normalized.dtype == torch.float64
assert torch.allclose(normalized, torch.tensor(expected_rms(values), dtype=torch.float64), rtol=1e-12, atol=1e-12)
combined = [[a + b for a, b in zip(row, res)] for row, res in zip(values, residual_values)]
normalized_residual, observed_residual = backend.rms_norm(obj, norm_x, residual)
assert normalized_residual.device.type == observed_residual.device.type == "cpu"
assert torch.equal(observed_residual, torch.tensor(combined, dtype=torch.float64))
assert torch.allclose(normalized_residual, torch.tensor(expected_rms(combined), dtype=torch.float64), rtol=1e-12, atol=1e-12)
assert torch.equal(norm_x, torch.tensor(values, dtype=torch.float64))
assert torch.equal(residual, torch.tensor(residual_values, dtype=torch.float64))
assert io_common.is_io_active() is False
@io_common.managed_inference_mode()
def evaluate():
    value = torch.tensor([2., -1.], dtype=torch.float32, device="cpu", requires_grad=True) * 3
    return {"grad_enabled": torch.is_grad_enabled(), "inference_enabled": torch.is_inference_mode_enabled(), "tensor_is_inference": value.is_inference(), "requires_grad": value.requires_grad, "device": str(value.device), "values": value.tolist()}
initial = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
with torch.enable_grad():
    outer = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
    observed = evaluate()
    assert observed == {"grad_enabled": False, "inference_enabled": True, "tensor_is_inference": True, "requires_grad": False, "device": "cpu", "values": [6., -3.]}
    assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == outer
    @io_common.managed_inference_mode()
    def raising():
        raise ValueError("real CPU state restoration probe")
    try:
        raising()
    except ValueError as error:
        assert str(error) == "real CPU state restoration probe"
    else:
        raise AssertionError("Decorator swallowed the real exception")
    assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == outer
assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == initial
assert not attempted, attempted
assert not any(name == "flag_gems" or name.startswith("flag_gems.") for name in sys.modules)
print("VLLM_FL_LOCAL_CPU_CONTRACT=" + json.dumps({"kind": "real_normal_site_cpu_contract", "torch_version": torch.__version__, "torch_origin": torch.__file__, "plugin_origin": vllm_fl.__file__, "utils_sha256": hashlib.sha256(pathlib.Path(utils.__file__).read_bytes()).hexdigest(), "silu_values": y.tolist(), "rms_values": normalized.tolist(), "rms_residual_values": normalized_residual.tolist(), "inference_non_io": observed, "state_restored_after_exception": True, "io_active_branch_tested": False, "flag_gems_import_attempts": attempted, "accelerator_acceptance": False}))
'''

_NORMAL_MISSING = r'''
import importlib.util, json, pathlib, sys, traceback
repo = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
assert sys.flags.no_site == 0
import torch
from vllm_fl import utils
from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
assert pathlib.Path(utils.__file__).resolve() == repo / "vllm_fl" / "utils.py"
assert pathlib.Path(sys.modules[FlagGemsBackend.__module__].__file__).resolve() == repo / "vllm_fl" / "dispatch" / "backends" / "flaggems" / "flaggems.py"
if importlib.util.find_spec("flag_gems") is not None:
    # A real installed backend needs its own device validation. Do not replace it.
    result = {"kind": "real_normal_site_utils_missing_dependency", "not_applicable": "Real FlagGems is installed; no missing-dependency outcome was fabricated", "flag_gems_operator_tested": False, "accelerator_acceptance": False}
else:
    backend = FlagGemsBackend()
    assert backend.is_available() is False
    errors = []
    for name in ("DeviceInfo", "get_flaggems_all_ops"):
        try:
            getattr(utils, name)()
        except ModuleNotFoundError as error:
            assert error.name == "flag_gems", (type(error).__name__, error.name, str(error))
            errors.append({"entrypoint": name, "error_type": type(error).__name__, "missing_module": error.name, "traceback": traceback.format_exc()})
        else:
            raise AssertionError(name + " swallowed the genuine missing dependency")
    result = {"kind": "real_normal_site_utils_missing_dependency", "available": False, "results": errors, "torch_version": torch.__version__, "utils_origin": utils.__file__, "flag_gems_operator_tested": False, "accelerator_acceptance": False}
print("VLLM_FL_LOCAL_CPU_CONTRACT=" + json.dumps(result))
'''


class TestUtilsImportBoundaries(unittest.TestCase):
    def _probe(self, source, arguments, *, no_site):
        command = [sys.executable, "-I", "-B"]
        if no_site:
            command.append("-S")
        command.extend(["-c", textwrap.dedent(source), *map(str, arguments)])
        environment = os.environ.copy()
        environment.pop("VLLM_FL_OP_CONFIG", None)
        # Other environment values, including plugin and GPU settings, are inherited.
        completed = subprocess.run(command, env=environment, capture_output=True,
                                   text=True, encoding="utf-8", errors="strict",
                                   timeout=CHILD_TIMEOUT_SECONDS, check=False)
        # Emit complete captured streams once, including successful-child warnings.
        sys.stdout.write(completed.stdout)
        sys.stdout.flush()
        sys.stderr.write(completed.stderr)
        sys.stderr.flush()
        self.assertEqual(completed.returncode, 0,
                         "Actual child exit {}; complete streams emitted above".format(
                             completed.returncode))
        records = [line[len(MARKER):] for line in completed.stdout.splitlines()
                   if line.startswith(MARKER)]
        self.assertEqual(len(records), 1, "Expected one result marker; complete streams emitted above")
        result = json.loads(records[0])
        self.assertIs(result["accelerator_acceptance"], False)
        return result

    def test_stdlib_config_defaults_and_cache_without_flaggems(self):
        with tempfile.TemporaryDirectory(prefix="vllm-fl-config-test-") as directory:
            path = Path(directory) / "valid.json"
            expected = {"rms_norm": "reference", "silu_and_mul": "flagos"}
            path.write_text(json.dumps(expected), encoding="utf-8")
            empty = Path(directory) / "empty.json"
            empty.write_text("{}", encoding="utf-8")
            cases = [{"name": "unset", "environment": None},
                     {"name": "blank", "environment": " \t"},
                     {"name": "valid", "environment": str(path)},
                     {"name": "empty_object", "environment": str(empty)}]
            result = self._probe(_FILE_CONFIG, [UTILS_FILE, json.dumps(cases)], no_site=True)
        self.assertEqual([row["value"] for row in result["results"]],
                         [None, None, expected, {}])
        self.assertTrue(all(row["cached_after_environment_change"] for row in result["results"]))
        self.assertEqual(result["flag_gems_import_attempts"], [])

    def test_stdlib_config_validation_errors_without_flaggems(self):
        with tempfile.TemporaryDirectory(prefix="vllm-fl-invalid-config-test-") as directory:
            base = Path(directory)
            cases = []
            for name, content in (("malformed", "{"), ("not_object", "[]"),
                                  ("null_object", "null"), ("non_string_value", '{"rms_norm": 1}')):
                path = base / (name + ".json")
                path.write_text(content, encoding="utf-8")
                cases.append({"name": name, "environment": str(path)})
            missing = base / "missing.json"
            cases.append({"name": "missing", "environment": str(missing)})
            result = self._probe(_FILE_CONFIG, [UTILS_FILE, json.dumps(cases)], no_site=True)
        self.assertEqual([row["error_type"] for row in result["results"]], ["ValueError"] * 5)
        self.assertEqual([row["message"] for row in result["results"]],
                         ["Invalid VLLM_FL_OP_CONFIG JSON file.",
                          "VLLM_FL_OP_CONFIG must be a JSON object.",
                          "VLLM_FL_OP_CONFIG must be a JSON object.",
                          "VLLM_FL_OP_CONFIG must map strings to strings.",
                          "VLLM_FL_OP_CONFIG file not found: " + str(missing)])
        self.assertEqual(result["flag_gems_import_attempts"], [])

    def test_no_site_real_missing_dependency_is_not_swallowed(self):
        result = self._probe(_FILE_MISSING, [UTILS_FILE], no_site=True)
        self.assertEqual([row["entrypoint"] for row in result["results"]],
                         ["DeviceInfo", "get_flaggems_all_ops"])
        self.assertEqual([row["missing_module"] for row in result["results"]], ["flag_gems"] * 2)

    def test_real_normal_site_package_reference_cpu_and_inference_states(self):
        result = self._probe(_NORMAL_CPU, [REPO_ROOT], no_site=False)
        self.assertEqual(result["kind"], "real_normal_site_cpu_contract")
        self.assertEqual(result["flag_gems_import_attempts"], [])
        self.assertIs(result["state_restored_after_exception"], True)
        self.assertIs(result["io_active_branch_tested"], False)

    def test_real_normal_site_utils_missing_dependency(self):
        result = self._probe(_NORMAL_MISSING, [REPO_ROOT], no_site=False)
        if "not_applicable" in result:
            self.skipTest(result["not_applicable"])
        self.assertEqual(result["kind"], "real_normal_site_utils_missing_dependency")
        self.assertIs(result["available"], False)
        self.assertEqual([row["entrypoint"] for row in result["results"]],
                         ["DeviceInfo", "get_flaggems_all_ops"])
        self.assertEqual([row["error_type"] for row in result["results"]], ["ModuleNotFoundError"] * 2)
        self.assertEqual([row["missing_module"] for row in result["results"]], ["flag_gems"] * 2)
        self.assertIs(result["flag_gems_operator_tested"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
