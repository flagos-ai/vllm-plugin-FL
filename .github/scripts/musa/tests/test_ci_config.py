import ast
import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[4]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_module = load(".github/scripts/load_config.py", "musa_config")
dependency_module = load(
    ".github/scripts/musa/dependency_profile.py", "musa_dependencies"
)
probe_module = load(".github/scripts/musa/probe_runner.py", "musa_probe")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / ".github/configs/musa.yml").read_text())
        self.config["ci_image"] = (
            "harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl@sha256:" + "a" * 64
        )
        # This fixture omits hardware defaults to exercise explicit allocations
        # and rejection without an implicit device fallback.
        self.config["musa_defaults"] = {
            "MUSA_CI_EMPTY_WHEEL": "/opt/musa/wheelhouse/vllm-0.28.0+empty-cp310-cp310-linux_x86_64.whl",
            "MUSA_CI_EMPTY_WHEEL_SHA256": "a" * 64,
            "MUSA_CI_DEPENDENCY_PROFILE": "latest",
            "MUSA_CI_DEPENDENCY_MANIFEST": "/opt/musa/dependency-manifest.json",
            "MUSA_CI_DEPENDENCY_MANIFEST_SHA256": "b" * 64,
            "MUSA_GATE_MODEL_27B": "/data/models/Qwen/Qwen3.6-27B",
            "MUSA_GATE_MODEL_35B_A3B": "/data/models/Qwen/Qwen3.6-35B-A3B",
        }
        self.allocation = {
            "runtime": "mthreads",
            "physical_devices": "2,3,6,7",
            "logical_devices": "0,1,2,3",
            "pci_bus_ids": "0000:31:00.0,0000:4b:00.0,0000:b1:00.0,0000:ca:00.0",
        }

    def resolve(self, allocation=None, env=None):
        values = dict(env or {})
        if allocation is not None:
            values["MUSA_CI_ALLOCATION"] = json.dumps(allocation)
        with patch.dict(os.environ, values, clear=True):
            return config_module.resolve_musa_config(copy.deepcopy(self.config))

    def test_repository_defaults_resolve_exact_four_devices_without_env(self):
        config = yaml.safe_load((ROOT / ".github/configs/musa.yml").read_text())
        with patch.dict(os.environ, {}, clear=True):
            result = config_module.resolve_musa_config(config)
        options = result["container_options"]
        self.assertEqual(result["runner_labels"], ["mt-cicd-vllm-plugin"])
        self.assertEqual(
            result["ci_image"],
            "harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:"
            "8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06",
        )
        for value in (
            "--runtime=mthreads",
            "--env MTHREADS_VISIBLE_DEVICES=4,5,6,7",
            "--env MUSA_VISIBLE_DEVICES=0,1,2,3",
            "--env MUSA_CI_PCI_BUS_IDS=0000:83:00.0,0000:a3:00.0,0000:c3:00.0,0000:e3:00.0",
            "--env MUSA_GATE_MODEL_27B=/data/models/Qwen/Qwen3.6-27B",
            "--env MUSA_GATE_MODEL_35B_A3B=/data/models/Qwen/Qwen3.6-35B-A3B",
            "--env MUSA_CI_DEPENDENCY_PROFILE=latest",
        ):
            self.assertIn(value, options)
        for value in ("--privileged", "--pid=host", "MTHREADS_VISIBLE_DEVICES=all"):
            self.assertNotIn(value, options)
        registry = yaml.safe_load((ROOT / ".github/configs/platforms.yml").read_text())
        self.assertIs(registry["platforms"]["musa"]["enabled"], True)

    def test_dispatch_uses_baked_wheels_and_exact_mask(self):
        result = self.resolve(self.allocation)
        options = result["container_options"]
        self.assertIn("--runtime=mthreads", options)
        self.assertIn("--env MTHREADS_VISIBLE_DEVICES=2,3,6,7", options)
        self.assertIn("--env MUSA_VISIBLE_DEVICES=0,1,2,3", options)
        self.assertIn("--env MUSA_CI_EMPTY_WHEEL=/opt/musa/wheelhouse/", options)
        self.assertIn("--env MUSA_CI_DEPENDENCY_PROFILE=latest", options)
        self.assertIn("--env MUSA_CI_DEPENDENCY_MANIFEST_SHA256=" + "b" * 64, options)
        self.assertNotIn("--privileged", options)
        self.assertNotIn("MTHREADS_VISIBLE_DEVICES=all", options)

    def test_repository_allocation_remains_supported(self):
        env = {
            "MUSA_CI_RUNTIME": self.allocation["runtime"],
            "MUSA_CI_PHYSICAL_DEVICES": self.allocation["physical_devices"],
            "MUSA_CI_LOGICAL_DEVICES": self.allocation["logical_devices"],
            "MUSA_CI_PCI_BUS_IDS": self.allocation["pci_bus_ids"],
        }
        result = self.resolve(env=env)
        self.assertIn(
            "--env MTHREADS_VISIBLE_DEVICES=2,3,6,7", result["container_options"]
        )

    def test_explicit_dispatch_can_override_runner_and_models(self):
        allocation = dict(
            self.allocation,
            runner_labels=["self-hosted", "mt-cicd-vllm-plugin"],
            model_27b="/data/models/27B",
        )
        result = self.resolve(allocation)
        self.assertEqual(result["runner_labels"], allocation["runner_labels"])
        self.assertIn(
            "--env MUSA_GATE_MODEL_27B=/data/models/27B", result["container_options"]
        )

    def test_missing_allocation_fails_without_gpu_fallback(self):
        with self.assertRaises(ValueError):
            self.resolve()

    def test_bad_allocation_inputs_rejected(self):
        bad = [
            ("physical_devices", "all"),
            ("physical_devices", "2,3,6,6"),
            ("physical_devices", "2,3,6,7 --privileged"),
            ("logical_devices", "2,3,6,7"),
            ("runtime", "mthreads; touch /tmp/bad"),
            ("pci_bus_ids", "0000:31:00.0"),
            ("runner_labels", ["mt-cicd-vllm-plugin", "--privileged"]),
            ("runner_labels", [{"label": "x"}]),
            ("model_27b", "/opt/musa/model"),
            ("model_27b", "/data/../etc/secret"),
            ("unknown_field", "x"),
        ]
        for key, value in bad:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.resolve(dict(self.allocation, **{key: value}))

    def test_duplicate_dispatch_fields_rejected(self):
        with (
            patch.dict(
                os.environ,
                {"MUSA_CI_ALLOCATION": '{"runtime":"mthreads","runtime":"other"}'},
                clear=True,
            ),
            self.assertRaises(ValueError),
        ):
            config_module.resolve_musa_config(copy.deepcopy(self.config))

    def test_incorrect_artifact_paths_and_hashes_fail(self):
        bad = [
            ("MUSA_CI_EMPTY_WHEEL", "/opt/musa/outside.whl"),
            ("MUSA_CI_EMPTY_WHEEL", "/opt/musa/wheelhouse/../outside.whl"),
            ("MUSA_CI_DEPENDENCY_MANIFEST_SHA256", "pending"),
        ]
        for key, value in bad:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.resolve(self.allocation, {key: value})

    def test_baseline_override_does_not_silently_consume_latest_manifest(self):
        result = self.resolve(
            self.allocation, {"MUSA_CI_DEPENDENCY_PROFILE": "baseline"}
        )
        self.assertNotIn(
            "--env MUSA_CI_DEPENDENCY_MANIFEST=", result["container_options"]
        )
        with self.assertRaises(ValueError):
            self.resolve(
                self.allocation,
                {
                    "MUSA_CI_DEPENDENCY_PROFILE": "baseline",
                    "MUSA_CI_DEPENDENCY_MANIFEST": "/opt/musa/dependency-manifest.json",
                },
            )

    def test_manifest_only_accepts_data_or_baked_wheelhouse(self):
        manifest = {
            "schema_version": 1,
            "profile": "latest",
            "wheels": [
                {
                    "name": name,
                    "version": "test",
                    "path": "/opt/musa/wheelhouse/" + name + ".whl",
                    "bytes": 1,
                    "sha256": "a" * 64,
                }
                for name in ["flag-gems", "flagtree", "pyyaml"]
            ],
            "vendor_versions": {
                "torch": "2.9.0",
                "torch_musa": "2.9.0",
                "musa_sdk": "4.3.5",
            },
            "vendor_files": [
                {
                    "role": role,
                    "path": "/usr/local/vendor/" + role,
                    "bytes": 1,
                    "sha256": "b" * 64,
                }
                for role in [
                    "torch_source",
                    "torch_musa_source",
                    "torch_metadata",
                    "torch_musa_metadata",
                    "sdk_version",
                ]
            ],
            "source_lineage": {"scope": "private candidate"},
        }
        dependency_module.validate_manifest(manifest)
        for path in [
            "/opt/musa/other/a.whl",
            "/opt/musa/wheelhouse/../a.whl",
            "/tmp/a.whl",
        ]:
            bad = copy.deepcopy(manifest)
            bad["wheels"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                dependency_module.validate_manifest(bad)

    def test_workflow_keeps_original_gate_and_generic_main_chain(self):
        workflow = yaml.load(
            (ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
        )
        inputs = workflow["on"]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["platform"]["options"], ["all", "musa"])
        self.assertEqual(
            workflow["jobs"]["test-musa"]["uses"],
            "./.github/workflows/_platform_test.yml",
        )
        platform = yaml.load(
            (ROOT / ".github/workflows/_platform_test.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        suite = platform["jobs"]["musa_suite"]
        self.assertEqual(suite["if"], "inputs.platform == 'musa'")
        self.assertEqual(suite["needs"], "setup")
        self.assertEqual(suite["timeout-minutes"], "1200")
        self.assertNotIn("runner.", json.dumps(suite["env"]))
        self.assertIn("GITHUB_ENV", suite["steps"][0]["run"])
        stage_commands = [
            step["run"]
            for step in suite["steps"]
            if step.get("run", "").startswith(
                "python -I -B .github/scripts/musa/run_ci_stage.py"
            )
        ]
        self.assertEqual(
            stage_commands,
            [
                "python -I -B .github/scripts/musa/run_ci_stage.py --scope " + scope
                for scope in ("unit", "functional", "e2e", "benchmark")
            ],
        )
        gate_steps = [
            step
            for step in suite["steps"]
            if step.get("run") == "bash .github/scripts/musa/run-adaptation-gate.sh"
        ]
        self.assertEqual(len(gate_steps), 1)
        gate = gate_steps[0]
        self.assertEqual(gate["timeout-minutes"], "500")
        self.assertIn("!cancelled()", gate["if"])
        for scope in ("unit", "functional", "e2e", "benchmark"):
            self.assertIn(
                "steps." + scope + ".outputs.gpu_stages_safe == 'true'", gate["if"]
            )
        self.assertIn("steps.devices.outcome == 'success'", gate["if"])
        self.assertIn("steps.install.outcome == 'success'", gate["if"])
        self.assertEqual(platform["jobs"]["e2e"]["strategy"]["max-parallel"], "256")
        self.assertIn("inputs.platform != 'musa'", platform["jobs"]["e2e"]["if"])

    def test_probe_is_separate_bare_host_workflow(self):
        workflow = yaml.load(
            (ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
        )
        probe = workflow["jobs"]["musa-runner-probe"]
        self.assertEqual(probe["runs-on"], "mt-cicd-vllm-plugin")
        self.assertNotIn("container", probe)
        selection = next(
            step
            for step in workflow["jobs"]["discover"]["steps"]
            if step.get("id") == "detect"
        )["env"]["REQUESTED_PLATFORM"]
        self.assertIn("inputs.musa_probe_only && 'musa'", selection)
        self.assertIn("inputs.musa_probe_only", workflow["jobs"]["test-musa"]["if"])
        self.assertIn("inputs.musa_probe_only", probe["if"])


class ProbeTests(unittest.TestCase):
    def test_overview_parser_discards_process_table_and_private_strings(self):
        sample = """mthreads-gmi:2.3.2           Driver Version:3.3.5-server
0    MTT S5000            |00000000:03:00.0    |23%   80408MiB(81920MiB)
4    MTT S5000            |00000000:83:00.0    |0%    2MiB(81920MiB)
7    MTT S5000            |00000000:e3:00.0    |94%   40405MiB(81920MiB)
Processes:
4    720902    /private/model/job.py SECRET_PASSWORD     2MiB
"""
        result = probe_module.parse_gmi(sample)
        self.assertEqual(result["driver_version"], "3.3.5-server")
        self.assertEqual([x["physical_id"] for x in result["devices"]], [0, 4, 7])
        self.assertEqual(result["devices"][1]["pci"], "0000:83:00.0")
        self.assertEqual(result["devices"][1]["memory_used_mib"], 2)
        self.assertEqual(
            [row["utilization_percent"] for row in result["devices"]], [23.0, 0.0, 94.0]
        )
        sanitized = json.dumps(result)
        self.assertNotIn("720902", sanitized)
        self.assertNotIn("SECRET_PASSWORD", sanitized)
        self.assertNotIn("/private", sanitized)

    def idle_resources(self):
        expected = ["0000:83:00.0", "0000:a3:00.0", "0000:c3:00.0", "0000:e3:00.0"]
        return expected, {
            "devices": [
                {
                    "physical_id": index + 4,
                    "pci": pci,
                    "memory_used_mib": 0,
                    "memory_total_mib": 81920,
                    "utilization_percent": 0.0,
                }
                for index, pci in enumerate(expected)
            ]
        }

    def test_selected_idle_cards_ignore_unselected_busy_and_keep_pci_order(self):
        expected, resources = self.idle_resources()
        resources["devices"].reverse()
        resources["devices"].append(
            {
                "physical_id": 0,
                "pci": "0000:03:00.0",
                "memory_used_mib": 80408,
                "memory_total_mib": 81920,
                "utilization_percent": 99.0,
            }
        )
        original = copy.deepcopy(resources)
        selected = probe_module.require_idle_devices(resources, expected)
        self.assertEqual([row["pci"] for row in selected], expected)
        self.assertEqual(resources, original)

    def test_busy_unknown_or_invalid_selected_usage_is_rejected(self):
        for field, value in (
            ("memory_used_mib", 1),
            ("memory_used_mib", None),
            ("memory_used_mib", False),
            ("memory_total_mib", 0),
            ("utilization_percent", 0.1),
            ("utilization_percent", None),
            ("utilization_percent", "0"),
            ("utilization_percent", False),
            ("utilization_percent", float("nan")),
        ):
            expected, resources = self.idle_resources()
            resources["devices"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                probe_module.require_idle_devices(resources, expected)

    def test_missing_or_duplicate_selected_card_is_rejected(self):
        expected, resources = self.idle_resources()
        resources["devices"].pop()
        with self.assertRaises(ValueError):
            probe_module.require_idle_devices(resources, expected)
        expected, resources = self.idle_resources()
        resources["devices"].append(copy.deepcopy(resources["devices"][0]))
        with self.assertRaises(ValueError):
            probe_module.require_idle_devices(resources, expected)
        expected, resources = self.idle_resources()
        resources["devices"][1]["physical_id"] = 4
        with self.assertRaises(ValueError):
            probe_module.require_idle_devices(resources, expected)
        for selected_pci in (expected[:3], [expected[0]] * 4, ["invalid"] * 4):
            with self.subTest(selected_pci=selected_pci), self.assertRaises(ValueError):
                probe_module.require_idle_devices(resources, selected_pci)

    def test_unknown_parser_utilization_cannot_pass_admission(self):
        expected, resources = self.idle_resources()
        rows = []
        for row in resources["devices"]:
            utilization = "N/A" if row["physical_id"] == 4 else "0%"
            rows.append(
                f"{row['physical_id']} MTT S5000 |{row['pci']} |{utilization} 0MiB(81920MiB)"
            )
        parsed = probe_module.parse_gmi("\n".join(rows))
        self.assertIsNone(parsed["devices"][0]["utilization_percent"])
        with self.assertRaises(ValueError):
            probe_module.require_idle_devices(parsed, expected)

    def test_resource_admission_precedes_accelerator_initialization(self):
        script = (ROOT / ".github/scripts/musa/check.sh").read_text()
        source = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        module = ast.parse(source)
        guard = next(
            node
            for node in module.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "sample"
        )
        torch_import = next(
            node
            for node in module.body
            if isinstance(node, ast.Import)
            and any(name.name == "torch" for name in node.names)
        )
        node_check = next(
            node
            for node in module.body
            if isinstance(node, ast.Assert) and "len(nodes)" in ast.unparse(node.test)
        )
        self.assertLess(node_check.lineno, guard.lineno)
        self.assertLess(guard.lineno, torch_import.lineno)
        self.assertEqual(ast.literal_eval(guard.iter), (1, 2))
        self.assertIn(
            "probe.require_idle_devices(resources, expected)", ast.unparse(guard)
        )
        self.assertIn("raise RuntimeError", ast.unparse(guard))
        self.assertNotIn("mthreads-gmi\n", script)

    def test_busy_fd_holder_prevents_idle_candidate(self):
        nodes = [{"node_id": 4, "pci": "0000:83:00.0"}]
        gmi = {
            "devices": [{"physical_id": 4, "pci": "0000:83:00.0", "memory_used_mib": 2}]
        }
        fd_report = {"scan_complete": True, "open_fd_counts": {4: 1}}
        self.assertEqual(probe_module.candidate_devices(nodes, gmi, fd_report), [])
        fd_report["open_fd_counts"][4] = 0
        self.assertEqual(len(probe_module.candidate_devices(nodes, gmi, fd_report)), 1)

    def test_incomplete_fd_scan_never_proposes_allocation(self):
        self.assertEqual(
            probe_module.candidate_devices(
                [], {"devices": []}, {"scan_complete": False}
            ),
            [],
        )

    def test_duplicate_vendor_rows_do_not_produce_mapping(self):
        row = "4    MTT S5000 |00000000:83:00.0 |0% 2MiB(81920MiB)\n"
        self.assertEqual(probe_module.parse_gmi(row + row)["devices"], [])


if __name__ == "__main__":
    unittest.main()
