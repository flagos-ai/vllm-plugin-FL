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
        # Artifact defaults are an isolated fixture; the repository may still
        # keep MUSA disabled while its image is being published.
        self.config["musa_defaults"] = {
            "MUSA_CI_EMPTY_WHEEL": "/opt/musa/wheelhouse/vllm-0.28.0+empty-cp310-cp310-linux_x86_64.whl",
            "MUSA_CI_EMPTY_WHEEL_SHA256": "a" * 64,
            "MUSA_CI_DEPENDENCY_PROFILE": "latest",
            "MUSA_CI_DEPENDENCY_MANIFEST": "/opt/musa/dependency-manifest.json",
            "MUSA_CI_DEPENDENCY_MANIFEST_SHA256": "b" * 64,
            "MUSA_GATE_MODEL_27B": "/data/models/Qwen3.6-27B",
            "MUSA_GATE_MODEL_35B_A3B": "/data/models/Qwen3.6-35B-A3B",
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
        gate = platform["jobs"]["musa_gate"]
        self.assertEqual(gate["timeout-minutes"], "500")
        self.assertNotIn("runner.", json.dumps(gate["env"]))
        self.assertIn("GITHUB_ENV", gate["steps"][0]["run"])
        self.assertIn("benchmark", gate["needs"])
        self.assertTrue(
            any(
                step.get("run") == "bash .github/scripts/musa/run-adaptation-gate.sh"
                for step in gate["steps"]
            )
        )
        self.assertEqual(
            platform["jobs"]["e2e"]["strategy"]["max-parallel"],
            "${{ inputs.platform == 'musa' && 1 || 256 }}",
        )

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
        sanitized = json.dumps(result)
        self.assertNotIn("720902", sanitized)
        self.assertNotIn("SECRET_PASSWORD", sanitized)
        self.assertNotIn("/private", sanitized)

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
