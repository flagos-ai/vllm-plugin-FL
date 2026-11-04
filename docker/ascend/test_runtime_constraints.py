# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU checks for dependency drift and vLLM empty-wheel validation."""

import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "ascend_runtime_constraints", Path(__file__).with_name("runtime_constraints.py")
)
stack = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stack)


def distribution(name, version, files=(), direct_url=None):
    return SimpleNamespace(
        metadata={"Name": name},
        version=version,
        files=[Path(path) for path in files],
        read_text=lambda _: json.dumps(direct_url or {}),
    )


def native_stack():
    return [
        distribution("torch", "2.10.0+cpu"),
        distribution("torch_npu", "2.10.0"),
        distribution("torchvision", "0.25.0+cpu"),
        distribution("flagtree", "0.6.1+ascend3.5"),
        distribution("flag_gems", "5.3.5"),
        distribution("cann-shmem", "1.6.0"),
        distribution("numpy", "1.26.4"),
    ]


_COMPILER_SNAPSHOT = {
    "executable": "/flagos/bin/python3.11",
    "prefix": "/flagos",
    "entrypoint": "/flagos/lib/python3.11/site-packages/triton/__init__.py",
    "sha256": "example",
    "owners": [{"name": "flagtree", "version": "0.6.1", "record_hash": None}],
}


class NativeStackTests(unittest.TestCase):
    def setUp(self):
        compiler_patch = patch.object(
            stack, "compiler_snapshot", return_value=_COMPILER_SNAPSHOT
        )
        compiler_patch.start()
        self.addCleanup(compiler_patch.stop)

    def test_capture_preserves_vendor_stack_but_allows_application_replacement(self):
        packages = native_stack() + [
            distribution("vllm", "0.24.0+flagos"),
            distribution("vllm-plugin-fl", "0.3.0"),
            distribution("vllm-ascend", "0.24.0"),
            distribution("flaggems-vllm", "0.1.0"),
            distribution("xgrammar", "0.1.23+flagos"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(
                stack.importlib.metadata, "distributions", return_value=packages
            ):
                stack.capture(directory)
            captured = json.loads((directory / "native-stack.json").read_text())
            self.assertEqual(captured["torch-npu"], "2.10.0")
            self.assertEqual(captured["xgrammar"], "0.1.23+flagos")
            for name in ("vllm", "vllm-plugin-fl", "vllm-ascend", "flaggems-vllm"):
                self.assertNotIn(name, captured)
            self.assertIn(
                "torch==2.10.0+cpu", (directory / "native-overrides.txt").read_text()
            )
            self.assertIn('"triton"', (directory / "uv.toml").read_text())

    def test_capture_keeps_selected_triton_distribution(self):
        packages = native_stack() + [distribution("triton", "3.5.1+ascend")]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(
                stack.importlib.metadata, "distributions", return_value=packages
            ):
                stack.capture(directory)
            self.assertEqual(
                (directory / "uv.toml").read_text(),
                'exclude-dependencies = ["triton-ascend"]\n',
            )

    def test_capture_rejects_wrong_compiler_environment(self):
        with (
            patch.object(
                stack.importlib.metadata,
                "distributions",
                return_value=[distribution("torch", "2.10.0+cpu")],
            ),
            self.assertRaisesRegex(RuntimeError, "FlagTree compiler"),
        ):
            stack.capture(Path("unused"))

    def test_conflicting_duplicate_native_distributions_fail(self):
        packages = native_stack() + [distribution("torch-npu", "2.11.0")]
        with (
            patch.object(
                stack.importlib.metadata, "distributions", return_value=packages
            ),
            self.assertRaisesRegex(RuntimeError, "Conflicting torch-npu"),
        ):
            stack.native_versions()

    def test_verify_rejects_changed_added_and_removed_native_packages(self):
        original = {
            normalize_name: version
            for normalize_name, version in (
                ("torch", "2.10.0+cpu"),
                ("torch-npu", "2.10.0"),
            )
        }
        variants = (
            {"torch": "2.11.0", "torch-npu": "2.10.0"},
            dict(original, **{"nvidia-cublas-cu12": "12.9.1"}),
            {"torch": "2.10.0+cpu"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "native-stack.json").write_text(json.dumps(original))
            for changed in variants:
                with (
                    self.subTest(changed=changed),
                    patch.object(stack, "native_versions", return_value=changed),
                    self.assertRaisesRegex(RuntimeError, "changed Ascend packages"),
                ):
                    stack.verify(directory)

    def test_verify_accepts_unchanged_stack_without_importing_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "native-stack.json").write_text('{"torch": "2.10.0+cpu"}')
            (directory / "compiler-entrypoint.json").write_text(
                json.dumps(_COMPILER_SNAPSHOT)
            )
            with (
                patch.object(
                    stack, "native_versions", return_value={"torch": "2.10.0+cpu"}
                ),
                patch.object(stack, "verify_imports") as imports,
            ):
                stack.verify(directory)
            imports.assert_not_called()


class CompilerEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.entrypoint = self.directory / "triton" / "__init__.py"
        self.entrypoint.parent.mkdir()
        self.content = b'__version__ = "3.5.1"\n'
        self.entrypoint.write_bytes(self.content)
        self.packages = [self.owner("flagtree", self.content)]
        spec_patch = patch.object(
            stack.importlib.util,
            "find_spec",
            side_effect=lambda _: SimpleNamespace(origin=str(self.entrypoint)),
        )
        packages_patch = patch.object(
            stack.importlib.metadata,
            "distributions",
            side_effect=lambda: self.packages,
        )
        spec_patch.start()
        packages_patch.start()
        self.addCleanup(spec_patch.stop)
        self.addCleanup(packages_patch.stop)

    def owner(self, name, recorded_content):
        path = importlib.metadata.PackagePath("triton/__init__.py")
        if recorded_content is not None:
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(recorded_content).digest())
                .decode()
                .rstrip("=")
            )
            path.hash = importlib.metadata.FileHash(f"sha256={digest}")
        return SimpleNamespace(
            metadata={"Name": name},
            version="0.6.1" if name == "flagtree" else "3.5.0",
            files=[path],
            locate_file=lambda _: self.entrypoint,
        )

    def save_manifest(self):
        (self.directory / "native-stack.json").write_text("{}")
        (self.directory / "compiler-entrypoint.json").write_text(
            json.dumps(stack.compiler_snapshot())
        )

    def assert_snapshot_changed(self):
        with (
            patch.object(stack, "native_versions", return_value={}),
            self.assertRaisesRegex(RuntimeError, "changed the Triton compiler"),
        ):
            stack.verify(self.directory)

    def test_captures_only_matching_owner_with_dual_compiler_distributions(self):
        self.packages.append(self.owner("triton", b"another compiler"))
        snapshot = stack.compiler_snapshot()
        self.assertEqual(snapshot["entrypoint"], str(self.entrypoint.resolve()))
        self.assertEqual(snapshot["sha256"], hashlib.sha256(self.content).hexdigest())
        self.assertEqual([owner["name"] for owner in snapshot["owners"]], ["flagtree"])

    def test_rejects_entrypoint_without_matching_metadata_owner(self):
        self.packages = [self.owner("triton", b"another compiler")]
        with self.assertRaisesRegex(RuntimeError, "no matching distribution RECORD"):
            stack.compiler_snapshot()

    def test_detects_overwrite_even_when_distribution_version_is_unchanged(self):
        # Editable/legacy metadata can omit RECORD hashes. The captured entrypoint
        # SHA256 must still detect an overwrite when metadata remains unchanged.
        self.packages = [self.owner("flagtree", None)]
        self.save_manifest()
        self.entrypoint.write_bytes(b"overwritten entrypoint")
        self.assert_snapshot_changed()

    def test_detects_selected_compiler_path_change(self):
        self.save_manifest()
        self.entrypoint = self.directory / "another" / "triton" / "__init__.py"
        self.entrypoint.parent.mkdir(parents=True)
        self.entrypoint.write_bytes(self.content)
        self.assert_snapshot_changed()

    def test_detects_interpreter_change(self):
        self.save_manifest()
        with patch.object(
            stack.sys, "executable", str(self.directory / "other-python")
        ):
            self.assert_snapshot_changed()

    def test_detects_environment_change_with_same_interpreter(self):
        self.save_manifest()
        with patch.object(stack.sys, "prefix", str(self.directory / "other-venv")):
            self.assert_snapshot_changed()

    def test_detects_metadata_owner_change_with_identical_bytes(self):
        self.save_manifest()
        self.packages = [self.owner("triton", self.content)]
        self.assert_snapshot_changed()


class EmptyWheelTests(unittest.TestCase):
    def setUp(self):
        self.distribution = distribution("vllm", "0.32.0+empty")
        self.distribution_patch = patch.object(
            stack.importlib.metadata,
            "distribution",
            side_effect=lambda name: self.distribution,
        )
        self.version_patch = patch.object(
            stack.importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError("vllm-ascend"),
        )
        self.spec_patch = patch.object(
            stack.importlib.util, "find_spec", return_value=None
        )
        self.distribution_patch.start()
        self.version_patch.start()
        self.spec_patch.start()
        self.addCleanup(self.distribution_patch.stop)
        self.addCleanup(self.version_patch.stop)
        self.addCleanup(self.spec_patch.stop)

    def test_accepts_ordinary_empty_wheel(self):
        stack.verify_empty_wheel("0.32.0")

    def test_rejects_wrong_version_and_nonempty_build(self):
        for version in ("0.28.0+empty", "0.32.0", "0.32.0+flagos"):
            with self.subTest(version=version):
                self.distribution = distribution("vllm", version)
                with self.assertRaisesRegex(
                    RuntimeError, "Expected the vLLM empty wheel"
                ):
                    stack.verify_empty_wheel("0.32.0")

    def test_rejects_editable_install(self):
        self.distribution = distribution(
            "vllm", "0.32.0+empty", direct_url={"dir_info": {"editable": True}}
        )
        with self.assertRaisesRegex(RuntimeError, "ordinary wheel"):
            stack.verify_empty_wheel("0.32.0")

    def test_rejects_cuda_native_extension(self):
        self.distribution = distribution(
            "vllm", "0.32.0+empty", files=["vllm/_C_stable_libtorch.abi3.so"]
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA extensions"):
            stack.verify_empty_wheel("0.32.0")

    def test_rejects_installed_vllm_ascend(self):
        with (
            patch.object(stack.importlib.metadata, "version", return_value="0.24.0"),
            self.assertRaisesRegex(RuntimeError, "Remove vllm-ascend"),
        ):
            stack.verify_empty_wheel("0.32.0")

    def test_rejects_residual_vllm_ascend_module(self):
        with (
            patch.object(stack.importlib.util, "find_spec", return_value=object()),
            self.assertRaisesRegex(RuntimeError, "residual vllm_ascend"),
        ):
            stack.verify_empty_wheel("0.32.0")


if __name__ == "__main__":
    unittest.main()
