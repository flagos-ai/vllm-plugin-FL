# Copyright 2026 FlagOS Contributors
"""CPU regressions for the native FlagOS vLLM CUDA installation."""

import base64
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cuda_ci_stack", HERE / "stack.py")
stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stack)


class NativeProviderTests(unittest.TestCase):
    def setUp(self):
        self.origin = HERE / "site-packages/vllm/__init__.py"

    def validate(self, **overrides):
        args = {
            "module_version": "0.32.0",
            "distribution_version": "0.32.0+flagos",
            "module_origin": self.origin,
            "distribution_origin": self.origin,
            "native_libraries": tuple(stack.NATIVE_VLLM_LIBRARIES.values()),
        }
        args.update(overrides)
        return stack.validate_vllm_provider(**args)

    def test_flagos_native_release_pair_is_accepted(self):
        result = self.validate()
        self.assertEqual(result["vllm_distribution_version"], "0.32.0+flagos")

    def test_empty_or_other_cuda_distribution_is_rejected(self):
        for version in ("0.32.0+empty", "0.32.0", "0.32.0+cu129"):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                self.validate(distribution_version=version)

    def test_old_and_dev_releases_are_rejected(self):
        for version in ("0.28.0", "0.32.0.dev1", "0.32.0+empty"):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                self.validate(module_version=version)

    def test_source_checkout_cannot_shadow_installed_wheel(self):
        with self.assertRaisesRegex(RuntimeError, "shadowing"):
            self.validate(module_origin=HERE / "source/vllm/__init__.py")

    def test_missing_module_origin_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.validate(module_origin=None)

    def test_editable_vllm_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "editable=True"):
            self.validate(editable=True)

    def test_every_required_native_extension_must_exist(self):
        for omitted in stack.NATIVE_VLLM_LIBRARIES.values():
            with self.subTest(omitted=omitted), self.assertRaises(RuntimeError):
                self.validate(
                    native_libraries=[
                        path
                        for path in stack.NATIVE_VLLM_LIBRARIES.values()
                        if path != omitted
                    ]
                )

    def test_required_payload_contract_includes_flagos_additions(self):
        extra = (
            "_deepselect_C.abi3.so",
            "_flashkda_C.abi3.so",
            "_qutlass_C.abi3.so",
            "fs_io_C.abi3.so",
            "spinloop.abi3.so",
            "third_party/deep_gemm/_C_extension.abi3.so",
        )
        self.assertTrue(set(extra).issubset(stack.NATIVE_VLLM_LIBRARIES.values()))

    def test_resolved_same_origin_is_accepted(self):
        self.validate(distribution_origin=self.origin.parent / "../vllm/__init__.py")

    def test_native_scan_finds_nested_and_unrecorded_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "_C_stable_libtorch.abi3.so",
                "vllm_flash_attn/_vllm_fa2_C.abi3.so",
                "libs/libcuda_payload.so.1",
                "_rust_test.abi3.so",
                "_custom_ops.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            found = stack.native_vllm_libraries(root)
            self.assertEqual(len(found), 3)
            self.assertTrue(any("_vllm_fa2_C" in path for path in found))
            self.assertTrue(any("_C_stable_libtorch" in path for path in found))


class NativePayloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records = []
        for relative in stack.NATIVE_VLLM_LIBRARIES.values():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = b"\x7fELF" + relative.encode()
            path.write_bytes(payload)
            entry = importlib.metadata.PackagePath(f"vllm/{relative}")
            digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            entry.hash = SimpleNamespace(
                mode="sha256", value=digest.rstrip(b"=").decode()
            )
            entry.size = len(payload)
            self.records.append(entry)
        self.distribution = SimpleNamespace(files=self.records)

    def test_all_recorded_native_payloads_are_accepted(self):
        origins = stack.validate_native_payload(self.distribution, self.root)
        self.assertEqual(len(origins), len(stack.NATIVE_VLLM_MODULES))
        self.assertIn("vllm._flashmla_C", origins)

    def test_unrecorded_extension_is_rejected(self):
        self.records.pop()
        with self.assertRaisesRegex(RuntimeError, "no wheel hash"):
            stack.validate_native_payload(self.distribution, self.root)

    def test_missing_hash_is_rejected(self):
        self.records[0].hash = None
        with self.assertRaisesRegex(RuntimeError, "no wheel hash"):
            stack.validate_native_payload(self.distribution, self.root)

    def test_modified_native_library_is_rejected(self):
        path = self.root / next(iter(stack.NATIVE_VLLM_LIBRARIES.values()))
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(RuntimeError, "differs from its wheel"):
            stack.validate_native_payload(self.distribution, self.root)

    def test_non_elf_placeholder_is_rejected(self):
        path = self.root / next(iter(stack.NATIVE_VLLM_LIBRARIES.values()))
        path.write_bytes(b"fake extension")
        with self.assertRaisesRegex(RuntimeError, "not ELF"):
            stack.validate_native_payload(self.distribution, self.root)


class NativeImportTests(unittest.TestCase):
    def setUp(self):
        self.name = "vllm._C_stable_libtorch"
        self.origin = str(HERE / "site-packages/vllm/_C_stable_libtorch.abi3.so")
        loader = importlib.machinery.ExtensionFileLoader(self.name, self.origin)
        self.module = SimpleNamespace(
            __file__=self.origin,
            __spec__=importlib.machinery.ModuleSpec(
                self.name, loader, origin=self.origin
            ),
        )

    def test_native_loader_and_origin_are_accepted(self):
        stack.validate_native_import(self.name, self.module, self.origin)

    def test_placeholder_module_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "genuine native extension"):
            stack.validate_native_import(self.name, SimpleNamespace(), self.origin)

    def test_python_shim_is_rejected_even_at_expected_path(self):
        self.module.__spec__.loader = importlib.machinery.SourceFileLoader(
            self.name, self.origin
        )
        with self.assertRaises(RuntimeError):
            stack.validate_native_import(self.name, self.module, self.origin)

    def test_shadowing_extension_is_rejected(self):
        self.module.__file__ = str(HERE / "source/_C_stable_libtorch.abi3.so")
        with self.assertRaises(RuntimeError):
            stack.validate_native_import(self.name, self.module, self.origin)


class RuntimeVersionTests(unittest.TestCase):
    def test_nccl_loaded_library_must_match_installed_cu13_metadata(self):
        stack.validate_nccl_version((2, 29, 7), "2.29.7")
        stack.validate_nccl_version((2, 30, 1), "2.30.1")
        with self.assertRaisesRegex(RuntimeError, "NCCL"):
            stack.validate_nccl_version((2, 27, 5), "2.29.7")

    def test_runtime_versions_report_updates_without_rejecting_them(self):
        versions = dict.fromkeys(stack.RUNTIME_PACKAGES, "a-new-version")
        self.assertEqual(stack.runtime_versions(versions), versions)

    def test_missing_runtime_package_is_not_silently_ignored(self):
        with self.assertRaises(KeyError):
            stack.runtime_versions({})

    def test_flagtree_compiler_updates_are_accepted_at_the_correct_origin(self):
        origin = Path("/opt/flagtree/triton/__init__.py")
        for version in ("3.6.0", "3.7.1"):
            stack.validate_compiler(version, origin, origin)
        with self.assertRaises(RuntimeError):
            stack.validate_compiler("3.7.1", Path("/other/triton/__init__.py"), origin)


class WheelInstallationTests(unittest.TestCase):
    def distribution(self, wheel="Wheel-Version: 1.0", editable=False):
        texts = {
            "WHEEL": wheel,
            "direct_url.json": '{"dir_info": {"editable": true}}' if editable else None,
        }
        return SimpleNamespace(read_text=texts.get)

    def test_normal_wheel_is_accepted(self):
        stack.validate_wheel_installation(self.distribution())

    def test_missing_wheel_metadata_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "wheel"):
            stack.validate_wheel_installation(self.distribution(wheel=None))

    def test_editable_installation_is_rejected_even_with_wheel_metadata(self):
        with self.assertRaisesRegex(RuntimeError, "Editable"):
            stack.validate_wheel_installation(self.distribution(editable=True))

    def test_build_check_rejects_source_and_editable_before_scanning_payload(self):
        for dist in (self.distribution(wheel=None), self.distribution(editable=True)):
            with (
                self.subTest(dist=dist),
                patch.object(stack.metadata, "distribution", return_value=dist),
                patch.object(stack, "validate_native_payload") as scan,
            ):
                with self.assertRaises(RuntimeError):
                    stack.check_build_payload()
                scan.assert_not_called()


class FlagGemsVllmProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "url": stack.FLAGGEMS_VLLM_SOURCE["repository"],
            "vcs_info": {
                "vcs": "git",
                "commit_id": stack.FLAGGEMS_VLLM_SOURCE["commit"],
            },
        }

    def test_exact_commit_is_accepted_despite_old_metadata_version(self):
        facts = stack.validate_flaggems_vllm_provenance("0.1.0", self.source)
        self.assertEqual(facts["release"], "v0.2.0")
        self.assertEqual(facts["version"], "0.1.0")
        self.source["url"] = self.source["url"].removesuffix(".git")
        stack.validate_flaggems_vllm_provenance("0.1.0", self.source)

    def test_metadata_without_git_provenance_is_rejected(self):
        for source in ({}, {"url": "file:///tmp/flaggems_vllm-0.1.0.whl"}):
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                stack.validate_flaggems_vllm_provenance("0.1.0", source)

    def test_wrong_commit_repository_or_vcs_is_rejected(self):
        for replacement in (
            {"url": "https://example.com/FlagGems-vllm.git"},
            {"vcs_info": {"vcs": "git", "commit_id": "b" * 40}},
            {
                "vcs_info": {
                    "vcs": "hg",
                    "commit_id": stack.FLAGGEMS_VLLM_SOURCE["commit"],
                }
            },
        ):
            source = {**self.source, **replacement}
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                stack.validate_flaggems_vllm_provenance("0.1.0", source)

    def test_editable_installation_is_rejected(self):
        self.source["dir_info"] = {"editable": True}
        with self.assertRaises(RuntimeError):
            stack.validate_flaggems_vllm_provenance("0.1.0", self.source)

    def test_checker_and_shared_installer_use_the_same_pin(self):
        spec = importlib.util.spec_from_file_location(
            "ci_install_flaggems_vllm", HERE.parent / "install_flaggems_vllm.py"
        )
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        self.assertEqual(
            stack.FLAGGEMS_VLLM_SOURCE,
            {
                "release": installer.RELEASE,
                "commit": installer.COMMIT,
                "repository": installer.REPOSITORY,
            },
        )


class FunctionalShutdownTests(unittest.TestCase):
    def run_shutdown(self, backend, status):
        spec = importlib.util.spec_from_file_location(
            "functional_shutdown",
            HERE.parents[2] / "tests/functional_tests/conftest.py",
        )
        module = importlib.util.module_from_spec(spec)
        pytest_stub = SimpleNamespace(
            hookimpl=lambda **kwargs: lambda function: function
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "pytest": pytest_stub,
                    "tests.utils.device_utils": SimpleNamespace(
                        get_backend=lambda: backend
                    ),
                    "torch._inductor.compile_worker": SimpleNamespace(
                        subproc_pool=None
                    ),
                },
            ),
            patch("os._exit") as force_exit,
        ):
            spec.loader.exec_module(module)
            module.pytest_sessionfinish(None, status)
            return force_exit.call_args_list

    def test_nvidia_keeps_normal_pytest_worker_shutdown(self):
        self.assertEqual(self.run_shutdown("nvidia", 0), [])

    def test_ascend_workaround_preserves_failure_exit_code(self):
        from unittest.mock import call

        self.assertEqual(self.run_shutdown("ascend", 1), [call(1)])


if __name__ == "__main__":
    unittest.main()
