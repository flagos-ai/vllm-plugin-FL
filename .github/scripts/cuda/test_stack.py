# Copyright 2026 FlagOS Contributors
"""CPU regressions for rejecting a full or shadowed vLLM in NVIDIA EMPTY CI."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cuda_ci_stack", HERE / "stack.py")
stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stack)


class EmptyProviderTests(unittest.TestCase):
    def setUp(self):
        self.origin = HERE / "site-packages/vllm/__init__.py"

    def validate(self, **overrides):
        args = {
            "module_version": "0.32.0",
            "distribution_version": "0.32.0+empty",
            "module_origin": self.origin,
            "distribution_origin": self.origin,
        }
        args.update(overrides)
        return stack.validate_vllm_provider(**args)

    def test_release_module_and_empty_distribution_are_accepted(self):
        result = self.validate()
        self.assertEqual(result["vllm_distribution_version"], "0.32.0+empty")

    def test_full_cuda_wheel_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "ordinary vLLM"):
            self.validate(distribution_version="0.32.0")

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

    def test_leftover_native_extensions_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "_C_stable_libtorch"):
            self.validate(native_libraries=["_C_stable_libtorch.abi3.so"])

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


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.provenance = {
            name: {**source, "wheel_sha256": "a" * 64}
            for name, source in stack.FL_SOURCES.items()
        }
        self.installed = {
            name: {"version": source["version"], "wheel_sha256": "a" * 64}
            for name, source in stack.FL_SOURCES.items()
        }

    def test_pinned_source_and_installed_wheel_are_accepted(self):
        stack.validate_fl_provenance(self.provenance, self.installed)

    def test_same_version_from_different_source_is_rejected(self):
        self.provenance["flagtree"]["commit"] = "b" * 40
        with self.assertRaisesRegex(RuntimeError, "source provenance"):
            stack.validate_fl_provenance(self.provenance, self.installed)

    def test_replaced_or_editable_installation_is_rejected(self):
        for replacement in ({"wheel_sha256": "b" * 64}, {"editable": True}):
            installed = copy.deepcopy(self.installed)
            installed["flag-gems"].update(replacement)
            with (
                self.subTest(replacement=replacement),
                self.assertRaisesRegex(RuntimeError, "not the wheel"),
            ):
                stack.validate_fl_provenance(self.provenance, installed)

    def test_release_wheel_cannot_impersonate_fixed_development_commit(self):
        self.installed["flagtree"]["version"] = "0.7.0"
        with self.assertRaisesRegex(RuntimeError, "not the wheel"):
            stack.validate_fl_provenance(self.provenance, self.installed)


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


if __name__ == "__main__":
    unittest.main()
