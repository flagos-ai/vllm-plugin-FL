# Copyright 2026 FlagOS Contributors
"""CPU regressions for the FlagOS Runtime 2.3.0 Ascend vLLM 0.32 stack checker."""

import base64
import hashlib
import importlib.metadata
import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "ascend_ci_stack", Path(__file__).with_name("stack.py")
)
stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stack)


class VllmProviderTests(unittest.TestCase):
    def setUp(self):
        self.origin = Path("/flagos/lib/python3.12/site-packages/vllm/__init__.py")

    def validate(self, **overrides):
        args = {
            "module_version": "0.32.0",
            "distribution_version": "0.32.0+flagos",
            "module_origin": self.origin,
            "distribution_origin": self.origin,
            "native_libraries": (),
        }
        args.update(overrides)
        return stack.validate_vllm_provider(**args)

    def test_flagos_pure_python_release_pair_is_accepted(self):
        result = self.validate()
        self.assertEqual(result["vllm_distribution_version"], "0.32.0+flagos")

    def test_old_or_other_vllm_distribution_is_rejected(self):
        for version in ("0.28.0+empty", "0.32.0+empty", "0.32.0", "0.32.0+cu129"):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                self.validate(distribution_version=version)

    def test_editable_or_shadowed_vllm_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.validate(editable=True)
        with self.assertRaises(RuntimeError):
            self.validate(module_origin=Path("/workspace/vllm/vllm/__init__.py"))

    def test_foreign_native_extensions_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.validate(native_libraries=("_C_stable_libtorch.abi3.so",))

    def test_resolved_same_origin_is_accepted(self):
        self.validate(distribution_origin=self.origin.parent / "../vllm/__init__.py")


class VllmRecordTests(unittest.TestCase):
    def test_wheel_record_hash_and_size_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "vllm"
            dist_info = root / "vllm-0.32.0+flagos.dist-info"
            pkg.mkdir()
            dist_info.mkdir()
            init_file = pkg / "__init__.py"
            payload = b"__version__ = '0.32.0'\n"
            init_file.write_bytes(payload)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
                .rstrip(b"=")
                .decode()
            )
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: vllm\nVersion: 0.32.0+flagos\n"
            )
            (dist_info / "RECORD").write_text(
                f"vllm/__init__.py,sha256={digest},{len(payload)}\n"
            )
            dist = importlib.metadata.PathDistribution(dist_info)
            result = stack.validate_vllm_record(dist, pkg)
            self.assertEqual(
                result["vllm_init_sha256"], hashlib.sha256(payload).hexdigest()
            )
            init_file.write_bytes(payload + b"# modified\n")
            with self.assertRaises(RuntimeError):
                stack.validate_vllm_record(dist, pkg)


class RuntimeAndCompilerTests(unittest.TestCase):
    def test_base_runtime_versions_are_accepted(self):
        versions = dict(stack.RUNTIME_VERSIONS)
        self.assertEqual(stack.validate_runtime_versions(versions), versions)

    def test_replaced_or_missing_base_package_is_rejected(self):
        for name in stack.RUNTIME_VERSIONS:
            versions = dict(stack.RUNTIME_VERSIONS)
            versions[name] = "0.0.0"
            with self.subTest(package=name), self.assertRaisesRegex(RuntimeError, name):
                stack.validate_runtime_versions(versions)

    def test_vendor_repack_versions_are_checked(self):
        versions = dict(stack.VENDOR_REPACK_VERSIONS)
        self.assertEqual(stack.validate_vendor_repack_versions(versions), versions)
        for name in stack.VENDOR_REPACK_VERSIONS:
            bad = dict(stack.VENDOR_REPACK_VERSIONS)
            bad[name] = "0.0.0"
            with self.subTest(package=name), self.assertRaisesRegex(RuntimeError, name):
                stack.validate_vendor_repack_versions(bad)

    def test_flagtree_compiler_origin_version_and_isolation_are_checked(self):
        origin = Path("/opt/flagtree/triton/__init__.py")
        self.assertEqual(
            stack.FLAGTREE_TRITON_VERSION,
            stack.FLAGTREE_TRITON_VERSIONS[stack.RUNTIME_VERSIONS["flagtree"]],
        )
        stack.validate_compiler(stack.FLAGTREE_TRITON_VERSION, origin, origin, {})
        with self.assertRaises(RuntimeError):
            stack.validate_compiler("3.6.0", origin, origin, {})
        with self.assertRaises(RuntimeError):
            stack.validate_compiler(
                stack.FLAGTREE_TRITON_VERSION,
                Path("/opt/triton/triton/__init__.py"),
                origin,
                {},
            )
        with self.assertRaises(RuntimeError):
            stack.validate_compiler(
                stack.FLAGTREE_TRITON_VERSION, origin, origin, {"triton": "3.6.0"}
            )


class FlagGemsVllmProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.direct_url = {
            "url": "https://github.com/flagos-ai/FlagGems-vllm.git",
            "vcs_info": {
                "vcs": "git",
                "commit_id": "b5c2c2249319277ccda4d41aa0f65164bd9b2a19",
            },
        }

    def test_pinned_v020_commit_is_accepted(self):
        result = stack.validate_flaggems_vllm_provenance("0.1.0", self.direct_url)
        self.assertEqual(result["release"], "v0.2.0")

    def test_wrong_commit_or_editable_install_is_rejected(self):
        bad_commit = {
            **self.direct_url,
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
        }
        with self.assertRaises(RuntimeError):
            stack.validate_flaggems_vllm_provenance("0.1.0", bad_commit)
        editable = {**self.direct_url, "dir_info": {"editable": True}}
        with self.assertRaises(RuntimeError):
            stack.validate_flaggems_vllm_provenance("0.1.0", editable)


if __name__ == "__main__":
    unittest.main()
