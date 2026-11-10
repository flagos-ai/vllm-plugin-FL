# Copyright 2026 FlagOS Contributors
"""CPU regressions for the pinned T-Head PPU stack checks."""

import importlib.util
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("thead_ci_stack", HERE / "stack.py")
stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stack)

CHECKED_PACKAGES = {"torch", "torchvision", "numpy", "vllm", "triton", "flagtree", "flag-gems"}


class VersionDictionaryTests(unittest.TestCase):
    def test_dictionary_covers_every_checked_package(self):
        self.assertEqual(set(stack.VERSIONS), CHECKED_PACKAGES)

    def test_every_entry_is_read_by_the_checks(self):
        source = (HERE / "stack.py").read_text()
        looped = "VERSIONS[name]" in source
        for name in CHECKED_PACKAGES:
            direct = 'VERSIONS["' + name + '"]' in source
            self.assertTrue(direct or (looped and '"' + name + '"' in source), name)

    def test_flag_packages_come_from_the_runtime_record(self):
        self.assertEqual(stack.VERSIONS["flag-gems"], stack.STACK["flaggems"]["version"])
        self.assertEqual(stack.VERSIONS["flagtree"], stack.STACK["flagtree"]["version"])

    def test_runtime_versions_match_the_reviewed_runtime(self):
        self.assertEqual(stack.VERSIONS["torch"], "2.10.0+v0.1.0.ppu2.1.1")
        self.assertEqual(stack.VERSIONS["vllm"], "0.32.0+flagos")
        self.assertEqual(stack.VERSIONS["triton"], "3.6.0")
        self.assertEqual(stack.VERSIONS["flag-gems"], "5.4.0")

    def test_no_version_is_compared_as_a_literal(self):
        source = (HERE / "stack.py").read_text()
        literals = re.findall(r'(?:==|!=)\s*"\d+\.\d+[^"]*"', source)
        self.assertEqual(literals, [])


class VllmProviderTests(unittest.TestCase):
    def call(self, module_version, distribution_version):
        return stack.validate_vllm_provider(
            module_version, distribution_version, HERE / "vllm", HERE / "vllm"
        )

    def test_runtime_pair_is_accepted(self):
        result = self.call("0.32.0", "0.32.0+flagos")
        self.assertEqual(result["vllm_distribution_version"], "0.32.0+flagos")

    def test_empty_build_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.call("0.28.0", "0.28.0+empty")

    def test_mismatched_distribution_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.call("0.32.0", "0.31.0+flagos")

    def test_module_and_distribution_must_share_an_origin(self):
        with self.assertRaises(RuntimeError):
            stack.validate_vllm_provider(
                "0.32.0", "0.32.0+flagos", HERE / "a/vllm", HERE / "b/vllm"
            )


class PublicVersionTests(unittest.TestCase):
    def test_local_suffix_is_stripped(self):
        self.assertEqual(stack.public_version("0.32.0+flagos"), "0.32.0")
        self.assertEqual(stack.public_version("2.10.0+v0.1.0.ppu2.1.1"), "2.10.0")

    def test_plain_version_is_unchanged(self):
        self.assertEqual(stack.public_version("1.26.4"), "1.26.4")


if __name__ == "__main__":
    unittest.main()
