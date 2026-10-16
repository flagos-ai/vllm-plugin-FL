#!/usr/bin/env python3
"""Portable profile selection and fail-closed tests; no target package imports."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "private_source_recipe", HERE / "prepare_private_source.py"
)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class FixedProfileContract(unittest.TestCase):
    def test_historical_default_and_latest_are_separate(self):
        historical, _ = recipe.load_profile(recipe.DEFAULT_PROFILE)
        latest, _ = recipe.load_profile("latest-bfc6")
        self.assertEqual(recipe.DEFAULT_PROFILE, "historical-cf174")
        self.assertEqual(
            historical["source_commit"], "cf174ab70ab2af934158c83385458fcc9c830712"
        )
        self.assertEqual(
            latest["source_commit"], "bfc6c49d2c4d06e5deac9784f203730c8372a84a"
        )
        self.assertEqual(
            latest["four_production_replacements"],
            historical["four_production_replacements"],
        )
        self.assertNotEqual(latest["archive"], historical["archive"])

    def test_ndsafe512_is_opt_in_with_six_whole_bound_postimages(self):
        latest, _ = recipe.load_profile("latest-bfc6")
        safe, _ = recipe.load_profile("latest-bfc6-ndsafe512")
        self.assertEqual(recipe.DEFAULT_PROFILE, "historical-cf174")
        self.assertEqual(safe["archive"], latest["archive"])
        self.assertEqual(
            safe["six_production_replacements"][:4],
            latest["four_production_replacements"],
        )
        self.assertEqual(safe["versions"]["master"], latest["versions"]["master"])
        self.assertEqual(
            safe["versions"]["corrected"], "5.4.0+archive.gbfc6c49d2c4d.pr6894.musafix2"
        )
        rows = safe["six_production_replacements"]
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["source_path"] for row in rows}), 6)
        files, proofs = recipe.load_patches("latest-bfc6-ndsafe512")
        self.assertEqual(set(files), {row["source_path"] for row in rows})
        self.assertEqual(proofs, recipe.PATCH_FILES)
        self.assertFalse(any(HERE.glob("patch-inputs/*.py.raw")))
        self.assertFalse(safe["compiler_root_cause_fixed"])
        self.assertFalse(safe["actual_CPU_ordinary_repack_and_install"]["GPU_executed"])

    def test_ndsafe512_manifest_tamper_is_rejected(self):
        name = "source-patch-manifest-latest-bfc6-ndsafe512.json"
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / name).write_bytes((HERE / name).read_bytes() + b" ")
            with (
                patch.object(recipe, "BUNDLE", Path(folder)),
                self.assertRaisesRegex(ValueError, "Fixed recipe manifest mismatch"),
            ):
                recipe.load_profile("latest-bfc6-ndsafe512")

    def test_unknown_profile_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-source"
            with self.assertRaisesRegex(ValueError, "Unknown fixed recipe profile"):
                recipe.prepare(
                    Path(folder) / "missing.tar.gz", output, "corrected", "unreviewed"
                )
            self.assertFalse(output.exists())

    def test_manifest_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "source-patch-manifest-latest-bfc6.json"
            manifest.write_bytes((HERE / manifest.name).read_bytes() + b" ")
            with (
                patch.object(recipe, "BUNDLE", Path(folder)),
                self.assertRaisesRegex(ValueError, "Fixed recipe manifest mismatch"),
            ):
                recipe.load_profile("latest-bfc6")

    def test_wrong_archive_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "wrong.tar.gz"
            archive.write_bytes(b"different archive")
            output = Path(folder) / "new-source"
            with self.assertRaisesRegex(ValueError, "Whole-file pin mismatch"):
                recipe.prepare(archive, output, "corrected", "latest-bfc6")
            self.assertFalse(output.exists())


class ExactPatchContract(unittest.TestCase):
    BODY = b"before\nold\nafter\n"
    PATCH = (
        b"--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n before\n-old\n+new\n after\n"
    )
    AFTER = {
        "bytes": 17,
        "sha256": hashlib.sha256(b"before\nnew\nafter\n").hexdigest(),
    }

    def apply(self, patch_body=None, body=None, expected=None):
        hunks = recipe.parse_patch(self.PATCH if patch_body is None else patch_body)
        return recipe.apply_exact_patch(
            self.BODY if body is None else body,
            hunks["file.py"],
            self.AFTER if expected is None else expected,
        )

    def test_exact_patch_preserves_bytes(self):
        self.assertEqual(self.apply(), b"before\nnew\nafter\n")

    def test_context_is_not_fuzzed(self):
        with self.assertRaisesRegex(ValueError, "Exact patch context"):
            self.apply(body=b"different\nold\nafter\n")

    def test_old_position_is_not_searched(self):
        with self.assertRaisesRegex(ValueError, "Exact patch context"):
            self.apply(body=b"prefix\n" + self.BODY)

    def test_new_position_is_exact(self):
        with self.assertRaisesRegex(ValueError, "Exact patch new position"):
            self.apply(patch_body=self.PATCH.replace(b"+1,3", b"+2,3"))

    def test_hunk_count_and_duplicate_file_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Exact patch hunk line counts"):
            self.apply(patch_body=self.PATCH.replace(b"-1,3", b"-1,4"))
        with self.assertRaisesRegex(ValueError, "Exact unique patch paths"):
            recipe.parse_patch(self.PATCH + self.PATCH)

    def test_wrong_postimage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Whole-file patch postimage mismatch"):
            self.apply(expected={"bytes": 17, "sha256": "0" * 64})

    def test_fixed_patch_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "production.patch").write_bytes(
                (HERE / "production.patch").read_bytes() + b" "
            )
            with (
                patch.object(recipe, "BUNDLE", Path(folder)),
                self.assertRaisesRegex(ValueError, "Whole-file pin mismatch"),
            ):
                recipe.load_patches("latest-bfc6")


if __name__ == "__main__":
    unittest.main()
