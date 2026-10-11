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
        for row in rows:
            body = (HERE / "patch-inputs" / row["input_name"]).read_bytes()
            self.assertEqual(
                {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
                row["after"],
            )
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


if __name__ == "__main__":
    unittest.main()
