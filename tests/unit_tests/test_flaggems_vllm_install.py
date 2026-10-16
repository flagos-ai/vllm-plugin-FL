# Copyright 2026 FlagOS Contributors
"""Check CI source refresh and dependency isolation without installing packages."""

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/install_flaggems_vllm.py"
)


class TestFlagGemsVllmInstall(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("ci_fused_install", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.source = {
            "url": self.module.REPOSITORY,
            "vcs_info": {"requested_revision": "main", "commit_id": "a" * 40},
        }

    def test_refreshes_main_even_when_already_installed(self):
        with (
            patch.object(self.module, "package_version", return_value="1.0"),
            patch.object(self.module, "installed_source", return_value=self.source),
            patch.object(self.module, "pip_install") as install,
        ):
            self.module.main()
        install.assert_called_once_with(
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            "flaggems-vllm @ git+https://github.com/flagos-ai/FlagGems-vllm.git@main",
        )

    def test_installs_missing_build_dependency_without_reinstalling_runtime(self):
        with (
            patch.object(
                self.module,
                "package_version",
                side_effect=lambda name: None if name == "pybind11" else "1.0",
            ),
            patch.object(self.module, "installed_source", return_value=self.source),
            patch.object(self.module, "pip_install") as install,
        ):
            self.module.main()
        self.assertEqual(install.call_count, 2)
        self.assertEqual(install.call_args_list[0].args, ("pybind11",))
        self.assertIn("--no-deps", install.call_args_list[1].args)

    def test_rejects_wrong_repository_or_revision_after_install(self):
        invalid = [
            {},
            dict(self.source, url="https://github.com/another/repository.git"),
            dict(
                self.source, vcs_info={"requested_revision": "v0.2.0", "commit_id": "a"}
            ),
            dict(self.source, vcs_info={"requested_revision": "main"}),
        ]
        for source in invalid:
            with (
                self.subTest(source=source),
                patch.object(self.module, "package_version", return_value="1.0"),
                patch.object(self.module, "installed_source", return_value=source),
                patch.object(self.module, "pip_install") as install,
                self.assertRaisesRegex(RuntimeError, "must come from .*@main"),
            ):
                self.module.main()
            install.assert_called_once()

    def test_runtime_dependency_change_is_rejected(self):
        state = {"updated": False}

        def version(name):
            return "2.0" if name == "torch" and state["updated"] else "1.0"

        with (
            patch.object(self.module, "package_version", side_effect=version),
            patch.object(self.module, "installed_source", return_value=self.source),
            patch.object(
                self.module,
                "pip_install",
                side_effect=lambda *args: state.update(updated=True),
            ),
            self.assertRaisesRegex(RuntimeError, "runtime dependencies changed"),
        ):
            self.module.main()

    def test_install_failure_propagates_without_retry(self):
        failure = subprocess.CalledProcessError(1, ["pip", "install"])
        with (
            patch.object(self.module, "package_version", return_value="1.0"),
            patch.object(self.module, "pip_install", side_effect=failure) as install,
            patch.object(self.module, "installed_source") as source,
            self.assertRaises(subprocess.CalledProcessError) as error,
        ):
            self.module.main()
        self.assertIs(error.exception, failure)
        install.assert_called_once()
        source.assert_not_called()

    def test_source_metadata_handles_absent_and_invalid_values(self):
        for text in (None, "not JSON", "[]", "null"):
            with (
                self.subTest(text=text),
                patch.object(
                    self.module,
                    "distribution",
                    return_value=SimpleNamespace(
                        read_text=lambda name, value=text: value
                    ),
                ),
            ):
                self.assertEqual(self.module.installed_source(), {})
        with patch.object(
            self.module,
            "distribution",
            side_effect=self.module.PackageNotFoundError("flaggems-vllm"),
        ):
            self.assertEqual(self.module.installed_source(), {})
        with patch.object(
            self.module,
            "distribution",
            return_value=SimpleNamespace(
                read_text=lambda name: json.dumps(self.source)
            ),
        ):
            self.assertEqual(self.module.installed_source(), self.source)
