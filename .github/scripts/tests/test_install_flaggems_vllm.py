"""CPU-only regression tests: python -m unittest discover -s .github/scripts/tests."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "installer", Path(__file__).resolve().parents[1] / "install_flaggems_vllm.py"
)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def metadata(self, source):
        dist = Mock()
        dist.read_text.return_value = json.dumps(source)
        return patch.object(installer, "distribution", return_value=dist)

    def test_vcs_match_and_mismatch(self):
        for commit, expected in [(installer.COMMIT, True), ("0" * 40, False)]:
            with (
                self.subTest(commit=commit),
                self.metadata(
                    {
                        "url": installer.REPOSITORY,
                        "vcs_info": {"commit_id": commit},
                    }
                ),
            ):
                self.assertEqual(installer.release_is_installed(), expected)

    def test_matching_runtime_never_installs(self):
        with (
            patch.object(installer, "package_version", return_value="1"),
            patch.object(installer, "release_is_installed", return_value=True),
            patch.object(installer, "pip_install") as pip,
        ):
            installer.main()
            pip.assert_not_called()

    def test_mismatch_replaces_once_without_dependencies(self):
        with (
            patch.object(installer, "package_version", return_value="1"),
            patch.object(installer, "release_is_installed", side_effect=[False, True]),
            patch.object(installer, "pip_install") as pip,
        ):
            installer.main()
            self.assertEqual(pip.call_count, 1)
            self.assertIn("--force-reinstall", pip.call_args.args)
            self.assertIn("--no-deps", pip.call_args.args)

    def test_archive_hash(self):
        with self.metadata(
            {
                "url": "https://example.org/source.tar.gz",
                "archive_info": {"hashes": {"sha256": "abc"}},
            }
        ):
            self.assertTrue(
                installer.release_is_installed(
                    source_archive="https://example.org/source.tar.gz",
                    archive_sha256="abc",
                )
            )
            self.assertFalse(
                installer.release_is_installed(
                    source_archive="https://example.org/source.tar.gz",
                    archive_sha256="def",
                )
            )

    def test_unknown_metadata_is_not_reused(self):
        with self.metadata({}):
            self.assertFalse(installer.release_is_installed())

    def test_real_editable_git_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", tmp, *args], text=True
                ).strip()

            git("init", "-q")
            git("config", "user.email", "test@example.org")
            git("config", "user.name", "Test")
            git("remote", "add", "origin", installer.REPOSITORY)
            (path / "source.py").write_text("original\n")
            git("add", ".")
            git("commit", "-qm", "test fixture")
            commit = git("rev-parse", "HEAD")
            with self.metadata({"url": path.as_uri(), "dir_info": {"editable": True}}):
                self.assertTrue(installer.release_is_installed(commit))
                self.assertFalse(installer.release_is_installed("0" * 40))
                (path / "source.py").write_text("modified\n")
                self.assertFalse(installer.release_is_installed(commit))
            with self.metadata({"url": path.as_uri(), "dir_info": {}}):
                self.assertFalse(installer.release_is_installed(commit))

    def test_iluvatar_reuses_image_without_installer(self):
        setup = Path(__file__).resolve().parents[1] / "iluvatar/setup.sh"
        text = setup.read_text()
        self.assertNotIn("install_flaggems_vllm.py", text)
        line = next(line for line in text.splitlines() if line.startswith("python -c "))
        code = line.split("'", 1)[1].rsplit("'", 1)[0]
        with (
            patch("importlib.metadata.version", return_value="0.1.0"),
            patch("subprocess.run") as run,
        ):
            exec(code, {})
            run.assert_not_called()
        with (
            patch(
                "importlib.metadata.version", side_effect=installer.PackageNotFoundError
            ),
            patch("subprocess.run") as run,
        ):
            with self.assertRaises(installer.PackageNotFoundError):
                exec(code, {})
            run.assert_not_called()

    def test_xdist_reuse_and_repair(self):
        setup = Path(__file__).resolve().parents[1] / "iluvatar/setup.sh"
        code = setup.read_text().split("python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
        for installed, should_install in [
            ("3.8.0", False),
            ("2.5.0", True),
            (None, True),
        ]:
            with (
                self.subTest(installed=installed),
                patch("importlib.metadata.version", return_value=installed) as version,
                patch("subprocess.run") as run,
            ):
                if installed is None:
                    version.side_effect = installer.PackageNotFoundError
                exec(code, {})
                self.assertEqual(run.called, should_install)


if __name__ == "__main__":
    unittest.main()
