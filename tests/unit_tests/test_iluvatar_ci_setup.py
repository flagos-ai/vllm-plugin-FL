"""CI source-install contract; real EMPTY build is a separate hardware check."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/iluvatar"


def test_setup_installs_target_before_plugin():
    setup = (SCRIPTS / "setup.sh").read_text()
    assert setup.index("unset PYTHONPATH") < setup.index("install_vllm.sh")
    assert "echo 'PYTHONPATH=' >> \"${GITHUB_ENV}\"" in setup
    assert setup.index("install_vllm.sh") < setup.index(
        "pip install --no-build-isolation --no-deps -e ."
    )
    assert "vllm.__version__.split('+')[0] == '0.32.0'" in setup
    for name in ("setup.sh", "install_vllm.sh"):
        subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


def test_install_rejects_corrupt_archive_before_pip(tmp_path):
    archive = tmp_path / "corrupt.tar.gz"
    archive.write_bytes(b"not an upstream archive")
    env = dict(os.environ, VLLM_SOURCE_ARCHIVE=str(archive))
    result = subprocess.run(
        ["bash", str(SCRIPTS / "install_vllm.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "FAILED" in result.stdout
    assert "pip install" not in result.stdout


def test_installer_keeps_vendor_dependencies():
    helper = (SCRIPTS / "install_vllm.sh").read_text()
    assert "VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.32.0" in helper
    assert "pip install --no-build-isolation --no-deps" in helper
    assert "sha256sum -c -" in helper
    assert "refs/tags/v0.32.0.tar.gz" in helper
