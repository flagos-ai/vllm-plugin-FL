# SPDX-License-Identifier: Apache-2.0
"""Keep Enflame CI supplied with the test runner's xdist dependency."""

import shlex
import subprocess
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def test_existing_image_setup_installs_xdist_without_runtime_extras():
    setup = ROOT / ".github/scripts/enflame/setup.sh"
    subprocess.run(["bash", "-n", str(setup)], check=True)
    commands = [
        shlex.split(line)
        for line in setup.read_text().splitlines()
        if line.startswith("python -m pip install")
    ]
    dependency_installs = [cmd for cmd in commands if "-e" not in cmd]
    assert len(dependency_installs) == 1
    install = dependency_installs[0]
    assert install[4:6] == ["--index-url", "https://mirrors.aliyun.com/pypi/simple"]
    requirements = [Requirement(arg) for arg in install[6:]]
    assert [r.name for r in requirements] == ["pytest-xdist"]
    assert "3.8.0" in requirements[0].specifier
    assert "4.0.0" not in requirements[0].specifier
    assert commands[-1] == [
        "python",
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "-e",
        ".",
    ]


def test_future_enflame_images_include_xdist():
    dockerfile = (ROOT / "docker/enflame/Dockerfile").read_text()
    dev = dockerfile.split("FROM base AS dev", 1)[1].split("FROM dev AS ci", 1)[0]
    assert '"pytest-xdist>=3.0.0,<4"' in dev
