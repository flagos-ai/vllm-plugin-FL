import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("check.sh")


@pytest.fixture
def probe(tmp_path):
    smi = tmp_path / "ixsmi"
    smi.write_text(
        '#!/bin/bash\nif [[ $SMI_MODE == hang ]]; then sleep 10; fi\nexit "${SMI_STATUS:-0}"\n'
    )
    smi.chmod(0o755)
    (tmp_path / "python").symlink_to(sys.executable)
    (tmp_path / "torch.py").write_text("""import os,time
if os.environ.get("TORCH_MODE") == "import_hang": time.sleep(10)
class Cuda:
 def is_available(self): return os.environ.get("TORCH_MODE") != "unavailable"
 def device_count(self): return int(os.environ.get("DEVICE_COUNT", "8"))
 def synchronize(self):
  if os.environ.get("TORCH_MODE") == "sync_hang": time.sleep(10)
 def get_device_name(self,i): return "mock BI-V150"
 def mem_get_info(self,i): return 1024,2048
cuda=Cuda()
class Tensor:
 device="cuda:0"
 shape=(32,32)
def ones(shape,device):
 if os.environ.get("TORCH_MODE") == "allocation_fail": raise RuntimeError("allocation failed")
 return Tensor()
""")

    def run(**kwargs):
        env = dict(
            os.environ,
            PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
            PYTHONPATH=str(tmp_path),
            ILUVATAR_SMI_TIMEOUT_SECONDS="0.2",
            ILUVATAR_TORCH_TIMEOUT_SECONDS="0.5",
            **kwargs,
        )
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )

    return run


def test_success(probe):
    r = probe()
    assert r.returncode == 0, r.stdout
    assert "Tensor smoke: cuda:0 (32, 32)" in r.stdout


@pytest.mark.parametrize(
    "mode,marker",
    [("import_hang", "Importing torch"), ("sync_hang", "Synchronizing cuda:0")],
)
def test_torch_timeout(probe, mode, marker):
    r = probe(TORCH_MODE=mode)
    assert r.returncode == 124, r.stdout
    assert marker in r.stdout
    assert "torch smoke test timed out" in r.stdout


def test_smi_timeout(probe):
    r = probe(SMI_MODE="hang")
    assert r.returncode == 124, r.stdout
    assert "management probe ixsmi timed out" in r.stdout
    assert "Importing torch" not in r.stdout


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"TORCH_MODE": "unavailable"}, "accelerator is unavailable"),
        ({"DEVICE_COUNT": "2"}, "At least 4 Iluvatar devices are required"),
        ({"TORCH_MODE": "allocation_fail"}, "allocation failed"),
    ],
)
def test_torch_failure_is_not_masked(probe, env, expected):
    r = probe(**env)
    assert r.returncode != 0, r.stdout
    assert expected in r.stdout
    assert "torch smoke test failed" in r.stdout


def test_management_error_still_requires_torch_success(probe):
    r = probe(SMI_STATUS="1")
    assert r.returncode == 0, r.stdout
    assert "continuing with the required torch smoke test" in r.stdout
    assert "Tensor smoke:" in r.stdout
    r = probe(SMI_STATUS="1", TORCH_MODE="unavailable")
    assert r.returncode != 0, r.stdout
