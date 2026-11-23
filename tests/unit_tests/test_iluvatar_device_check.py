# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU-only subprocess checks for the Iluvatar CI health-check watchdog.

Run directly with Python; no vLLM, Torch, pytest or accelerator is required.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/iluvatar/check.sh"
BASH = shutil.which("bash")
if BASH is None and os.name == "nt":
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    if candidate.is_file():
        BASH = str(candidate)

FAKE_TORCH = """
import os
import time
mode = os.environ.get("PROBE_TEST_MODE", "success")
if mode == "import-hang":
    time.sleep(60)
class Cuda:
    def is_available(self):
        return mode != "unavailable"
    def device_count(self):
        return 4
    def synchronize(self):
        if mode == "sync-hang":
            time.sleep(60)
    def get_device_name(self, index):
        return "mock BI-V150"
    def mem_get_info(self, index):
        return 1024, 2048
cuda = Cuda()
class Tensor:
    device = "cuda:0"
    shape = (32, 32)
def ones(shape, device):
    if mode == "allocation-fail":
        raise RuntimeError("mock allocation failure")
    return Tensor()
"""


@unittest.skipUnless(BASH, "Bash is required for CI shell regression tests")
class IluvatarDeviceCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        (self.directory / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
        python = Path(sys.executable).as_posix()
        (self.bin / "python").write_text(
            f'#!/bin/bash\nexec "{python}" "$@"\n', encoding="utf-8"
        )
        self.ixsmi = self.bin / "ixsmi"
        self.ixsmi.write_text('#!/bin/bash\necho "mock ixsmi"\n', encoding="utf-8")
        for path in self.bin.iterdir():
            path.chmod(0o755)

    def run_probe(self, mode="success", duration="10"):
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(self.directory),
            PROBE_TEST_MODE=mode,
            ILUVATAR_DEVICE_CHECK_TIMEOUT_SECONDS=duration,
        )
        # Set PATH inside Bash: Windows PATH separators otherwise hide mock tools.
        command = (
            'mock_bin="$1"; '
            "if command -v cygpath >/dev/null 2>&1; then "
            'mock_bin=$(cygpath -u "$mock_bin"); fi; '
            'export PATH="$mock_bin:$PATH"; bash "$2"'
        )
        started = time.monotonic()
        result = subprocess.run(
            [BASH, "-c", command, "probe-test", self.bin.as_posix(), SCRIPT.as_posix()],
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )
        return result, time.monotonic() - started

    def test_success_logs_every_stage(self):
        result, _ = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for stage in (
            "management-tool",
            "torch-import",
            "cuda-availability",
            "device-count",
            "tensor-allocation",
            "cuda-synchronize",
            "device-name",
        ):
            self.assertIn(f"[device-check] {stage}: start", result.stdout)
            self.assertIn(f"[device-check] {stage}: complete", result.stdout)
        self.assertIn("Tensor smoke: cuda:0 (32, 32)", result.stdout)
        self.assertIn("[device-check] complete", result.stdout)

    def test_unavailable_device_remains_failure(self):
        result, _ = self.run_probe("unavailable")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Iluvatar accelerator is unavailable", result.stderr)
        self.assertIn("failed with exit code 1", result.stdout)
        self.assertNotIn("[device-check] complete\n", result.stdout)

    def test_allocation_failure_is_not_masked_by_diagnostics(self):
        result, _ = self.run_probe("allocation-fail")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("mock allocation failure", result.stderr)
        self.assertNotIn("tensor-allocation: complete", result.stdout)

    def assert_timeout(self, mode, stage):
        result, elapsed = self.run_probe(mode, "1")
        self.assertIn(result.returncode, (124, 137), result.stdout + result.stderr)
        self.assertLess(elapsed, 20)
        self.assertIn(f"[device-check] {stage}: start", result.stdout)
        self.assertNotIn(f"[device-check] {stage}: complete", result.stdout)
        self.assertIn("timed out after 1s", result.stdout)

    def test_torch_import_hang_is_bounded_and_start_log_is_flushed(self):
        self.assert_timeout("import-hang", "torch-import")

    def test_cuda_sync_hang_is_bounded_and_start_log_is_flushed(self):
        self.assert_timeout("sync-hang", "cuda-synchronize")

    def test_management_tool_hang_is_also_bounded(self):
        self.ixsmi.write_text("#!/bin/bash\nsleep 60\n", encoding="utf-8")
        self.ixsmi.chmod(0o755)
        self.assert_timeout("success", "management-tool")

    def test_management_tool_failure_remains_best_effort(self):
        self.ixsmi.write_text("#!/bin/bash\nexit 9\n", encoding="utf-8")
        self.ixsmi.chmod(0o755)
        result, _ = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Tensor smoke:", result.stdout)

    def test_invalid_timeout_fails_before_probe(self):
        result, _ = self.run_probe(duration="0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be a positive integer", result.stdout)
        self.assertNotIn("management-tool: start", result.stdout)


if __name__ == "__main__":
    unittest.main()
