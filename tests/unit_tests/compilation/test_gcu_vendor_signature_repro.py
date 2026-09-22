"""Plugin-free GCU inductor repros (#557): evidence + workaround candidate.

Both probes run torch-only subprocesses (no vllm/vllm_fl import), so the
plugin is structurally excluded from whatever happens.

test_gcu_default_compile_hits_vendor_signature_error documents the bug:
while #557 is open the default compile dies with the vendor
GCUTritonConfigGenerator.persistent_reduction_configs() TypeError. The
assertion is inverted on purpose — it PASSES while the bug reproduces,
and fails (alerting us) once the vendor backend is fixed.

test_gcu_persistent_reductions_disabled_compiles probes the workaround
candidate: TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0 makes the scheduler
avoid persistent-reduction kernels (torch/_inductor/config.py reads it,
ir.py should_use_persistent_reduction consults it), so the crashing
persistent_reduction() factory is never called. If this probe passes,
graph-mode cases can be re-enabled with that knob set for GCU.
"""

import os
import subprocess
import sys

import pytest
import torch

# Deliberately torch-only: importing vllm/vllm_fl here would invalidate the
# experiment (the plugin must not be in the failing process).
_SCRIPT = """
import torch
import torch.nn.functional as F

x = torch.randn(1024, 64, device="gcu")

f1 = torch.compile(lambda t: F.rms_norm(t * 2.0, [t.shape[-1]]))
f1(x)
f2 = torch.compile(lambda t: (t * 2.0).sum(-1))
f2(x)
f3 = torch.compile(lambda t: (t.float() * 1.5).mean(-1))
f3(x)

torch.gcu.synchronize()
print("PLUGIN_FREE_COMPILE_OK")
"""


def _gcu_available() -> bool:
    return hasattr(torch, "gcu") and torch.gcu.is_available()


def _run_probe(extra_env: dict[str, str] | None = None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


@pytest.mark.skipif(
    not _gcu_available(), reason="GCU vendor-signature experiment (#557)"
)
def test_gcu_default_compile_hits_vendor_signature_error():
    proc = _run_probe()
    assert proc.returncode != 0, (
        "default plugin-free compile SUCCEEDED on GCU — #557 looks fixed "
        "or this probe no longer selects a persistent-reduction kernel; "
        f"stdout={proc.stdout[-500:]}"
    )
    assert "GCUTritonConfigGenerator.persistent_reduction_configs" in (proc.stderr), (
        "plugin-free compile died of something other than the known #557 "
        f"vendor error — investigate:\n{proc.stderr[-2000:]}"
    )


@pytest.mark.skipif(not _gcu_available(), reason="GCU workaround probe (#557)")
def test_gcu_persistent_reductions_disabled_compiles():
    proc = _run_probe({"TORCHINDUCTOR_PERSISTENT_REDUCTIONS": "0"})
    assert proc.returncode == 0, (
        "workaround candidate TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0 did "
        f"not unblock the plugin-free compile:\n{proc.stderr[-2000:]}"
    )
    assert "PLUGIN_FREE_COMPILE_OK" in proc.stdout
