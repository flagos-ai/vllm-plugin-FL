"""Plugin-free repro of the GCU inductor config-generator signature bug (#557).

Runs a torch-only persistent-reduction compile in a subprocess that never
imports vllm or vllm_fl, so the plugin is physically excluded from the
failing path. On the S60 stack the subprocess is expected to die with:

    TypeError: GCUTritonConfigGenerator.persistent_reduction_configs()
    takes from 2 to 4 positional arguments but 5 were given

The test asserts the *fixed* state (subprocess succeeds). While #557 is
open it xfails on GCU; an XPASS either means the vendor backend was fixed
(re-enabling the graph cases) or that none of the probes selected a
persistent-reduction kernel — check the #557 log before concluding.
"""

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

# Three persistent-reduction-prone probes; any of them compiling is enough.
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


@pytest.mark.skipif(
    not _gcu_available(), reason="GCU vendor-signature experiment (#557)"
)
@pytest.mark.xfail(
    strict=False,
    reason=(
        "#557: torch_gcu's GCUTritonConfigGenerator.persistent_reduction_configs "
        "signature predates this torch; kernel-module import fails before the "
        "plugin is ever loaded"
    ),
)
def test_gcu_plugin_free_persistent_reduction_compiles():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        "plugin-free torch.compile failed on GCU "
        "(expected while #557 is open):\n"
        f"{proc.stderr[-2000:]}"
    )
    assert "PLUGIN_FREE_COMPILE_OK" in proc.stdout
