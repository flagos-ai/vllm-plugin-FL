"""Real FlagGems activation contract under fullgraph compilation."""

import pytest
import torch


@pytest.mark.parametrize("shape", [(2, 256), (1, 512), (3, 4, 256)])
def test_iluvatar_silu_custom_op_contract(shape):
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        pytest.skip("requires Iluvatar FlagGems device")
    from flaggems_vllm import silu_and_mul

    from vllm_fl.dispatch.backends.flaggems.impl.activation import silu_and_mul_flaggems
    from vllm_fl.dispatch.backends.vendor.iluvatar.activation import (
        _iluvatar_gems_silu_and_mul,
    )

    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    d = x.shape[-1] // 2
    expected = silu_and_mul(x[..., :d], x[..., d:])
    compiled = torch.compile(
        lambda value: silu_and_mul_flaggems(None, value),
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)
    torch.library.opcheck(_iluvatar_gems_silu_and_mul, (x,))


def test_iluvatar_silu_cold_fullgraph_import():
    """First invocation may occur inside Dynamo, before the op is registered."""
    import subprocess
    import sys

    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        pytest.skip("requires Iluvatar FlagGems device")
    code = """
import sys
import torch
from vllm_fl.dispatch.backends.flaggems.impl.activation import silu_and_mul_flaggems
assert "vllm_fl.dispatch.backends.vendor.iluvatar.activation" not in sys.modules
x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
compiled = torch.compile(lambda x: silu_and_mul_flaggems(None, x),
                         backend="eager", fullgraph=True, dynamic=True)
expected = torch.nn.functional.silu(x[..., :128]) * x[..., 128:]
torch.testing.assert_close(compiled(x), expected, rtol=0.02, atol=0.02)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
