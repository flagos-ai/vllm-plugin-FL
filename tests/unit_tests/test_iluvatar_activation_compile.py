"""Real FlagGems activation contract under fullgraph compilation."""
import pytest
import torch


@pytest.mark.parametrize('shape', [(2, 256), (1, 512), (3, 4, 256)])
def test_iluvatar_silu_custom_op_contract(shape):
    from vllm.platforms import current_platform
    if getattr(current_platform, 'vendor_name', None) != 'iluvatar':
        pytest.skip('requires Iluvatar FlagGems device')
    from flaggems_vllm import silu_and_mul
    from vllm_fl.dispatch.backends.flaggems.impl.activation import (
        _iluvatar_gems_silu_and_mul, silu_and_mul_flaggems,
    )
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    d = x.shape[-1] // 2
    expected = silu_and_mul(x[..., :d], x[..., d:])
    compiled = torch.compile(lambda value: silu_and_mul_flaggems(None, value),
                             backend='eager', fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)
    torch.library.opcheck(_iluvatar_gems_silu_and_mul, (x,))
