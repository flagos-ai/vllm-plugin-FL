# Copyright (c) 2026 BAAI. All rights reserved.
"""The MetaX exclusion keeps strided AA interpolation on native ATen."""

import pytest
import torch
import torch.nn.functional as F

from tests.utils.device_utils import get_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_backend() != "metax" or not torch.cuda.is_available(),
        reason="MetaX GPU and MACA runtime required",
    ),
]


@pytest.mark.parametrize("layout", ["channels_last", "sliced"])
@torch.inference_mode()
def test_strided_bicubic_aa_exclusion_matches_cpu(layout):
    import flag_gems

    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert not whitelist
    assert "_upsample_bicubic2d_aa" in blacklist
    generator = torch.Generator().manual_seed(817)
    if layout == "channels_last":
        cpu = torch.randn(1, 64, 64, 768, generator=generator).to(torch.bfloat16)
        source = cpu.cuda().permute(0, 3, 1, 2).float()
    else:
        cpu = torch.randn(1, 32, 128, 128, generator=generator)
        source = cpu.cuda()[:, :, ::2, ::2]
    assert not source.is_contiguous()
    expected = F.interpolate(
        source.cpu(),
        size=(40, 40),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    )
    with flag_gems.use_gems(exclude=blacklist):
        actual = F.interpolate(
            source,
            size=(40, 40),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        )
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=1e-5)
