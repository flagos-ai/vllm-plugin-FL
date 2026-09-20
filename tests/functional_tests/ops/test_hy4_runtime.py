# SPDX-License-Identifier: Apache-2.0
"""First execution of HY4 portable providers against independent Torch math."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

from vllm_fl.patches import hy_v4_runtime as runtime

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not current_platform.is_cuda(), reason="HY4 currently supports NVIDIA only"
    ),
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("qk_width,v_width", [(256, 256), (192, 128)])
def test_portable_prefill_matches_torch_at_mla_dimensions(causal, qk_width, v_width):
    torch.manual_seed(23)
    from flag_gems import flash_attn_varlen_func

    cls = runtime._make_hy4_flaggems_mla_prefill_backend(flash_attn_varlen_func)
    backend = cls(2, qk_width**-0.5, 512, qk_width - 64, 64, v_width, None)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    backend.prepare_metadata(
        SimpleNamespace(
            query_start_loc=offsets,
            max_query_len=3,
            chunked_context=SimpleNamespace(cu_seq_lens=[offsets], max_seq_lens=[3]),
        )
    )
    q, k = [
        torch.randn(5, 2, qk_width, dtype=torch.bfloat16, device="cuda")
        for _ in range(2)
    ]
    v = torch.randn(5, 2, v_width, dtype=torch.bfloat16, device="cuda")
    expected, expected_lse = [], []
    for start, end in ((0, 3), (3, 5)):
        # CPU FP64 reference is independent of the FlagGems provider/dispatch.
        qr, kr, vr = [x[start:end].cpu().double().transpose(0, 1) for x in (q, k, v)]
        scores = (qr @ kr.transpose(-1, -2)) * backend.scale
        if causal:
            mask = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, -float("inf"))
        expected.append((scores.softmax(-1) @ vr).transpose(0, 1))
        expected_lse.append(scores.logsumexp(-1))
    if causal:
        out = torch.empty_like(v)
        actual, lse = backend.run_prefill_new_tokens(q, k, v, True, out=out)
        assert actual is out
    else:
        actual, lse = backend.run_prefill_context_chunk(0, q, k, v)
    torch.testing.assert_close(
        actual.cpu().double(), torch.cat(expected), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        lse.cpu().double(), torch.cat(expected_lse, dim=1), rtol=0.005, atol=0.005
    )


@pytest.mark.parametrize("ue8m0", [False, True])
def test_planned_query_quantizer_matches_torch(ue8m0):
    # FlagGems defaults to FP32 output before Hopper; that is not the HY4
    # FP8 query path exercised by this numerical reference.
    if not current_platform.has_device_capability(90):
        pytest.skip("HY4 FP8 query quantization is validated on Hopper or newer")

    import flag_gems

    quantizer = runtime._flaggems_query_quantizer(flag_gems.per_token_group_quant_fp8)
    torch.manual_seed(29)
    x = torch.randn(5, 256, dtype=torch.bfloat16, device="cuda")
    quantized, scales = quantizer(x, 128, use_ue8m0=ue8m0)
    assert quantized.dtype == torch.float8_e4m3fn
    xr = x.cpu().float().reshape(5, 2, 128)
    expected_scale = (
        xr.abs().amax(-1).clamp_min(1e-10) / torch.finfo(quantized.dtype).max
    )
    if ue8m0:
        expected_scale = torch.exp2(torch.ceil(torch.log2(expected_scale)))
    expected = (
        (xr / expected_scale[..., None]).to(quantized.dtype).float().reshape_as(x)
    )
    torch.testing.assert_close(scales.cpu(), expected_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(quantized.cpu().float(), expected, rtol=0, atol=0)
