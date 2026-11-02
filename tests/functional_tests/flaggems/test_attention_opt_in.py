# Copyright 2026 FlagOS Contributors

import os
import subprocess
import sys

import pytest


@pytest.mark.gpu
@pytest.mark.flaggems
def test_metax_attention_opt_in(request):
    from vllm.platforms import current_platform

    if request.config.getoption("--platform") not in ("", "metax") or getattr(
        current_platform, "vendor_name", None
    ) not in ("metax", "maca"):
        pytest.skip("This integration check requires actual MetaX hardware")

    # Imports bind the package at startup. Isolate the opt-in environment from
    # the default MetaX functional job and from any previously imported module.
    code = r"""
from types import SimpleNamespace
import torch
from vllm.v1.attention.backend import AttentionType
from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
from vllm_fl.dispatch.backends.flaggems.impl import attention

assert FlagGemsBackend().attention_backend().endswith("AttentionFLBackend")
assert attention.flash_attn_varlen_func.__module__.startswith("flaggems_vllm.")
torch.manual_seed(17)
q = torch.randn(16, 4, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(16, 2, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
cu = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
slots = torch.arange(16, device="cuda", dtype=torch.int64)
layer = SimpleNamespace(**{
    f"_{name}_scale": torch.tensor(1.0, device="cuda") for name in ("q", "k", "v")
})
metadata = attention.AttentionFLMetadata(
    num_actual_tokens=16, max_query_len=16, query_start_loc=cu,
    max_seq_len=16, seq_lens=cu[1:],
    block_table=torch.tensor([[0]], device="cuda", dtype=torch.int32),
    slot_mapping=slots, use_cascade=False, common_prefix_len=0,
    cu_prefix_query_lens=None, prefix_kv_lens=None, suffix_kv_lens=None,
)
for kind in (AttentionType.ENCODER_ONLY, AttentionType.DECODER):
    impl = attention.AttentionFLImpl(4, 64, 64**-0.5, 2, None, None, "auto", attn_type=kind)
    impl.dcp_world_size = 1
    cache = torch.empty(2, 1, 16, 2, 64, device="cuda", dtype=q.dtype)
    impl.do_kv_cache_update(layer, k, v, cache, slots)
    metadata.causal = kind == AttentionType.DECODER
    actual = impl.forward(layer, q, k, v, cache, metadata, output=torch.empty_like(q))
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0).float(),
        k.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0).float(),
        v.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0).float(),
        is_causal=metadata.causal,
    ).squeeze(0).transpose(0, 1).to(q.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
print("MetaX fused opt-in encoder and paged decoder: PASS")
"""
    env = dict(
        os.environ,
        VLLM_PLUGINS="fl",
        VLLM_FL_USE_FLAGGEMS_VLLM="1",
        VLLM_FL_USE_FLAGGEMS_ATTN="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        # The functional session exit hook bypasses pytest's failure summary.
        # Emit the child traceback now so CI preserves the opt-in failure.
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
