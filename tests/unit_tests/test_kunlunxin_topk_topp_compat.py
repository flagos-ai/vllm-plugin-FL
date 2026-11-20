# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.topk_topp_compat import (
    apply_topk_topp_compat,
)
from vllm_fl.utils import get_flag_gems_whitelist_blacklist


@pytest.mark.parametrize(
    "device_type", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_v2_top_p_keeps_threshold_crossing_token(monkeypatch, backend, device_type):
    if device_type == "cuda" and backend != "kunlunxin":
        pytest.skip("requires a Kunlunxin accelerator")

    sampler_ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")
    v2 = pytest.importorskip("vllm.v1.worker.gpu.sample.sampler")
    # V2 holds the dispatcher imported before applying the platform patch.
    dispatcher = v2.apply_top_k_top_p
    monkeypatch.setattr(sampler_ops, "HAS_TRITON", True)
    monkeypatch.setattr(sampler_ops, "apply_top_k_top_p", dispatcher)
    monkeypatch.setattr(sampler_ops, "xpu_sampler_supported", lambda: True)
    monkeypatch.setattr(v2, "xpu_sampler_supported", lambda: True)
    monkeypatch.setattr(
        v2, "gumbel_sample", lambda logits, *args, **kwargs: logits.argmax(-1)
    )
    apply_topk_topp_compat()
    apply_topk_topp_compat()

    assert v2.apply_top_k_top_p is dispatcher
    assert not sampler_ops.xpu_sampler_supported()
    assert not v2.xpu_sampler_supported()

    base = torch.tensor(
        [[0.5, 0.3, 0.2], [0.22, 0.2, 0.58], [0.1, 0.77, 0.13]],
        device=device_type,
    ).log()
    top_k = torch.tensor([3, 2, 3], device=device_type, dtype=torch.int32)
    top_p = torch.tensor([0.7, 0.9, 1.0], device=device_type)
    slots = torch.arange(3, device=device_type)
    sampler = object.__new__(v2.Sampler)
    sampler.use_fp64_gumbel = False
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=torch.ones_like(top_p)),
        seeds=SimpleNamespace(gpu=slots),
    )
    sampled, filtered = sampler._sample_random(
        base.clone(), slots, np.arange(3), slots, top_k, top_p, False
    )

    keep = torch.tensor(
        [[True, True, False], [True, False, True], [True, True, True]],
        device=device_type,
    )
    torch.testing.assert_close(filtered, base.masked_fill(~keep, -torch.inf))
    assert sampled.tolist() == [0, 2, 1]


@pytest.mark.gpu
@pytest.mark.flaggems
def test_v2_full_vocab_top_k_top_p_matches_native_policy(monkeypatch, backend):
    if backend != "kunlunxin":
        pytest.skip("requires a Kunlunxin accelerator")
    flag_gems = pytest.importorskip("flag_gems")
    if flag_gems.vendor_name != "kunlunxin":
        pytest.skip("requires Kunlunxin FlagGems")
    sampler_ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")
    v2 = pytest.importorskip("vllm.v1.worker.gpu.sample.sampler")
    for name in (
        "VLLM_FL_CONFIG",
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        monkeypatch.delenv(name, raising=False)
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert whitelist is None
    assert blacklist

    # Exercise the same V2 filtering entry as runtime. Capture the logits
    # passed to Gumbel sampling so randomness cannot hide a mask regression.
    monkeypatch.setattr(sampler_ops, "HAS_TRITON", True)
    monkeypatch.setattr(sampler_ops, "xpu_sampler_supported", lambda: True)
    monkeypatch.setattr(v2, "xpu_sampler_supported", lambda: True)
    monkeypatch.setattr(
        v2, "gumbel_sample", lambda logits, *args, **kwargs: logits.argmax(-1)
    )
    apply_topk_topp_compat()
    batch_size, vocab_size = 4, 248320
    generator = torch.Generator().manual_seed(2030)
    logits = torch.randn((batch_size, vocab_size), generator=generator)
    logits[:, :3] = torch.tensor([13.0, 12.0, 11.0])
    logits = logits.cuda()
    top_k = torch.full((batch_size,), 50, dtype=torch.int32, device=logits.device)
    top_p = torch.full((batch_size,), 0.7, device=logits.device)
    slots = torch.arange(batch_size, device=logits.device)
    sampler = object.__new__(v2.Sampler)
    sampler.use_fp64_gumbel = False
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=torch.ones_like(top_p)),
        seeds=SimpleNamespace(gpu=slots),
    )
    expected = sampler_ops.apply_top_k_top_p_pytorch(logits.clone(), top_k, top_p)
    # Top-k leaves 50 values before the cumulative distribution; top-p must
    # reduce them further instead of leaving the entire top-k set after NaNs.
    assert torch.isfinite(expected).sum(dim=-1).tolist() == [2] * batch_size

    with flag_gems.use_gems(exclude=blacklist):
        sampled, filtered = sampler._sample_random(
            logits, slots, np.arange(batch_size), slots, top_k, top_p, False
        )
        torch.cuda.synchronize()

    torch.testing.assert_close(filtered, expected, rtol=0, atol=0)
    assert sampled.tolist() == [0] * batch_size
