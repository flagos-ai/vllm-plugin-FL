# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-GPU numerical coverage for the tilelang mHC broadcast backport.

The specialized kernel is a public symbol of
``vllm_fl.dispatch.backends.tilelang.impl.mhc_broadcast`` (exported via
``__all__``). A100 machines cannot import the adapter module
``tilelang/mhc_broadcast.py`` (it requires SM90 for the NVIDIA reference
path), so the backport module is imported directly and its JIT kernel is fed
gemm outputs computed in plain torch — exactly the decomposition the host
API performs through DeepGEMM's ``tf32_hc_prenorm_gemm``:

    out = x.float() @ fn_broadcast.T       # (n_splits, T, hc_mult3)
    sqrsum = x.float().square().sum(-1)    # (n_splits, T)

The host API's zero-token early return (which precedes any DeepGEMM call)
and its fp32/strided norm_weight conversion are exercised through
``mhc_pre_broadcast_tilelang`` directly.

Shapes are deliberately small (hidden 512, tokens <= 64, hc_mult 4) so the
tilelang JIT compile stays well inside the 30 s per-test budget.
"""

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _load_backport():
    from vllm_fl.dispatch.backends.tilelang.impl import mhc_broadcast

    return (
        mhc_broadcast.mhc_pre_big_fuse_broadcast_with_norm_tilelang,
        mhc_broadcast.mhc_pre_broadcast_tilelang,
    )


def _make_inputs(device, tokens, hidden=512, hc_mult=4, seed=20260827):
    torch.manual_seed(seed + tokens)
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * (2 + hc_mult)
    residual = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    fn = (torch.randn(hc_mult3, hc_mult * hidden, device=device) * 0.01).contiguous()
    fn_broadcast = fn.view(hc_mult3, hc_mult, hidden).sum(dim=1).contiguous()
    hc_scale = torch.tensor([0.8, 0.7, 0.6], device=device)
    hc_base = (torch.randn(hc_mult3, device=device) * 0.1).contiguous()
    norm_weight = (
        (1 + torch.randn(hidden, device=device) * 0.05).to(torch.bfloat16).contiguous()
    )
    return {
        "residual": residual,
        "fn": fn,
        "fn_broadcast": fn_broadcast,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "norm_weight": norm_weight,
        "hc_mult": hc_mult,
        "hidden": hidden,
        "hc_mult2": hc_mult2,
        "hc_mult3": hc_mult3,
    }


def _torch_gemm(values, n_splits=1):
    """Compute the gemm outputs the way tf32_hc_prenorm_gemm defines them."""
    residual = values["residual"].float()
    fn_broadcast = values["fn_broadcast"]
    out = residual @ fn_broadcast.T  # (T, hc_mult3)
    sqrsum = residual.square().sum(-1)  # (T,)
    return (
        out.unsqueeze(0).expand(n_splits, -1, -1).contiguous(),
        sqrsum.unsqueeze(0).expand(n_splits, -1).contiguous(),
    )


def _reference(
    values,
    gemm_out_mul,
    gemm_out_sqrsum,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_eps,
):
    """Independent mirror of the fused kernel's math, all in fp32 on device."""
    residual = values["residual"]
    hc_mult = values["hc_mult"]
    hidden = values["hidden"]
    tokens = residual.shape[0]
    scale, base = values["hc_scale"], values["hc_base"]
    norm_weight = values["norm_weight"].float()

    # rms across splits then mixed coefficients.
    rms = torch.rsqrt(gemm_out_sqrsum.sum(0) / hidden + rms_eps)
    mixes = gemm_out_mul.sum(0) * rms.unsqueeze(-1)  # (T, hc_mult3)

    # post_mix: sigmoid on the scale/base slice [hc_mult : 2*hc_mult).
    post_mix = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )  # (T, hc_mult)

    # comb_mix: sinkhorn-normalized mixing matrix from the last hc_mult2 rows.
    cm = mixes[:, 2 * hc_mult :].view(tokens, hc_mult, hc_mult) * scale[2] + base[
        2 * hc_mult :
    ].view(hc_mult, hc_mult)
    row_max = cm.max(dim=-1, keepdim=True).values
    cm = torch.exp(cm - row_max)
    row_sum = cm.sum(dim=-1, keepdim=True)
    cm = cm / row_sum + hc_sinkhorn_eps
    col_sum = cm.sum(dim=-2, keepdim=True)
    cm = cm / (col_sum + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        row_sum = cm.sum(dim=-1, keepdim=True)
        cm = cm / (row_sum + hc_sinkhorn_eps)
        col_sum = cm.sum(dim=-2, keepdim=True)
        cm = cm / (col_sum + hc_sinkhorn_eps)
    comb_mix = cm.reshape(tokens, hc_mult * hc_mult)

    # pre_mix and the broadcast residual expansion.
    pre_mix = (
        torch.sigmoid(mixes[:, :hc_mult] * scale[0] + base[:hc_mult]) + hc_pre_eps
    )  # (T, hc_mult)
    residual_out = residual.unsqueeze(1).expand(tokens, hc_mult, hidden).contiguous()

    # All streams see the same input, so the mixed result is a single row
    # scaled by the sum of the stream coefficients before RMS normalization.
    mixed = residual.float() * pre_mix.sum(-1, keepdim=True)
    mean_square = mixed.square().mean(-1, keepdim=True)
    layer_input = (
        mixed.to(torch.bfloat16).float()
        * torch.rsqrt(mean_square + norm_eps)
        * norm_weight
    ).to(torch.bfloat16)
    return residual_out, post_mix, comb_mix, layer_input


class TestMhcBroadcastKernel:
    """The tilelang JIT kernel matches the independent torch reference."""

    @pytest.mark.parametrize("tokens", [8, 33, 64])
    def test_kernel_matches_reference(self, device, tokens):
        """The JIT kernel output matches the torch mirror for each shape."""
        kernel, _ = _load_backport()
        values = _make_inputs(device, tokens)
        gemm_mul, gemm_sqrsum = _torch_gemm(values, n_splits=1)

        hc_mult, hidden = values["hc_mult"], values["hidden"]
        residual_out = torch.empty(
            tokens, hc_mult, hidden, device=device, dtype=torch.bfloat16
        )
        post_mix = torch.empty(tokens, hc_mult, device=device)
        comb_mix = torch.empty(tokens, hc_mult * hc_mult, device=device)
        layer_input = torch.empty(tokens, hidden, device=device, dtype=torch.bfloat16)
        kernel(
            gemm_mul,
            gemm_sqrsum,
            values["hc_scale"],
            values["hc_base"],
            values["residual"],
            residual_out,
            post_mix,
            comb_mix,
            layer_input,
            values["norm_weight"],
            hidden,
            1e-5,  # rms_eps
            1e-6,  # hc_pre_eps
            1e-6,  # hc_sinkhorn_eps
            2.5,  # hc_post_mult_value
            20,  # sinkhorn_repeat
            1e-5,  # norm_eps
            1,  # n_splits
            hc_mult,
        )
        torch.cuda.synchronize()

        ref = _reference(
            values,
            gemm_mul,
            gemm_sqrsum,
            1e-5,
            1e-6,
            1e-6,
            2.5,
            20,
            1e-5,
        )
        # residual_out is an exact bf16 broadcast copy.
        assert torch.equal(residual_out, ref[0])
        torch.testing.assert_close(post_mix, ref[1], rtol=2e-3, atol=1e-3)
        torch.testing.assert_close(comb_mix, ref[2], rtol=2e-3, atol=1e-3)
        torch.testing.assert_close(layer_input, ref[3], rtol=2e-2, atol=4e-2)

    def test_sinkhorn_repeat_one(self, device):
        """sinkhorn_repeat=1 skips the refinement loop after the first pass."""
        kernel, _ = _load_backport()
        values = _make_inputs(device, 8)
        gemm_mul, gemm_sqrsum = _torch_gemm(values, n_splits=1)
        hc_mult, hidden = values["hc_mult"], values["hidden"]
        tokens = 8
        residual_out = torch.empty(
            tokens, hc_mult, hidden, device=device, dtype=torch.bfloat16
        )
        post_mix = torch.empty(tokens, hc_mult, device=device)
        comb_mix = torch.empty(tokens, hc_mult * hc_mult, device=device)
        layer_input = torch.empty(tokens, hidden, device=device, dtype=torch.bfloat16)
        kernel(
            gemm_mul,
            gemm_sqrsum,
            values["hc_scale"],
            values["hc_base"],
            values["residual"],
            residual_out,
            post_mix,
            comb_mix,
            layer_input,
            values["norm_weight"],
            hidden,
            1e-5,
            1e-6,
            1e-6,
            2.5,
            1,
            1e-5,
            1,
            hc_mult,
        )
        torch.cuda.synchronize()
        ref = _reference(values, gemm_mul, gemm_sqrsum, 1e-5, 1e-6, 1e-6, 2.5, 1, 1e-5)
        assert torch.equal(residual_out, ref[0])
        torch.testing.assert_close(comb_mix, ref[2], rtol=2e-3, atol=1e-3)
        torch.testing.assert_close(layer_input, ref[3], rtol=2e-2, atol=4e-2)

    def test_multi_split_gemm_inputs_sum(self, device):
        """n_splits>1 sums per-split gemm outputs before normalizing."""
        kernel, _ = _load_backport()
        values = _make_inputs(device, 16)
        single_mul, single_sqrsum = _torch_gemm(values, n_splits=1)
        n_splits = 2
        # Split the projection across splits by dividing the residual rows'
        # contribution: each split carries half of every element.
        half_mul = (single_mul * 0.5).repeat(n_splits, 1, 1).contiguous()
        half_sqrsum = (single_sqrsum * 0.5).repeat(n_splits, 1).contiguous()
        hc_mult, hidden = values["hc_mult"], values["hidden"]
        tokens = 16
        residual_out = torch.empty(
            tokens, hc_mult, hidden, device=device, dtype=torch.bfloat16
        )
        post_mix = torch.empty(tokens, hc_mult, device=device)
        comb_mix = torch.empty(tokens, hc_mult * hc_mult, device=device)
        layer_input = torch.empty(tokens, hidden, device=device, dtype=torch.bfloat16)
        kernel(
            half_mul,
            half_sqrsum,
            values["hc_scale"],
            values["hc_base"],
            values["residual"],
            residual_out,
            post_mix,
            comb_mix,
            layer_input,
            values["norm_weight"],
            hidden,
            1e-5,
            1e-6,
            1e-6,
            2.5,
            20,
            1e-5,
            n_splits,
            hc_mult,
        )
        torch.cuda.synchronize()
        # The sums reconstruct the single-split gemm outputs (fp32 sums of
        # halves are exact for these magnitudes).
        ref = _reference(
            values, single_mul, single_sqrsum, 1e-5, 1e-6, 1e-6, 2.5, 20, 1e-5
        )
        assert torch.equal(residual_out, ref[0])
        torch.testing.assert_close(post_mix, ref[1], rtol=2e-3, atol=1e-3)
        torch.testing.assert_close(layer_input, ref[3], rtol=2e-2, atol=4e-2)


class TestMhcBroadcastHostApi:
    """The host API contract: asserts, weight conversion, empty outputs."""

    def test_zero_tokens_returns_empty_outputs(self, device):
        """num_tokens==0 short-circuits before any gemm/kernel launch."""
        _, host = _load_backport()
        values = _make_inputs(device, 0)
        residual_out, post_mix, comb_mix, layer_input = host(
            values["residual"],
            values["fn"],
            values["hc_scale"],
            values["hc_base"],
            1e-5,
            1e-6,
            1e-6,
            2.5,
            20,
            norm_weight=values["norm_weight"],
            norm_eps=1e-5,
            fn_broadcast=values["fn_broadcast"],
        )
        hidden, hc_mult = values["hidden"], values["hc_mult"]
        assert residual_out.shape == (0, hc_mult, hidden)
        assert residual_out.dtype == torch.bfloat16
        assert post_mix.shape == (0, hc_mult, 1)
        assert post_mix.dtype == torch.float32
        assert comb_mix.shape == (0, hc_mult, hc_mult)
        assert comb_mix.dtype == torch.float32
        assert layer_input.shape == (0, hidden)
        assert layer_input.dtype == torch.bfloat16
        assert layer_input.device == values["residual"].device

    def test_zero_tokens_converts_fp32_strided_norm_weight(self, device):
        """The zero-token path first normalizes weight dtype/contiguity."""
        _, host = _load_backport()
        values = _make_inputs(device, 0)
        weight = values["norm_weight"].to(torch.float32)
        storage = torch.empty(weight.numel(), 2, device=device)
        storage[:, 0].copy_(weight)
        strided = storage[:, 0]
        assert not strided.is_contiguous()

        residual_out, _, _, _ = host(
            values["residual"],
            values["fn"],
            values["hc_scale"],
            values["hc_base"],
            1e-5,
            1e-6,
            1e-6,
            2.5,
            20,
            norm_weight=strided,
            norm_eps=1e-5,
            fn_broadcast=values["fn_broadcast"],
        )
        assert residual_out.shape == (0, values["hc_mult"], values["hidden"])
        # The strided fp32 weight must not be mutated by the conversion.
        assert torch.allclose(storage[:, 0], weight)
