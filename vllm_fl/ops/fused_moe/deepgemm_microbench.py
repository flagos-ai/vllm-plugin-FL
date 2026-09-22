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

"""Microbenchmark for the PPU-native DeepGEMM BF16 MoE path.

Validates, on the real PPU device and Qwen3.5-MoE shapes, that
``deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad`` can drive the two expert
GEMMs of an unquantized BF16 MoE.

Uses a self-contained pure-torch argsort permute (dtype-agnostic, no dependency
on vLLM's fp8-oriented ep_scatter nor the missing
get_mk_alignment_for_contiguous_layout). This isolates the grouped GEMM math and
speed; the serving path uses the vendor ``deepgemm_moe_permute`` kernel.

Run:  python -m vllm_fl.ops.fused_moe.deepgemm_microbench
"""

import deep_gemm
import torch
import torch.nn.functional as F

E = 256  # experts (TP, no EP -> all experts local)
I = 256  # intermediate size per partition (moe_intermediate_size / TP2)
H = 2048  # hidden size
TOPK = 8
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _grouped_gemm_nopad(lhs, rhs, m_indices, m_rows):
    """lhs [m,k] @ rhs[G,n,k]^T grouped by m_indices -> out [m,n]."""
    m, k = lhs.shape
    G, n, k2 = rhs.shape
    assert k == k2
    out = torch.empty(m, n, dtype=DTYPE, device=DEVICE)
    deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(lhs, rhs, out, m_indices, m_rows)
    return out


def _permute(hidden, topk_ids):
    """Pure-torch permute: group (token,expert) rows contiguously by expert.

    Returns a1[m,H], m_indices[m] int32, m_rows[E] int32, order, tok_of_row.
    """
    M = hidden.shape[0]
    flat_expert = topk_ids.reshape(-1).to(torch.int32)  # [M*topk]
    tok_idx = torch.arange(M, device=DEVICE).repeat_interleave(TOPK)  # [M*topk]
    order = torch.argsort(flat_expert)  # group by expert
    m_indices = flat_expert[order].contiguous()
    tok_of_row = tok_idx[order].contiguous()
    a1 = hidden.index_select(0, tok_of_row).contiguous()  # [m,H]
    m_rows = torch.bincount(flat_expert, minlength=E).to(torch.int32)
    return a1, m_indices, m_rows, order, tok_of_row


def _run(M, w1, w2, gen, check_numeric):
    hidden = torch.randn(M, H, dtype=DTYPE, device=DEVICE, generator=gen)
    topk_ids = torch.randint(
        0, E, (M, TOPK), device=DEVICE, generator=gen, dtype=torch.int64
    )
    topk_w = torch.rand(M, TOPK, dtype=torch.float32, device=DEVICE, generator=gen)

    def pipeline():
        a1, m_indices, m_rows, order, tok_of_row = _permute(hidden, topk_ids)
        mm1 = _grouped_gemm_nopad(a1, w1, m_indices, m_rows)  # [m, 2I]
        act = (
            F.silu(mm1[:, :I].float()).to(DTYPE) * mm1[:, I:]
        )  # silu_and_mul -> [m, I]
        act = act.contiguous()
        mm3 = _grouped_gemm_nopad(act, w2, m_indices, m_rows)  # [m, H]
        # unpermute + weighted reduce
        tmp = torch.empty_like(mm3)
        tmp[order] = mm3
        tmp = tmp.view(M, TOPK, H)
        out = (tmp.float() * topk_w.unsqueeze(-1)).sum(1).to(DTYPE)
        return out

    out = pipeline()
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(out).all())
    print(f"\n=== M={M} (m={M * TOPK}) === out {tuple(out.shape)} finite={ok}")

    if check_numeric:
        # reference via gathered per-row bmm (bounded memory: only small M)
        a1, m_indices, m_rows, order, tok_of_row = _permute(hidden, topk_ids)
        w1g = w1.index_select(0, m_indices)  # [m,2I,H]
        gate_up = torch.bmm(w1g.float(), a1.float().unsqueeze(-1)).squeeze(-1)  # [m,2I]
        ref_act = F.silu(gate_up[:, :I]) * gate_up[:, I:]  # [m,I]
        w2g = w2.index_select(0, m_indices)  # [m,H,I]
        ref_row = torch.bmm(w2g.float(), ref_act.unsqueeze(-1)).squeeze(-1)  # [m,H]
        tmp = torch.empty(M * TOPK, H, device=DEVICE)
        tmp[order] = ref_row
        ref = (tmp.view(M, TOPK, H) * topk_w.unsqueeze(-1)).sum(1)
        diff = (out.float() - ref).abs()
        rel = diff.max().item() / (ref.abs().max().item() + 1e-6)
        print(
            f"  numeric vs torch ref: max_abs={diff.max().item():.3f} "
            f"max_rel={rel:.4f} -> {'PASS' if rel < 2e-2 else 'CHECK'}"
        )

    # timing
    for _ in range(5):
        pipeline()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        pipeline()
    e.record()
    torch.cuda.synchronize()
    print(f"  full pipeline: {s.elapsed_time(e) / 20 * 1000:.1f} us/iter")


def main():
    print("device:", torch.cuda.get_device_name(0))
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    w1 = torch.randn(E, 2 * I, H, dtype=DTYPE, device=DEVICE, generator=gen) * 0.02
    w2 = torch.randn(E, H, I, dtype=DTYPE, device=DEVICE, generator=gen) * 0.02

    # decode-ish (small M -> should hit gemv path) with numeric check
    _run(16, w1, w2, gen, check_numeric=True)
    # prefill-ish (large M -> gemm path), finite + latency only
    _run(2048, w1, w2, gen, check_numeric=False)
    print("\nValidation complete: see finite/PASS results above")


if __name__ == "__main__":
    main()
