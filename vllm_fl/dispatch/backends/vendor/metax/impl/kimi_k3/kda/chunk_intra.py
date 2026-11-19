# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
"""K3 compatibility differences from the pinned vLLM KDA implementation.

Upstream computation is reused through local launch bindings. Host wrappers
remain local where they select those bindings or the retained gate cumsum.
"""

import torch

from vllm.models.kimi_k3.nvidia.ops.third_party.kda import chunk_intra as native_intra
from vllm.models.kimi_k3.nvidia.ops.third_party.kda.chunk_intra_token_parallel import (
    chunk_kda_fwd_intra_token_parallel,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.triton_utils import triton

from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.launch import local_kernel

chunk_kda_fwd_kernel_inter_solve_fused = local_kernel(
    native_intra.chunk_kda_fwd_kernel_inter_solve_fused,
    heuristics={"IS_VARLEN": lambda args: args["cu_seqlens"] is not None},
    configs=[
        triton.Config({"BK": BK, "DEFAULT_DOT_PRECISION": "ieee"}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [4]
    ],
    key=["H", "HV", "K", "BC"],
)

chunk_kda_fwd_kernel_intra_sub_chunk = local_kernel(
    native_intra.chunk_kda_fwd_kernel_intra_sub_chunk,
    heuristics={"IS_VARLEN": lambda args: args["cu_seqlens"] is not None},
    configs=[
        triton.Config(
            {"DEFAULT_DOT_PRECISION": "ieee"},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_warps in [4]
        for num_stages in [1]
    ],
    key=["BK", "NC", "BT", "HV"],
)


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
):
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT = chunk_size
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Separate fp32 buffer for diagonal 16x16 blocks (for precision in solve_tril)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)

    # Compute diagonal blocks into Akkd in fp32.
    if safe_gate:
        grid = (NT, NC, B * HV)
        BK = triton.next_power_of_2(K)
        chunk_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            # The validated MetaX path uses the native direct-load branch.
            USE_GATHER=False,
        )
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: Fused inter + solve_tril (works for both fixed-len and varlen)
    solve_tril_dot_precision = (
        "tf32"
        if current_platform.is_cuda() and current_platform.has_device_capability(80)
        else "ieee"
    )
    grid = (NT, B * HV)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        USE_SAFE_GATE=safe_gate,
        SOLVE_TRIL_DOT_PRECISION=solve_tril_dot_precision,
    )
    return Aqk, Akk
