# SPDX-License-Identifier: Apache-2.0
"""TP headpacking belongs to FL, not the local attention operator library."""

from typing import Optional, Tuple
import torch
import torch.distributed as dist
import triton
from vllm_fl import dsv4_config as _dsv4
from vllm_fl.dispatch.backends.vendor.metax.impl.dsv4.attention import prefill as _kernels


def _try_headpack(
    q, kv, indices, sm_scale, DV, attn_sink, topk_length, output, max_logits, lse
):
    if not _dsv4.current.prefill_headpack:
        return False
    (SQ, HQ, DQK) = q.shape
    (_, HKV, _) = kv.shape
    TOPK = indices.shape[-1]
    if not (
        SQ >= 8
        and HQ == 16
        and (HKV == 1)
        and DQK == 512
        and (DV == 512)
        and (TOPK > 0)
        and q.is_contiguous()
        and indices.is_contiguous()
    ):
        return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if torch.compiler.is_compiling():
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        return False
    world = dist.get_world_size()
    if world < 2 or world % 2 != 0:
        return False
    SKV = kv.shape[0]
    rank = dist.get_rank()
    peer = rank ^ 1
    T2 = SQ // 2
    if rank % 2 == 0:
        (lo, hi, plo, phi) = (0, T2, T2, SQ)
    else:
        (lo, hi, plo, phi) = (T2, SQ, 0, T2)
    T_loc = hi - lo
    T_peer = phi - plo
    dev = q.device
    REAL = HQ // 2
    q_send = q[plo:phi, :REAL].contiguous()
    q_recv = torch.empty((T_loc, REAL, DQK), dtype=q.dtype, device=dev)
    plan = [dist.P2POp(dist.isend, q_send, peer), dist.P2POp(dist.irecv, q_recv, peer)]
    if attn_sink is not None:
        sink_send = attn_sink[:REAL].contiguous()
        sink_recv = torch.empty(REAL, dtype=attn_sink.dtype, device=dev)
        plan.append(dist.P2POp(dist.isend, sink_send, peer))
        plan.append(dist.P2POp(dist.irecv, sink_recv, peer))
    for req in dist.batch_isend_irecv(plan):
        req.wait()
    q16 = torch.empty((T_loc, HQ, DQK), dtype=q.dtype, device=dev)
    q16[:, :REAL] = q[lo:hi, :REAL]
    q16[:, REAL:] = q_recv
    if attn_sink is not None:
        sink16 = torch.cat((attn_sink[:REAL], sink_recv)).contiguous()
    else:
        sink16 = None
    out16 = torch.empty((T_loc, HQ, DV), dtype=output.dtype, device=dev)
    ml16 = torch.empty((T_loc, HQ), dtype=torch.float32, device=dev)
    lse16 = torch.empty((T_loc, HQ), dtype=torch.float32, device=dev)
    idx_loc = indices[lo:hi].contiguous()
    topk_loc = topk_length[lo:hi].contiguous() if topk_length is not None else None
    grid = (T_loc,)
    _kernels.sparse_prefill_local(
        q16, kv, idx_loc, sm_scale, sink16, topk_loc, out16, ml16, lse16
    )
    output[lo:hi, :REAL] = out16[:, :REAL]
    o_send = out16[:, REAL:].contiguous()
    o_recv = torch.empty((T_peer, REAL, DV), dtype=output.dtype, device=dev)
    for req in dist.batch_isend_irecv(
        [dist.P2POp(dist.isend, o_send, peer), dist.P2POp(dist.irecv, o_recv, peer)]
    ):
        req.wait()
    output[plo:phi, :REAL] = o_recv
    return True


def sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    max_logits_out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Framework-owned TP headpacking; the matching TP-specialized arithmetic stays in FL."""
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and (indices.dtype == torch.int32)
    )
    (SQ, HQ, DQK) = q.shape
    (SKV, HKV, _) = kv.shape
    assert d_v == 512, "Unsupported d_v"
    DV = d_v
    assert kv.shape[-1] == DQK
    (_, _, TOPK) = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,), "topk_length error shape"
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 16 or HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"
    if out is None:
        output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    else:
        assert out.device == q.device and out.dtype == q.dtype
        assert out.shape == (SQ, HQ, DV), (
            f"out error shape {tuple(out.shape)} != {(SQ, HQ, DV)}"
        )
        output = out
    if max_logits_out is None:
        max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    else:
        assert (
            max_logits_out.device == q.device and max_logits_out.dtype == torch.float32
        )
        assert max_logits_out.shape == (SQ, HQ), "max_logits_out error shape"
        max_logits = max_logits_out
    if lse_out is None:
        lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    else:
        assert lse_out.device == q.device and lse_out.dtype == torch.float32
        assert lse_out.shape == (SQ, HQ), "lse_out error shape"
        lse = lse_out
    if _try_headpack(
        q, kv, indices, sm_scale, DV, attn_sink, topk_length, output, max_logits, lse
    ):
        return (output, max_logits, lse)
    return _kernels.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        out=output,
        max_logits_out=max_logits,
        lse_out=lse,
        real_num_heads=_dsv4.current.real_query_heads,
    )
