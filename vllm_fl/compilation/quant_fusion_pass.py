# SPDX-License-Identifier: Apache-2.0
"""`QuantFusionPass`: Vendor-neutral post-grad FX `InductorPass` for W8A8 execution.

Rewrites post-AOTAutograd FX subgraphs into fused FlagGems / vLLM custom ops:
1. `RMSNorm` / `FusedAddRMSNorm` + `flag_gems.quant_input` (both `vllm_ir` custom
   ops and Dynamo/Inductor-decomposed primitive ATen chains) -> `rmsnorm_quant` /
   `fused_add_rmsnorm_quant` (or standalone in-place `fl_fused_add_rms_norm`).
2. `GateUp` (`quant_gemm_w8a8(..., epilogue_mode=1)`) + `SiluAndMul` + `quant_input`
   -> interleaved 64-column SwiGLU+Hadamard GEMM epilogue (`epilogue_mode=2`) +
   Hadamard-free `quant_input`, or `flag_gems.silu_mul_quant` when non-interleaved.
3. `auto_functionalized(vllm.unified_attention_with_output)` + `aten.empty` +
   `quant_input` -> `vllm.fl_unified_attention_quant`, eliminating functionalization
   buffer copies and delegating Pure Decode steps directly to `impl.forward_quant`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import operator
from typing import Any

import torch
from torch import fx
from vllm.compilation.passes.inductor_pass import InductorPass, get_pass_context
from vllm.utils.torch_utils import LayerNameType, _resolve_layer_name, direct_register_custom_op

import flag_gems  # noqa: F401 - ensures public `flag_gems::*` custom ops are registered
from flag_gems.fused.fused_add_rms_norm import fused_add_rms_norm as _fg_fused_add_rms_norm
from flag_gems.utils.quant_precondition import (
    ACT_NONE,
    _alloc_q_and_s,
    _fake_q_and_s,
    _launch_fused_act_quant,
)

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.fl.quant_fusion_pass")
except Exception:
    logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantFusionConfig:
    """Compile-time static configuration for `QuantFusionPass`.

    Attributes:
        norm_do_hadamard: Enable block-64 Walsh-Hadamard smoothing inside fused
            `rmsnorm_quant` / `fused_add_rmsnorm_quant` when requested by `quant_input`.
        down_do_hadamard: Enable Walsh-Hadamard smoothing inside `silu_mul_quant`
            prior to `down_proj` when non-interleaved SwiGLU fusion is used.
        down_num_scale_groups: Minimum per-row dynamic INT8 scale group count
            emitted for `down_proj` activation quantization.
        fuse_interleaved_swiglu: Rewrite `quant_gemm_w8a8(..., epilogue_mode=1)` +
            SwiGLU + Hadamard `quant_input` into `epilogue_mode=2` (requires offline
            64-column `[Gate, Up]` weight interleaving from `QuantTransformConfig`).
        fuse_attn_out_quant: Fuse `auto_functionalized(unified_attention_with_output)`
            + `quant_input` into `vllm.fl_unified_attention_quant`.
    """

    norm_do_hadamard: bool = True
    down_do_hadamard: bool = True
    down_num_scale_groups: int = 1
    fuse_interleaved_swiglu: bool = False
    fuse_attn_out_quant: bool = False


_GET_ATTN_CTX_FN: Any = None
_GET_DEVICE_FN = getattr(torch._C, "_cuda_getDevice", torch.cuda.current_device)
_RAW_STREAM_FN = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_IS_CAPTURING_FN = getattr(torch._C, "_cuda_isCurrentStreamCapturing", torch.cuda.is_current_stream_capturing)
# Reusable BF16/FP16 scratch buffer keyed by (device_idx, raw_stream_ptr, dtype)
# to avoid per-layer `aten.empty` allocations during eager Prefill/Mixed fallback.
_ATTN_OUT_SCRATCH: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}


def _fl_fused_add_rms_norm_impl(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """In-place `FusedAddRMSNorm` custom op body mutating both `x` and `residual`."""
    _fg_fused_add_rms_norm(x, residual, (int(weight.shape[0]),), weight, float(eps))


def _fl_fused_add_rms_norm_fake(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """FakeTensor meta implementation for `vllm::fl_fused_add_rms_norm`."""
    return None


if not hasattr(torch.ops.vllm, "fl_fused_add_rms_norm"):
    direct_register_custom_op(
        op_name="fl_fused_add_rms_norm",
        op_func=_fl_fused_add_rms_norm_impl,
        mutates_args=["x", "residual"],
        fake_impl=_fl_fused_add_rms_norm_fake,
    )


def _fl_unified_attention_quant_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    do_hadamard: bool = False,
    num_scale_groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute unified attention and directly emit `(out_int8, out_scale)` for `o_proj`.

    Functional custom op (`mutates_args=[]`) that replaces AOTAutograd's
    `auto_functionalized(unified_attention_with_output)` + `aten.empty` +
    `quant_input` sequence. On Pure Decode steps (or whenever the attention
    backend provides `impl.forward_quant`), fuses attention epilogue/Stage-2
    reduction directly into INT8 output quantization without materializing an
    intermediate BF16 tensor in HBM. On Prefill/Mixed fallback steps, writes
    BF16 attention output into a stream-cached scratch buffer (or graph-pool
    tensor during CUDA Graph capture) and launches `_launch_fused_act_quant`.
    """
    global _GET_ATTN_CTX_FN
    # `kv_cache_dummy_dep` is an FX data-dependency token ordering KV-cache
    # updates before attention; it carries no runtime payload.
    del kv_cache_dummy_dep
    if _GET_ATTN_CTX_FN is None:
        from vllm.model_executor.layers.attention.attention import get_attention_context

        _GET_ATTN_CTX_FN = get_attention_context
    attn_metadata, attn_layer, kv_cache, _ = _GET_ATTN_CTX_FN(_resolve_layer_name(layer_name))
    m, nh, v_hd = int(query.shape[0]), int(query.shape[1]), int(value.shape[2])
    k, nsg = nh * v_hd, int(num_scale_groups)
    dev = query.device
    if m == 0:
        out_3d = torch.empty((0, nh, v_hd), dtype=query.dtype, device=dev)
        attn_layer.impl.forward(attn_layer, query, key, value, kv_cache, attn_metadata, output=out_3d)
        return _alloc_q_and_s(query, 0, k, nsg)

    # Fast path: delegate directly to backend's fused attention + INT8 quantizer
    # (e.g., Pure Decode Stage-2/single-stage epilogue fusion). Returns `None`
    # when the current step contains Prefill tokens and requires BF16 fallback.
    if (fwd_q := getattr(attn_layer.impl, "forward_quant", None)) is not None:
        fused_pair = fwd_q(attn_layer, query, key, value, kv_cache, attn_metadata, bool(do_hadamard), nsg)
        if fused_pair is not None:
            return fused_pair

    # Fallback path: materialize BF16 attention output in a CUDA-Graph-pool tensor
    # (when capturing) or a per-(device, stream, dtype) reusable scratch buffer.
    need = m * k
    if bool(_IS_CAPTURING_FN()):
        out_2d = torch.empty((m, k), dtype=query.dtype, device=dev)
    else:
        dev_idx = dev.index if dev.index is not None else _GET_DEVICE_FN()
        s_key = (dev_idx, _RAW_STREAM_FN(dev_idx) if _RAW_STREAM_FN is not None else 0, query.dtype)
        buf = _ATTN_OUT_SCRATCH.get(s_key)
        if buf is None or buf.numel() < need:
            if s_key not in _ATTN_OUT_SCRATCH and len(_ATTN_OUT_SCRATCH) >= 8:
                _ATTN_OUT_SCRATCH.pop(next(iter(_ATTN_OUT_SCRATCH)))
            buf = torch.zeros((max(need, 2048 * k),), dtype=query.dtype, device=dev)
            _ATTN_OUT_SCRATCH[s_key] = buf
        out_2d = buf[:need].view(m, k)
    attn_layer.impl.forward(attn_layer, query, key, value, kv_cache, attn_metadata, output=out_2d.view(m, nh, v_hd))
    out_q, out_s = _alloc_q_and_s(query, m, k, nsg)
    _launch_fused_act_quant(out_2d, None, None, out_q, out_s, 1e-5, ACT_NONE, bool(do_hadamard), nsg)
    return out_q, out_s


def _fl_unified_attention_quant_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    do_hadamard: bool = False,
    num_scale_groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FakeTensor meta shape/dtype inference for `vllm::fl_unified_attention_quant`."""
    _ = (key, layer_name, kv_cache_dummy_dep, do_hadamard)
    return _fake_q_and_s(query, query.shape[0], int(query.shape[1]) * int(value.shape[2]), int(num_scale_groups))


if not hasattr(torch.ops.vllm, "fl_unified_attention_quant"):
    direct_register_custom_op(
        op_name="fl_unified_attention_quant",
        op_func=_fl_unified_attention_quant_impl,
        mutates_args=[],
        fake_impl=_fl_unified_attention_quant_fake,
    )


def _is_op(node: Any, *targets: Any) -> bool:
    """Return True if `node` is an FX `call_function` matching any target in `targets`."""
    return isinstance(node, fx.Node) and node.op == "call_function" and node.target in targets


def _unwrap_cast(node: Any) -> Any:
    """Peel consecutive `prims.convert_element_type` casts to reach the underlying value node."""
    while _is_op(node, torch.ops.prims.convert_element_type.default):
        node = node.args[0]
    return node


def _unwrap_reshape(node: Any) -> Any:
    """Peel consecutive `reshape` / `view` / `alias` nodes to reach the source tensor node."""
    while _is_op(node, torch.ops.aten.reshape.default, torch.ops.aten.view.default, torch.ops.aten.alias.default):
        node = node.args[0]
    return node


def _match_silu_branch(node: Any) -> fx.Node | None:
    """Match `silu(g)` or decomposed `g * sigmoid(g)` (in either operand order)."""
    n = _unwrap_cast(node)
    if _is_op(n, torch.ops.aten.silu.default):
        return _unwrap_cast(n.args[0])
    if _is_op(n, torch.ops.aten.mul.Tensor):
        a, b = n.args[0], n.args[1]
        if _is_op(b, torch.ops.aten.sigmoid.default) and b.args[0] is a:
            return _unwrap_cast(a)
        if _is_op(a, torch.ops.aten.sigmoid.default) and a.args[0] is b:
            return _unwrap_cast(b)
    return None


def _extract_quant_args(qnode: fx.Node) -> tuple[bool, int]:
    """Extract `(do_hadamard, num_scale_groups)` from a `flag_gems.quant_input` node."""
    do_had = bool(qnode.args[1]) if len(qnode.args) > 1 else bool(qnode.kwargs.get("do_hadamard", False))
    nsg = int(qnode.args[2]) if len(qnode.args) > 2 else int(qnode.kwargs.get("num_scale_groups", 1))
    return do_had, nsg


def _unwrap_single_cast(node: Any) -> Any:
    """Peel single-user `prims.convert_element_type` casts without bypassing shared nodes."""
    while _is_op(node, torch.ops.prims.convert_element_type.default) and len(node.users) == 1:
        node = node.args[0]
    return node


def _try_fuse_decomposed_aten_rmsnorm(
    graph: fx.Graph, qnode: fx.Node, node_to_idx: dict[fx.Node, int], cfg: QuantFusionConfig,
    donated_ids: set[int] | None = None,
) -> bool:
    """Match and fuse Dynamo/AOT-decomposed ATen `RMSNorm` / `AddRMSNorm` feeding `qnode`.

    When custom `vllm_ir.rms_norm` ops are not emitted (e.g., on model layers or
    backends where `RMSNorm.forward_native` is traced into primitive ATen ops),
    matches the canonical FP32-upcast normalization tree:
        `y = (x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)) * weight`
    under any association order of the three multiplicative factors `(x, rstd, weight)`.

    Safety invariants enforced before rewriting:
    - Every intermediate arithmetic/view node inside the normalization tree has
      exactly 1 user (no side consumers).
    - If `x` is itself `aten.add.Tensor(a, b)` (residual add), fused in-place
      `fused_add_rmsnorm_quant(x_in, res_arg, ...)` is selected only when all
      external consumers of the residual sum appear strictly after `qnode` in
      topological order, `res_in` has no conflicting uses between `add` and
      `qnode`, and non-donated graph `placeholder` buffers are defensively cloned
      via `aten._to_copy` to uphold Inductor's input-immutability contract.
    """
    cur: Any = qnode.args[0]
    while _is_op(
        cur, torch.ops.aten.reshape.default, torch.ops.aten.view.default,
        torch.ops.aten.alias.default, torch.ops.prims.convert_element_type.default,
    ):
        if len(cur.users) != 1:
            return False
        cur = cur.args[0]
    if not _is_op(cur, torch.ops.aten.mul.Tensor) or len(cur.users) != 1:
        return False

    # Recover the three multiplicative leaves `(x, rstd, weight)` regardless of
    # whether `(x * rstd) * weight` or `x * (rstd * weight)` was traced.
    leaves: tuple[Any, Any, Any] | None = None
    for i_in in (0, 1):
        cand_inner = _unwrap_single_cast(cur.args[i_in])
        if _is_op(cand_inner, torch.ops.aten.mul.Tensor) and len(cand_inner.users) == 1:
            leaves = (cand_inner.args[0], cand_inner.args[1], cur.args[1 - i_in])
            mul_inner = cand_inner
            break
    if leaves is None:
        return False

    # Verify the `rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)` variance branch.
    rstd_idx = next(
        (i for i, lf in enumerate(leaves) if _is_op(_unwrap_single_cast(lf), torch.ops.aten.rsqrt.default) and len(_unwrap_single_cast(lf).users) == 1),
        -1,
    )
    if rstd_idx < 0:
        return False
    rsqrt_n = _unwrap_single_cast(leaves[rstd_idx])
    eps_add = rsqrt_n.args[0]
    if not _is_op(eps_add, torch.ops.aten.add.Tensor) or len(eps_add.users) != 1 or eps_add.kwargs.get("alpha", 1) != 1:
        return False
    if isinstance(eps_add.args[1], (int, float)) and not isinstance(eps_add.args[1], bool):
        var_n, eps = eps_add.args[0], float(eps_add.args[1])
    elif isinstance(eps_add.args[0], (int, float)) and not isinstance(eps_add.args[0], bool):
        var_n, eps = eps_add.args[1], float(eps_add.args[0])
    else:
        return False

    keepdim = bool(var_n.args[2]) if (isinstance(var_n, fx.Node) and len(var_n.args) > 2) else bool(getattr(var_n, "kwargs", {}).get("keepdim", False))
    if not _is_op(var_n, torch.ops.aten.mean.dim) or len(var_n.users) != 1 or not keepdim or list(var_n.args[1]) not in ([-1], [1]):
        return False
    sq_n = var_n.args[0]
    if not isinstance(sq_n, fx.Node) or len(sq_n.users) != 1:
        return False
    if _is_op(sq_n, torch.ops.aten.pow.Tensor_Scalar) and sq_n.args[1] in (2, 2.0):
        xf_var = sq_n.args[0]
    elif _is_op(sq_n, torch.ops.aten.mul.Tensor) and sq_n.args[0] is sq_n.args[1]:
        xf_var = sq_n.args[0]
    else:
        return False

    rem = [leaves[i] for i in range(3) if i != rstd_idx]
    x_base = _unwrap_cast(xf_var)
    if _unwrap_cast(rem[0]) is x_base:
        xf_mul, w_raw = rem[0], rem[1]
    elif _unwrap_cast(rem[1]) is x_base:
        xf_mul, w_raw = rem[1], rem[0]
    else:
        return False
    w = _unwrap_cast(w_raw)
    w_val = w.meta.get("val") if isinstance(w, fx.Node) else None
    if not isinstance(x_base, fx.Node) or w_val is None or w_val.ndim != 1:
        return False
    # Ensure intermediate FP32 upcasts of `x` and `w` have no consumers outside this RMSNorm tree.
    norm_consumers = {sq_n, mul_inner, cur}
    for cast_n in (xf_var, xf_mul, w_raw):
        if isinstance(cast_n, fx.Node) and cast_n not in (x_base, w):
            if any(u not in norm_consumers and not (_is_op(u, torch.ops.prims.convert_element_type.default) and set(u.users) <= norm_consumers) for u in cast_n.users):
                return False

    q_had, q_nsg = _extract_quant_args(qnode)
    do_had = bool(q_had and cfg.norm_do_hadamard)
    if donated_ids is None:
        try:
            donated_ids = getattr(get_pass_context(), "donated_input_ids", set())
        except (AssertionError, RuntimeError, ValueError, TypeError):
            donated_ids = set()

    # Check if `x_base` originates from a residual `aten.add.Tensor(a, b)` that can
    # be fused into in-place `fused_add_rmsnorm_quant` without violating consumer order.
    add_cand = x_base if _is_op(x_base, torch.ops.aten.add.Tensor) else (xf_var if _is_op(xf_var, torch.ops.aten.add.Tensor) else None)
    if add_cand is not None and add_cand.kwargs.get("alpha", 1) == 1:
        a_in, b_in = _unwrap_cast(add_cand.args[0]), _unwrap_cast(add_cand.args[1])
        a_val = a_in.meta.get("val") if isinstance(a_in, fx.Node) else None
        b_val = b_in.meta.get("val") if isinstance(b_in, fx.Node) else None
        add_val = add_cand.meta.get("val")
        if (
            a_val is not None and b_val is not None and add_val is not None
            and a_val.ndim == 2 and a_val.shape == b_val.shape and w_val.shape == (a_val.shape[1],)
            and a_val.dtype in (torch.bfloat16, torch.float16) and a_val.dtype == b_val.dtype and w_val.dtype == a_val.dtype
        ):
            add_idx, q_idx = node_to_idx.get(add_cand, 0), node_to_idx.get(qnode, 0)
            int_u = {xf_var, xf_mul, *norm_consumers}
            ext_direct, ext_casts, ext_ok = [], [], True
            for u in add_cand.users:
                if u in int_u:
                    continue
                if add_val.dtype == a_val.dtype and node_to_idx.get(u, 0) > q_idx:
                    ext_direct.append(u)
                elif _is_op(u, torch.ops.prims.convert_element_type.default) and u.args[1] == a_val.dtype and all(node_to_idx.get(cu, 0) > q_idx for cu in u.users):
                    ext_casts.append(u)
                else:
                    ext_ok = False
                    break
            if ext_ok:
                # Prefer mutating an internal tensor over a non-donated graph
                # input placeholder whenever either addend can serve as `res_in`.
                cands = [(a_in, b_in), (b_in, a_in)]
                cands.sort(key=lambda p: 1 if (p[1].op == "placeholder" and node_to_idx.get(p[1], -1) not in donated_ids) else 0)
                for x_in, res_in in cands:
                    in_casts = {c for c in add_cand.args[:2] if isinstance(c, fx.Node) and _unwrap_cast(c) is res_in and len(c.users) == 1}
                    if all(node_to_idx.get(u, q_idx) < add_idx for u in res_in.users if u is not add_cand and u not in in_casts):
                        res_arg = res_in
                        if res_in.op == "placeholder" and node_to_idx.get(res_in, -1) not in donated_ids:
                            # Clone non-donated placeholder before in-place residual update.
                            with graph.inserting_before(qnode):
                                res_arg = graph.call_function(torch.ops.aten._to_copy.default, args=(res_in,))
                                res_arg.meta["val"] = res_in.meta.get("val")
                                node_to_idx[res_arg] = q_idx
                        with graph.inserting_before(qnode):
                            pair = graph.call_function(torch.ops.flag_gems.fused_add_rmsnorm_quant.default, args=(x_in, res_arg, w, eps, do_had, q_nsg))
                            pair.meta["val"] = qnode.meta.get("val")
                            node_to_idx[pair] = q_idx
                        qnode.replace_all_uses_with(pair)
                        for cu in ext_casts:
                            cu.replace_all_uses_with(res_arg)
                        for du in ext_direct:
                            du.replace_input_with(add_cand, res_arg)
                        graph.erase_node(qnode)
                        return True

    # Standalone RMSNorm + quant_input (no residual add or residual could not be mutated safely).
    x_val = x_base.meta.get("val")
    if x_val is None or x_val.ndim != 2 or w_val.shape != (x_val.shape[1],) or x_val.dtype not in (torch.bfloat16, torch.float16) or w_val.dtype != x_val.dtype:
        return False
    with graph.inserting_before(qnode):
        pair = graph.call_function(torch.ops.flag_gems.rmsnorm_quant.default, args=(x_base, w, eps, do_had, q_nsg))
        pair.meta["val"] = qnode.meta.get("val")
        node_to_idx[pair] = node_to_idx.get(qnode, 0)
    qnode.replace_all_uses_with(pair)
    graph.erase_node(qnode)
    return True


class QuantFusionPass(InductorPass):
    """Post-grad FX fusion pass for W8A8 Linear + RMSNorm + SwiGLU + Attention-Out quantization."""

    def __init__(self, cfg: QuantFusionConfig | None = None) -> None:
        """Initialize the pass with `cfg` (or default `QuantFusionConfig`)."""
        self.cfg = cfg or QuantFusionConfig()

    def uuid(self) -> str:
        """Return a JSON-safe digest hashing this class, module helpers, and config.

        `InductorPass.hash_source(self)` only hashes `QuantFusionPass` methods;
        module-level matcher/runtime functions MUST be included explicitly so any
        change to pattern-matching or custom-op dispatch invalidates stale
        TorchInductor / vLLM disk caches, and `hash_dict` produces a plain hex
        string safe for JSON serialization in Inductor metadata.
        """
        return InductorPass.hash_dict({
            "src": InductorPass.hash_source(
                self, _match_silu_branch, _try_fuse_decomposed_aten_rmsnorm, _fl_unified_attention_quant_impl
            ),
            "cfg": asdict(self.cfg),
        })

    def __call__(self, graph: fx.Graph) -> None:
        """Apply RMSNorm, SwiGLU, and Attention-Out quantization fusions in place."""
        n_rmsnorm = self._fuse_rmsnorm_and_quant(graph)
        n_swiglu = self._fuse_swiglu_and_quant(graph)
        n_attn_q = self._fuse_attn_out_and_quant(graph) if self.cfg.fuse_attn_out_quant else 0
        if n_rmsnorm or n_swiglu or n_attn_q:
            graph.eliminate_dead_code()
            logger.info(
                "[FL-QuantFusion] Fused %d RMSNorm/AddRMSNorm ops, %d SwiGLU ops, and %d AttnOut+Quant ops.",
                n_rmsnorm,
                n_swiglu,
                n_attn_q,
            )

    def _fuse_rmsnorm_and_quant(self, graph: fx.Graph) -> int:
        """Fuse `vllm_ir` or decomposed ATen RMSNorm/AddRMSNorm nodes with `quant_input`."""
        rms_op, add_rms_op = torch.ops.vllm_ir.rms_norm.default, torch.ops.vllm_ir.fused_add_rms_norm.default
        quant_op = torch.ops.flag_gems.quant_input.default
        node_to_idx = {n: idx for idx, n in enumerate(graph.nodes)}
        try:
            donated_ids = getattr(get_pass_context(), "donated_input_ids", set())
        except (AssertionError, RuntimeError, ValueError, TypeError):
            donated_ids = set()

        def _find_quant_user(src: fx.Node) -> tuple[fx.Node | None, fx.Node | None]:
            """Return `(quant_node, optional_reshape_node)` if `src` exclusively feeds `quant_input`."""
            if len(src.users) != 1:
                return None, None
            u0 = next(iter(src.users))
            if _is_op(u0, quant_op):
                return u0, None
            if _is_op(u0, torch.ops.aten.reshape.default, torch.ops.aten.view.default) and len(u0.users) == 1:
                u1 = next(iter(u0.users))
                if _is_op(u1, quant_op):
                    return u1, u0
            return None, None

        def _ensure_mutable(arg: fx.Node, before: fx.Node) -> fx.Node:
            """Clone non-donated graph placeholders before in-place mutation to preserve Inductor invariants."""
            if arg.op == "placeholder" and node_to_idx.get(arg, -1) not in donated_ids:
                with graph.inserting_before(before):
                    cloned = graph.call_function(torch.ops.aten._to_copy.default, args=(arg,))
                    cloned.meta["val"] = arg.meta.get("val")
                    node_to_idx[cloned] = node_to_idx.get(before, 0)
                    return cloned
            return arg

        fused = 0
        for node in list(graph.nodes):
            # Case 1: `vllm_ir.rms_norm` -> `flag_gems.rmsnorm_quant`.
            if _is_op(node, rms_op):
                x, w, eps = node.args[0], node.args[1], node.args[2]
                var_ov = node.args[3] if len(node.args) > 3 else None
                qnode, reshape_mid = _find_quant_user(node) if (w is not None and var_ov is None) else (None, None)
                x_val = x.meta.get("val") if isinstance(x, fx.Node) else None
                w_val = w.meta.get("val") if isinstance(w, fx.Node) else None
                if (
                    qnode is None or x_val is None or w_val is None
                    or x_val.ndim != 2 or w_val.ndim != 1 or w_val.shape != (x_val.shape[1],)
                    or x_val.dtype not in (torch.bfloat16, torch.float16) or w_val.dtype != x_val.dtype
                ):
                    continue
                q_had, q_nsg = _extract_quant_args(qnode)
                with graph.inserting_before(qnode):
                    pair = graph.call_function(
                        torch.ops.flag_gems.rmsnorm_quant.default,
                        args=(x, w, float(eps), bool(q_had and self.cfg.norm_do_hadamard), q_nsg),
                    )
                    pair.meta["val"] = qnode.meta.get("val")
                    node_to_idx[pair] = node_to_idx.get(qnode, 0)
                qnode.replace_all_uses_with(pair)
                graph.erase_node(qnode)
                if reshape_mid is not None:
                    graph.erase_node(reshape_mid)
                graph.erase_node(node)
                fused += 1

            # Case 2: `vllm_ir.fused_add_rms_norm` -> `fused_add_rmsnorm_quant` (if
            # followed by `quant_input`) or in-place `fl_fused_add_rms_norm` (e.g., final norm).
            elif _is_op(node, add_rms_op):
                x, res, w, eps = node.args[0], node.args[1], node.args[2], node.args[3]
                var_ov = node.args[4] if len(node.args) > 4 else None
                if w is None or var_ov is not None or not (isinstance(x, fx.Node) and isinstance(res, fx.Node)):
                    continue
                x_val, res_val = x.meta.get("val"), res.meta.get("val")
                w_val = w.meta.get("val") if isinstance(w, fx.Node) else None
                if (
                    x_val is None or res_val is None or w_val is None
                    or x_val.ndim != 2 or res_val.shape != x_val.shape
                    or w_val.ndim != 1 or w_val.shape != (x_val.shape[1],)
                    or x_val.dtype not in (torch.bfloat16, torch.float16)
                    or res_val.dtype != x_val.dtype or w_val.dtype != x_val.dtype
                    or any(not _is_op(u, operator.getitem) for u in node.users)
                ):
                    continue
                umap = {int(u.args[1]): u for u in node.users}
                g0, g1 = umap.get(0), umap.get(1)
                qnode, reshape_mid = _find_quant_user(g0) if g0 is not None else (None, None)

                if qnode is not None:
                    q_had, q_nsg = _extract_quant_args(qnode)
                    res_arg = _ensure_mutable(res, node)
                    with graph.inserting_before(qnode):
                        pair = graph.call_function(
                            torch.ops.flag_gems.fused_add_rmsnorm_quant.default,
                            args=(x, res_arg, w, float(eps), bool(q_had and self.cfg.norm_do_hadamard), q_nsg),
                        )
                        pair.meta["val"] = qnode.meta.get("val")
                        node_to_idx[pair] = node_to_idx.get(qnode, 0)
                    qnode.replace_all_uses_with(pair)
                    if g1 is not None:
                        g1.replace_all_uses_with(res_arg)
                    graph.erase_node(qnode)
                    if reshape_mid is not None:
                        graph.erase_node(reshape_mid)
                    for u in list(node.users):
                        graph.erase_node(u)
                    graph.erase_node(node)
                    fused += 1
                elif x_val.dtype in (torch.bfloat16, torch.float16) and x_val.dtype == res_val.dtype:
                    cur_idx = node_to_idx.get(node, 0)
                    if any(node_to_idx.get(u, cur_idx + 1) > cur_idx for u in (*x.users, *res.users) if u is not node):
                        continue
                    x_arg, res_arg = _ensure_mutable(x, node), _ensure_mutable(res, node)
                    with graph.inserting_before(node):
                        fn_n = graph.call_function(torch.ops.vllm.fl_fused_add_rms_norm.default, args=(x_arg, res_arg, w, float(eps)))
                        fn_n.meta["val"] = None
                        node_to_idx[fn_n] = cur_idx
                    if g0 is not None:
                        g0.replace_all_uses_with(x_arg)
                    if g1 is not None:
                        g1.replace_all_uses_with(res_arg)
                    for u in list(node.users):
                        graph.erase_node(u)
                    graph.erase_node(node)
                    fused += 1
        node_to_idx = {n: idx for idx, n in enumerate(graph.nodes)}
        # Case 3: Decomposed primitive ATen RMSNorm / AddRMSNorm chains feeding `quant_input`.
        for qnode in list(graph.nodes):
            if _is_op(qnode, quant_op) and _try_fuse_decomposed_aten_rmsnorm(graph, qnode, node_to_idx, self.cfg, donated_ids):
                fused += 1
        return fused

    def _fuse_swiglu_and_quant(self, graph: fx.Graph) -> int:
        """Fuse `GateUp + SiluAndMul + quant_input` into GEMM epilogue or `silu_mul_quant`.

        Matches decomposed SwiGLU (`silu(x[:, :I]) * x[:, I:]`) feeding `quant_input`:
        - Interleaved GEMM epilogue path (`fuse_interleaved_swiglu=True`, `q_had=True`,
          and `src_node` is `quant_gemm_w8a8` with standard `epilogue_mode=1`):
          rewrites `quant_gemm_w8a8` to `epilogue_mode=2`, which computes SwiGLU and
          block-64 Walsh-Hadamard transform in registers before writing BF16 `[M, I]`
          (relying on offline 64-column `[Gate, Up]` weight interleaving), followed by
          lightweight `quant_input(mid_h64, do_hadamard=False, nsg_out)`.
        - Non-interleaved fallback: fuses SwiGLU + optional Hadamard + INT8 dynamic
          quantization into a single `flag_gems.silu_mul_quant` kernel.
        """
        quant_op, gemm_op = torch.ops.flag_gems.quant_input.default, torch.ops.flag_gems.quant_gemm_w8a8.default
        silu_quant_op = torch.ops.flag_gems.silu_mul_quant.default
        fused = 0
        for qnode in list(graph.nodes):
            if not _is_op(qnode, quant_op):
                continue
            q_had, q_nsg = _extract_quant_args(qnode)
            mul_node = _unwrap_reshape(qnode.args[0])
            if not _is_op(mul_node, torch.ops.aten.mul.Tensor) or len(mul_node.users) != 1:
                continue
            q_in_val = qnode.args[0].meta.get("val") if isinstance(qnode.args[0], fx.Node) else None
            if q_in_val is None:
                q_in_val = mul_node.meta.get("val")
            k_inter = int(q_in_val.shape[-1]) if (q_in_val is not None and q_in_val.ndim >= 1 and isinstance(q_in_val.shape[-1], int)) else 0
            cfg_nsg = int(self.cfg.down_num_scale_groups)
            if q_nsg > 1:
                nsg_out = q_nsg
            elif cfg_nsg != 1 and k_inter > 0:
                from vllm_fl.quantization.quant_transform import resolve_valid_scale_groups

                # Inspect the downstream `down_proj` W8A8 GEMM weight to obtain `out_n`
                # so `resolve_valid_scale_groups` can verify output-dimension divisibility.
                out_n = 0
                for u_get in qnode.users:
                    if _is_op(u_get, operator.getitem):
                        for u_cons in u_get.users:
                            if _is_op(u_cons, gemm_op) and len(u_cons.args) >= 3 and isinstance(u_cons.args[2], fx.Node):
                                w_val = u_cons.args[2].meta.get("val")
                                if w_val is not None and w_val.ndim == 2 and isinstance(w_val.shape[0], int):
                                    out_n = int(w_val.shape[0])
                                    break
                nsg_out = resolve_valid_scale_groups(
                    k_inter, cfg_nsg, 64 if (q_had and self.cfg.down_do_hadamard) else 0, out_n
                )
            else:
                nsg_out = max(q_nsg, 1)

            src_node: fx.Node | None = None
            for silu_cand, up_cand in ((mul_node.args[0], mul_node.args[1]), (mul_node.args[1], mul_node.args[0])):
                g_slice, u_slice = _match_silu_branch(silu_cand), _unwrap_cast(up_cand)
                if g_slice is None or not (_is_op(g_slice, torch.ops.aten.slice.Tensor) and _is_op(u_slice, torch.ops.aten.slice.Tensor)):
                    continue
                if g_slice.args[:2] != u_slice.args[:2] or not isinstance(g_slice.args[0], fx.Node):
                    continue
                s0 = 0 if len(g_slice.args) < 3 or g_slice.args[2] is None else int(g_slice.args[2])
                if s0 == 0 and int(g_slice.args[3]) == int(u_slice.args[2]) and set(g_slice.args[0].users) == {g_slice, u_slice}:
                    src_node = g_slice.args[0]
                    break
            if src_node is None:
                continue

            is_std_gemm = (
                _is_op(src_node, gemm_op)
                and len(src_node.args) > 6
                and int(src_node.args[6]) == 1
            )
            if self.cfg.fuse_interleaved_swiglu and q_had and is_std_gemm:
                with graph.inserting_before(qnode):
                    mid_h64 = graph.call_function(gemm_op, args=(*src_node.args[:6], 2))
                    mid_h64.meta["val"] = mul_node.meta.get("val")
                    new_q = graph.call_function(quant_op, args=(mid_h64, False, nsg_out))
                    new_q.meta["val"] = qnode.meta.get("val")
                qnode.replace_all_uses_with(new_q)
                graph.erase_node(qnode)
                fused += 1
            elif not is_std_gemm:
                src_val = src_node.meta.get("val")
                if src_val is None or src_val.ndim != 2:
                    continue
                with graph.inserting_before(qnode):
                    new_q = graph.call_function(silu_quant_op, args=(src_node, bool(q_had and self.cfg.down_do_hadamard), nsg_out))
                    new_q.meta["val"] = qnode.meta.get("val")
                qnode.replace_all_uses_with(new_q)
                graph.erase_node(qnode)
                fused += 1
        return fused

    def _fuse_attn_out_and_quant(self, graph: fx.Graph) -> int:
        """Fuse `auto_functionalized(unified_attention_with_output) + quant_input` into `fl_unified_attention_quant`.

        Eliminates the `auto_functionalized` higher-order op wrapper and its
        `aten.empty` output buffer allocation (left dead and cleaned up by DCE),
        forwarding `(query, key, value, layer_name, kv_cache_dummy_dep)` directly
        into `vllm.fl_unified_attention_quant` so Pure Decode steps can emit
        quantized `(out_q, out_s)` directly from the attention kernel.
        """
        quant_op = torch.ops.flag_gems.quant_input.default
        attn_op = getattr(getattr(torch.ops, "vllm", None), "unified_attention_with_output", None)
        attn_default = getattr(attn_op, "default", None)
        if attn_default is None:
            return 0
        af_ops = tuple(
            op
            for op in (
                getattr(torch.ops.higher_order, "auto_functionalized", None),
                getattr(torch.ops.higher_order, "auto_functionalized_v2", None),
            )
            if op is not None
        )
        fused_op = torch.ops.vllm.fl_unified_attention_quant.default
        fused = 0
        for qnode in list(graph.nodes):
            if not _is_op(qnode, quant_op):
                continue
            src = qnode.args[0]
            view_nodes: list[fx.Node] = []
            while _is_op(src, torch.ops.aten.reshape.default, torch.ops.aten.view.default, torch.ops.aten.alias.default):
                if len(src.users) != 1:
                    break
                view_nodes.append(src)
                src = src.args[0]
            # `auto_functionalized` returns `(None, mutated_output)`; index 1 is the attention output.
            if not (_is_op(src, operator.getitem) and len(src.users) == 1 and src.args[1] == 1):
                continue
            af_node = src.args[0]
            if not (
                bool(af_ops)
                and _is_op(af_node, *af_ops)
                and len(af_node.args) >= 1
                and af_node.args[0] == attn_default
                and all(u is src or len(u.users) == 0 for u in af_node.users)
            ):
                continue
            if af_node.kwargs.get("output_scale") is not None or af_node.kwargs.get("output_block_scale") is not None:
                continue
            q_3d, k_3d, v_3d = af_node.kwargs.get("query"), af_node.kwargs.get("key"), af_node.kwargs.get("value")
            layer_name, kv_dep = af_node.kwargs.get("layer_name"), af_node.kwargs.get("kv_cache_dummy_dep")
            if not (isinstance(q_3d, fx.Node) and isinstance(k_3d, fx.Node) and isinstance(v_3d, fx.Node) and layer_name is not None):
                continue
            q_val, v_val, qin_val = q_3d.meta.get("val"), v_3d.meta.get("val"), qnode.args[0].meta.get("val")
            if (
                q_val is None or v_val is None or qin_val is None
                or q_val.ndim != 3 or v_val.ndim != 3 or qin_val.ndim != 2
                or int(qin_val.shape[1]) != int(q_val.shape[1]) * int(v_val.shape[2])
            ):
                continue
            q_had, q_nsg = _extract_quant_args(qnode)
            with graph.inserting_before(qnode):
                pair = graph.call_function(
                    fused_op,
                    args=(q_3d, k_3d, v_3d, layer_name, kv_dep, bool(q_had), int(q_nsg)),
                )
                pair.meta["val"] = qnode.meta.get("val")
            qnode.replace_all_uses_with(pair)
            graph.erase_node(qnode)
            for vn in view_nodes:
                graph.erase_node(vn)
            for u in list(af_node.users):
                graph.erase_node(u)
            graph.erase_node(af_node)
            fused += 1
        return fused


__all__ = ["QuantFusionConfig", "QuantFusionPass"]
