# SPDX-License-Identifier: Apache-2.0
"""`RopeKvFusionPass`: Vendor-neutral post-grad FX `InductorPass` fusing NeoX RoPE + KV cache write.

Matches AOTAutograd-decomposed NeoX rotary position embedding (`RoPE`) subgraphs
on `(query, key)` together with the layer's KV-cache write (either an explicit
`vllm.unified_kv_cache_update` node or an inline attention backend KV-cache
update) and replaces the 4+ separate kernels with a single custom op dispatch
(`vllm::fl_rope_and_kv_cache_update` by default).
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import operator
from typing import Any

import torch
from torch import fx
from vllm.compilation.passes.inductor_pass import InductorPass
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.utils.torch_utils import LayerNameType, _resolve_layer_name, direct_register_custom_op

from .quant_fusion_pass import _is_op, _unwrap_cast, _unwrap_single_cast

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.fl.rope_kv_fusion_pass")
except Exception:
    logger = logging.getLogger(__name__)

_GET_ATTN_CTX_FN: Any = None
_EMPTY_DEP_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_empty_dep(query: torch.Tensor) -> torch.Tensor:
    """Return a cached 0-element tensor on `(query.device, query.dtype)` for FX dependency ordering."""
    dep_key = (query.device, query.dtype)
    if (dep := _EMPTY_DEP_CACHE.get(dep_key)) is None:
        if len(_EMPTY_DEP_CACHE) >= 16:
            _EMPTY_DEP_CACHE.pop(next(iter(_EMPTY_DEP_CACHE)))
        dep = _EMPTY_DEP_CACHE[dep_key] = torch.empty(
            0, device=query.device, dtype=query.dtype
        )
    return dep


def dispatch_layer_rope_and_kv_cache_update(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool, layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Delegate fused Q/K NeoX RoPE + paged KV-cache write to `attn_layer.impl`.

    Returns `(q_rot, empty_dep)`, where `empty_dep` is a zero-element tensor
    cached per `(device, dtype)` and threaded into downstream attention's
    `kv_cache_dummy_dep` argument so FX / Inductor preserves strict topological
    ordering between KV-cache mutation and attention reads without `mutates_args`
    functionalization overhead.
    """
    global _GET_ATTN_CTX_FN
    if _GET_ATTN_CTX_FN is None:
        from vllm.model_executor.layers.attention.attention import get_attention_context

        _GET_ATTN_CTX_FN = get_attention_context
    attn_metadata, attn_layer, kv_cache, slot_mapping = _GET_ATTN_CTX_FN(_resolve_layer_name(layer_name))
    q_rot = attn_layer.impl.do_rope_and_kv_cache_update(
        attn_layer, query, key, value, positions, cos_sin_cache, is_neox, kv_cache, slot_mapping, attn_metadata
    )
    return q_rot, _get_empty_dep(query)


def fake_layer_rope_and_kv_cache_update(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool, layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FakeTensor meta shape/dtype inference for `vllm::fl_rope_and_kv_cache_update`."""
    return query.new_empty((query.shape[0], query.shape[1], query.shape[2])), query.new_empty((0,))


def dispatch_layer_qk_norm_rope_and_kv_cache_update(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor, rms_eps: float,
    positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool, layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Delegate fused per-head QK-Norm + Q/K NeoX RoPE + paged KV-cache write to `attn_layer.impl`."""
    global _GET_ATTN_CTX_FN
    if _GET_ATTN_CTX_FN is None:
        from vllm.model_executor.layers.attention.attention import get_attention_context

        _GET_ATTN_CTX_FN = get_attention_context
    attn_metadata, attn_layer, kv_cache, slot_mapping = _GET_ATTN_CTX_FN(_resolve_layer_name(layer_name))
    q_rot = attn_layer.impl.do_qk_norm_rope_and_kv_cache_update(
        attn_layer, query, key, value, q_norm_weight, k_norm_weight, rms_eps,
        positions, cos_sin_cache, is_neox, kv_cache, slot_mapping, attn_metadata,
    )
    return q_rot, _get_empty_dep(query)


def fake_layer_qk_norm_rope_and_kv_cache_update(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor, rms_eps: float,
    positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool, layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FakeTensor meta shape/dtype inference for `vllm::fl_qk_norm_rope_and_kv_cache_update`."""
    return query.new_empty((query.shape[0], query.shape[1], query.shape[2])), query.new_empty((0,))


if not hasattr(torch.ops.vllm, "fl_rope_and_kv_cache_update"):
    direct_register_custom_op(
        op_name="fl_rope_and_kv_cache_update",
        op_func=dispatch_layer_rope_and_kv_cache_update,
        mutates_args=[],
        fake_impl=fake_layer_rope_and_kv_cache_update,
    )

if not hasattr(torch.ops.vllm, "fl_qk_norm_rope_and_kv_cache_update"):
    direct_register_custom_op(
        op_name="fl_qk_norm_rope_and_kv_cache_update",
        op_func=dispatch_layer_qk_norm_rope_and_kv_cache_update,
        mutates_args=[],
        fake_impl=fake_layer_qk_norm_rope_and_kv_cache_update,
    )


def _unwrap_full_last_slice(node: Any, head_dim: int) -> fx.Node:
    """Peel no-op full-width `slice(..., 0, head_dim)` and `alias` wrappers on the last dimension."""
    while True:
        if _is_op(node, torch.ops.aten.alias.default):
            node = node.args[0]
        elif _is_op(node, torch.ops.aten.slice.Tensor) and node.args[1] in (-1, 2):
            s0 = 0 if len(node.args) < 3 or node.args[2] is None else int(node.args[2])
            if s0 == 0 and int(node.args[3]) >= head_dim:
                node = node.args[0]
                continue
            break
        else:
            break
    return node


def _match_half_split_pair(x1: Any, x2: Any) -> tuple[fx.Node, int] | None:
    """Match equal-half split or contiguous slice pair `(x1, x2)` back to `(source_node, full_dim)`."""
    x1, x2 = _unwrap_cast(x1), _unwrap_cast(x2)
    if _is_op(x1, operator.getitem) and _is_op(x2, operator.getitem) and x1.args[0] is x2.args[0] and x1.args[1] == 0 and x2.args[1] == 1:
        sp = x1.args[0]
        if _is_op(sp, torch.ops.aten.split.Tensor) and sp.args[2] in (-1, 2, 1):
            return sp.args[0], int(sp.args[1]) * 2
        if _is_op(sp, torch.ops.aten.split_with_sizes.default) and sp.args[2] in (-1, 2, 1) and len(sp.args[1]) == 2 and sp.args[1][0] == sp.args[1][1]:
            return sp.args[0], int(sp.args[1][0]) * 2
    if _is_op(x1, torch.ops.aten.slice.Tensor) and _is_op(x2, torch.ops.aten.slice.Tensor) and x1.args[:2] == x2.args[:2]:
        s0 = 0 if len(x1.args) < 3 or x1.args[2] is None else int(x1.args[2])
        if s0 == 0 and int(x1.args[3]) == int(x2.args[2]) and int(x1.args[3]) > 0:
            return x1.args[0], int(x1.args[3]) * 2
    return None


def _match_mul_pair(mul_node: Any, target_x: Any) -> fx.Node | None:
    """If `mul_node` multiplies `target_x` with another operand, return that other operand."""
    if not _is_op(mul_node, torch.ops.aten.mul.Tensor):
        return None
    a, b, t = _unwrap_cast(mul_node.args[0]), _unwrap_cast(mul_node.args[1]), _unwrap_cast(target_x)
    return mul_node.args[1] if a is t else (mul_node.args[0] if b is t else None)


def _match_neox_rope_branch(attn_in_3d: fx.Node, max_outer_users: int) -> tuple[fx.Node, fx.Node, fx.Node, int] | None:
    """Match a decomposed NeoX RoPE rotation tree producing `attn_in_3d`.

    Recognizes the canonical half-rotated NeoX identity:
        `cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)`
    where `(x1, x2)` are equal-half splits/slices of `raw_3d` along `head_dim`,
    transparently unwrapping single-user view/clone/cast nodes and zero-width
    rotary pass-through `cat` suffixes emitted when `rotary_dim == head_dim`.
    Returns `(raw_3d, cos_node, sin_node, head_dim)` on match, else `None`.
    """
    if len(attn_in_3d.users) > max_outer_users:
        return None
    cur: Any = attn_in_3d
    while True:
        if _is_op(cur, torch.ops.aten.reshape.default, torch.ops.aten.view.default, torch.ops.aten.alias.default, torch.ops.aten.clone.default):
            if cur is not attn_in_3d and len(cur.users) > 1:
                return None
            cur = cur.args[0]
        elif (
            _is_op(cur, torch.ops.aten.cat.default) and len(cur.args[0]) == 2
            and (len(cur.args) < 2 or cur.args[1] in (-1, 2))
            and isinstance(cur.args[0][1], fx.Node) and getattr(cur.args[0][1].meta.get("val"), "shape", (1,))[-1] == 0
        ):
            if cur is not attn_in_3d and len(cur.users) > 1:
                return None
            cur = cur.args[0][0]
        else:
            break
    cur = _unwrap_cast(cur)
    if not _is_op(cur, torch.ops.aten.cat.default) or len(cur.users) > (max_outer_users if cur is attn_in_3d else 1):
        return None
    parts, dim = cur.args[0], (-1 if len(cur.args) < 2 else cur.args[1])
    if len(parts) != 2 or dim not in (-1, 2):
        return None
    sub_n, add_n = _unwrap_cast(parts[0]), _unwrap_cast(parts[1])
    if not (_is_op(sub_n, torch.ops.aten.sub.Tensor) and _is_op(add_n, torch.ops.aten.add.Tensor)):
        return None
    m_c1, m_s2 = sub_n.args[0], sub_n.args[1]
    if not (_is_op(m_c1, torch.ops.aten.mul.Tensor) and _is_op(m_s2, torch.ops.aten.mul.Tensor)):
        return None
    for idx1 in (0, 1):
        cand_x1, cand_cos = m_c1.args[idx1], m_c1.args[1 - idx1]
        for idx2 in (0, 1):
            cand_x2, cand_sin = m_s2.args[idx2], m_s2.args[1 - idx2]
            if (sp := _match_half_split_pair(cand_x1, cand_x2)) is None:
                continue
            raw_3d, head_dim = _unwrap_full_last_slice(sp[0], sp[1]), sp[1]
            a0, a1 = add_n.args[0], add_n.args[1]
            for p0, p1 in (((cand_x2, cand_cos), (cand_x1, cand_sin)), ((cand_x1, cand_sin), (cand_x2, cand_cos))):
                c0, c1 = _match_mul_pair(a0, p0[0]), _match_mul_pair(a1, p1[0])
                if c0 is not None and c1 is not None and _unwrap_cast(c0) is _unwrap_cast(p0[1]) and _unwrap_cast(c1) is _unwrap_cast(p1[1]):
                    return raw_3d, _unwrap_cast(cand_cos), _unwrap_cast(cand_sin), head_dim
    return None


def _match_cos_sin_lookup(cos_u: fx.Node, sin_u: fx.Node, head_dim: int) -> tuple[fx.Node, fx.Node] | None:
    """Match `(cos, sin)` half-splits back to a shared `cos_sin_cache[positions]` lookup."""
    c = _unwrap_cast(cos_u.args[0]) if _is_op(cos_u, torch.ops.aten.unsqueeze.default) else cos_u
    s = _unwrap_cast(sin_u.args[0]) if _is_op(sin_u, torch.ops.aten.unsqueeze.default) else sin_u
    if (sp := _match_half_split_pair(c, s)) is None or sp[1] != head_dim:
        return None
    cs_sel = _unwrap_cast(sp[0])
    if _is_op(cs_sel, torch.ops.aten.index.Tensor) and len(cs_sel.args[1]) == 1:
        cs_cache, pos = cs_sel.args[0], cs_sel.args[1][0]
    elif _is_op(cs_sel, torch.ops.aten.index_select.default) and cs_sel.args[1] == 0:
        cs_cache, pos = cs_sel.args[0], cs_sel.args[2]
    else:
        return None
    if not (isinstance(cs_cache, fx.Node) and isinstance(pos, fx.Node)):
        return None
    cs_val, pos_val = cs_cache.meta.get("val"), pos.meta.get("val")
    if (
        cs_val is None
        or pos_val is None
        or cs_val.ndim != 2
        or cs_val.shape[1] != head_dim
        or cs_val.dtype not in (torch.bfloat16, torch.float16)
        or pos_val.ndim != 1
    ):
        return None
    return pos, cs_cache


def _all_live_paths_reach(src: fx.Node, sink: fx.Node, max_visits: int = 32) -> bool:
    """Return True iff `src` has users and every live consumer chain terminates at `sink`."""
    if not src.users:
        return False
    stack, seen, reached = list(src.users), {src}, False
    while stack:
        u = stack.pop()
        if u is sink:
            reached = True
            continue
        if u in seen:
            continue
        seen.add(u)
        if len(seen) > max_visits or u.op != "call_function":
            return False
        if not u.users:
            continue
        stack.extend(u.users)
    return reached


def _match_per_head_rmsnorm_3d(raw_3d: fx.Node, attn_in_3d: fx.Node, head_dim: int) -> tuple[fx.Node, fx.Node, float] | None:
    """Match an upstream per-head 3D `RMSNorm(..., head_dim)` feeding only `attn_in_3d`."""
    if not _all_live_paths_reach(raw_3d, attn_in_3d):
        return None
    prev: Any = None
    cur: Any = raw_3d
    while _is_op(cur, torch.ops.aten.reshape.default, torch.ops.aten.view.default, torch.ops.aten.alias.default):
        if cur is not raw_3d and any(len(u.users) > 0 for u in cur.users if u is not prev):
            return None
        prev, cur = cur, cur.args[0]

    rms_ir_op = getattr(getattr(getattr(torch.ops, "vllm_ir", None), "rms_norm", None), "default", None)
    if rms_ir_op is not None and _is_op(cur, rms_ir_op):
        if cur is not raw_3d and sum(1 for u in cur.users if u.users) != 1:
            return None
        if len(cur.args) > 3 and cur.args[3] is not None:
            return None
        x_in, w_in, eps = cur.args[0], cur.args[1], float(cur.args[2])
    else:
        mul_out = _unwrap_single_cast(cur)
        if not _is_op(mul_out, torch.ops.aten.mul.Tensor) or (mul_out is not raw_3d and sum(1 for u in mul_out.users if u.users) != 1):
            return None
        leaves: tuple[Any, Any, Any] | None = None
        for i_in in (0, 1):
            cand_in = _unwrap_single_cast(mul_out.args[i_in])
            if _is_op(cand_in, torch.ops.aten.mul.Tensor) and len(cand_in.users) == 1:
                leaves, mul_inner = (cand_in.args[0], cand_in.args[1], mul_out.args[1 - i_in]), cand_in
                break
        if leaves is None:
            return None
        rstd_idx = next((i for i, lf in enumerate(leaves) if _is_op(_unwrap_single_cast(lf), torch.ops.aten.rsqrt.default) and len(_unwrap_single_cast(lf).users) == 1), -1)
        if rstd_idx < 0:
            return None
        eps_add = _unwrap_single_cast(leaves[rstd_idx]).args[0]
        if not _is_op(eps_add, torch.ops.aten.add.Tensor) or len(eps_add.users) != 1 or eps_add.kwargs.get("alpha", 1) != 1:
            return None
        if isinstance(eps_add.args[1], (int, float)) and not isinstance(eps_add.args[1], bool):
            var_n, eps = eps_add.args[0], float(eps_add.args[1])
        elif isinstance(eps_add.args[0], (int, float)) and not isinstance(eps_add.args[0], bool):
            var_n, eps = eps_add.args[1], float(eps_add.args[0])
        else:
            return None
        keepdim = bool(var_n.args[2]) if (isinstance(var_n, fx.Node) and len(var_n.args) > 2) else bool(getattr(var_n, "kwargs", {}).get("keepdim", False))
        if not _is_op(var_n, torch.ops.aten.mean.dim) or len(var_n.users) != 1 or not keepdim or list(var_n.args[1]) not in ([-1], [2]):
            return None
        sq_n = var_n.args[0]
        if not isinstance(sq_n, fx.Node) or len(sq_n.users) != 1:
            return None
        if _is_op(sq_n, torch.ops.aten.pow.Tensor_Scalar) and sq_n.args[1] in (2, 2.0):
            xf_var = sq_n.args[0]
        elif _is_op(sq_n, torch.ops.aten.mul.Tensor) and sq_n.args[0] is sq_n.args[1]:
            xf_var = sq_n.args[0]
        else:
            return None
        rem = [leaves[i] for i in range(3) if i != rstd_idx]
        x_in = _unwrap_cast(xf_var)
        if _unwrap_cast(rem[0]) is x_in:
            xf_mul, w_raw = rem[0], rem[1]
        elif _unwrap_cast(rem[1]) is x_in:
            xf_mul, w_raw = rem[1], rem[0]
        else:
            return None
        w_in = _unwrap_cast(w_raw)
        norm_consumers = {sq_n, mul_inner, mul_out}
        for cast_n in (xf_var, xf_mul, w_raw):
            if isinstance(cast_n, fx.Node) and cast_n not in (x_in, w_in):
                if any(u not in norm_consumers and not (_is_op(u, torch.ops.prims.convert_element_type.default) and set(u.users) <= norm_consumers) for u in cast_n.users):
                    return None

    if not (isinstance(x_in, fx.Node) and isinstance(w_in, fx.Node)):
        return None
    x_val, w_val = x_in.meta.get("val"), w_in.meta.get("val")
    if (
        x_val is None or w_val is None or x_val.ndim != 3 or x_val.shape[2] != head_dim
        or x_val.dtype not in (torch.bfloat16, torch.float16) or x_val.stride(1) != head_dim or x_val.stride(2) != 1
        or w_val.ndim != 1 or w_val.shape != (head_dim,) or w_val.dtype != x_val.dtype or w_val.stride(0) != 1
    ):
        return None
    return x_in, w_in, eps


def _parse_rope_kv_op_spec(rope_kv_op: Any = None) -> tuple[str, str, str]:
    """Normalize a custom op handle or string into a picklable `(ns, op, overload)` tuple.

    During `vllm serve` multiprocess startup (`APIServer -> EngineCore` under
    `spawn`), vLLM serializes `vllm_config` (including `inductor_compile_config`
    and its attached `InductorPass` instances) via `cloudpickle.dumps(vllm_config)`.
    C++ `torch._ops.OpOverload` / `OpOverloadPacket` objects cannot be pickled
    across process boundaries, so `RopeKvFusionPass` stores only a pure-Python
    `tuple[str, str, str]` and resolves `torch.ops.<ns>.<op>.<overload>` lazily
    inside the worker process when compiling the FX graph.
    """
    if rope_kv_op is None:
        return ("vllm", "fl_rope_and_kv_cache_update", "default")
    if isinstance(rope_kv_op, tuple) and len(rope_kv_op) == 3 and all(isinstance(s, str) for s in rope_kv_op):
        return (str(rope_kv_op[0]), str(rope_kv_op[1]), str(rope_kv_op[2]))
    if isinstance(rope_kv_op, str):
        s = rope_kv_op.strip()
        if "::" in s:
            ns, rest = s.split("::", 1)
            op_name, overload = rest.split(".", 1) if "." in rest else (rest, "default")
            return (str(ns), str(op_name), str(overload))
        parts = [p for p in s.split(".") if p]
        if parts[:2] == ["torch", "ops"]:
            parts = parts[2:]
        if len(parts) == 2:
            return (str(parts[0]), str(parts[1]), "default")
        if len(parts) == 3:
            return (str(parts[0]), str(parts[1]), str(parts[2]))
        raise ValueError(f"Invalid rope_kv_op string spec: {rope_kv_op!r}")
    name_fn = getattr(rope_kv_op, "name", None)
    if callable(name_fn):
        qname = str(name_fn())
        if "::" in qname:
            ns, rest = qname.split("::", 1)
            op_name = rest.split(".", 1)[0]
            overload = str(getattr(rope_kv_op, "_overloadname", "") or "default")
            return (str(ns), str(op_name), overload)
    return _parse_rope_kv_op_spec(str(rope_kv_op))


class RopeKvFusionPass(InductorPass):
    """Orthogonal FX pass fusing NeoX RoPE + KV-cache write across dual-chip backends.

    Stores `rope_kv_op_spec` as a cloudpickle-safe `(namespace, op_name, overload)`
    3-tuple and supports both attention consumer forms (`unified_attention_with_output`
    and `fl_unified_attention_quant` emitted by `QuantFusionPass`) as well as both
    KV-cache update conventions (`has_kv_upd` explicit `unified_kv_cache_update` vs
    `inline_kv_upd` where `attn_backend.forward_includes_kv_cache_update` is True).
    """

    def __init__(
        self,
        rope_kv_op: str | tuple[str, str, str] | Callable[..., Any] | None = None,
        qk_norm_rope_kv_op: str | tuple[str, str, str] | Callable[..., Any] | None = None,
    ) -> None:
        """Initialize pass with cloudpickle-safe op specs for RoPE+KV and optional QK-Norm+RoPE+KV."""
        self.rope_kv_op_spec: tuple[str, str, str] = _parse_rope_kv_op_spec(rope_kv_op)
        self.qk_norm_rope_kv_op_spec: tuple[str, str, str] | None = (
            _parse_rope_kv_op_spec(qk_norm_rope_kv_op) if qk_norm_rope_kv_op is not None else None
        )

    def _resolve_rope_kv_op(self) -> Any:
        """Lazily resolve `(ns, op_name, overload)` to the live `torch.ops` overload."""
        ns, op_name, overload = self.rope_kv_op_spec
        return getattr(getattr(getattr(torch.ops, ns), op_name), overload)

    def _resolve_qk_norm_rope_kv_op(self) -> Any | None:
        """Lazily resolve optional QK-Norm+RoPE+KV op spec to its live `torch.ops` overload."""
        if self.qk_norm_rope_kv_op_spec is None:
            return None
        ns, op_name, overload = self.qk_norm_rope_kv_op_spec
        return getattr(getattr(getattr(torch.ops, ns), op_name), overload)

    def uuid(self) -> str:
        """Return a JSON-safe digest hashing pass source, matchers, and target op spec."""
        return InductorPass.hash_dict({
            "src": InductorPass.hash_source(self, _match_neox_rope_branch, _match_cos_sin_lookup, _match_per_head_rmsnorm_3d),
            "op": list(self.rope_kv_op_spec),
            "qkn_op": list(self.qk_norm_rope_kv_op_spec) if self.qk_norm_rope_kv_op_spec is not None else None,
        })

    def __call__(self, graph: fx.Graph) -> None:
        """Fuse matched NeoX RoPE (+ optional QK-Norm) + KV-cache update subgraphs and run DCE."""
        n_fused, n_qkn = self._fuse_rope_and_kv_cache(graph)
        if n_fused:
            graph.eliminate_dead_code()
            if n_qkn:
                logger.info("[FL-RopeKVFusion] RopeKvFusionPass fused %d RoPE+KV-Cache ops (%d with fused QK-Norm).", n_fused, n_qkn)
            else:
                logger.info("[FL-RopeKVFusion] RopeKvFusionPass fused %d RoPE+KV-Cache ops.", n_fused)

    def _fuse_rope_and_kv_cache(self, graph: fx.Graph) -> tuple[int, int]:
        """Match attention nodes and rewrite upstream Q/K RoPE (+ optional QK-Norm) + KV write."""
        rope_kv_op = self._resolve_rope_kv_op()
        qkn_rope_kv_op = self._resolve_qk_norm_rope_kv_op()
        attn_op, kv_upd_op = torch.ops.vllm.unified_attention_with_output.default, torch.ops.vllm.unified_kv_cache_update.default
        attn_q_op = getattr(getattr(torch.ops.vllm, "fl_unified_attention_quant", None), "default", None)
        af_ops = tuple(op for op in (getattr(torch.ops.higher_order, "auto_functionalized", None), getattr(torch.ops.higher_order, "auto_functionalized_v2", None)) if op is not None)
        ctx_layers: dict[str, Any] = {}
        try:
            if is_forward_context_available():
                ctx_layers = get_forward_context().no_compile_layers or {}
        except Exception:
            ctx_layers = {}
        if not ctx_layers:
            try:
                ctx_layers = get_current_vllm_config().compilation_config.static_forward_context or {}
            except Exception:
                ctx_layers = {}

        fused, fused_qkn = 0, 0
        for attn_node in list(graph.nodes):
            # Support three attention call shapes:
            # 1) `is_af`: `auto_functionalized(unified_attention_with_output)`
            # 2) `is_aq`: `fl_unified_attention_quant` (already rewritten by `QuantFusionPass`)
            # 3) raw `unified_attention_with_output` call
            is_af = bool(af_ops) and _is_op(attn_node, *af_ops) and len(attn_node.args) >= 1 and attn_node.args[0] == attn_op
            is_aq = attn_q_op is not None and _is_op(attn_node, attn_q_op) and len(attn_node.args) >= 5
            if is_af:
                q_3d, k_3d, v_3d = attn_node.kwargs.get("query"), attn_node.kwargs.get("key"), attn_node.kwargs.get("value")
                layer_name, kv_dep = attn_node.kwargs.get("layer_name"), attn_node.kwargs.get("kv_cache_dummy_dep")
            elif is_aq:
                q_3d, k_3d, v_3d, layer_name, kv_dep = attn_node.args[:5]
            elif _is_op(attn_node, attn_op) and len(attn_node.args) >= 5:
                q_3d, k_3d, v_3d, _, layer_name = attn_node.args[:5]
                kv_dep = attn_node.args[7] if len(attn_node.args) > 7 else attn_node.kwargs.get("kv_cache_dummy_dep")
            else:
                continue
            if not (isinstance(q_3d, fx.Node) and isinstance(k_3d, fx.Node) and isinstance(v_3d, fx.Node)):
                continue
            attn_layer = ctx_layers.get(_resolve_layer_name(layer_name)) if (ctx_layers and layer_name is not None) else None
            # Dual-backend KV-cache update handling:
            # - `has_kv_upd`: backends with `forward_includes_kv_cache_update=False`
            #   (e.g., FlashAttention) emit an explicit `unified_kv_cache_update(k, v)`
            #   node passed via `kv_cache_dummy_dep`; rotated `k_3d` has 2 users.
            # - `inline_kv_upd`: backends with `forward_includes_kv_cache_update=True`
            #   (e.g., TritonAttention) update KV inside `forward` when not pre-populated
            #   and leave `kv_dep=None`; rotated `k_3d` has 1 user.
            has_kv_upd = _is_op(kv_dep, kv_upd_op) and len(kv_dep.users) == 1 and kv_dep.args[:2] == (k_3d, v_3d)
            inline_kv_upd = (
                (kv_dep is None) and (attn_layer is not None)
                and bool(getattr(getattr(attn_layer, "attn_backend", None), "forward_includes_kv_cache_update", False))
            )
            if not (has_kv_upd or inline_kv_upd):
                continue
            impl = getattr(attn_layer, "impl", None) if attn_layer is not None else None
            if attn_layer is not None:
                if not getattr(impl, "_fl_layer_ok", False) or getattr(impl, "kv_sharing_target_layer_name", None) is not None or not hasattr(impl, "do_rope_and_kv_cache_update"):
                    continue

            k_max_users = 2 if has_kv_upd else 1
            qm, km = _match_neox_rope_branch(q_3d, max_outer_users=1), _match_neox_rope_branch(k_3d, max_outer_users=k_max_users)
            if qm is None or km is None or qm[3] != km[3] or qm[3] not in (64, 128, 256):
                continue
            q_raw_3d, cos_q, sin_q, hd_q = qm
            k_raw_3d, cos_k, sin_k, _ = km
            if (cs_q := _match_cos_sin_lookup(cos_q, sin_q, hd_q)) is None or cs_q != _match_cos_sin_lookup(cos_k, sin_k, hd_q):
                continue
            pos_1d, cos_sin_cache = cs_q
            q_val, qr_val, kr_val = q_3d.meta.get("val"), q_raw_3d.meta.get("val"), k_raw_3d.meta.get("val")
            if (
                q_val is None
                or qr_val is None
                or kr_val is None
                or qr_val.ndim != 3
                or kr_val.ndim != 3
                or q_val.dtype not in (torch.bfloat16, torch.float16)
                or qr_val.dtype != q_val.dtype
                or kr_val.dtype != q_val.dtype
            ):
                continue

            # Optionally absorb upstream per-head QK-Norm (e.g., Qwen3) when both Q and K
            # share identical RMSNorm epsilon and the backend implements fused QK-Norm+RoPE+KV.
            qkn_match: tuple[tuple[fx.Node, fx.Node, float], tuple[fx.Node, fx.Node, float]] | None = None
            if qkn_rope_kv_op is not None and (attn_layer is None or hasattr(impl, "do_qk_norm_rope_and_kv_cache_update")):
                qn_m = _match_per_head_rmsnorm_3d(q_raw_3d, q_3d, hd_q)
                kn_m = _match_per_head_rmsnorm_3d(k_raw_3d, k_3d, hd_q) if qn_m is not None else None
                if qn_m is not None and kn_m is not None and abs(qn_m[2] - kn_m[2]) <= 1e-12:
                    qkn_match = (qn_m, kn_m)

            dep_val = kv_dep.meta.get("val") if has_kv_upd else q_val.new_empty((0,))
            with graph.inserting_before(attn_node):
                if qkn_match is not None:
                    (q_un_3d, q_nw, qk_eps), (k_un_3d, k_nw, _) = qkn_match
                    rope_kv = graph.call_function(
                        qkn_rope_kv_op,
                        args=(q_un_3d, k_un_3d, v_3d, q_nw, k_nw, float(qk_eps), pos_1d, cos_sin_cache, True, layer_name),
                    )
                    k_dummy = k_un_3d
                    fused_qkn += 1
                else:
                    rope_kv = graph.call_function(rope_kv_op, args=(q_raw_3d, k_raw_3d, v_3d, pos_1d, cos_sin_cache, True, layer_name))
                    k_dummy = k_raw_3d
                q_rot_3d = graph.call_function(operator.getitem, args=(rope_kv, 0))
                kv_dep_new = graph.call_function(operator.getitem, args=(rope_kv, 1))
                q_rot_3d.meta["val"], kv_dep_new.meta["val"] = q_val, dep_val
                rope_kv.meta["val"] = (q_val, dep_val)

            if is_af:
                attn_node.kwargs = {**attn_node.kwargs, "query": q_rot_3d, "key": k_dummy, "kv_cache_dummy_dep": kv_dep_new}
            elif is_aq:
                new_args = list(attn_node.args)
                new_args[0], new_args[1], new_args[4] = q_rot_3d, k_dummy, kv_dep_new
                attn_node.args = tuple(new_args)
            else:
                new_args = list(attn_node.args)
                new_args[0], new_args[1] = q_rot_3d, k_dummy
                if len(new_args) > 7:
                    new_args[7] = kv_dep_new
                    attn_node.args = tuple(new_args)
                    if "kv_cache_dummy_dep" in attn_node.kwargs:
                        attn_node.kwargs = {k: v for k, v in attn_node.kwargs.items() if k != "kv_cache_dummy_dep"}
                else:
                    attn_node.args = tuple(new_args)
                    attn_node.kwargs = {**attn_node.kwargs, "kv_cache_dummy_dep": kv_dep_new}
            if has_kv_upd:
                graph.erase_node(kv_dep)
            fused += 1
        return fused, fused_qkn


__all__ = [
    "RopeKvFusionPass",
    "dispatch_layer_qk_norm_rope_and_kv_cache_update",
    "dispatch_layer_rope_and_kv_cache_update",
    "fake_layer_qk_norm_rope_and_kv_cache_update",
    "fake_layer_rope_and_kv_cache_update",
]
