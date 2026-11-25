# Copyright (c) 2026 BAAI. All rights reserved.
"""GLM-5.2 evas MLA backend — subclass attention + metadata/builder/impl.

Option C: ``EvasGLMMLAAttention`` subclasses upstream ``DeepseekV2MLAAttention``,
installed by ``_model_mla``; it runs the upstream ``__init__`` and overrides
``forward`` for the LPU path (real compute here). Also provides
``EvasGLMMLABackend`` slot class + metadata/builder/impl per ``kv_cache_spec``
group, sharing MLA choices (block 256, bf16, ``"CUSTOM"``) from
``EvasMLAAttentionBackend``, overriding the GLM 4D layout
``(num_blocks, 1, block_size, head_size)`` + no-op impl stub.

"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import torch

from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLAAttention as _UpstreamDeepseekV2MLAAttention,
)
from vllm.v1.attention.backend import (
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
)

from vllm_fl.dispatch.backends.vendor.evas.impl.attention.mla_attention import (
    EvasMLAAttentionBackend,
)

# Numeric guards on the GLM MLA path catch NaN/inf collapse (DSA + dummy
# weights, or bf16 overflow).  Enabled by default; ``VLLM_FL_GLM_GUARD=0``
# gates only their host reads (the diagnostics), not the unconditional
# clamp/nan_to_num, so disabling is numerically transparent.
_GLM_GUARD_ON = os.environ.get("VLLM_FL_GLM_GUARD", "1") == "1"
_HS_CLAMP_BOUND = 100000.0


def is_glm_config(config) -> bool:
    """True iff ``config`` is a GLM-5.2 HF config (``model_type == "glm_moe_dsa"``).

    Accepts the raw HF config (has ``model_type``) or a ``VllmConfig``
    (unwrapped via ``.model_config.hf_config``). Guards the shared
    ``DeepseekV2MLAAttention``/``Indexer`` rebinding so deepseek-v3.1 and
    other models sharing those classes keep pure upstream behavior.
    """
    hf = getattr(config, "model_type", None)
    if hf is None:
        mc = getattr(config, "model_config", None)
        hf = getattr(getattr(mc, "hf_config", None), "model_type", None)
    return hf == "glm_moe_dsa"


def _lpu_pace(tag: str = "") -> None:
    """Drain the LPU async backlog (deliberate pacing — not a correctness fix).

    No-op (AttributeError) off-LPU. Re-raises RuntimeError: a backlog failure
    is already fatal before the next GEMM. Shared by the GLM forward's
    pre-kv_b_proj and post-o_proj drains.

    With ``VLLM_FL_GLM_PACE_GRID=1`` the forward also inserts diagnostic drains
    around this one, and every drain that blocks >= 500 ms reports its tag +
    elapsed ms. The 32.768 s firmware watchdog releases a drain whose window
    holds a poisoned async task, so the *tag* of the slow drain brackets the
    culprit (a healthy drain waits ~2.75 ms).
    """
    import time as _time

    t0 = _time.monotonic()
    try:
        torch.lpu.synchronize()
    except AttributeError:
        pass
    except RuntimeError as e:
        import sys as _sys
        _sys.stderr.write(f"[GLM_MLA_SYNC] {tag} sync RAISED RuntimeError: {e!r}\n")
        _sys.stderr.flush()
        raise
    else:
        dt_ms = (_time.monotonic() - t0) * 1000.0
        import os as _os
        if dt_ms >= 500.0 and _os.environ.get("VLLM_FL_GLM_PACE_GRID", "0") == "1":
            import sys as _sys
            _sys.stderr.write(
                f"[GLM_MLA_PACE] {tag} drained in {dt_ms:.1f}ms\n")
            _sys.stderr.flush()


def _guard_hs(hidden_states, layer_id: int, seq_len: int, tag: str = "hs"):
    """Clamp extreme/NaN hidden states (DSA + dummy weights -> bf16 overflow).

    Logs when triggered; numerically a no-op for real weights.  ``abs().amax()``
    is a single host read that carries both tripwires -- NaN propagates through
    ``amax`` (unlike ``max``, which drops it on LPU) and inf/huge saturates the
    bound -- replacing the former min/max/isnan trio (three reads).  The clamp
    below is unconditional either way; ``VLLM_FL_GLM_GUARD=0`` gates only the
    check (the host read), not the clamp.
    """
    import sys as _sys
    if _GLM_GUARD_ON:
        hs_amax = hidden_states.abs().amax().item()
        if hs_amax != hs_amax or hs_amax > _HS_CLAMP_BOUND:
            _sys.stderr.write(
                f"[HS_CLAMP] {tag} L{layer_id} S={seq_len} "
                f"min={hidden_states.min().item():.6g} "
                f"max={hidden_states.max().item():.6g} "
                f"nan={bool(torch.isnan(hidden_states).any().item())}\n")
            _sys.stderr.flush()
    return hidden_states.clamp(-_HS_CLAMP_BOUND, _HS_CLAMP_BOUND)


def _guard_out(out, layer_id: int, seq_len: int, tag: str,
               lo: float = -5000.0, hi: float = 5000.0):
    """nan_to_num + clamp on an attention/projection output (pre-o_proj / pre-residual).

    One host read on the clean path: ``abs().amax()`` flags both NaN (propagates
    through ``amax``) and inf (saturates it), replacing the isnan/isinf pair.
    """
    import sys as _sys
    if _GLM_GUARD_ON:
        out_amax = out.abs().amax().item()
        if out_amax != out_amax or out_amax == float("inf"):
            _sys.stderr.write(
                f"[HS_CLAMP] {tag} L{layer_id} S={seq_len} "
                f"nan={torch.isnan(out).sum().item()} inf={torch.isinf(out).sum().item()}\n")
            _sys.stderr.flush()
    out = torch.nan_to_num(out, nan=0.0)
    return out.clamp(lo, hi)


def _drop_unused_glm_wrapper(attn, prefix: str) -> bool:
    """Drop the unused upstream CUDA wrapper stack on the GLM LPU path.

    Upstream ``DeepseekV2MLAAttention.__init__`` (run by Option C for
    construction fidelity) always builds ``MultiHeadLatentAttentionWrapper``
    → inner ``MLAAttention``. The LPU ``forward`` override never calls it
    (pristine own-stack set ``self.mla_attn = None``). Left alive it causes
    two phantom side effects per layer:

    1. ``MLAAttention.__init__`` **self-registers**
       ``static_forward_context[f"{prefix}.attn"] = self``
       (vllm ``model_executor/layers/attention/mla_attention.py:489-491``) and
       ``get_kv_cache_spec()`` returns a **real** ``MLAAttentionSpec`` — so the
       KV-cache manager sees phantom per-layer spec entries (second cache
       group / bound ``kv_cache`` the LPU path never uses) instead of the
       pristine own-stack's single uniform group.
    2. live wrapper/prefill-backend objects on device (~0.4 GiB profile-peak
       delta measured on the GLM 6-layer true-DSA run).

    This structural delta correlates with the true-DSA regression
    (warm-up ``kv_b_proj`` einsum ``evError2102`` die-copy: pristine own-stack
    passes / Option C crashes 3/3; the dense bring-up runs never exercised the
    Indexer×wrapper combination). Dropping it restores pristine topology
    while keeping the "upstream class + patch" construction shape.

    Returns True if a wrapper was present and dropped (idempotent: no-op
    when ``mla_attn`` is already None).
    """
    wrapper = getattr(attn, "mla_attn", None)
    if wrapper is None:
        return False
    attn.mla_attn = None
    key = f"{prefix}.attn"
    try:
        from vllm.config import get_current_vllm_config
        sfc = get_current_vllm_config().compilation_config.static_forward_context
    except Exception:  # noqa: BLE001 — off-engine unit tests: attr drop is enough
        return True
    registered = getattr(wrapper, "mla_attn", None)
    # Identity-guarded pop: remove exactly the inner MLAAttention the upstream
    # init registered under "<prefix>.attn", never a foreign entry.
    if key in sfc and (registered is None or sfc[key] is registered):
        sfc.pop(key)
    return True


class EvasGLMMLABackend(EvasMLAAttentionBackend):
    """GLM-5.2 evas MLA backend (4D paged cache, subclass forward).

    The shared base fixes block_size=256, bf16, ``"CUSTOM"`` enum slot.  GLM
    diverges only in the 4D cache layout and the GLM metadata/impl classes.
    """

    @staticmethod
    def get_impl_cls() -> type:
        return EvasGLMMLAAttentionImpl

    @staticmethod
    def get_builder_cls() -> type:
        return EvasGLMMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Single KV latent vector per position (num_kv_heads=1, no dim-0 k/v
        # split).  Matches the MLAAttentionSpec and per-slot helpers.
        return num_blocks, num_kv_heads, block_size, head_size

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # C-contiguous: the first physical dim is the block index (vLLM MLA
        # backend invariant).
        return (0, 1, 2, 3)


@dataclass
class EvasGLMMLAMetadata:
    """Per-layer attention metadata for the GLM MLA backend.

    Carries the common fields derived from ``CommonAttentionMetadata`` that
    the paged gather/write helpers (``EvasGLMMLAAttention``) need. vLLM
    allocates one builder per ``kv_cache_spec`` group, so a single instance is
    shared across the layers in that group.
    """

    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True

    # Per-layer prefixes for the group (all share the same logical metadata);
    # reserved for future per-layer lookup.
    layer_prefixes: list[str] = field(default_factory=list)

    def get_layer_metadata(self, prefix: str) -> "EvasGLMMLAMetadata":
        """All layers in a spec group share the same metadata — return self."""
        return self


def glm_indexer_causal_window(attn_metadata, n_q: int, n_kv: int, device
                              ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-query-row visible window ``[lo, hi)`` in global ``k_full`` columns.

    Upstream keeps a per-query-row causal limit on the indexer (its metadata
    field is documented as ``seq_lens: per-token effective context lengths`` and
    the decode kernel derives ``seq_len - max_decode_len + local_idx + 1``), so a
    query may score only its own causal prefix.

    Both bounds are needed because the LPU gather concatenates every request's
    full ``seq_len`` history in request order (``_indexer_gather_k`` /
    ``_glm_paged_gather`` both ``torch.cat`` per-request segments). Global query
    row ``g`` — request ``r``, local index ``i`` — may therefore attend only
    columns ``[kv_off[r], kv_off[r] + (seq_lens[r] - num_q[r]) + i + 1)``, where
    ``kv_off[r]`` is the running sum of ``seq_lens``. The upper bound alone would
    leave the lower one open: request ``r > 0`` would still see every earlier
    request's KV, the same entanglement the dense branch removes with one varlen
    segment per request (it must not be reintroduced through the indexer, since
    the sparse SDPA below runs with ``is_causal=False`` and no other mask).

    Returns two float32 ``[n_q]`` tensors (float, not int: LPU serves no int
    compare/clamp the mask build could use).

    Returns ``None`` (caller keeps the previous dense-equivalent behaviour) when
    there is no metadata, the metadata is not causal, or it does not agree with
    ``n_q`` / ``n_kv`` — the cache was not bound, so the gather fell through and
    returned the current tokens, and building a window from unhonoured metadata
    would slice past the end of ``k_full``.
    """
    if attn_metadata is None:
        return None
    if not getattr(attn_metadata, "causal", True):
        return None
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    qsl = getattr(attn_metadata, "query_start_loc", None)
    if seq_lens is None or qsl is None:
        return None
    kv_lens = [int(x) for x in seq_lens.tolist()]
    num_reqs = len(kv_lens)
    if num_reqs == 0 or sum(kv_lens) != n_kv:
        return None
    q_bounds = [int(x) for x in qsl.tolist()]
    if len(q_bounds) != num_reqs + 1 or q_bounds[-1] != n_q:
        return None
    los: list[float] = []
    his: list[float] = []
    kv_off = 0
    for r in range(num_reqs):
        n_q_r = q_bounds[r + 1] - q_bounds[r]
        length = kv_lens[r]
        # Local query i is the (length - n_q_r + i)-th token of the request, so
        # it may attend kv_off..(kv_off + length - n_q_r + i) inclusive.
        base = kv_off + (length - n_q_r) + 1
        for i in range(n_q_r):
            los.append(float(kv_off))
            his.append(float(min(base + i, kv_off + length)))
        kv_off += length
    return (torch.tensor(los, dtype=torch.float32, device=device),
            torch.tensor(his, dtype=torch.float32, device=device))


class EvasGLMMLAMetadataBuilder(
    AttentionMetadataBuilder[EvasGLMMLAMetadata]
):
    """Builds ``EvasGLMMLAMetadata`` from ``CommonAttentionMetadata``.

    One builder per ``kv_cache_spec`` group; all layers in a group share the
    same logical metadata, so we copy the common fields verbatim (cloning the
    per-token-read tensors, see ``build``).
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> EvasGLMMLAMetadata:
        # CLONE the four per-token-read tensors (query_start_loc/seq_lens/
        # block_table/slot_mapping). evas reads them in pure-Python per-token
        # loops spanning the whole model forward (_glm_paged_write/gather:
        # .item() on slot_mapping, seq_lens, block_table), but
        # common_attn_metadata hands them over as VIEWS into the worker's GPU/
        # pinned buffers, which model_runner asynchronously overwrites for the
        # next step's input prep while async GPU ops are still in flight. LPU
        # symptom: seq_lens flipped [1] -> 0x31FB0001 (800GiB OOM). Upstream
        # backends also hold the views but consume them inside a single kernel
        # launch (timing-safe); evas's Python loops are timing-fragile.
        def _clone_if_tensor(t):
            if t is None:
                return None
            # CPU round-trip to guarantee independent storage: LPU .clone()
            # may return a tensor whose storage is still asynchronously shared
            # with the original (seen as seq_lens corruption mid-forward in GLM
            # true DSA step 7+).
            return t.cpu().clone()

        return EvasGLMMLAMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=_clone_if_tensor(common_attn_metadata.query_start_loc),
            seq_lens=_clone_if_tensor(common_attn_metadata.seq_lens),
            block_table=_clone_if_tensor(common_attn_metadata.block_table_tensor),
            slot_mapping=_clone_if_tensor(common_attn_metadata.slot_mapping),
            causal=common_attn_metadata.causal,
            layer_prefixes=list(self.layer_names),
        )


class EvasGLMMLAAttentionImpl(AttentionImpl):
    """No-op attention impl for the GLM MLA backend.

    The real attention compute lives in ``EvasGLMMLAAttention``
    (this file), which reads the bound ``kv_cache`` tensors from the
    ``AttentionLayerBase`` modules (``EvasGLMMLAAttentionLayer``, indexer)
    directly.  This class exists only to satisfy the vLLM ``AttentionBackend``
    ABC.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: EvasGLMMLAMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Warm-up / profiling path.
        if attn_metadata is None:
            return output
        # Real compute is handled by EvasGLMMLAAttention; this stub just
        # passes the pre-allocated output through unchanged.
        return output


class EvasGLMMLAAttention(_UpstreamDeepseekV2MLAAttention):
    """LPU GLM-5.2 MLA attention — upstream construction + LPU forward (Option C).

    ``__init__`` always runs the upstream body (fused_qkv_a_proj, layernorms,
    q_b/kv_b/o_proj, rope, Indexer via the rebind factory, MLAModules +
    MultiHeadLatentAttentionWrapper), then — only for GLM-5.2 — adds the LPU
    paged-cache layer and derived aliases. Non-GLM models (deepseek-v3.1 and
    others share this upstream class) get pure upstream behavior: ``forward``
    delegates to ``super()``.
    """

    def __init__(self, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config") or (args[0] if args else None)
        hf_config = kwargs.get("config") or (args[1] if len(args) > 1 else None)
        prefix = kwargs.get("prefix", "")
        topk_indices_buffer = kwargs.get("topk_indices_buffer")
        # Plain bool is safe before super(): nn.Module.__setattr__ only
        # rejects Parameter/Module/Tensor pre-init, not object attributes.
        self._fl_is_glm = is_glm_config(vllm_config)
        super().__init__(*args, **kwargs)
        if not self._fl_is_glm:
            return

        # ---- GLM-only post-wiring (DSV4-style upstream class plus patch) ----
        # Lazy import: glm_layers imports this module at scope (backend ref),
        # so a module-scope import here would be circular.
        from vllm_fl.dispatch.backends.vendor.evas.impl.attention.glm_layers import (
            EvasGLMMLAAttentionLayer,
            glm_layer_runs_dsa,
        )

        m = re.search(r"layers\.(\d+)\.", prefix)
        self.layer_id = int(m.group(1)) if m else 0
        self.prefix = prefix
        self.index_topk = getattr(hf_config, "index_topk", None)
        # Factory already applied HF indexer_types semantics — single source
        # of truth for "this layer computes its own topk".
        self.has_indexer = self.indexer is not None
        # ... and for "this layer runs the sparse path at all": a "shared"
        # layer reuses the previous full layer's topk, while a configuration
        # with no DSA type slot for this layer (the glm-5 stub carries
        # index_topk but no indexer_types) stays dense.  index_topk alone is
        # not that test -- it is config-wide, so reading it as the trigger
        # puts every layer of such a model behind a buffer no full layer
        # ever writes.
        self._fl_dsa_layer = glm_layer_runs_dsa(hf_config, self.layer_id)
        # Own-stack alias (forward + gather use softmax_scale; == qk_head_dim**-0.5).
        self.softmax_scale = self.scaling
        # Upstream stores the buffer on MLAModules/wrapper, not on attention —
        # the LPU forward needs the direct reference (full writes / shared reads).
        self.topk_indices_buffer = topk_indices_buffer
        self._cache_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # Paged 4D KV cache layer: a real AttentionLayerBase with a KVCacheSpec
        # so vLLM allocates + binds kv_cache and builds block_table/slot_mapping
        # metadata (without it: "model without KVCache", chunked prefill off,
        # first generate step hangs). Unbound kv_cache (profile/warm-up) makes
        # the gather/write helpers no-op.
        self._attn = EvasGLMMLAAttentionLayer(
            head_dim=self._cache_head_dim, dtype=torch.bfloat16, prefix=prefix)
        if self.indexer is not None:
            # Wire the Indexer op's 128-dim buffer to MIRROR the MLA cache slot
            # layout; the op reuses the MLA layer's attn_metadata (head-size-
            # independent). Replaces a vLLM-managed Indexer KV cache —
            # get_kv_cache_spec→None keeps the cache group uniform → no LPU
            # die-copy watchdog (see EvasGLMIndexerCacheLayer doc).
            self.indexer._evas_indexer_op.mla_cache_layer = self._attn
        # Drop the unused upstream CUDA wrapper + its phantom
        # static_forward_context["<prefix>.attn"] registration (see helper
        # docstring): restores the pristine own-stack topology (`mla_attn =
        # None`, single uniform KV-cache spec group) that the true-DSA e2e
        # passed with.
        _drop_unused_glm_wrapper(self, prefix)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._fl_is_glm:
            # deepseek-v3.1 / every non-GLM model sharing this class: byte-for-
            # byte upstream path (mla_attn wrapper).
            return super().forward(positions, hidden_states, llama_4_scaling)

        # ================= GLM LPU path =================
        # Projection calls go through super()'s modules (fused_qkv single GEMM;
        # upstream linears default return_bias=True → tuple unwraps via _out1).
        import flag_gems
        from vllm.forward_context import get_forward_context

        def _out1(y):
            return y[0] if isinstance(y, tuple) else y

        layer_id = self.layer_id

        # vLLM v1 passes 2D [num_tokens, hidden]; normalize to 3D [1, S, H] for
        # the forward body, restore on return.
        orig_is_2d = hidden_states.dim() == 2
        if orig_is_2d:
            hidden_states = hidden_states.unsqueeze(0)  # [1, S, H]
        bsz, seq_len, _ = hidden_states.shape

        # Numerical guard: DSA + dummy weights can drive hidden_states to bf16
        # overflow in the projections (NaN -> all-zero output). Clamp to a wide
        # range (~2000x typical RMS ~250) — a no-op for real weights.
        hidden_states = _guard_hs(hidden_states, layer_id, seq_len, "hs_in")

        # GLM's 3 large GEMMs/layer (q_b_proj/kv_b_proj/o_proj) let the async
        # backlog accumulate across layers until the LPU firmware watchdog
        # trips. One synchronize at the end of each layer's MLA drains the
        # pipeline (deliberate LPU-only pacing, not a correctness change).
        import os as _os
        _sync_dbg = _os.environ.get("VLLM_FL_GLM_MLA_TRACE", "0") == "1"

        def _trace(tag, *shapes):
            if not _sync_dbg:
                return
            import sys as _sys
            _sys.stderr.write(
                f"[GLM_MLA_TRACE] L{layer_id} S={seq_len} {tag} shapes={shapes}\n")
            _sys.stderr.flush()

        _trace("enter", hidden_states.shape)

        # Diagnostic-only drain grid (VLLM_FL_GLM_PACE_GRID=1). The
        # pre-kv_b_proj drain below is the one that hits
        # the firmware watchdog, but its window spans [previous layer's
        # MLP/MoE + this layer's input_layernorm + fused_qkv_a_proj + q_a_layernorm
        # + q_b_proj + rope + kv_a_layernorm] -- too wide to name the culprit.
        # These extra drains bisect it; the slow one's tag names the segment.
        # Off by default (they change queue depth, which is itself a fault
        # variable) and no-ops with the rest of the pace on non-LPU builds.
        _pace_grid = _os.environ.get("VLLM_FL_GLM_PACE_GRID", "0") == "1"
        if _pace_grid:
            _lpu_pace(f"L{layer_id} S={seq_len} pre-attn")

        # ---- q_lora + kv_lora via upstream fused_qkv_a_proj (one GEMM) ----
        # Old own-stack: separate q_a_proj + kv_a_proj_with_mqa ReplicatedLinears.
        # Upstream fused: [S, q_lora_rank + kv_lora_rank + qk_rope_head_dim].
        # Keep the original rank ([1, S, D]) through q_b_proj: the LPU
        # F.linear→_einsum_linear patch caches a resident transposed copy
        # (contiguous column slabs, same total bytes as a plain
        # weight.T.contiguous(): 64 MiB for q_b [16384, 2048] bf16) ONLY for
        # 2D input; a 3D input takes the uncached batched einsum branch
        # instead (gemm.py 2D-vs-3D split).
        # Pristine own-stack (attn_op.py) also called q_b_proj on 3D and only
        # viewed to heads AFTER q_b — matching it here drops our own
        # 6 × 64 MiB = 384 MiB profile peak.
        qkv = _out1(self.fused_qkv_a_proj(hidden_states))  # keep rank (3D [1,S,D])
        _trace("fused_qkv_a_proj", qkv.shape)
        if _pace_grid:
            _lpu_pace(f"L{layer_id} S={seq_len} post-fused_qkv_a_proj")
        # Slice q_a on the last dim, keep leading dims (3D) so q_b_proj sees
        # 3D → no cached weight.T. RMSNorm normalizes the trailing dim only
        # (rank-agnostic; _evas/ops/rms_norm + ir.ops.rms_norm both do).
        q_a_raw = qkv[..., : self.q_lora_rank]  # pre-layernorm (Indexer qr source)
        q = self.q_a_layernorm(q_a_raw)
        q = _out1(self.q_b_proj(q))  # 3D input → uncached einsum (matches pristine)
        _trace("q_b_proj", q.shape)
        if _pace_grid:
            _lpu_pace(f"L{layer_id} S={seq_len} post-q_b_proj")
        q = q.view(seq_len, self.num_local_heads, self.qk_head_dim)  # AFTER q_b
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # kv_lora path: view to 2D after slicing (pristine views kv after its
        # proj too) — the split below and everything downstream expect 2D.
        kv = qkv[..., self.q_lora_rank :]       # [S, kv_lora_rank + qk_rope_head_dim]
        kv = kv.view(seq_len, -1)
        kv_a, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Save the pre-RoPE k_pe for the cache write: the cache stores the raw
        # latent (position-invariant); RoPE is re-applied at gather time.
        # Must be a real copy, not an alias: the rope below mutates its inputs
        # in place (RotaryEmbeddingFL.forward_oot -> flaggems ->
        # _evas/fused/rotary_embedding.py, inplace=True writes through the
        # strided k_pe view), so `k_pe_raw = k_pe` would silently see the
        # ROTATED values -- the cache would store rotated k_pe and
        # _glm_paged_gather would rotate it a second time (double RoPE on the
        # k side). Proved on CPU against the production rope chain:
        # the returned tensor aliases the input, and an aliased snapshot reads
        # back rotated. clone() freezes the raw values before the rope runs.
        k_pe_raw = k_pe.clone()
        if _pace_grid:
            import sys as _sys3
            _re = self.rotary_emb
            _sys3.stderr.write(
                f"[GLM_MLA_WIN] L{layer_id} pos={tuple(positions.shape)}:{positions.dtype}"
                f" csc={tuple(_re.cos_sin_cache.shape)}:{_re.cos_sin_cache.dtype}"
                f" neox={_re.is_neox_style} rd={_re.rotary_dim} hs={_re.head_size}"
                f" q={tuple(q.shape)}:c={q.is_contiguous()}"
                f" q_nope={tuple(q_nope.shape)}:c={q_nope.is_contiguous()}"
                f" q_pe={tuple(q_pe.shape)}:c={q_pe.is_contiguous()}"
                f" kv_a={tuple(kv_a.shape)}:c={kv_a.is_contiguous()}"
                f" k_pe={tuple(k_pe.shape)}:c={k_pe.is_contiguous()}\n")
            _sys3.stderr.flush()
            _lpu_pace(f"L{layer_id} S={seq_len} post-qkv-split")

        # RoPE on the rope slices (GLM interleave)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        _trace("rotary", q_pe.shape, k_pe.shape)
        q_pe = q_pe.reshape(seq_len, self.num_local_heads, self.qk_rope_head_dim)
        k_pe = k_pe.reshape(seq_len, 1, self.qk_rope_head_dim)
        if _pace_grid:
            _lpu_pace(f"L{layer_id} S={seq_len} post-rope")

        # Rebuild [nope|rope] without torch.cat: q_nope and q_pe are two
        # non-contiguous last-dim slices of this same q buffer (row stride
        # qk_head_dim), and catting strided views makes the LPU materialize a
        # per-call strided copy that trips the firmware watchdog. With the PACE
        # grid the drain immediately after this cat absorbed the WaitSync
        # overtime while the rope drain before it stayed well under it, i.e.
        # this copy is the faulting kernel; the trap then
        # surfaces at the next host<->device copy (kv_a_layernorm rms_norm
        # dispatch, evError2102). q already holds the rotated tail in place --
        # RotaryEmbeddingFL.forward_oot calls apply_rotary_pos_emb with
        # inplace=True, which writes through the q_pe view -- so
        # cat([q[..., :nope], q[..., nope:]]) is just q; clone() keeps the
        # fresh-tensor semantics of the old cat with one contiguous copy.
        q = q.clone()                                    # [S, H, 256]
        if _pace_grid:
            _lpu_pace(f"L{layer_id} S={seq_len} post-cat")

        # Expand kv_a to k_nope + v for the current tokens.
        kv_a_normed = self.kv_a_layernorm(kv_a)
        # LPU watchdog workaround: drain the stream so kv_b_proj's bmm does not
        # trip the firmware overtime check from the async GEMM backlog above.
        # No-op on CUDA.
        _lpu_pace(f"L{layer_id} S={seq_len} pre-kv_b_proj")
        if _sync_dbg:
            import sys as _sys2
            _sys2.stderr.write(
                f"[GLM_MLA_SYNC] L{layer_id} S={seq_len} pre-kv_b_proj sync OK\n")
            _sys2.stderr.flush()
        kv_expanded = _out1(self.kv_b_proj(kv_a_normed))
        _trace("kv_b_proj", kv_expanded.shape)
        kv_expanded = kv_expanded.view(
            seq_len, self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(
            kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, self.num_local_heads, -1)], dim=-1
        )  # [S, H, 256]

        # Paged KV: WRITE current tokens first, THEN gather full KV — the
        # gather reads all positions (incl. the current token's) from cache and
        # ignores the local k/v args, so the current tokens must be cached first
        # (matches DSV4's write-before-gather pattern).
        #
        # attn_metadata is a dict[str, AttentionMetadata] keyed by layer prefix
        # (one EvasGLMMLAMetadata per KVCacheSpec group); None during warm-up /
        # profile_run, when the gather/write helpers no-op and return k/v as-is.
        ctx_attn_metadata = get_forward_context().attn_metadata
        if isinstance(ctx_attn_metadata, dict):
            attn_metadata = ctx_attn_metadata.get(self.prefix)
        else:
            attn_metadata = ctx_attn_metadata
        if _sync_dbg:
            import sys as _sys
            if attn_metadata is None:
                _sys.stderr.write(f"[MLA_DBG] L{layer_id} attn_metadata=None\n")
            else:
                _sl = getattr(attn_metadata, "seq_lens", None)
                _sys.stderr.write(
                    f"[MLA_DBG] L{layer_id} prefix={self.prefix} "
                    f"seq_lens={_sl.tolist() if _sl is not None and hasattr(_sl,'tolist') else _sl} "
                    f"num_reqs={getattr(attn_metadata,'num_reqs','?')}\n")
            import sys as _sys_ctx; _sys_ctx.stderr.flush()
        self._glm_paged_write(kv_a, k_pe_raw, attn_metadata)
        _trace("paged_write")
        k_full, v_full = self._glm_paged_gather(
            k, v, attn_metadata)
        _trace("paged_gather", k_full.shape, v_full.shape)

        # ---- DSA branch selection (HF-aligned) ----
        #   indexer type "full"   -> compute topk, write buffer, sparse
        #   indexer type "shared" -> read buffer topk,          sparse
        #   no DSA type slot      -> dense causal
        # The branch follows straight from the layer's HF ``indexer_types``
        # slot, the same predicate the factory used to decide whether to hand
        # this layer an Indexer.
        do_sparse = bool(self.has_indexer or self._fl_dsa_layer)

        # Note on the sparse branch: the Indexer's ``topk_indices`` need no
        # per-request OFFSET -- ``layers.py`` gathers the cache with the same
        # request-order ``for r in range(num_reqs)`` / ``torch.cat`` and runs
        # top-k over the resulting global axis, so each query already points at
        # the right global rows and offsetting them again would push each
        # request past its own rows.  They do need the causal WINDOW mask
        # though: this SDPA runs with is_causal=False and no padding mask, so
        # causality and request isolation rest entirely on the indexer having
        # excluded positions the row may not see.  ``glm_indexer_causal_window``
        # does that inside the Indexer, before the top-k.
        if do_sparse:
            if self.has_indexer:
                # Full layer: run the Indexer to compute this layer's topk, using
                # qr = the q_lora representation (post q_a_layernorm) from above.
                qr = self.q_a_layernorm(q_a_raw)  # q_a_raw saved pre-layernorm
                _n = hidden_states.reshape(-1, self.hidden_size).shape[0]
                # The Indexer returns the WHOLE shared buffer (the upstream
                # contract), which is [max_num_batched_tokens, topk] -- only its
                # first _n rows are this forward's. Take those: everything below
                # (the belt-and-suspenders re-write, and the sparse gather) is
                # sized to the S_q query rows of this forward. With
                # max_num_batched_tokens == 1 the two coincide, which is why the
                # mismatch only shows on a multi-token or decode-after-prefill
                # step.
                topk_indices = self.indexer(
                    hidden_states.reshape(-1, self.hidden_size), qr,
                    positions.reshape(-1) if positions.dim() > 1 else positions,
                    self.indexer_rope_emb)[:_n]  # [S_q, topk] int32
                _trace("indexer_topk", topk_indices.shape)
                # Also write the shared buffer here (belt-and-suspenders; the
                # Indexer forward already writes it).
                if self.topk_indices_buffer is not None:
                    self.topk_indices_buffer[
                        :_n, : self.indexer.topk_tokens] = topk_indices
                _topk_width = self.indexer.topk_tokens
            else:
                # Shared layer: no Indexer; reuse the previous "full" layer's
                # topk from the model-level shared buffer (the side-channel).
                _n = hidden_states.reshape(-1, self.hidden_size).shape[0]
                _topk_width = self.index_topk
                if self.topk_indices_buffer is not None:
                    topk_indices = self.topk_indices_buffer[
                        :_n, : _topk_width]
                else:
                    topk_indices = None
                # Warm-up guard: the buffer is torch.empty until a "full" layer
                # writes it; if the indices are insane (out of KV range, e.g.
                # abnormal-order profile runs), fall back to dense causal for
                # this step rather than indexing garbage. A lone -1 is now
                # legitimate (the Indexer marks a column outside the row's
                # visible window, or the tail padding, with -1), so only < -1
                # counts as garbage.
                _s_total = int(k_full.shape[0])
                if topk_indices is None or topk_indices.numel() == 0 or \
                        bool(topk_indices.min().item() < -1) or \
                        bool(topk_indices.max().item() >= _s_total):
                    _trace("shared_topk_uninit_fallback_dense",
                           getattr(topk_indices, "shape", None))
                    do_sparse = False  # fall through to dense branch below
                else:
                    _trace("shared_topk_reused", topk_indices.shape)

        if do_sparse:
            # Truncate -1 padding columns from the Indexer's fixed-width [N,
            # topk_tokens] output (padding when S_total < topk_tokens): keeping
            # them would blow sparse SDPA to O(topk_tokens). real_topk =
            # min(topk_tokens, S_total); the same truncation is safe for shared
            # layers' reused topk (real position indices).
            real_topk = min(_topk_width, int(k_full.shape[0]))
            topk_indices = topk_indices[:, :real_topk]
            # Sparse gather: each query token attends to its topk positions in
            # k_full/v_full (position-aligned with the Indexer's 128-dim cache).
            # A -1 marks a column the Indexer rejected: the tail padding when
            # S_total < topk_tokens, or -- with the causal window now in the
            # Indexer -- a position outside this query row's visible window
            # (beyond its causal prefix, or inside an earlier request's
            # segment; the sparse SDPA below runs with is_causal=False, so the
            # Indexer alone enforces both, and it leaves such holes whenever a
            # row's visible set is smaller than the topk width, e.g. every early
            # row of a prefill chunk or any row of the second request in a
            # batch). The dense branch keeps requests apart with one varlen
            # segment per request; this is the sparse equivalent. Redirect -1 to
            # column 0 for the gather and mask it out of the SDPA. Float
            # arithmetic + one cast: LPU serves no int masked_fill / clamp.
            idx = topk_indices.long()
            idx_valid = idx.to(torch.float32) >= 0.0
            idx_safe = (idx.to(torch.float32) * idx_valid.to(torch.float32)).long()
            k_sparse = k_full[idx_safe]                  # [S_q, real_topk, H, D]
            v_sparse = v_full[idx_safe]                  # [S_q, real_topk, H, D]
            # flag_gems SDPA is 4D [1, B, Sq, D] and cannot take a per-query
            # varying key set (5D rejected). Flatten (H, S_q) into the batch
            # dim: each (head, query) is one attention problem with real_topk
            # keys.
            H = self.num_local_heads
            D_qk = q.shape[-1]                # qk_head_dim (q and k share this)
            D_v = v_sparse.shape[-1]          # v_head_dim (differs from D_qk)
            real_topk = k_sparse.shape[1]      # may be < topk_tokens after truncation
            # q: [S_q, H, D_qk] -> [H, S_q, D_qk] -> [H*S_q, 1, D_qk]
            q_flat = q.transpose(0, 1).reshape(H * seq_len, 1, D_qk).contiguous()
            # k_sparse: [S_q, real_topk, H, D_qk] -> [H, S_q, real_topk, D_qk] -> [H*S_q, real_topk, D_qk]
            k_flat = k_sparse.permute(2, 0, 1, 3).reshape(H * seq_len, -1, D_qk).contiguous()
            # v_sparse: [S_q, real_topk, H, D_v] -> [H, S_q, real_topk, D_v] -> [H*S_q, real_topk, D_v]
            v_flat = v_sparse.permute(2, 0, 1, 3).reshape(H * seq_len, -1, D_v).contiguous()
            # Additive attention mask (0 keep / -inf drop) for the -1 columns,
            # per query row and shared across its heads: the evas SDPA adds it
            # to the fp32 scores. All-real rows stay all-zero, so a fully valid
            # sparse step is numerically what it was before the cap.
            #
            # Rank is the standard SDPA one, [B, H, S_q, S_k]: the adopted evas
            # kernel reads attn_mask.stride(3) before launching, so a 3-D mask
            # raises IndexError on the kernel arm (this call site is the pin's
            # only masked consumer; see the scaled_dot_product_attention pin
            # evidence in FlagGems _evas/fused/_adoption.py). The pure twin
            # broadcasts the same 4-D form down to its [H*S_q, 1, real_topk]
            # scores, so one rank serves both arms.
            _mask = torch.where(
                idx_valid, torch.zeros_like(idx_valid, dtype=torch.float32),
                torch.full_like(idx_valid, float("-inf"), dtype=torch.float32))
            attn_mask = _mask.unsqueeze(0).expand(H, seq_len, real_topk).reshape(
                1, H * seq_len, 1, real_topk)
            _trace("pre_sdpa_sparse", q_flat.shape, k_flat.shape, v_flat.shape,
                   attn_mask.shape)
            out = flag_gems.scaled_dot_product_attention(
                q_flat.unsqueeze(0), k_flat.unsqueeze(0), v_flat.unsqueeze(0),
                attn_mask=attn_mask,
                is_causal=False,
            ).squeeze(0)  # [H*S_q, 1, v_head]
            out = out.reshape(H, seq_len, -1).transpose(0, 1)  # [S_q, H, v_head]
            # NaN/inf sanitising and the NaN log for both branches are owned by
            # the shared _guard_out("sdpa_out") below; a second isnan read plus
            # nan_to_num here was a redundant host sync on the sparse path.
            _trace("sdpa_sparse", out.shape)
        if not do_sparse:
            # Dense branch: attend to all KV, causal. Reached on layers the
            # configuration gives no DSA type slot, or a shared layer whose
            # topk buffer is not sane yet at warm-up.
            #
            # flag_gems.flash_attn_varlen_func (non-paged varlen) replaces the
            # former flag_gems.scaled_dot_product_attention call: the evas
            # varlen override's _sdpa_fp32 uses matmul, so v_head_dim (128) may
            # differ from qk_head_dim (256) — no F.pad needed (unlike CUDA FA's
            # _flash_attn_varlen_diff_headdims, flaggems/impl/mla.py:93-120).
            # q/k_full/v_full are already [T, H, D].  One varlen segment **per
            # request**: segment r pairs request r's query tokens with request
            # r's rows of ``k_full``.  The evas override loops over segments and
            # applies the FA bottom-right causal convention inside each one
            # (query i sees keys j <= i + Sk - Sq, attention.py:149-154), so each
            # request attends exactly its own causal prefix.
            #
            # The former single segment ([0, S_q] / [0, S_total]) declared the
            # whole concatenation one sequence: with seq_lens=[4,4] and one
            # decode query each, S_total=8, S_q=2 and the causal offset 6 let
            # request 0's query read keys 0..6 and request 1's read 0..7 --
            # every request saw every other request's KV.  That is the
            # multi-request entanglement (one request generating another's token
            # stream); single-request runs were unaffected because local ==
            # global there.
            # Derived here rather than up front: only this branch needs it, and
            # the sparse path must not pick up a new exception surface.
            q_counts, kv_lens = self._per_request_kv_segments(
                attn_metadata, hidden_states, k_full)
            cu_q = [0]
            for count in q_counts:
                cu_q.append(cu_q[-1] + count)
            cu_k = [0]
            for length in kv_lens:
                cu_k.append(cu_k[-1] + length)
            out = flag_gems.flash_attn_varlen_func(
                q.contiguous(), k_full.contiguous(), v_full.contiguous(),
                max_seqlen_q=max(q_counts),
                cu_seqlens_q=torch.tensor(
                    cu_q, dtype=torch.int32, device=q.device),
                max_seqlen_k=max(kv_lens),
                cu_seqlens_k=torch.tensor(
                    cu_k, dtype=torch.int32, device=q.device),
                softmax_scale=self.softmax_scale,
                causal=True,
            )  # [S_q, H, v_head]
            _trace("sdpa_varlen", out.shape, "segs", len(q_counts))
        out = out.reshape(seq_len, -1)
        # Numerical guard: clamp the SDPA output before o_proj to prevent bf16
        # overflow in the projection (DSA + dummy weights -> random attention
        # weighted sums that can overflow through bf16 matmul).
        out = _guard_out(out, layer_id, seq_len, "sdpa_out")
        out = _out1(self.o_proj(out))
        _trace("o_proj", out.shape)
        # Drain the async bmm backlog accumulated by this layer's 3 large GEMMs
        # + SDPA bmm so it cannot pile up across layers into a firmware
        # watchdog trap. See the pacing-sync note above the q_lora path.
        # Fail-fast on RuntimeError (see the _lpu_pace docstring).
        _lpu_pace(f"L{layer_id} S={seq_len} post-o_proj")
        # o_proj output guard: clamp + nan_to_num to prevent extreme values
        # from overflowing the post-attention MLP (residual + RMSNorm + bf16).
        out = _guard_out(out, layer_id, seq_len, "oproj_out")
        _trace("sync_done")
        out = out.view(bsz, seq_len, -1)
        if orig_is_2d:
            out = out.squeeze(0)  # restore [num_tokens, hidden]
        return out

    # ------------------------------------------------------------------
    # GLM paged KV gather/write (standard DeepSeek MLA, no SWA/Compressor).
    # Uses the _read_contiguous_run/_write_paged_runs helpers from
    # impl/attention/cache.py.
    #
    # Cache layout: each slot stores raw kv_a latent (pre-norm, kv_lora_rank)
    # + k_pe (pre-RoPE, qk_rope_head_dim) = _cache_head_dim dims. On gather we
    # read the raw latent, apply kv_a_layernorm + kv_b_proj to expand to
    # k_nope + v, and apply RoPE to k_pe.
    # ------------------------------------------------------------------

    @staticmethod
    def _per_request_kv_segments(attn_metadata, hidden_states, k_full):
        """Per-request (query count, KV length) of the concatenated ``k_full``.

        ``_glm_paged_gather`` concatenates every request's KV into one
        ``k_full`` in request order, so request ``r`` owns rows
        ``sum(kv_lens[:r]) .. sum(kv_lens[:r+1])``.  Query tokens are laid out in
        request order too: one per request on decode, or ``query_start_loc``
        -segmented on prefill.  Both are needed to hand the dense varlen call
        one segment per request instead of one global segment.

        Degrades to a single segment covering ``k_full`` when there is no
        metadata (profile_run / warm-up) or when ``k_full`` does not agree with
        it (cache not yet bound, so the gather fell through and returned the
        current k/v): building ``cu_seqlens`` from metadata the gather did not
        honour would slice past the end of ``k_full``.
        """
        hs = hidden_states.reshape(-1, hidden_states.shape[-1])
        n_q = int(hs.shape[0])
        n_kv = int(k_full.shape[0])
        if attn_metadata is not None:
            kv_lens = [int(x) for x in attn_metadata.seq_lens.tolist()]
            num_reqs = len(kv_lens)
            qsl = getattr(attn_metadata, "query_start_loc", None)
            if n_q == num_reqs:
                q_counts = [1] * num_reqs                # decode: one each
            elif qsl is not None and int(qsl[-1]) == n_q:
                b = [int(x) for x in qsl.tolist()]
                q_counts = [b[i + 1] - b[i] for i in range(num_reqs)]
            else:
                raise ValueError(
                    f"cannot map {n_q} query tokens onto {num_reqs} requests: "
                    f"query_start_loc={None if qsl is None else qsl.tolist()} "
                    f"seq_lens={kv_lens}")
            if (num_reqs > 0 and sum(q_counts) == n_q and sum(kv_lens) == n_kv):
                return q_counts, kv_lens
        return [n_q], [n_kv]

    def _glm_paged_gather(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather full KV from the paged MLA cache for all positions.

        Returns (k_full, v_full) of shape [S_total, H, D] over the full
        sequence seen so far (prefill: all positions; decode: history +
        current). With no cache (warm-up / profile_run / metadata None /
        cache not bound) returns the current k/v as-is.
        """
        if attn_metadata is None:
            return k, v
        if self._attn is None or self._attn.kv_cache.numel() == 0:
            return k, v

        from vllm_fl.dispatch.backends.vendor.evas.impl.attention.cache import (
            _read_contiguous_run,
        )

        def _out1(y):
            return y[0] if isinstance(y, tuple) else y

        kv_cache = self._attn.kv_cache
        block_table = attn_metadata.block_table  # [num_reqs, max_blocks]
        seq_lens = attn_metadata.seq_lens        # [num_reqs]
        num_reqs = attn_metadata.num_reqs
        storage_block_size = getattr(self._attn, 'kv_block_size', 256)

        if num_reqs == 0:
            return k, v

        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []

        for r in range(num_reqs):
            seq_len_r = int(seq_lens[r].item())
            kv_a_r = torch.zeros(
                seq_len_r, self.kv_lora_rank, dtype=k.dtype, device=k.device)
            k_pe_raw_r = torch.zeros(
                seq_len_r, self.qk_rope_head_dim, dtype=k.dtype, device=k.device)

            # Batch-read via contiguous block-aligned runs (fewer Python-loop
            # iterations across block boundaries than per-slot _read_slot).
            s = 0
            while s < seq_len_r:
                manager_idx = s // storage_block_size
                run_end = min((manager_idx + 1) * storage_block_size, seq_len_r)
                chunk = _read_contiguous_run(
                    kv_cache, block_table, r, s, run_end,
                    storage_block_size, flattened=False)
                kv_a_r[s:run_end] = chunk[:, :self.kv_lora_rank]
                k_pe_raw_r[s:run_end] = chunk[:, self.kv_lora_rank:]
                s = run_end

            # Expand kv_a -> k_nope + v
            kv_a_normed = self.kv_a_layernorm(kv_a_r)
            kv_expanded = _out1(self.kv_b_proj(kv_a_normed))
            kv_expanded = kv_expanded.view(
                seq_len_r, self.num_local_heads,
                self.qk_nope_head_dim + self.v_head_dim)
            k_nope_r, v_r = torch.split(
                kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # Apply RoPE to k_pe: rotary_emb(positions, q, k) expects q/k shaped
            # like the forward path's (q=[S, qk_rope], k=[S, 1, qk_rope], the
            # unsqueeze(1) is the single KV head). Pass a zero placeholder q
            # (its result is discarded) and k_pe_raw_r as k.
            pos_r = torch.arange(seq_len_r, device=k.device)
            _q_dummy = torch.zeros(
                seq_len_r, self.qk_rope_head_dim, dtype=k.dtype, device=k.device)
            _, k_pe_r = self.rotary_emb(
                pos_r, _q_dummy, k_pe_raw_r.unsqueeze(1))
            k_pe_r = k_pe_r.reshape(seq_len_r, 1, self.qk_rope_head_dim)

            k_r = torch.cat(
                [k_nope_r, k_pe_r.expand(-1, self.num_local_heads, -1)], dim=-1)
            k_list.append(k_r)
            v_list.append(v_r)

        k_full = torch.cat(k_list, dim=0)  # [total_tokens, H, D_k]
        v_full = torch.cat(v_list, dim=0)  # [total_tokens, H, D_v]
        return k_full, v_full

    def _glm_paged_write(
        self,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata,
    ) -> None:
        """Write the new tokens' raw kv_a + k_pe latent into the paged cache.

        kv_a: [S, kv_lora_rank] pre-norm, pre-RoPE
        k_pe: [S, qk_rope_head_dim] pre-RoPE (stored raw; RoPE is applied
              during gather using the position)
        """
        if attn_metadata is None:
            return
        if self._attn is None or self._attn.kv_cache.numel() == 0:
            return

        from vllm_fl.dispatch.backends.vendor.evas.impl.attention.cache import (
            _write_paged_runs,
        )

        kv_cache = self._attn.kv_cache
        slot_mapping = attn_metadata.slot_mapping    # [num_tokens_in_chunk]
        storage_block_size = getattr(self._attn, 'kv_block_size', 256)

        # Concatenate kv_a + k_pe into the raw latent vector. k_pe arrives as
        # [S, 1, qk_rope] (rotary_emb output of the unsqueezed input) while
        # kv_a is [S, kv_lora_rank]; collapse k_pe to [S, qk_rope] so cat on
        # dim=-1 yields [S, _cache_head_dim].
        k_pe_2d = k_pe.reshape(kv_a.shape[0], self.qk_rope_head_dim)
        latent = torch.cat([kv_a, k_pe_2d], dim=-1)  # [S, _cache_head_dim]
        # Cache-write guard: sanitize the latent before it persists. One decode
        # step's NaN projection (DSA + random weights → extreme attention out
        # overflowing the post-attention MLP) would otherwise poison the cache
        # for all subsequent decode steps. No-op for real weights. One host read
        # on the clean path: abs().amax() carries both NaN and inf.
        if _GLM_GUARD_ON:
            latent_amax = latent.abs().amax().item()
            if latent_amax != latent_amax or latent_amax == float("inf"):
                import sys as _nan_latent_sys
                _nan_latent_sys.stderr.write(
                    f"[CACHE_GUARD] L{self.layer_id} latent has "
                    f"nan={torch.isnan(latent).sum().item()} "
                    f"inf={torch.isinf(latent).sum().item()}\n")
                _nan_latent_sys.stderr.flush()
        latent = torch.nan_to_num(latent, nan=0.0)
        latent = latent.clamp(-50000.0, 50000.0)

        # Group consecutive slot_mapping entries into contiguous runs and write
        # each with a single slice ``copy_`` (vs a per-token Python loop), capped
        # at the physical block boundary to stay in-bounds (avoids LPU
        # advanced-indexing write errors). Shared with the DSV3 backend's
        # do_kv_cache_update via _write_paged_runs.
        _write_paged_runs(kv_cache, latent, slot_mapping, storage_block_size)
