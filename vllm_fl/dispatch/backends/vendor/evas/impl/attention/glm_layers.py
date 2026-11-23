# Copyright (c) 2026 BAAI. All rights reserved.
"""GLM-5.2 evas LPU MLA cache layers + sparse Indexer op — paged KV-cache holders.

GLM cache-layer stack, one file: ``EvasMLACacheLayerBase`` (common init: empty
``kv_cache``, prefix guard, self-registration into ``static_forward_context``,
``get_attn_backend``); paged holders ``EvasGLMMLAAttentionLayer`` +
``EvasGLMIndexerCacheLayer``; and ``_EvasGLMSparseIndexerOp``, the pure-PyTorch
SparseAttnIndexer replacement whose 128-dim scoring buffer mirrors the MLA
cache's slot layout (the base is used only by this stack).

"""
from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import extract_layer_index

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_fl.dispatch.backends.vendor.evas.impl.attention.cache import (
    _read_contiguous_run,
    _write_paged_runs,
)

from vllm_fl.dispatch.backends.vendor.evas.impl.attention.glm_mla_attention import (
    EvasGLMMLABackend,
    glm_indexer_causal_window,
    is_glm_config,
)

# vLLM manager block size for MLA paged caches.  GLM uses block_size=256
# (compress_ratio=1 -> direct layout) so every layer merges into one
# UniformTypeKVCacheSpecs group.
MLA_KV_BLOCK_SIZE = 256


_GLM_DSA_LAYER_TYPES = ("full", "shared")


def glm_indexer_type(config, layer_id: int):
    """The checkpoint's HF ``indexer_types`` slot for ``layer_id``, or None.

    None means the configuration declares no DSA layer here at all: either it
    carries no ``index_topk`` (not a v3.2-style config) or it has no type slot
    for this layer.  One predicate serves both questions below so the factory
    and the attention forward cannot drift apart.
    """
    if getattr(config, "index_topk", None) is None:
        return None
    indexer_types = getattr(config, "indexer_types", []) or []
    if layer_id >= len(indexer_types):
        return None
    return indexer_types[layer_id]


def glm_indexer_required(config, layer_id: int) -> bool:
    """HF ``indexer_types`` semantics: does this layer build an Indexer?

    Only a "full" layer does; a "shared" layer reuses the previous full
    layer's topk through the shared buffer instead.
    """
    return glm_indexer_type(config, layer_id) == "full"


def glm_layer_runs_dsa(config, layer_id: int) -> bool:
    """HF ``indexer_types`` semantics: does this layer run the sparse path?

    True for "full" (computes its own topk) and "shared" (reads the previous
    full layer's).  False where the configuration declares no DSA layer: such
    a model stays on the dense causal branch instead of reading a topk buffer
    that no full layer ever writes.  ``index_topk`` alone is not that test --
    it is a config-wide field, so a checkpoint that carries it without
    ``indexer_types`` would otherwise land on the sparse path everywhere.
    """
    return glm_indexer_type(config, layer_id) in _GLM_DSA_LAYER_TYPES


class EvasMLACacheLayerBase(nn.Module, AttentionLayerBase):
    """Common base for GLM MLA paged KV-cache holders.

    Concrete layers: ``EvasGLMMLAAttentionLayer`` (the MLA paged cache) and
    ``EvasGLMIndexerCacheLayer`` (registration-only, ``get_kv_cache_spec``
    returns ``None``, buffer managed by ``_EvasGLMSparseIndexerOp``).

    Provides: empty ``kv_cache`` (vLLM binds the real one), ``kv_block_size``
    exposure for gather/write helpers, duplicate-prefix guard +
    self-registration into ``compilation_config.static_forward_context[prefix]``,
    ``get_attn_backend`` returning ``EvasGLMMLABackend``, and a no-op
    ``forward`` (real compute lives in ``EvasGLMMLAAttention.forward``).
    """

    def __init__(self, head_dim: int, dtype: torch.dtype, prefix: str,
                 block_size: int = MLA_KV_BLOCK_SIZE):
        super().__init__()
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        self.kv_block_size = block_size
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        # EvasGLMMLABackend is imported at module scope in this file (the lazy
        # import from the standalone layers module is gone with the merge).
        return EvasGLMMLABackend

    def forward(self):
        # Cache-holding module: no-op.  The real attention compute lives in
        # ``EvasGLMMLAAttention``, which reads ``self.kv_cache``.
        ...


class EvasGLMMLAAttentionLayer(EvasMLACacheLayerBase):
    """Paged KV-cache holder for one GLM MLA layer on LPU.

    GLM stores one latent vector per token (compress_ratio=1). Each slot stores
    the raw ``kv_a`` latent (pre-norm, kv_lora_rank=512) + ``k_pe`` (pre-RoPE,
    qk_rope_head_dim=64) = 576 dims; ``kv_b_proj`` expansion and RoPE are applied
    during gather (see ``EvasGLMMLAAttention._glm_paged_gather``).
    """

    def __init__(self, head_dim: int, dtype: torch.dtype, prefix: str):
        super().__init__(head_dim, dtype, prefix)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return EvasGLMMLABackend

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=MLA_KV_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=1,
        )

    def forward(self):
        # Cache-holding module: no-op. The real attention compute lives in
        # EvasGLMMLAAttention, which reads self.kv_cache.
        ...


class EvasGLMIndexerCacheLayer(EvasMLACacheLayerBase):
    """Registration-only holder for one GLM Indexer's 128-dim scoring k on LPU.

    The Indexer scores query tokens against a 128-dim MQA k (index_head_dim)
    to pick topk KV positions. A vLLM-managed paged KV cache
    (``get_kv_cache_spec`` returning an ``MLAAttentionSpec``) is not usable: the 128-dim
    spec cannot merge with the 576-dim MLA spec (different ``model_version``
    forces separation) -> a **non-uniform cache group** of two physically
    separate page buffers — the necessary trigger of the LPU firmware
    ``double_die_copy`` watchdog during warm-up (memory
    ``glm5-indexer-coverage-blocked``).

    Fix: this layer returns ``None`` from ``get_kv_cache_spec``, the
    upstream-legal pattern (``DeepseekV4MLAAttention`` returns None for SWA
    layers, ``deepseek_v4_attention.py:576-580``). The worker skips
    allocating/binding a kv_cache for this layer (``kv_cache`` stays the
    ``torch.tensor([])`` placeholder) and builds no attn_metadata for its
    prefix. The 128-dim buffer is instead lazily allocated by
    ``_EvasGLMSparseIndexerOp`` as a plain paged-layout tensor that mirrors the
    MLA cache's slot layout (same ``storage_block_size``, same
    ``block_table``/``slot_mapping`` semantics), so the Indexer reuses the MLA
    layer's attn_metadata (position-aligned, head-size-independent) to
    read/write its 128-dim k. Cache group stays uniform (576-dim MLA only) ->
    no die-copy trigger, while preserving correct gather.
    """

    def __init__(self, head_dim: int, dtype: torch.dtype, prefix: str,
                 cache_config):
        super().__init__(head_dim, dtype, prefix)
        self.cache_config = cache_config

    def get_attn_backend(self) -> type[AttentionBackend]:
        return EvasGLMMLABackend

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec | None:
        # Returning None keeps this layer OUT of every KV cache group -> the
        # group stays uniform (576-dim MLA only) -> no LPU firmware die-copy
        # watchdog trigger (memory glm5-indexer-coverage-blocked). This also
        # sidesteps the former ``model_version="glm_indexer"`` forced-separation
        # design, which avoided the merge-collapse bug (two MLAAttentionSpecs
        # with different head_size would merge and hand the Indexer a 576-dim
        # view, crashing copy_(k[:128])) by forcing merge to fail — but that
        # non-uniform group was exactly what tripped the watchdog. Returning
        # None is upstream-legal (``DeepseekV4MLAAttention.get_kv_cache_spec``,
        # deepseek_v4_attention.py:576-580; ``model_runner.py`` skips None
        # specs). The 128-dim buffer is lazily allocated by
        # ``_EvasGLMSparseIndexerOp._ensure_buffer`` mirroring the MLA layout.
        return None

    def forward(self):
        # Cache-holding module: no-op. Scoring compute lives in
        # _EvasGLMSparseIndexerOp, which reads self.kv_cache.
        ...


class _EvasGLMSparseIndexerOp(nn.Module):
    """Pure-PyTorch replacement for the CUDA ``SparseAttnIndexer`` on LPU.

    The CUDA ``sparse_attn_indexer`` op has no LPU path (forward_native raises
    NotImplementedError). Computes MQA logits in pure PyTorch and selects topk
    KV positions per query token via ``torch.topk``.

    Inputs (produced by the patched ``Indexer.forward`` pre-op, mirroring
    upstream deepseek_v2.py:681-727):
      - q: [num_tokens, n_head, head_dim] bf16 (scale folded into ``weights``
            upstream-style if the FP8 quant path is taken).
      - k: [num_tokens, head_dim] current tokens' 128-dim MQA k, post-RoPE,
            post-k_norm (to be written into the Indexer paged cache).
      - weights: [num_tokens, n_head] (contains ``softmax_scale * n_head**-0.5``,
            folded by the pre-op per upstream line 722-725).

    Returns: topk_indices [num_tokens, topk_tokens] int32, position indices
    into the MLA paged cache (0..S-1); -1 marks a non-selection, either the
    tail padding when S_total < topk_tokens or a position outside the query
    row's visible window (see ``forward``).

    Math (spec §4.5-A, aligned to upstream Indexer.forward):
      logits[t,h,s] = (q[t,h,:] . k_full[s,:]) * weights[t,h]
      aggregate: logits.sum(dim=head) -> [num_tokens, S_total]
      topk_indices = torch.topk(logits_summed, topk, dim=-1).indices

    The sum-over-heads aggregation is the MQA "32 heads vote on each KV
    position" semantics. (Spec §4.5-B: the most-likely divergence point from
    the opaque ``top_k_per_row`` CUDA kernel; if real-weight numerics show a
    per-head max/mean instead, adjust here. Dummy weights cannot validate this.)
    """

    def __init__(self, k_cache, topk_tokens: int, head_dim: int,
                 n_head: int, softmax_scale: float, quant_block_size: int,
                 *, mla_cache_layer=None, device=None):
        super().__init__()
        self.k_cache = k_cache              # EvasGLMIndexerCacheLayer | None
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.n_head = n_head
        self.softmax_scale = softmax_scale
        self.quant_block_size = quant_block_size
        # Op-managed 128-dim scoring buffer (replaces the former vLLM-managed
        # paged KV cache), lazily allocated by ``_ensure_buffer`` to MIRROR the
        # MLA cache's direct layout [num_blocks, 1, block_size, head_dim]. The
        # MLA layer's attn_metadata (block_table/slot_mapping/seq_lens/
        # query_start_loc, all head-size-independent) then indexes it correctly
        # via (block, offset) pairs. ``mla_cache_layer`` is the MLA
        # ``EvasGLMMLAAttentionLayer``; its bound kv_cache shape
        # [num_blocks, 1, 256, 576] gives the exact dims the op buffer must
        # match. The MLA kv_cache is unbound at construction (bound later by the
        # worker), so the buffer is sized lazily on first forward; ``device``
        # pins the buffer device.
        self.mla_cache_layer = mla_cache_layer
        self.device = device
        self._flat_buffer = None            # [num_blocks, 1, block_size, head_dim] bf16

    def _ensure_buffer(self, ref: torch.Tensor) -> torch.Tensor | None:
        """Lazily allocate the op-managed 128-dim scoring buffer if absent.

        Sized to MATCH the MLA cache's full shape ``(num_blocks, 1, 256,
        head_dim)`` (direct layout with kernel_block_size=256), so the MLA
        layer's ``block_table``/``slot_mapping`` (slot ``blk*256 + offset``)
        land on the right 128-dim row. Returns ``None`` (caller falls back to
        single-step) when the MLA cache is unbound (``kv_cache.numel() == 0``),
        e.g. warm-up/profile_run or unit tests without a real MLA cache —
        mirrors the MLA gather/write no-cache fallback.
        """
        if self._flat_buffer is not None:
            return self._flat_buffer
        mla = self.mla_cache_layer
        if mla is None or mla.kv_cache.numel() == 0:
            return None
        # Mirror the MLA cache's full shape (num_blocks, 1, block_size, head_dim)
        # so the same (block, offset) indexing works for both caches.
        num_blocks = mla.kv_cache.shape[0]
        block_size = mla.kv_cache.shape[2]
        device = self.device if self.device is not None else ref.device
        self._flat_buffer = torch.zeros(
            num_blocks, 1, block_size, self.head_dim,
            dtype=torch.bfloat16, device=device)
        return self._flat_buffer

    def forward(self, hidden_states, q, k, weights, attn_metadata) -> torch.Tensor:
        k_full = self._indexer_gather_k(k, attn_metadata)

        N = q.shape[0]
        S_total = k_full.shape[0]

        # mqa_logits: [N, n_head, S_total] = q . k_full^T, scaled by weights.
        # fp32 accumulate for LPU bf16 correctness.
        q_f = q.to(torch.float32)
        k_f = k_full.to(torch.float32)
        logits = torch.einsum('thd,sd->ths', q_f, k_f)  # [N, n_head, S_total]
        w = weights.to(torch.float32).unsqueeze(-1)      # [N, n_head, 1]
        logits = logits * w                              # broadcast over S

        # Aggregate over heads (MQA voting) -> [N, S_total], then topk.
        logits_summed = logits.sum(dim=1)                # [N, S_total]
        topk = min(self.topk_tokens, S_total)
        if topk <= 0:
            return torch.full((N, self.topk_tokens), -1, dtype=torch.int32,
                              device=q.device)
        # Per-query-row visible window. The sparse SDPA downstream runs with
        # is_causal=False and no padding mask, so causality AND request isolation
        # rest entirely on this topk excluding positions the query may not see --
        # upstream keeps the causal limit in its indexer metadata (per-token
        # effective context lengths) and the LPU path must too: the gather
        # concatenates every request's rows in request order, so without the
        # window a prefill chunk's early row could select its own later tokens,
        # and any row of a later request could select an earlier request's rows.
        win = glm_indexer_causal_window(attn_metadata, N, S_total, q.device)
        if win is not None:
            lo, hi = win
            col = torch.arange(S_total, device=q.device,
                               dtype=torch.float32).unsqueeze(0)
            # Two masked_fill calls rather than a combined bool mask: the LPU
            # aten surface is narrow and a single comparison per call is the
            # shape already exercised on device.
            logits_summed = logits_summed.masked_fill(
                col < lo.unsqueeze(1), float("-inf"))
            logits_summed = logits_summed.masked_fill(
                col >= hi.unsqueeze(1), float("-inf"))
        topk_vals, topk_idx = torch.topk(logits_summed, topk, dim=-1)
        if win is not None:
            # A row whose visible set is smaller than the topk width still
            # returns topk picks, the surplus being -inf-scored columns outside
            # the window. They are not real selections: mark them -1 so the
            # consumer drops (masks) them. The finite picks rank first, so their
            # count is min(width, topk); float arithmetic + one cast avoids the
            # int masked_fill / clamp the LPU does not serve.
            width = torch.clamp(hi - lo, max=float(topk))
            keep = torch.arange(topk, device=q.device, dtype=torch.float32) \
                .unsqueeze(0) < width.unsqueeze(1)
            keepf = keep.to(torch.float32)
            topk_idx = (topk_idx.to(torch.float32) * keepf - (1.0 - keepf)) \
                .to(torch.int32)
        else:
            topk_idx = topk_idx.to(torch.int32)
        # Pad to topk_tokens columns with -1 if S_total < topk_tokens.
        if topk < self.topk_tokens:
            pad = torch.full((N, self.topk_tokens - topk), -1,
                             dtype=torch.int32, device=q.device)
            topk_idx = torch.cat([topk_idx, pad], dim=-1)
        return topk_idx  # [N, topk_tokens] int32, -1 non-selection

    def _indexer_gather_k(self, k: torch.Tensor, attn_metadata) -> torch.Tensor:
        """Write current k into the Indexer 128-dim buffer then gather history.

        The buffer is the op-managed paged-layout tensor from ``_ensure_buffer``
        (NOT a vLLM-managed kv_cache: ``get_kv_cache_spec`` returns None, so
        none is bound). It mirrors the MLA cache's direct
        layout, so the MLA layer's attn_metadata (all head-size-independent)
        indexes it correctly via (block, offset) pairs. With no buffer
        (``attn_metadata`` None / unit tests with ``k_cache=None``) returns the
        passed ``k`` as the full history (single-step) — mirrors
        ``EvasGLMMLAAttention._glm_paged_gather``'s no-cache fallback.
        """
        if attn_metadata is None:
            return k
        kv_cache = self._ensure_buffer(k)
        if kv_cache is None:
            # No sizing info (test path / max_model_len==0): single-step.
            return k

        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        num_reqs = attn_metadata.num_reqs
        slot_mapping = attn_metadata.slot_mapping
        # Take the block size from the buffer being indexed (``_ensure_buffer``
        # mirrors the MLA cache's shape[2]), not from the module constant, so
        # the (blk, offset) split and the run capping can never disagree with
        # the tensor they address. Direct layout -> the shared helpers' direct
        # path (cache[blk, 0, offset]) lands on the right 128-dim row.
        storage_block_size = kv_cache.shape[2]

        # Write current tokens (whole 128-dim vector per slot, no pe split) via
        # the shared run-grouped helper: it moves slot_mapping to the host in a
        # single read and coalesces consecutive slots into one slice copy, where
        # a per-token loop would pay one D2H slot read plus one copy_ per token.
        _write_paged_runs(kv_cache, k, slot_mapping, storage_block_size)

        # Gather full history per request (position-aligned with MLA cache),
        # block-aligned so each host sync covers a whole physical block instead
        # of a single slot (same shape as EvasGLMMLAAttention's gather).
        k_list: list[torch.Tensor] = []
        for r in range(num_reqs):
            seq_len_r = int(seq_lens[r].item())
            k_r = torch.zeros(seq_len_r, self.head_dim, dtype=k.dtype,
                              device=k.device)
            s = 0
            while s < seq_len_r:
                manager_idx = s // storage_block_size
                run_end = min((manager_idx + 1) * storage_block_size, seq_len_r)
                chunk = _read_contiguous_run(
                    kv_cache, block_table, r, s, run_end,
                    storage_block_size, False)
                k_r[s:run_end] = chunk[:, :self.head_dim]
                s = run_end
            k_list.append(k_r)
        return torch.cat(k_list, dim=0) if k_list else k


def _get_mla_attn_metadata_for_indexer(indexer):
    """Resolve this Indexer's MLA layer attn_metadata (via wired op)."""
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return None
    ctx = get_forward_context().attn_metadata
    if not isinstance(ctx, dict):
        return ctx
    evas_op = getattr(indexer, "_evas_indexer_op", None)
    mla_layer = getattr(evas_op, "mla_cache_layer", None)
    mla_prefix = getattr(mla_layer, "prefix", None)
    if mla_prefix is None:
        return None
    return ctx.get(mla_prefix)


class EvasGLMIndexer(nn.Module):
    """LPU GLM Indexer — upstream field layout, bf16 plain cache, pure-PyTorch op.

    Does NOT call the upstream ``Indexer.__init__``: that builds FP8
    ``DeepseekV32IndexerCache`` (non-uniform cache group → LPU die-copy
    watchdog) and CUDA ``SparseAttnIndexer`` (no LPU path). Field names mirror
    upstream so ``load_weights`` (wq_b / wk_weights_proj / k_norm) and the
    attention-side call site stay identical. Signature matches upstream
    ``Indexer.__init__`` so the rebind factory is a drop-in.

    ``indexer_rope_emb`` is intentionally absent: upstream attention sets it
    on *self* before constructing the Indexer (deepseek_v2.py:1030).
    """

    def __init__(self, vllm_config, config, hidden_size, q_lora_rank,
                 quant_config, cache_config, topk_indices_buffer,
                 prefix="", is_inplace_rope=False):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = None
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim ** -0.5
        # Fields upstream/impl helpers may read (upstream Indexer sets both
        # too):
        # max_model_len guards buffer sizing, scale_fmt documents the FP8 no-op
        # contract of this bf16 path (quant override is a no-op on evas).
        self.scale_fmt = "ue8m0"
        self.max_model_len = vllm_config.model_config.max_model_len
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer
        self.prefix = prefix
        self.is_inplace_rope = is_inplace_rope

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank, self.head_dim * self.n_head, bias=False,
            quant_config=None, prefix=f"{prefix}.wq_b")
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size, [self.head_dim, self.n_head], bias=False,
            quant_config=None, disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj")
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)

        self.k_cache = EvasGLMIndexerCacheLayer(
            head_dim=self.head_dim, dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache", cache_config=cache_config)
        self._evas_indexer_op = _EvasGLMSparseIndexerOp(
            k_cache=self.k_cache, topk_tokens=self.topk_tokens,
            head_dim=self.head_dim, n_head=self.n_head,
            softmax_scale=self.softmax_scale,
            quant_block_size=self.quant_block_size)
        self.indexer_op = None

    def forward(self, hidden_states, qr, positions, rotary_emb):
        # Pure-PyTorch MQA logits path; quant override is a no-op on evas bf16
        # so no per_token_group_quant_fp8 call here.
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, _ = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim:]
        k = self.k_norm(k)
        k_pe, _ = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        # RoPE is applied in place: RotaryEmbeddingFL.forward_oot calls
        # _rotary_embedding with inplace=True and writes through the strided
        # q_pe / k_pe views, so the rotated head lands back inside q and k. The
        # returned tensors alias those views and carry nothing extra, hence the
        # call is made for its side effect only.
        rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Rebuild [rope|nope] without torch.cat: q_pe and the nope half are two
        # non-contiguous last-dim slices of this same q buffer, and catting
        # strided views makes the LPU materialize a per-call strided copy that
        # trips the 32.768 s firmware watchdog (same class as the MLA fix in
        # mla_attention.py; note the split order here is pe-first). q already
        # holds the rotated head, so clone() keeps the fresh-tensor semantics of
        # the old cat in one contiguous copy.
        q = q.clone().view(-1, self.head_dim)
        q_fp8 = q
        q_scale = torch.ones(
            q.shape[0], 1, dtype=torch.bfloat16, device=q.device)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head ** -0.5
        ).squeeze(-1)
        q_bf16 = q_fp8.to(torch.bfloat16)
        # Same strided-cat hazard as q above: k_pe is a slice of this k buffer
        # that the rope already rewrote in place, so clone() over cat().
        k = k.clone()
        topk_indices = self._evas_indexer_op(
            hidden_states, q_bf16, k, weights,
            attn_metadata=_get_mla_attn_metadata_for_indexer(self))
        n = hidden_states.shape[0]
        self.topk_indices_buffer[:n, : self.topk_tokens] = topk_indices
        return self.topk_indices_buffer


def make_evas_glm_indexer_factory(orig_indexer_cls):
    """Wrap the upstream ``Indexer`` class with GLM/LPU dispatch (rebind target).

    Dispatch (call site = upstream attention ``__init__``, deepseek_v2.py:1036):
      - non-GLM config  → ``orig_indexer_cls(...)`` unchanged (deepseek-v3.1 etc.)
      - GLM, not needed → ``None`` (shared layer — HF ``indexer_types``
        semantics; None avoids phantom ``static_forward_context`` registration)
      - GLM, needed     → ``EvasGLMIndexer(...)``
    """
    def _fl_evas_glm_indexer_factory(
        vllm_config, config, hidden_size, q_lora_rank, quant_config,
        cache_config, topk_indices_buffer, prefix="", is_inplace_rope=False,
    ):
        if not is_glm_config(vllm_config):
            return orig_indexer_cls(
                vllm_config, config, hidden_size, q_lora_rank, quant_config,
                cache_config, topk_indices_buffer, prefix=prefix,
                is_inplace_rope=is_inplace_rope)
        layer_id = extract_layer_index(prefix)
        if not glm_indexer_required(config, layer_id):
            return None
        return EvasGLMIndexer(
            vllm_config, config, hidden_size, q_lora_rank, quant_config,
            cache_config, topk_indices_buffer, prefix=prefix,
            is_inplace_rope=is_inplace_rope)

    _fl_evas_glm_indexer_factory._fl_evas_glm_factory = True
    return _fl_evas_glm_indexer_factory
