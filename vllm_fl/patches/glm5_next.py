# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next patches for FlagOS platforms (Hygon DCU and friends).

vLLM ships GLM-5-Next as an NVIDIA-only implementation
(``vllm/models/glm5next/nvidia/``). The model code itself is almost entirely
Triton and portable; what is not portable is the sparse kpool *indexer*
(``model_executor/layers/sparse_attn_indexer_kpool.py``), which reaches for
three things this platform does not have:

1. ``vllm._custom_ops`` -- bound only under ``is_cuda_alike()`` / ``is_xpu()``.
   PlatformFL is neither (it reports ``PlatformEnum.OOT``), so the module-level
   name ``ops`` is left *undefined* and the first call raises ``NameError``.
2. DeepGEMM ``fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits`` -- the core
   sparse-logits kernels, hand-tuned CUDA.
3. ``torch.ops._C.top_k_per_row_{prefill,decode}`` -- radix top-k CUDA kernels.
   This vLLM is built with ``VLLM_TARGET_DEVICE=empty`` (zero ``.so``), so the
   plugin registers their *schemas* but no implementations.

FlagGems provides a Triton implementation of every one of them, with signatures
that already match vLLM's call sites (they were written against this exact API).
So these patches are pure rebinding -- no new kernels.

Finally, ``SparseAttnIndexerKpool.forward_native`` is not a native
implementation at all: it re-dispatches to ``forward_cuda``/``forward_hip`` and
raises otherwise. Since ``CustomOp.dispatch_forward`` routes out-of-tree
platforms to ``forward_oot`` -> ``forward_native``, DCU hit that raise. We bind
``forward_oot`` to ``forward_cuda`` directly; the underlying custom op is
FlagGems-backed once the rebinds above are in place.
"""

from vllm.logger import init_logger

# Use vLLM's logger, not a bare logging.getLogger: vLLM installs its own
# dictConfig for the "vllm" hierarchy, so a plain module logger produces no
# output and these patches would apply invisibly.
logger = init_logger(__name__)


def _patch_indexer_kpool_ops() -> bool:
    """Point the kpool indexer's CUDA/DeepGEMM symbols at FlagGems Triton."""
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as mod

    if getattr(mod, "_fl_patched_kpool", False):
        return True

    import flag_gems

    # (1) `ops` -- the module never bound it on this platform. Provide a shim
    # exposing just the two entry points the indexer actually uses.
    class _FlagGemsIndexerOps:
        """Stand-in for ``vllm._custom_ops`` covering the kpool indexer."""

        indexer_k_quant_and_cache = staticmethod(flag_gems.indexer_k_quant_and_cache)
        cp_gather_indexer_k_quant_cache = staticmethod(
            flag_gems.cp_gather_indexer_k_quant_cache
        )

    mod.ops = _FlagGemsIndexerOps

    # (2) DeepGEMM sparse-logits kernels. FlagGems keeps `schedule_metadata` in
    # its signature for API compatibility but does not read it -- which also
    # means the zeroed metadata buffer this platform produces (vLLM only fills
    # it under `is_cuda()`) cannot silently corrupt results.
    mod.fp8_fp4_mqa_logits = flag_gems.fp8_fp4_mqa_logits
    # Take the capture-safe wrapper if it is already installed.
    _patch_paged_mqa_logits_capture_safe()
    mod.fp8_fp4_paged_mqa_logits = flag_gems.fp8_fp4_paged_mqa_logits

    mod._fl_patched_kpool = True
    logger.info(
        "FL: kpool indexer now uses FlagGems (indexer_k_quant_and_cache, "
        "cp_gather_indexer_k_quant_cache, fp8_fp4_mqa_logits, "
        "fp8_fp4_paged_mqa_logits)"
    )
    return True


def _patch_indexer_forward_oot() -> None:
    """Route the OOT dispatch to the real implementation.

    ``forward_native`` only re-dispatches and raises on unknown platforms, so
    the default ``forward_oot -> forward_native`` chain is a dead end here.
    ``forward_cuda`` just calls ``torch.ops.vllm.sparse_attn_indexer_kpool``,
    whose internals are FlagGems-backed after ``_patch_indexer_kpool_ops``.
    """
    from vllm.model_executor.layers.sparse_attn_indexer_kpool import (
        SparseAttnIndexerKpool,
    )

    if getattr(SparseAttnIndexerKpool, "_fl_patched_forward_oot", False):
        return

    SparseAttnIndexerKpool.forward_oot = SparseAttnIndexerKpool.forward_cuda
    SparseAttnIndexerKpool._fl_patched_forward_oot = True
    logger.info("FL: SparseAttnIndexerKpool.forward_oot -> forward_cuda")


def _register_topk_per_row_ops() -> None:
    """Supply CUDA implementations for the two ``_C`` top-k schemas.

    ``vllm_fl.ops._C_ops_registry`` defines the schemas so torch.compile can
    see them, but leaves them unimplemented; the kpool indexer calls both on
    the non-XPU path. Both FlagGems functions write into ``indices`` in place
    and return None, matching the ``-> ()`` schema exactly.
    """
    import torch

    if getattr(_register_topk_per_row_ops, "_done", False):
        return

    import flag_gems

    lib = torch.library.Library("_C", "FRAGMENT")

    def _prefill(logits, row_starts, row_ends, indices, num_rows, stride0, stride1,
                 top_k):
        flag_gems.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )
        # FlagGems writes *every* top_k slot, sourcing them from a
        # zero-initialised candidate buffer -- so a row with fewer than top_k
        # valid columns gets index 0 in the tail rather than a negative
        # sentinel. vLLM allocates this buffer with torch.empty and treats any
        # non-negative entry as a real selection (`pool_ids >= 0`), so that
        # padding silently becomes hundreds of duplicate references to pool 0.
        # With kpool the tail dominates short prompts (select_k = 2048/4 = 512
        # slots vs. a handful of real pools), which wrecks attention: measured
        # 510/512 phantom entries on a 5-token prompt, giving finite-but-wrong
        # logits (no NaN, so nothing downstream complains).
        counts = (
            row_ends[:num_rows].to(torch.long) - row_starts[:num_rows].to(torch.long)
        ).clamp_(min=0, max=top_k)
        ar = torch.arange(top_k, device=indices.device).unsqueeze(0)
        indices[:num_rows].masked_fill_(ar >= counts.unsqueeze(1), -1)

    def _decode_batched(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    ):
        """Batched torch top-k per row for the decode path.

        FlagGems' radix kernel asserts ``num_rows == 1``, but vLLM calls this
        with one row per (request, spec-token) pair -- up to max_num_seqs, and
        CUDA-graph capture exercises the full range. Fall back to torch here.

        Masking is not optional: the caller invokes paged-MQA logits with
        ``clean_logits=False``, so positions past a row's valid length hold
        ``0.0`` rather than ``-inf`` and would otherwise outrank genuinely
        negative logits.

        The output buffer is ``torch.empty``, and the consumer treats any
        negative entry as "no pool selected" (``pool_ids >= 0``), so every slot
        must be written -- valid indices first, then ``-1`` padding.
        """
        n_cols = logits.shape[1]

        # Row -> valid length. Mirrors FlagGems' fp8_fp4_paged_mqa_logits:
        # seq_lens is (B, next_n) for native spec decode, (B, 1) otherwise.
        if seq_lens.dim() == 2:
            lens = seq_lens.reshape(-1)[:num_rows]
        else:
            lens = seq_lens.repeat_interleave(max(next_n, 1))[:num_rows]
        lens = lens.to(torch.long).clamp_(min=0, max=n_cols)

        cols = torch.arange(n_cols, device=logits.device)
        valid = cols.unsqueeze(0) < lens.unsqueeze(1)
        masked = torch.where(
            valid,
            logits[:num_rows].to(torch.float32),
            torch.full((), float("-inf"), device=logits.device, dtype=torch.float32),
        )

        k = min(top_k, n_cols)
        _, idx = torch.topk(masked, k, dim=1)

        # Exactly min(lens[i], k) of row i's picks are real; the rest are the
        # -inf padding we just inserted. Derive validity from the counts rather
        # than comparing against -inf (a row with no valid columns has none).
        ar = torch.arange(k, device=logits.device).unsqueeze(0)
        keep = ar < lens.clamp(max=k).unsqueeze(1)
        indices[:num_rows, :k] = torch.where(
            keep, idx.to(torch.int32), torch.full_like(idx, -1, dtype=torch.int32)
        )
        if k < top_k:
            indices[:num_rows, k:] = -1

    def _decode(logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k):
        # Always take the batched torch path. FlagGems' radix decode kernel not
        # only asserts num_rows == 1, it was also observed to leave the output
        # buffer *completely untouched* for a short-context single row -- and
        # since vLLM allocates that buffer with torch.empty, the indexer would
        # then consume uninitialised memory. The torch version below is checked
        # against a per-row torch.topk reference, including -1 padding.
        _decode_batched(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    for name, fn in (
        ("top_k_per_row_prefill", _prefill),
        ("top_k_per_row_decode", _decode),
    ):
        try:
            lib.impl(name, fn, "CUDA")
        except Exception as e:  # already implemented by a real extension
            logger.debug("FL: skip _C.%s impl: %s", name, e)

    # Keep the Library alive; it deregisters when garbage collected.
    _register_topk_per_row_ops._lib = lib
    _register_topk_per_row_ops._done = True
    logger.info("FL: registered FlagGems top_k_per_row_{prefill,decode} for _C")


def _install_lazy_indexer_hook() -> None:
    """Apply the indexer patches the moment that module finishes importing.

    We cannot import ``sparse_attn_indexer_kpool`` from ``register_model()``:
    that module imports ``vllm.models.glm5next.nvidia.ops.kpool_compress``,
    and ``vllm/models/glm5next/__init__.py`` imports the model, which imports
    the indexer again -- a cycle that only resolves if the *model* package is
    what starts the chain. Importing it ourselves first raises
    ``KeyError: 'vllm.models.glm5next'``.

    So instead of forcing the import, install a meta-path hook that fires
    right after vLLM imports the module on its own (during model load).
    If it is somehow already imported, patch immediately.
    """
    import sys

    if "vllm.model_executor.layers.sparse_attn_indexer_kpool" in sys.modules:
        _patch_indexer_kpool_ops()
        _patch_indexer_forward_oot()
        return

    import importlib.abc

    target = "vllm.model_executor.layers.sparse_attn_indexer_kpool"

    class _IndexerPatchFinder(importlib.abc.MetaPathFinder):
        """No-op finder that patches the target module post-import."""

        def find_module(self, fullname, path=None):  # legacy API, unused
            return None

        def find_spec(self, fullname, path=None, target_module=None):
            if fullname != target:
                return None
            # Let the normal finders build the spec, then wrap its loader so we
            # run right after execution completes.
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None

            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    _patch_indexer_kpool_ops()
                    _patch_indexer_forward_oot()
                except Exception as e:  # pragma: no cover
                    logger.error("FL: kpool indexer patch failed: %s", e,
                                 exc_info=True)
                finally:
                    if self in sys.meta_path:
                        sys.meta_path.remove(self)

            spec.loader.exec_module = exec_module
            return spec

    import importlib.util  # noqa: F401  (used above)

    sys.meta_path.insert(0, _IndexerPatchFinder())
    logger.info("FL: armed GLM-5-Next kpool indexer patch (applies on import)")


def _register_cache_ops() -> None:
    """Register the ``_C_cache_ops`` namespace, backed by FlagGems.

    ``vllm/_custom_ops.py`` routes MLA KV-cache writes through
    ``torch.ops._C_cache_ops.*``, but the plugin's ``_C_ops_registry`` declares
    every schema -- including the cache ops -- in the ``_C`` namespace. Nothing
    noticed until now because the FlagGems attention backend calls
    ``flag_gems.reshape_and_cache_flash`` directly and never goes through
    ``_custom_ops``. MLA does: ``AttentionImpl.do_kv_cache_update`` calls
    ``ops.concat_and_cache_mla``, which hit an empty ``_C_cache_ops`` namespace.

    Register the ops GLM-5-Next's MLA path needs, with FlagGems Triton kernels.
    Signatures already line up with vLLM's wrappers.
    """
    import torch

    if getattr(_register_cache_ops, "_done", False):
        return

    import flag_gems

    SCHEMAS = {
        "concat_and_cache_mla": (
            "concat_and_cache_mla(Tensor kv_c, Tensor k_pe, Tensor! kv_cache, "
            "Tensor slot_mapping, str kv_cache_dtype, Tensor scale) -> ()",
            "concat_and_cache_mla",
        ),
        "indexer_k_quant_and_cache": (
            "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache, "
            "Tensor slot_mapping, int quant_block_size, str kv_cache_dtype) -> ()",
            "indexer_k_quant_and_cache",
        ),
        "cp_gather_indexer_k_quant_cache": (
            "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, "
            "Tensor! dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()",
            "cp_gather_indexer_k_quant_cache",
        ),
    }

    lib = torch.library.Library("_C_cache_ops", "FRAGMENT")
    registered = []
    for op_name, (schema, gems_name) in SCHEMAS.items():
        gems_fn = getattr(flag_gems, gems_name, None)
        if gems_fn is None:
            logger.warning("FL: FlagGems has no %s; skipping", gems_name)
            continue
        if hasattr(torch.ops._C_cache_ops, op_name):
            continue
        try:
            lib.define(schema)
            # Bind through a lambda so the FlagGems symbol is resolved lazily
            # and the returned value is discarded (schemas declare `-> ()`).
            def _make(fn):
                def _impl(*args):
                    fn(*args)

                return _impl

            lib.impl(op_name, _make(gems_fn), "CUDA")
            registered.append(op_name)
        except Exception as e:
            logger.warning("FL: could not register _C_cache_ops.%s: %s", op_name, e)

    _register_cache_ops._lib = lib
    _register_cache_ops._done = True
    if registered:
        logger.info("FL: registered _C_cache_ops via FlagGems: %s",
                    ", ".join(registered))


def _constrain_sparse_mla_autotune() -> None:
    """Keep FlagGems' sparse-MLA autotune configs inside the DCU LDS budget.

    ``flag_gems.fused.flashmla_sparse.triton_flash_mla_sparse_fwd`` only offers
    ``BK=64, BH=64`` configs at ``num_stages`` 2 and 4. Those were tuned for
    NVIDIA's 228 KB shared memory; software pipelining double-buffers the q/kv
    tiles, which needs 72 KB and blows past this hardware's 64 KB limit:

        OutOfResources: shared memory, Required: 73728, Hardware limit: 65536

    Measured on gfx9xx (BK, BH, num_stages): every ``num_stages=1`` variant
    fits, including the full 64/64 tile; 64/16 already needs 66 KB at
    ``num_stages=2``. So clamp ``num_stages`` to 1 and keep the tile shape --
    that costs pipelining, not tile efficiency, and avoids shrinking BK/BH.

    Applied only when the device really is short on shared memory, so this is a
    no-op on hardware that can pipeline.
    """
    if getattr(_constrain_sparse_mla_autotune, "_done", False):
        return

    import torch
    import triton

    import importlib

    try:
        # importlib, not `from flag_gems.fused import ...`: some submodules are
        # shadowed by same-named function re-exports in that package's __init__.
        fms = importlib.import_module("flag_gems.fused.flashmla_sparse")
    except Exception as e:
        logger.warning("FL: FlagGems flashmla_sparse unavailable: %s", e)
        return

    kern = getattr(fms, "triton_flash_mla_sparse_fwd", None)
    configs = getattr(kern, "configs", None)
    if not configs:
        return

    limit = getattr(
        torch.cuda.get_device_properties(0), "shared_memory_per_block", 0
    )
    # NVIDIA SM90+ has >= 200 KB; only clamp on the tighter budgets.
    if limit and limit > 96 * 1024:
        return

    clamped, seen = [], set()
    for c in configs:
        key = (tuple(sorted(c.kwargs.items())), c.num_warps)
        if key in seen:
            continue
        seen.add(key)
        clamped.append(
            triton.Config(dict(c.kwargs), num_warps=c.num_warps, num_stages=1)
        )
    if not clamped:
        return

    kern.configs = clamped
    if hasattr(kern, "cache"):
        kern.cache.clear()
    _constrain_sparse_mla_autotune._done = True
    logger.info(
        "FL: clamped sparse-MLA autotune to num_stages=1 for the %d KiB "
        "shared-memory limit (%d config(s))",
        limit // 1024,
        len(clamped),
    )


def _patch_paged_mqa_logits_capture_safe() -> None:
    """Make FlagGems' paged-MQA logits safe under CUDA-graph capture.

    ``flag_gems.fused.fp8_fp4_paged_mqa_logits`` derives its launch shape from
    the *runtime* context lengths::

        max_ctx = int(context_lens.max().item())     # _preprocess_kv_cache
        BLOCK_KV, _ = _select_block_kv(max_ctx, block_size)
        grid = (triton.cdiv(max_ctx, BLOCK_KV), total_rows)

    Two problems when vLLM captures the FULL decode graph:

    1. ``.item()`` is a device->host sync, which HIP rejects on a capturing
       stream ("operation not permitted when stream is capturing"). The
       DeepGEMM kernel this replaces has no such sync, so upstream never hit
       it. ``eager_break_during_capture`` does not help -- it runs the op
       inline once ``cudagraph_runtime_mode`` is FULL.
    2. More importantly, the grid is *baked into the captured graph*. Capture
       runs with dummy (short) context lengths, so a grid sized from them would
       cover only part of the context at replay and silently drop KV tiles.

    Both are fixed the same way: while capturing, size the launch from
    ``max_model_len`` -- the true upper bound for any replay -- instead of the
    live lengths. The kernel already masks per row against the on-device
    ``ctx_len`` (``end_pos = min(kv_start + BLOCK_KV, ctx_len)``), so the extra
    tiles are no-ops.

    ``context_lens`` is used for nothing else inside ``_preprocess_kv_cache``,
    so we hand the original function a small CPU tensor carrying that bound --
    ``.max().item()`` on CPU costs nothing and never touches the stream. This
    keeps FlagGems as the single source of truth for the actual reshaping.
    """
    if getattr(_patch_paged_mqa_logits_capture_safe, "_done", False):
        return

    import torch

    import importlib

    try:
        # NOTE: `from flag_gems.fused import fp8_fp4_paged_mqa_logits` returns
        # the *function*, not the module -- flag_gems/fused/__init__.py
        # re-exports a function with the same name as the submodule and the
        # binding wins. Go through importlib to get the module itself.
        fmod = importlib.import_module("flag_gems.fused.fp8_fp4_paged_mqa_logits")
    except Exception as e:
        logger.warning("FL: FlagGems fp8_fp4_paged_mqa_logits unavailable: %s", e)
        return

    orig_pre = getattr(fmod, "_preprocess_kv_cache", None)
    orig_fn = getattr(fmod, "fp8_fp4_paged_mqa_logits", None)
    if orig_pre is None or orig_fn is None:
        logger.warning(
            "FL: cannot make paged-MQA logits capture-safe "
            "(_preprocess_kv_cache=%s, fp8_fp4_paged_mqa_logits=%s); "
            "FULL cudagraph capture will fail on the .item() sync.",
            orig_pre is not None,
            orig_fn is not None,
        )
        return

    # Communicates the current call's max_model_len to the wrapped
    # _preprocess_kv_cache without changing its signature.
    state = {"bound": None}

    def _preprocess_kv_cache(kv_cache, block_tables, context_lens, total_rows,
                             next_n_val):
        if torch.cuda.is_current_stream_capturing() and state["bound"]:
            context_lens = torch.tensor(
                [state["bound"]], dtype=torch.int32
            )  # CPU tensor: .max().item() does not sync the device
        return orig_pre(kv_cache, block_tables, context_lens, total_rows,
                        next_n_val)

    def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
                                 schedule_metadata, max_model_len,
                                 clean_logits=False):
        state["bound"] = max_model_len
        try:
            return orig_fn(q, kv_cache, weights, context_lens, block_tables,
                           schedule_metadata, max_model_len,
                           clean_logits=clean_logits)
        finally:
            state["bound"] = None

    fmod._preprocess_kv_cache = _preprocess_kv_cache
    fmod.fp8_fp4_paged_mqa_logits = fp8_fp4_paged_mqa_logits

    # flag_gems re-exports the symbol at package level, and the kpool indexer
    # binds it by value; refresh both so everyone sees the wrapper.
    import flag_gems

    flag_gems.fp8_fp4_paged_mqa_logits = fp8_fp4_paged_mqa_logits
    import sys

    kpool = sys.modules.get(
        "vllm.model_executor.layers.sparse_attn_indexer_kpool"
    )
    if kpool is not None:
        kpool.fp8_fp4_paged_mqa_logits = fp8_fp4_paged_mqa_logits

    _patch_paged_mqa_logits_capture_safe._done = True
    logger.info("FL: paged-MQA logits made CUDA-graph-capture safe")


def _patch_bind_kv_cache_oot() -> None:
    """Allow multiple attention layers per layer index on OOT platforms.

    ``bind_kv_cache`` walks layers grouped by layer *index*; when one index owns
    several attention layers it only tolerates that on cuda-alike / xpu / cpu
    and raises ``NotImplementedError`` otherwise. GLM-5-Next hits this on every
    MLA layer -- the main MLA attention, the kpool indexer cache and the tail
    cache all share one index.

    The upstream comment says explicitly that the GPU runner is not impacted by
    this case (``runner_kv_caches`` ordering is only consumed by test code), and
    PlatformFL *is* a GPU runner -- it just reports ``PlatformEnum.OOT`` rather
    than cuda-alike. Extend the allowlist to out-of-tree platforms.
    """
    import vllm.v1.worker.utils as u
    from vllm.platforms import current_platform

    if getattr(u, "_fl_patched_bind_kv_cache", False):
        return
    if not current_platform.is_out_of_tree():
        return

    _orig_is_cuda_alike = current_platform.is_cuda_alike

    def _bind_kv_cache(*args, **kwargs):
        # Only widen the predicate for the duration of the call, so nothing
        # else observes a fake is_cuda_alike().
        current_platform.__class__.is_cuda_alike = lambda self: True
        try:
            return _orig_bind(*args, **kwargs)
        finally:
            current_platform.__class__.is_cuda_alike = _cls_is_cuda_alike

    _orig_bind = u.bind_kv_cache
    _cls_is_cuda_alike = current_platform.__class__.is_cuda_alike
    u.bind_kv_cache = _bind_kv_cache

    # NOTE: vllm_fl/worker/model_runner.py deliberately calls this through the
    # module (`vllm.v1.worker.utils.bind_kv_cache`) rather than the name it
    # imported at module scope, so this patch reaches it even though the runner
    # is imported after register_model() runs.

    u._fl_patched_bind_kv_cache = True
    logger.info("FL: bind_kv_cache accepts multi-layer-per-index on OOT platform")


def _patch_mhc_norm_weight() -> None:
    """Restore the fused RMSNorm that mHC's torch fallback silently drops.

    ``Glm5NextDecoderLayer`` has no standalone layernorm call on the mHC path.
    It passes ``input_layernorm.weight`` / ``post_attention_layernorm.weight``
    into mHC as ``norm_weight`` and relies on the kernel to fold the RMSNorm
    into ``layer_input``. The TileLang kernel documents that contract:

        "When norm_weight is provided, the layer_input_cur output is the
         RMSNorm'd activation (fused into the kernel); otherwise it is the
         raw pre-norm activation as before."

    But ``MHCPreOp.forward_native`` / ``MHCFusedPostPreOp.forward_native``
    accept ``norm_weight``/``norm_eps`` and never forward them -- and
    ``mhc_pre_torch`` has no such parameters at all. On DCU every mHC op
    dispatches out-of-tree -> ``forward_native``, so all 45 layers x 2 sites
    (pre-attn + pre-FFN) lose their RMSNorm.

    Measured on GLM-5-Next layer 0: the probe records an MLP input of
    rms=7.5244e-03, and a CPU replay of hc_pre *without* the norm gives
    7.5309e-03 -- a 0.1% match -- while *with* the norm it is 8.7061e-02
    (11.6x larger). Because SiLU degenerates to x^2/2 near zero, that 11.6x
    linear shortfall becomes ~134x at the activation and ~29x at the MLP
    output, which is exactly the suppression seen on every layer. Dummy
    weights hide it: with a near-constant norm_weight the omission is a pure
    global rescale that softmax washes out.
    """
    import torch
    from vllm.model_executor.layers import mhc as _mhc

    if getattr(_mhc, "_fl_patched_mhc_norm", False):
        return

    def _rmsnorm(x, norm_weight, norm_eps):
        if norm_weight is None:
            return x
        orig_dtype = x.dtype
        f = x.float()
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + norm_eps)
        return (f * norm_weight.float()).to(orig_dtype)

    _orig_pre = _mhc.MHCPreOp.forward_native

    def _pre_native(self, residual, fn, hc_scale, hc_base, rms_eps,
                    hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                    sinkhorn_repeat, n_splits=1, norm_weight=None,
                    norm_eps=0.0):
        post_mix, comb_mix, layer_input = _orig_pre(
            self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits,
            norm_weight, norm_eps,
        )
        return post_mix, comb_mix, _rmsnorm(layer_input, norm_weight, norm_eps)

    _mhc.MHCPreOp.forward_native = _pre_native

    _orig_fused = _mhc.MHCFusedPostPreOp.forward_native

    def _fused_native(self, x, residual, post_layer_mix, comb_res_mix, fn,
                      hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                      hc_post_mult_value, sinkhorn_repeat, n_splits=1,
                      tile_n=1, norm_weight=None, norm_eps=0.0):
        residual_cur, post_mix, comb_mix, layer_input = _orig_fused(
            self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
            hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
            sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps,
        )
        return (residual_cur, post_mix, comb_mix,
                _rmsnorm(layer_input, norm_weight, norm_eps))

    _mhc.MHCFusedPostPreOp.forward_native = _fused_native

    _mhc._fl_patched_mhc_norm = True
    logger.info("FL: mHC forward_native now applies the fused RMSNorm "
                "(norm_weight was being dropped)")


def apply_model_patches() -> None:
    """GLM-5-Next patches applied at model-registration time.

    Must run after vLLM's model/layer modules are importable, i.e. from
    ``register_model()`` rather than ``register()``.
    """
    try:
        _patch_mhc_norm_weight()
        _register_topk_per_row_ops()
        _register_cache_ops()
        _constrain_sparse_mla_autotune()
        _patch_paged_mqa_logits_capture_safe()
        _install_lazy_indexer_hook()
        _patch_bind_kv_cache_oot()
        from vllm_fl.patches.layer_probe import install_layer_probe
        install_layer_probe()
    except Exception as e:  # pragma: no cover
        logger.error("FL: GLM-5-Next patches failed: %s", e, exc_info=True)
