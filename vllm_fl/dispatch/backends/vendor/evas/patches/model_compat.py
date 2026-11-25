# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU model-level compatibility patches (merged).

Two families, both applied by ``apply_lpu_model_compat_patch()``: GLM-5.2
rebinds upstream ``deepseek_v2.py`` components and DSV4 patches the
per-platform ``vllm.models.deepseek_v4`` package -- class swaps, kernel
rebinds and weight-loader fixes installed as module-attribute patches, so
upstream files stay untouched. DSV4 items are skipped when that package is
absent.

"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


# ---- deepseek_v2_mla ----


def _model_mla() -> None:
    """Rebind ``DeepseekV2MLAAttention`` → ``EvasGLMMLAAttention`` (class swap).

    DSV4-style upstream class plus patch: the subclass always runs the upstream
    ``__init__`` (fused_qkv_a_proj / layernorms / q_b·kv_b·o_proj / rope /
    Indexer via factory / MLAModules + wrapper) and overrides ``forward``:
    non-GLM configs delegate to ``super()``; GLM-5.2 runs the LPU path (paged
    4-D cache + flag_gems varlen/SDPA). Replaces the former ``__init__`` /
    ``forward`` monkey-patch pair and the nested own-stack weight layout.
    """
    import vllm.model_executor.models.deepseek_v2 as dsv2
    from vllm_fl.dispatch.backends.vendor.evas.impl.attention.glm_mla_attention import (
        EvasGLMMLAAttention,
    )
    if dsv2.DeepseekV2MLAAttention is EvasGLMMLAAttention:
        return
    dsv2.DeepseekV2MLAAttention = EvasGLMMLAAttention


# ---- deepseek_v2_indexer ----


def _model_indexer() -> None:
    """Rebind ``Indexer`` → GLM/LPU factory (``EvasGLMIndexer`` | ``None`` | upstream).

    Factory (``glm_layers.make_evas_glm_indexer_factory``): non-GLM config →
    upstream class; GLM shared layer → ``None`` (no phantom
    ``static_forward_context`` registration; HF ``indexer_types`` semantics);
    GLM full layer → ``EvasGLMIndexer`` (bf16 ``EvasGLMIndexerCacheLayer`` +
    pure-PyTorch op; never builds FP8 ``DeepseekV32IndexerCache`` → uniform
    cache group → no die-copy watchdog; no CUDA ``SparseAttnIndexer``).
    """
    import vllm.model_executor.models.deepseek_v2 as dsv2
    from vllm_fl.dispatch.backends.vendor.evas.impl.attention.glm_layers import (
        make_evas_glm_indexer_factory,
    )
    if getattr(dsv2.Indexer, "_fl_evas_glm_factory", False):
        return
    dsv2.Indexer = make_evas_glm_indexer_factory(dsv2.Indexer)


def _glm_filter_weights(weights, param_names):
    """Drop weight names that land on no live param (names byte-identical).

    The LPU form has no own-stack nesting (Option C); the base loader's
    ``stacked_params_mapping`` fuses q_a/kv_a and wk/weights_proj itself. Two
    drop rules: (1) a ``.indexer.`` name with no live param is dropped unless
    its fused form (``wk|weights_proj -> wk_weights_proj``) exists — then the
    ORIGINAL name is kept so the base stacked mapping can do the fusion; (2) a
    ``model.layers.<N>.`` name whose layer has NO live param at all is dropped
    — ``VLLM_FL_NUM_LAYERS`` truncation (the e2e script's hf override sets
    ``num_hidden_layers=6`` while the checkpoint ships 79) and the disabled
    MTP head (``num_nextn_predict_layers=0`` vs ckpt layer 78) leave
    ``layers.6..78.*`` keys the base loader would ``params_dict[name]``
    KeyError on (observed on a TP16 GLM-5 run:
    ``KeyError: 'layers.10.input_layernorm.weight'``).  Rule 2 is LAYER
    granularity, not per-name: stacked ckpt names that are not themselves live
    params (``q_a_proj`` -> fused ``fused_qkv_a_proj``) must survive when a
    layer-mate param proves the layer is built.
    """
    import re as _re
    pd = set(param_names)
    layer_re = _re.compile(r"^(?:model\.)?layers\.(\d+)\.")
    live_layers = {
        int(m.group(1)) for n in pd if (m := layer_re.match(n))
    }
    kept = []
    for name, tensor in weights:
        m = layer_re.match(name)
        if m is not None and int(m.group(1)) not in live_layers:
            continue
        if ".indexer." in name and name not in pd:
            fused = _re.sub(
                r"\.(wk|weights_proj)(\.weight)$", r".wk_weights_proj\2", name
            )
            if fused == name or fused not in pd:
                continue
        kept.append((name, tensor))
    return kept


# ---- glm load_weights (indexer weight filtering) ----


def _glm_load_weights_filter() -> None:
    """Base vllm ``GlmMoeDsaForCausalLM.load_weights`` doesn't know two GLM/LPU facts.

    The LPU subclass rebind puts MLA weights at plain ``self_attn.*`` (no
    own-stack nesting — Option C), so the base ``stacked_params_mapping`` fuses
    q_a/kv_a and wk/weights_proj as usual. Still unknown to the base loader:
    (1) **shared layers** — HF ``indexer_types="shared"`` layers have no
    Indexer params, so the "full"/MTP layers' ``layers.*.self_attn.indexer.*``
    weights they ship have no home → KeyError; drop the ``.indexer.`` names
    unless a live (direct or fused ``wk_weights_proj``) param exists.
    (2) **truncated / MTP layers** — layer-granularity drop of
    ``model.layers.<N>.`` names with no live layer: ``VLLM_FL_NUM_LAYERS``
    truncation (script hf override num_hidden_layers=6 vs ckpt 79 layers) and
    the disabled nextn head (num_nextn_predict_layers=0 vs ckpt layer 78);
    see ``_glm_filter_weights``. Also handles tie_word_embeddings (tied
    checkpoints ship no ``lm_head.weight``; upstream never ties it).
    """
    import vllm.model_executor.models.deepseek_v2 as _dsv2_mod

    _orig = _dsv2_mod.GlmMoeDsaForCausalLM.load_weights
    if getattr(_orig, "_fl_glm_filter_wrapped", False):
        return

    from vllm_fl.utils import DeviceInfo

    def _wrapped(self, weights):
        if DeviceInfo().device_type != "lpu":
            return _orig(self, weights)
        # Tied checkpoints (glm-5 tiny: tie_word_embeddings=True) ship no
        # lm_head.weight; upstream builds a separate random ParallelLMHead and
        # never ties it.  Share embed_tokens' weight instead.
        if getattr(getattr(self, "config", None), "tie_word_embeddings", False):
            embed = getattr(getattr(self, "model", None), "embed_tokens", None)
            lm = getattr(self, "lm_head", None)
            if (
                embed is not None
                and lm is not None
                and isinstance(getattr(lm, "weight", None), nn.Parameter)
                and isinstance(getattr(embed, "weight", None), nn.Parameter)
            ):
                lm.weight = embed.weight
        filtered = _glm_filter_weights(
            weights,
            (n for n, _ in self.named_parameters()),
        )
        return _orig(self, filtered)

    _wrapped._fl_glm_filter_wrapped = True
    _dsv2_mod.GlmMoeDsaForCausalLM.load_weights = _wrapped


# ---- deepseek_v4_decoder_layer ----


def _dsv4_decoder_layer() -> None:
    """Gate the tilelang mhc import in DeepseekV4DecoderLayer.__init__ on LPU.

    Upstream ``DeepseekV4DecoderLayer.__init__`` does
    ``import vllm.model_executor.layers.mhc`` (tilelang), which crashes on LPU
    (tilelang has no working LPU backend/driver). On LPU we wrap
    ``__init__`` to install a temporary dummy ``sys.modules`` entry for that
    module around the upstream init call, making the import a no-op, then
    restore. Off LPU, ``__init__`` is unchanged (real import happens).
    """
    import sys
    import types
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer
    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return  # CUDA: leave upstream import unchanged

    _orig_init = DeepseekV4DecoderLayer.__init__

    _MHC_MOD = "vllm.model_executor.layers.mhc"

    def _patched_decoder_init(self, *args, **kwargs):
        # Install a dummy module so the upstream `import ... mhc` is a no-op.
        prior = sys.modules.get(_MHC_MOD)
        installed = False
        if prior is None:
            sys.modules[_MHC_MOD] = types.ModuleType(_MHC_MOD)
            installed = True
        try:
            _orig_init(self, *args, **kwargs)
        finally:
            if installed:
                # Remove our stub; do not clobber a real entry if one appeared.
                if sys.modules.get(_MHC_MOD) is not None and \
                        getattr(sys.modules[_MHC_MOD], "__name__", "") == _MHC_MOD \
                        and not hasattr(sys.modules[_MHC_MOD], "__file__"):
                    sys.modules.pop(_MHC_MOD, None)

    DeepseekV4DecoderLayer.__init__ = _patched_decoder_init


def lpu_hc_head_fused_kernel(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
    """LPU stand-in for upstream ``hc_head_fused_kernel_tilelang``.

    Module level on purpose: both ``nvidia/model.py`` and ``nvidia/mtp.py``
    bind this name from ``model_executor.kernels.mhc.tilelang`` and each
    binding has to be rebound in its own module namespace, so keeping one
    implementation avoids a second source of truth.

    Return the callee result, not a local ``out``.  The evas kernel allocates
    its own double-die buffer (``split_dim=0``) on the dual-die path and
    returns it without touching the caller buffer, so returning ``out`` here
    would hand back uninitialized memory.  The pure-PyTorch default writes
    ``out`` in place and returns it, so this is a no-op while the op stays on
    the default path.
    """
    import flag_gems

    num_tokens, hc_mult, hidden_size = hs_flat.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )
    if num_tokens == 0:
        return out
    return flag_gems.hc_head_fused_kernel(
        hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps,
        hc_mult,
    )


def _patch_dsv4_mhc_for_lpu() -> None:
    """Route base 0.24 DSV4 tilelang MHC kernels to flaggems evas on LPU.

    ``vllm.models.deepseek_v4.nvidia.model`` imports the four tilelang MHC
    kernels at module scope (``mhc_pre_tilelang`` / ``mhc_fused_post_pre_tilelang``
    / ``mhc_post_tilelang`` / ``hc_head_fused_kernel_tilelang``).  They import
    fine but crash at call time on LPU (evas-triton-opt).  flaggems ``_evas``
    provides the pure-PyTorch ``mhc_pre`` / ``mhc_post`` / ``hc_head_fused_kernel``
    (validated token-identical).  Rebind the module-level names:

    - ``mhc_pre_tilelang`` → flaggems ``mhc_pre`` + fused attn/ffn norm (the
      upstream base passes ``norm_weight``/``norm_eps`` to fuse the RMSNorm into
      the kernel; flaggems ``mhc_pre`` has no norm, so apply it after).
    - ``mhc_fused_post_pre_tilelang`` → ``mhc_post`` then ``mhc_pre``+norm
      (matches the fused kernel's ``(residual_cur, post_mix_cur, comb_mix_cur,
      layer_input_cur)`` return).
    - ``mhc_post_tilelang`` → flaggems ``mhc_post`` (drop-in).
    - ``hc_head_fused_kernel_tilelang`` → flaggems ``hc_head_fused_kernel``
      adapter (evas takes an explicit ``out`` + ``hidden_size``/``hc_mult``, and
      on the dual-die path returns its own buffer instead of filling ``out``).
    """
    import vllm.models.deepseek_v4.nvidia.model as _dsv4_model

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return

    import flag_gems

    def _lpu_mhc_pre(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1,
        norm_weight=None, norm_eps=1e-6,
    ):
        post_mix, res_mix, x = flag_gems.mhc_pre(
            residual=residual, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
            rms_eps=rms_eps, hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat, n_splits=n_splits,
        )
        if norm_weight is not None:
            x = flag_gems.rms_norm(x, (x.shape[-1],), norm_weight, norm_eps)
        return post_mix, res_mix, x

    def _lpu_mhc_fused_post_pre(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None, norm_eps=1e-6,
    ):
        residual_cur = flag_gems.mhc_post(
            x=x, residual=residual, post_layer_mix=post_layer_mix,
            comb_res_mix=comb_res_mix,
        )
        post_mix_cur, comb_mix_cur, layer_input_cur = _lpu_mhc_pre(
            residual=residual_cur, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
            rms_eps=rms_eps, hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat, n_splits=n_splits,
            norm_weight=norm_weight, norm_eps=norm_eps,
        )
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    _dsv4_model.mhc_pre_tilelang = _lpu_mhc_pre
    _dsv4_model.mhc_fused_post_pre_tilelang = _lpu_mhc_fused_post_pre
    _dsv4_model.mhc_post_tilelang = flag_gems.mhc_post
    _dsv4_model.hc_head_fused_kernel_tilelang = lpu_hc_head_fused_kernel


def _patch_dsv4_attention_execute_for_lpu() -> None:
    """Replace the base DSV4 attention's CUDA bits on LPU.

    - ``execute_in_parallel`` (``attn_gemm_parallel_execute``) and
      ``maybe_execute_in_parallel`` (``attention_impl``) touch
      ``torch.cuda.current_stream()`` even in their sequential fallback —
      ``Found no NVIDIA driver`` on LPU.  Rebind both module-level names to
      pure sequential versions (fn0/fn1 run in order; no CUDA events/streams).
    - ``fused_q_kv_rmsnorm`` (called at ``DeepseekV4Attention.forward``) is a
      direct module import of the base Triton kernel (common/ops), which
      bypasses the flaggems whitelist dispatch and fails to compile on LPU
      (evas-triton-opt ``math.rsqrt`` type mismatch).  Rebind to the flaggems
      evas op of the same signature.
    - Both bindings go through the public ``flag_gems.<op>`` names, so the
      implementation behind them stays the backend's decision; the call site
      knows nothing about kernels versus pure-PyTorch twins.
    """
    import vllm.models.deepseek_v4.attention as _dsv4_attn_mod

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return

    def _lpu_execute_in_parallel(
        default_fn, aux_fns, start_event, done_events, aux_streams=None,
        enable=False,
    ):
        del start_event, done_events, aux_streams, enable  # no CUDA on LPU
        default_result = default_fn()
        aux_results = [fn() if fn is not None else None for fn in aux_fns]
        return default_result, aux_results

    def _lpu_maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream=None):
        del event0, event1, aux_stream  # no CUDA on LPU
        res0 = fn0()
        res1 = fn1() if fn1 is not None else None
        return res0, res1

    _dsv4_attn_mod.execute_in_parallel = _lpu_execute_in_parallel
    _dsv4_attn_mod.maybe_execute_in_parallel = _lpu_maybe_execute_in_parallel

    # Bind the public ``flag_gems.<op>`` names.  Neither the vendored kernel nor
    # the pure-PyTorch twin is imported here: which implementation serves the
    # call site is FlagGems' decision, and the plugin layer must not hold a
    # reference to either half.
    import flag_gems

    # Upstream calls this at ``DeepseekV4Attention.forward`` on a column view of
    # the merged qkv projection; signature (q, kv, q_weight, kv_weight, eps).
    _dsv4_attn_mod.fused_q_kv_rmsnorm = flag_gems.fused_q_kv_rmsnorm
    # The DSV4 indexer's Q rope+quant: base's has_cutedsl() branch imports a
    # cutedsl CUDA kernel (vllm.vllm_flash_attn.cute), absent on LPU.  The
    # flaggems evas version has the same signature (positions, index_q,
    # cos_sin_cache, weights, softmax_scale, head_scale, use_fp4).
    _dsv4_attn_mod.fused_indexer_q_rope_quant = flag_gems.fused_indexer_q_rope_quant


def _patch_dsv4_attn_cls_for_lpu() -> None:
    """Make ``_select_dsv4_attn_cls`` return the evas attention on LPU.

    The base ``vllm.models.deepseek_v4.nvidia.model._select_dsv4_attn_cls``
    returns ``DeepseekV4FlashMLAAttention`` (flashmla, CUDA) for non-Explicit
    backends — its ``_o_proj``/``forward_mqa`` use deep_gemm /
    fused_inv_rope_fp8_quant Triton kernels that don't run on LPU.  Route LPU
    to ``EvasDSV4Attention`` (flaggems evas pure-PyTorch impl) regardless of
    the configured backend.
    """
    import vllm.models.deepseek_v4.nvidia.model as _dsv4_model

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return

    from vllm_fl.dispatch.backends.vendor.evas.impl.attention.dsv4_attention import (
        EvasDSV4Attention,
    )

    _orig_select = _dsv4_model._select_dsv4_attn_cls

    def _lpu_select(vllm_config):
        del vllm_config
        return EvasDSV4Attention

    _dsv4_model._select_dsv4_attn_cls = _lpu_select


def _patch_dsv4_embed_repeat_for_lpu() -> None:
    """Fix LPU ``repeat`` NaN on the DSV4 embed expansion.

    The 0.24 model expands the token embedding to ``hc_mult`` streams with
    ``hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)``.  LPU's
    ``repeat`` on a non-contiguous (unsqueezed) view is nondeterministically
    broken: the kernel writes only part of the output, leaving stale NaN from
    the pooled allocator (probe: embed output finite, repeat output NaN at
    small batch T; contiguous-input repeat and ``expand().contiguous()`` both
    clean in the same allocator state).  Replace ``DeepseekV4Model.forward``
    with a copy that materializes via the equivalent ``expand().contiguous()``.
    """
    import vllm.models.deepseek_v4.nvidia.model as _dsv4_model

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return

    from itertools import islice

    from vllm.distributed import get_pp_group
    from vllm.model_executor.models.utils import IntermediateTensors

    def _safe_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).expand(
                hidden_states.shape[0], self.hc_mult, hidden_states.shape[1]
            ).contiguous()
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        residual, post_mix, res_mix = None, None, None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        if layer is not None:
            hidden_states = _dsv4_model.mhc_post_tilelang(
                hidden_states, residual, post_mix, res_mix
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = _dsv4_model.hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    _dsv4_model.DeepseekV4Model.forward = _safe_forward


def _evas_topk_softplus_sqrt():
    """Return the flaggems evas ``topk_softplus_sqrt``, or None when absent.

    The flaggems adoption layer installs this symbol when ``import flag_gems``
    runs, so it is looked up (not latched) on each use.
    """
    try:
        import flag_gems
    except ImportError:
        return None
    return getattr(flag_gems, "topk_softplus_sqrt", None)


def _topk_softplus_sqrt_lpu(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
):
    """LPU drop-in for ``vllm_topk_softplus_sqrt`` (DSV4 sqrtsoftplus gate).

    Calls the flaggems evas op ``flag_gems.topk_softplus_sqrt`` and falls back
    to the upstream pure-torch fallback (``_topk_softplus_sqrt_torch``) when
    flaggems is unavailable.

    Two things about the callee must stay in mind:

    1. Its signature mirrors the CUDA operator
       ``_moe_C.topk_softplus_sqrt``, whose 6th parameter is
       ``routed_scaling_factor`` while this python wrapper's 6th is
       ``e_score_correction_bias``.  The call is therefore re-mapped by
       keyword -- a plain rebind to that function silently swaps the two.
    2. It writes ``topk_weights``/``topk_indices`` in place and returns None,
       while the router returns this wrapper's result and its caller unpacks
       the pair, so return them here.
    """
    op = _evas_topk_softplus_sqrt()
    if op is None:
        from vllm.model_executor.layers.fused_moe.router import (
            fused_topk_bias_router as _mod,
        )
        return _mod._topk_softplus_sqrt_torch(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )
    op(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias=e_score_correction_bias,
        input_ids=input_tokens,
        tid2eid=hash_indices_table,
    )
    return topk_weights, topk_indices


def _patch_topk_softplus_sqrt_for_lpu() -> None:
    """Route ``vllm_topk_softplus_sqrt`` (DSV4 sqrtsoftplus routing) to evas.

    The upstream wrapper uses the pure-PyTorch fallback only on XPU and
    otherwise calls ``torch.ops._moe_C.topk_softplus_sqrt``, absent on LPU
    (vllm 0.24 ships no compiled ``vllm._C``).  Rebind it to the flaggems evas
    override for the LPU path.
    """
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as _mod

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return

    if getattr(_mod.vllm_topk_softplus_sqrt, "_fl_lpu_wrapped", False):
        return

    _mod.vllm_topk_softplus_sqrt = _topk_softplus_sqrt_lpu
    _mod.vllm_topk_softplus_sqrt._fl_lpu_wrapped = True


def _topk_sigmoid_lpu(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    """Pure-PyTorch drop-in for ``torch.ops._moe_C.topk_sigmoid`` (minimax-m2).

    Semantics match the CUDA ``moeTopK`` kernel with ``SCORING_SIGMOID``
    (vllm csrc/libtorch_stable/moe/topk_softmax_kernels.cu), verified against
    the source:
      - ``scores = sigmoid(gating_output)`` first.
      - ``e_score_correction_bias`` is added to ``scores`` ONLY for the top-k
        selection; the stored weights are the UNBIASED sigmoid values
        ("Return the unbiased scores for output weights").
      - ``token_expert_indices[i, j] = j * num_tokens + i`` (column-major flat
        source rows, same convention as topk_softmax).
      - renormalize divides by the sum of the unbiased selected scores.

    Mutates the buffers in-place and returns ``(topk_weights, topk_ids)`` to
    match the ``vllm_topk_sigmoid`` contract (callers unpack it).
    """
    num_tokens = gating_output.size(0)
    topk = topk_weights.size(-1)
    scores = gating_output.sigmoid()
    sel = scores
    if e_score_correction_bias is not None:
        sel = scores + e_score_correction_bias
    vals, ids = torch.topk(sel, k=topk, dim=-1)
    topk_weights.copy_(scores.gather(1, ids))
    topk_ids.copy_(ids.to(topk_ids.dtype))
    rows = torch.arange(num_tokens, dtype=torch.int32, device=gating_output.device)
    for j in range(topk):
        token_expert_indices[:, j] = rows + j * num_tokens
    if renormalize:
        topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8))
    return topk_weights, topk_ids


def _patch_topk_sigmoid_for_lpu() -> None:
    """Route ``vllm_topk_sigmoid`` (sigmoid-MoE routing, minimax-m2) to the
    pure-PyTorch fallback.

    vllm 0.24 has no compiled ``vllm._C`` on LPU, so
    ``torch.ops._moe_C.topk_sigmoid`` is absent (``_moe_C`` namespace comes
    from mcoplib but lacks the op).  Wrap ``vllm_topk_sigmoid`` in both router
    modules (``fused_topk_router`` / ``fused_topk_bias_router``), mirroring
    ``_patch_topk_softplus_sqrt_for_lpu``.
    """
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as _bias_mod
    import vllm.model_executor.layers.fused_moe.router.fused_topk_router as _mod

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return

    for _m in (_mod, _bias_mod):
        if getattr(_m.vllm_topk_sigmoid, "_fl_lpu_wrapped", False):
            continue
        _m.vllm_topk_sigmoid = _topk_sigmoid_lpu
        _m.vllm_topk_sigmoid._fl_lpu_wrapped = True


# ---- DSV4 sparse-indexer dispatch ----

# Module-level name DeepseekV4Attention's indexer selection (deepseek_v4.py
# L1203-1208) uses for the FP8 branch: on LPU it is repointed to
# ``EvasDSV4SparseAttnIndexer`` (real FP8 Q quant + FP8 132B cache + FP8
# logits), which is the only supported route.
_DSV4_INDEXER_DISPATCHED = False


def _dsv4_indexer_dispatch(vllm_config=None) -> None:
    """Repoint the DSV4 sparse-indexer op to the LPU custom op (FP8 route).

    ``DeepseekV4Attention.__init__`` (``vllm/models/deepseek_v4/attention.py``
    L748) constructs ``indexer_op`` from the module-level ``SparseAttnIndexer``
    name (imported at L24 from
    ``vllm.model_executor.layers.sparse_attn_indexer``), i.e. the CUDA base
    class.  On LPU the CUDA base's native forward is NotImplementedError, so we
    repoint that name to ``EvasDSV4SparseAttnIndexer`` (whose ``forward_oot`` calls
    the LPU ``evas_dsv4_sparse_attn_indexer`` custom op, living in
    ``vllm_fl.dispatch.backends.vendor.evas.impl.attention.dsv4_sparse_attn_indexer``).

    Called from the DeepseekV4ForCausalLM ``__init__`` hook (weights patch);
    idempotent via ``_DSV4_INDEXER_DISPATCHED``.
    """
    global _DSV4_INDEXER_DISPATCHED
    if _DSV4_INDEXER_DISPATCHED:
        return
    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return

    from vllm_fl.dispatch.backends.vendor.evas.impl.attention.dsv4_sparse_attn_indexer import (
        EvasDSV4SparseAttnIndexer,
    )
    import vllm.models.deepseek_v4.attention as _dsv4_attn_mod

    if _dsv4_attn_mod.SparseAttnIndexer is not EvasDSV4SparseAttnIndexer:
        _dsv4_attn_mod.SparseAttnIndexer = EvasDSV4SparseAttnIndexer
    _DSV4_INDEXER_DISPATCHED = True


# ---- Public entry point ----


def _patch_fa_utils_for_lpu() -> None:
    """Route vllm 0.24's MLA prefill dispatch to the flaggems evas varlen.

    vllm 0.24 centralizes MLA prefill in ``MLAPrefillBackend``; on LPU the
    selector picks ``FlashAttnPrefillBackend`` (capability 9,0), which gates on
    ``fa_utils.is_flash_attn_varlen_func_available()`` — False for non-CUDA/XPU/
    ROCm. flaggems ``_evas`` provides a same-signature pure-PyTorch varlen, so
    make fa_utils expose it. Idempotent.
    """
    import vllm.v1.attention.backends.fa_utils as fa_utils
    if fa_utils.is_flash_attn_varlen_func_available():
        return
    import flag_gems
    fa_utils.flash_attn_varlen_func = flag_gems.flash_attn_varlen_func
    fa_utils.is_flash_attn_varlen_func_available = lambda: True


def apply_lpu_model_compat_patch() -> None:
    """Apply all LPU model-level compatibility patches.  Idempotent.

    GLM-5.2 patches (deepseek_v2.py based): mla -> indexer (matching the layer
    construction order in upstream DeepseekV2DecoderLayer).  The GLM MoE needs
    no patch (upstream FusedMoE path; ``_model_moe`` deleted), nor do the DSV4
    mega-MoE branches (deleted with the ``impl/moe`` migration).

    DSV4 patches (base vllm 0.24 per-platform package + patches): decoder_layer
    first (its mhc-import gate must be in place before any decoder layer is
    constructed).  DSV4 attention is the base 0.24 ``DeepseekV4Attention`` ABC
    with ``EvasDSV4Attention`` swapped in as the concrete class; the surrounding
    CUDA-only pieces (parallel-execute helpers, fused_q_kv_rmsnorm,
    fused_indexer_q_rope_quant) are rebound to LPU-safe versions here.  MoE goes
    through the generic FusedMoE path (use_mega_moe=False), so it needs no
    patch.  The real-FP8 path needs no extra wrapper patch: LPU reports
    capability (9, 0) via ``vllm_fl/platform.py``, so the upstream recipe
    selection takes the SM90-style path the evas FP8 overrides consume.

    All DSV4 classes are imported from base vllm's ``vllm.models.deepseek_v4``,
    the module ModelRegistry resolves.
    """
    global _PATCHED
    if _PATCHED:
        return
    # vllm 0.24 MLA prefill dispatch: route fa_utils.flash_attn_varlen_func to
    # the flaggems evas varlen so FlashAttnPrefillBackend works on LPU. Needed
    # by every MLA model (deepseek-v3.1 / DSV4 / GLM), so apply first.
    _patch_fa_utils_for_lpu()
    # MoE softplus-sqrt routing (DSV4 gate): no compiled _moe_C on 0.24.
    _patch_topk_softplus_sqrt_for_lpu()
    # MoE sigmoid routing (minimax-m2): torch.ops._moe_C.topk_sigmoid absent.
    _patch_topk_sigmoid_for_lpu()
    # GLM-5.2 model patches (deepseek_v2.py based)
    _model_mla()
    _model_indexer()
    # GLM-5.x load_weights: base 0.24 loader doesn't drop the indexer weights
    # of "shared" layers.  Applied before the DSV4 gate so it stays effective
    # while the DSV4 fork isn't ported yet.
    _glm_load_weights_filter()
    # DSV4 model patches (base vllm 0.24 per-platform package + patches)
    # decoder_layer first: its mhc-import gate must be in place before any
    # decoder layer is constructed.  vllm 0.24: DSV4 lives in base vllm's
    # vllm.models.deepseek_v4; skip the decoder-layer patch if that package is
    # absent so non-DSV4 models are unaffected.
    try:
        import vllm.models.deepseek_v4  # noqa: F401
    except ImportError:
        # Not a declined install: the GLM/config-level half above is already in
        # place, and this package is absent on cores or builds without DSV4.
        logger.info(
            "[model-compat] vllm.models.deepseek_v4 absent; DSV4 half of the "
            "merged patch skipped (GLM/config half applied)"
        )
        return
    _dsv4_decoder_layer()
    _patch_dsv4_mhc_for_lpu()
    _patch_dsv4_attention_execute_for_lpu()
    _patch_dsv4_attn_cls_for_lpu()
    _patch_dsv4_embed_repeat_for_lpu()
    # Indexer dispatch is deferred to the DeepseekV4ForCausalLM __init__ hook
    # (weights patch) so the checkpoint's expert_dtype decides whether the real
    # FP8 indexer is used — no env knob. See _dsv4_indexer_dispatch.
    _PATCHED = True
