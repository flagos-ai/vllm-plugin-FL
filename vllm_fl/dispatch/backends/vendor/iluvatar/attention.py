"""vLLM wiring for the accepted FlagGems MiniCPM operators. No compute kernels."""
from __future__ import annotations
from flag_gems.fused.minicpm import attention as _gems
from flag_gems.fused.minicpm.attention import (
    Optional,
    SimpleNamespace,
    StepPlan,
    _DEC_NP,
    _DEC_UNROLL,
    _HEAD,
    _I8_OK,
    _LOG2E,
    _PAGE,
    _PF_BLOCK_Q,
    _PF_QK_INT8,
    _PF_T,
    _PF_WARPS,
    _QPK,
    _check_case,
    _current_plan,
    _decode_kernel,
    _decode_kernel_i8,
    _i8_views,
    _logged,
    _once,
    _pf_range,
    _pf_range8,
    _pf_tile,
    _pf_tile8,
    _prefill_kernel,
    _prefill_kernel8,
    _quantize_cache,
    _reduce_kernel,
    _reference,
    _unp,
    _ws_copy_kernel,
    _zero_tail_kernel,
    dataclass,
    decode_splits,
    hc_views,
    logger,
    logging,
    plan_step,
    run_decode,
    run_prefill,
    self_check,
    stock_to_hc_codes,
    tl,
    torch,
    triton
)

def _patch_int8_cache_layout(ta) -> None:
    """Make TritonAttentionImpl keep its per-token-head int8 cache head-contiguous: scale views and cache writes use hc_views.
    The scales start as 1.0 (an unwritten token must not carry an arbitrary bit pattern into p * scale)."""
    impl = ta.TritonAttentionImpl
    if getattr(impl, "_iluvatar_hc", False):
        return

    def ensure(self, kv_cache):
        if self._k_scale_cache is not None:
            return
        _, _, ksc, vsc = hc_views(kv_cache)
        ksc.fill_(1.0)
        vsc.fill_(1.0)
        self._k_scale_cache, self._v_scale_cache = ksc, vsc

    stock_update = impl.do_kv_cache_update

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if not self._is_per_token_head_quant or kv_cache.dtype != torch.int8:
            return stock_update(self, layer, key, value, kv_cache, slot_mapping)
        ensure(self, kv_cache)
        kc, vc, _, _ = hc_views(kv_cache)
        ta.triton_reshape_and_cache_flash_per_token_head_quant(key, value, kc, vc, self._k_scale_cache, self._v_scale_cache, slot_mapping)

    impl._ensure_scale_caches = ensure
    impl.do_kv_cache_update = do_kv_cache_update
    impl._iluvatar_hc = True
    logger.info("BI-V150 own attention: int8 KV cache is head-contiguous (E028 layout)")


def patch_triton_attn_for_iluvatar(vllm_config=None) -> None:
    """Route TRITON_ATTN through the Iluvatar kernels. Must run in every Worker
    process before the first forward pass. Idempotent."""
    try:
        import importlib

        ta = importlib.import_module("vllm.v1.attention.backends.triton_attn")
        if getattr(ta.unified_attention, "_iluvatar_stock", None) is not None:
            return
        err = self_check(torch.device("cuda", torch.cuda.current_device()))
        if not err <= 2e-2:
            logger.warning("BI-V150 own attention: self-check failed (max rel err %.3g), stock kernel kept", err)
            return
        logger.info("BI-V150 own attention: self-check passed (max rel err %.2e)", err)
        global _I8_OK
        try:
            err8 = self_check(torch.device("cuda", torch.cuda.current_device()), quant=True)
        except Exception as e:
            err8 = float("inf")
            logger.warning("BI-V150 own attention: int8 KV self-check raised %s: %s", type(e).__name__, e)
        _I8_OK = bool(err8 <= 2e-2)
        _gems._I8_OK = _I8_OK
        if _I8_OK:
            logger.info("BI-V150 own attention: int8 KV (per-token-head scales) self-check passed (max rel err %.2e)", err8)
        else:
            logger.warning("BI-V150 own attention: int8 KV self-check failed (max rel err %.3g), stock kernel for int8 KV", err8)
        ta.unified_attention = make_unified_attention(ta.unified_attention)
        if _I8_OK:
            _patch_int8_cache_layout(ta)
        from vllm_fl import kv4_probe
        probe = False
        if kv4_probe.int4_active():
            from . import int4_kv
            kv4_probe.patch_page_budget()
            r4 = int4_kv.self_check(torch.device("cuda", torch.cuda.current_device()))
            ok4 = (r4["rot"] < 1e-2 and r4["rot_inv"] < 1e-2 and r4["write_mismatch"] < 1e-3 and r4["window_tags"] == 0.0
                   and r4["decode"] < 2e-2 and r4["prefill"] < 2e-2 and r4["decode_vs_no_window_ref"] > 2 * r4["decode"])
            if not ok4:
                raise RuntimeError(f"E040 int4 KV self-check failed: {r4}")
            logger.warning("E040 int4 KV self-check passed: %s", {k: round(v, 5) for k, v in r4.items()})
            int4_kv.install(ta, vllm_config)
        builder = ta.TritonAttentionMetadataBuilder
        stock_build = builder.build

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            md = stock_build(self, common_prefix_len, common_attn_metadata, fast_build)
            if common_prefix_len == 0 and not probe:
                ws_dtype = self.kv_cache_spec.dtype
                i8k = ws_dtype in (torch.int8, torch.uint8) and _I8_OK and _PF_QK_INT8
                if ws_dtype in (torch.int8, torch.uint8):          # int8 KV: V (and K unless int8 QK) are dequantized to bf16 in the workspace
                    ws_dtype = torch.bfloat16
                plan_step(common_attn_metadata, self.num_heads_kv, self.headdim, ws_dtype, self.device, i8k)
            return md

        builder.build = build
        logger.info("E050 FlagGems operators active: %s", _gems.__file__)
        logger.info("BI-V150 attention: own prefill (BLOCK_Q %d, T %d, %d warps) and split-KV decode kernels",
                    _PF_BLOCK_Q, _PF_T, _PF_WARPS)
    except Exception as e:
        logger.warning("patch_triton_attn_for_iluvatar: %s", e)
        from vllm_fl import kv4_probe
        if kv4_probe.int4_active():                         # the int4 page budget is already in force: no stock fallback exists
            raise


def own_attention_applies(q, k, v, causal, window_size, softcap, q_descale, alibi_slopes, output_scale,
                          qq_bias, sinks, mm_prefix_range, kv_quant_mode, use_td,
                          k_scale_cache=None, v_scale_cache=None) -> bool:
    from vllm.v1.kv_cache_interface import KVQuantMode

    common = (
        causal is True
        and window_size[0] < 0
        and not softcap
        and q_descale is None
        and alibi_slopes is None
        and output_scale is None
        and qq_bias is None
        and sinks is None
        and mm_prefix_range is None
        and not use_td
        and q.shape[2] == _HEAD
        and k.shape[1] == _PAGE
        and q.shape[1] == _QPK * k.shape[2]
        and k.stride(3) == 1 and v.stride(3) == 1 and q.stride(2) == 1
    )
    if not common:
        return False
    if kv_quant_mode == KVQuantMode.NONE:
        return q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype
    return (
        _gems._I8_OK
        and kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD
        and q.dtype == torch.bfloat16
        and k.dtype == torch.int8 and v.dtype == torch.int8
        and k.shape[3] == _HEAD
        and _i8_views(k, v, k_scale_cache, v_scale_cache) is not None
    )


def make_unified_attention(stock):
    def iluvatar_unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                                   softmax_scale, causal, window_size, block_table, softcap, q_descale,
                                   k_descale, v_descale, alibi_slopes=None, output_scale=None, qq_bias=None,
                                   sinks=None, mm_prefix_range=None, kv_quant_mode=None, use_td: bool = False,
                                   **kwargs):
        from vllm.v1.kv_cache_interface import KVQuantMode

        if kv_quant_mode is None:
            kv_quant_mode = KVQuantMode.NONE
        ksc, vsc = kwargs.get("k_scale_cache"), kwargs.get("v_scale_cache")
        hc = k.dtype == torch.int8 and k.shape[3] == _HEAD + 4
        if hc:                                   # stock [blocks, 16, heads, 132] views of the cache halves -> head-contiguous code views
            k, v = stock_to_hc_codes(k, v)
        ok = own_attention_applies(q, k, v, causal, window_size, softcap, q_descale, alibi_slopes, output_scale,
                                   qq_bias, sinks, mm_prefix_range, kv_quant_mode, use_td, ksc, vsc)
        plan = _gems._current_plan
        if ok and max_seqlen_q > 1 and (plan is None or plan.cu is not cu_seqlens_q):
            ok = False
            _once("noplan", "BI-V150 own attention: no step plan for a prefill batch, stock kernel used")
        if not ok and hc:
            raise RuntimeError("BI-V150 own attention: int8 head-contiguous KV cache but the own kernels do not apply to this call")
        if not ok:
            return stock(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
                         seqused_k=seqused_k, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
                         causal=causal, window_size=window_size, block_table=block_table, softcap=softcap,
                         q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                         alibi_slopes=alibi_slopes, output_scale=output_scale, qq_bias=qq_bias, sinks=sinks,
                         mm_prefix_range=mm_prefix_range, kv_quant_mode=kv_quant_mode, use_td=use_td,
                         **kwargs)
        if max_seqlen_q <= 1:
            _once("dec", "BI-V150 own attention: decode kernel (rows=%d, splits=%d)",
                  seqused_k.shape[0], decode_splits(seqused_k.shape[0]))
            run_decode(q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale, seqused_k.shape[0], ksc, vsc)
            return
        _once("pf", "BI-V150 own attention: prefill kernel (prefill seqs=%d, decode rows=%d)",
              plan.n_p, plan.n_decode)
        if plan.n_decode:
            run_decode(q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale, plan.n_decode, ksc, vsc)
        if plan.n_p:
            run_prefill(q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale, plan, ksc, vsc)

    iluvatar_unified_attention._iluvatar_stock = stock
    return iluvatar_unified_attention

