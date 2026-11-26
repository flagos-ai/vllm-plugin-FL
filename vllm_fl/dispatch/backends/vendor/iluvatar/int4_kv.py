"""vLLM wiring for the accepted FlagGems MiniCPM operators. No compute kernels."""
from __future__ import annotations
from flag_gems.fused.minicpm import _deferred_kv_writer as deferred_writer
from flag_gems.fused.minicpm import int4_kv as _gems
from flag_gems.fused.minicpm import pyramid_prefill as _PP
from flag_gems.fused.minicpm.int4_kv import (
    D,
    HKV_EXP,
    LAST_DIM,
    OFF_REC,
    OFF_V,
    Optional,
    PAGE,
    PAGE_BYTES,
    QPK,
    SLOTS,
    WIN,
    WIN_ON,
    WIN_WARPS,
    native_window as Window,
    _DEC14_ASM,
    _DEC14_CONS,
    _HAD,
    _INV_SQRT_D,
    _LOG2E,
    _ROW_SLOT,
    _TAGS,
    _V2_DEFAULT,
    _V2_SET,
    _check_prefill,
    _dec14_asm,
    _dec14_kernel,
    _e5m3_dec,
    _e5m3_dec_ref,
    _e5m3_up,
    _e5m3_up_ref,
    _fill_seqs,
    _had_pm,
    _hi32,
    _hw_k,
    _hw_v,
    _hw_v2,
    _lo32,
    _meta,
    _reduce_inv_kernel,
    _ref_attn,
    _rot2_kernel,
    _rot_kernel,
    _unp8,
    _win4_kernel,
    _win_ref,
    _write_kernel,
    _ws4_kernel,
    _ws_fresh_kernel,
    _wstart,
    attend,
    cache_views,
    dequant_pages,
    int4_splits,
    is_v2_layer,
    logger,
    logging,
    math,
    quant_k_ref,
    quant_v_ref,
    re,
    rot_matrix,
    rotate,
    rotate2,
    run_decode4,
    run_prefill4,
    self_check,
    set_rows,
    shared_state,
    tl,
    torch,
    triton,
    v2_layer_set,
    write
)

def _layer_md(layer):
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return None
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = md.get(getattr(layer, "layer_name", None))
    return md if md is not None and hasattr(md, "query_start_loc") else None


def _supports_deferred_writes(vllm_config) -> bool:
    if vllm_config is None or vllm_config.model_config is None or not WIN_ON:
        return False
    model = vllm_config.model_config.hf_text_config
    parallel = vllm_config.parallel_config
    return (
        (getattr(model, "num_hidden_layers", None), getattr(model, "hidden_size", None),
         getattr(model, "vocab_size", None))
        == (deferred_writer.LAYERS, 2048, 130560)
        and parallel.pipeline_parallel_size == parallel.tensor_parallel_size == parallel.data_parallel_size == 1
        and not parallel.use_ubatching
        and vllm_config.kv_transfer_config is None
    )


def install(ta, vllm_config=None) -> None:
    """Route TritonAttentionImpl through the int4 cache: page shape, cache write, attention; window buffers at init."""
    impl, backend = ta.TritonAttentionImpl, ta.TritonAttentionBackend
    if getattr(impl, "_e040_int4", False):
        return
    defer_writes = _supports_deferred_writes(vllm_config)
    stock_shape = backend.get_kv_cache_shape

    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        if cache_dtype_str == "int8_per_token_head" and head_size == D and num_kv_heads == HKV_EXP and block_size == PAGE:
            return (num_blocks, 2, PAGE, HKV_EXP, LAST_DIM)
        return stock_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)
    backend.get_kv_cache_shape = staticmethod(get_kv_cache_shape)
    stock_init = impl.__init__

    def __init__(self, *a, **k):
        stock_init(self, *a, **k)
        self._e040 = self.num_kv_heads == HKV_EXP and self.head_size == D and self.num_heads == QPK * HKV_EXP
        self._deferred_writer = None
        self._deferred_pending = False
        if self._e040:
            dev = torch.device("cuda", torch.cuda.current_device())
            self._win = Window(dev)                     # before memory profiling: charged against the KV pool
            shared_state(dev)
            self._views = None
            if defer_writes:
                self._deferred_writer = deferred_writer.state(dev)

    def v2_of(self, layer):
        v = getattr(self, "_v2", None)
        if v is None:
            v = self._v2 = is_v2_layer(layer)
            _gems.bind_sparse_layer(self._win, layer)
            if v and not _HAD.get("v2_logged"):
                _HAD["v2_logged"] = True
                logger.warning("E045 V2 KV active: V at 2 bit (g128 mid-rise, e5m3 scale) in %d of 42 layers %s", len(_gems._V2_SET), sorted(_gems._V2_SET))
        return v

    def views(self, kv_cache):
        if self._views is None or self._views[0] is not kv_cache:
            self._views = (kv_cache,) + cache_views(kv_cache)
        return self._views[1], self._views[2]

    def write_for_layer(self, kr, vr, slot_mapping, c8, c16, md, v2):
        seq_use = md.seq_lens if md else None
        if md is not None:
            st_idx = getattr(self, "_pyramid_stage", None)
            if st_idx is None:
                li = getattr(self._win, "_crosslayer_index", None)
                st_idx = _PP.stage_of_layer(li)
                if li is not None:
                    self._pyramid_stage = st_idx
            ds = _PP.get_decode_state()
            if st_idx > 0 and ds is not None and ds.active and seq_use is not None and seq_use.shape[0] <= ds.max_reqs:
                seq_use = ds.seq_lens[st_idx - 1, :seq_use.shape[0]]
        batch = deferred_writer.BATCH
        if (
            self._deferred_writer is not None
            and md is not None
            and md.max_query_len == 1
            and md.query_start_loc.numel() == batch + 1
            and slot_mapping.shape == (batch,)
            and slot_mapping.dtype == torch.int64
            and kr.shape == vr.shape == (batch, HKV_EXP, D)
            and kr.stride() == vr.stride() == (HKV_EXP * D, D, 1)
            and self._win.native_split_meta
            and self._win.native_layout == "token"
        ):
            self._deferred_pending = True
            self._deferred_writer.write(
                self._win._crosslayer_index, kr, vr, slot_mapping, c8, c16,
                self._win, md.query_start_loc, seq_use, v2,
            )
        else:
            write(kr, vr, slot_mapping, c8, c16, self._win, md.query_start_loc if md else None,
                  seq_use, v2)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self._deferred_pending = False
        if kv_cache.dim() != 5 or kv_cache.numel() == 0:
            return
        n = slot_mapping.shape[0]
        if key.dtype == torch.float32:                   # E043: fused rope + rotation upstream already produced rotated fp32 K / V
            kr, vr = key[:n], value[:n]
            self._kr, self._vr = kr, vr
            c8, c16 = views(self, kv_cache)
            md = _layer_md(layer)
            write_for_layer(self, kr, vr, slot_mapping, c8, c16, md, v2_of(self, layer))
            return
        kr = torch.empty((n, HKV_EXP, D), dtype=torch.float32, device=key.device)
        vr = torch.empty_like(kr)
        if key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16:
            rotate2(key[:n], value[:n], kr, vr)          # E043 mechanism: one launch for K and V
            if not _HAD.get("rot2_logged"):
                _HAD["rot2_logged"] = True
                logger.warning("E043 fused k/v rotation active (rotate2, one launch per layer)")
        else:
            rotate(key[:n], kr)
            rotate(value[:n], vr)
        self._kr, self._vr = kr, vr
        c8, c16 = views(self, kv_cache)
        md = _layer_md(layer)
        write_for_layer(self, kr, vr, slot_mapping, c8, c16, md, v2_of(self, layer))

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)
        n = attn_metadata.num_actual_tokens
        hq = self.num_heads
        q3 = query[:n] if query.dim() == 3 else query[:n].view(n, hq, D)
        if key is not None and key.dtype == torch.float32:     # E043: q already rotated by the fused rope op
            qr = q3
        else:
            qr = torch.empty((n, hq, D), dtype=torch.bfloat16, device=query.device)
            rotate(q3, qr)
        c8, c16 = views(self, kv_cache)
        v2 = v2_of(self, layer)
        if attn_metadata.max_query_len <= 1 and WIN_ON and output.dtype == torch.bfloat16:   # E043: decode-only step
            if not _HAD.get("redinv_logged"):
                _HAD["redinv_logged"] = True
                logger.warning("E043 fused reduce + inverse rotation active (decode)")
            seq_use = attn_metadata.seq_lens
            tbl_use = attn_metadata.block_table
            st_idx = getattr(self, "_pyramid_stage", None)
            if st_idx is None:
                li = getattr(self._win, "_crosslayer_index", None)
                st_idx = _PP.stage_of_layer(li)
                if li is not None:
                    self._pyramid_stage = st_idx
            ds = _PP.get_decode_state()
            if st_idx > 0 and ds is not None and ds.active and seq_use.shape[0] <= ds.max_reqs and tbl_use.shape[1] <= ds.max_blocks:
                seq_use, tbl_use = ds.get_stage_views(st_idx, seq_use.shape[0], tbl_use.shape[1])
            attend(qr, self._kr, self._vr, c8, c16, self._win, None, attn_metadata.query_start_loc, seq_use,
                   tbl_use, self.scale, attn_metadata.max_query_len, final=output[:n].view(n, hq, D), v2=v2)
        else:
            seq_use = attn_metadata.seq_lens
            tbl_use = attn_metadata.block_table
            st_idx = getattr(self, "_pyramid_stage", None)
            if st_idx is None:
                li = getattr(self._win, "_crosslayer_index", None)
                st_idx = _PP.stage_of_layer(li)
                if li is not None:
                    self._pyramid_stage = st_idx
            ds = _PP.get_decode_state()
            if st_idx > 0 and ds is not None and ds.active and seq_use.shape[0] <= ds.max_reqs and tbl_use.shape[1] <= ds.max_blocks:
                seq_use, tbl_use = ds.get_stage_views(st_idx, seq_use.shape[0], tbl_use.shape[1])
                if not _HAD.get("e488_mixed_pyr"):
                    _HAD["e488_mixed_pyr"] = True
                    logger.warning("E488 mixed-step pyramid decode tables active (fixes stages 1..3 KV corruption)")
            outr = torch.empty((n, hq, D), dtype=torch.float32, device=query.device)
            attend(qr, self._kr, self._vr, c8, c16, self._win, outr, attn_metadata.query_start_loc, seq_use,
                   tbl_use, self.scale, attn_metadata.max_query_len, v2=v2)
            rotate(outr, output[:n].view(n, hq, D), inv=True)
        if self._deferred_pending and self._win._crosslayer_index == deferred_writer.LAYERS - 1:
            self._deferred_writer.flush()
        return output

    impl.__init__ = __init__
    impl.do_kv_cache_update = do_kv_cache_update
    impl.forward = forward
    impl._e040_int4 = True
    logger.warning("E040 int4 KV: layout v14 (register image, inline asm page loop)")
    logger.warning("E040 int4 KV: page %d B (int4 K g64 zp + V sym, e5m3 scales, Hadamard rotation, clip search), "
                   "int8 window %d tokens x %d slots", PAGE_BYTES, WIN, SLOTS)

