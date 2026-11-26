"""vLLM wiring for the accepted FlagGems MiniCPM operators. No compute kernels."""
from __future__ import annotations
from flag_gems.fused.minicpm import w8a8 as _gems,_native_pair_mlp as _native,_packed_pair_reader as _packed,_h128_basis as _h128_module
from flag_gems.fused.minicpm._group_a8_mlp import bind_capture_layer
from flag_gems.fused.minicpm.w8a8 import (
    KINDS,
    N_QUANTIZED,
    ROT_BLOCK,
    ROT_BM,
    ROT_BM_PREFILL,
    ROT_KIND,
    ROT_WARPS,
    _H128,
    _ONE_TILE_MAX,
    _ONE_TILE_WARPS,
    _PREFILL_BIG_CFG,
    _PREFILL_BIG_M,
    _PREFILL_CFG,
    _QBLOCK,
    _cfg,
    _gemm_kernel,
    _h128,
    _quant1_kernel,
    _quant_from1_kernel,
    _quant_from_kernel,
    _quant_kernel,
    _rot_kernel,
    _swizzle,
    _w8a8_linear_fake,
    hadamard,
    logger,
    logging,
    quant_act,
    quant_rows_ref,
    quantize_weight,
    quantize_weight_ref,
    reference,
    rot_cols,
    self_check,
    shapes_ok,
    tl,
    torch,
    triton,
    w8a8_linear
)

def patch_linear_w8a8_for_iluvatar() -> None:
    """Quantize qkv/o/gate_up/down after loading and run them through the int8 op. Must run in
    every Worker process before the model loads. Idempotent."""
    global N_QUANTIZED
    try:
        from vllm.model_executor.layers import linear as L
        from vllm.utils.torch_utils import direct_register_custom_op

        UL = L.UnquantizedLinearMethod
        if getattr(UL.apply, "_iluvatar_stock", None) is not None:
            return
        dev = torch.device("cuda", torch.cuda.current_device())
        err = self_check(dev)
        if not err <= 2e-2:
            logger.warning("BI-V150 W8A8: self-check failed once (max rel err %.3g), repeating", err)
            err = self_check(dev)          # E027: one transient non-finite output was seen in a serve start-up, not reproducible offline
        if not err <= 2e-2:
            logger.warning("BI-V150 W8A8: self-check failed (max rel err %.3g), layers stay bf16", err)
            return
        direct_register_custom_op("iluvatar_w8a8_linear", w8a8_linear, fake_impl=_w8a8_linear_fake)
        op = torch.ops.vllm.iluvatar_w8a8_linear
        direct_register_custom_op("iluvatar_packed_gu_linear",_packed.linear,fake_impl=_packed.linear_fake)
        direct_register_custom_op("iluvatar_h128_down_linear",_h128_module.linear,fake_impl=_h128_module.linear_fake)
        op_packed=torch.ops.vllm.iluvatar_packed_gu_linear
        op_down=torch.ops.vllm.iluvatar_h128_down_linear
        stock_pw, stock_apply = UL.process_weights_after_loading, UL.apply

        def process_weights_after_loading(self, layer):
            global N_QUANTIZED
            stock_pw(self, layer)
            kind = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
            w = getattr(layer, "weight", None)
            if kind not in KINDS or w is None or w.dim() != 2 or w.dtype != torch.bfloat16:
                return
            rot = kind == ROT_KIND
            if not shapes_ok(w.shape[1], w.shape[0], rot):
                return
            if kind=='gate_up_proj':
                base_q,base_s=quantize_weight(w.data,False)
                wq,ws=_native.prepare_weight(base_q,base_s)
                layer._e110_packed_gu=True
                logger.info('E110 single packed GU basis loaded: %s bytes=%d',layer.prefix,wq.numel())
            elif kind=='down_proj':
                wq,ws=_h128_module.quant_weight(w.data)
                layer._e110_h128_down=True
                logger.info('E110 shared H128 down basis loaded: %s bytes=%d',layer.prefix,wq.numel())
            else:wq,ws=quantize_weight(w.data,rot)
            layer.weight = torch.nn.Parameter(wq, requires_grad=False)
            if kind=="gate_up_proj":bind_capture_layer(layer.weight, layer.prefix)
            layer.register_buffer("w8a8_scale", ws, persistent=False)
            layer.w8a8_rot = int(rot)
            N_QUANTIZED += 1
            if N_QUANTIZED == 1 or N_QUANTIZED % 42 == 0:
                logger.info("BI-V150 W8A8: %d linear layers int8 (last %s, rot=%d)", N_QUANTIZED, layer.prefix, rot)

        def apply(self, layer, x, bias=None):
            ws = getattr(layer, "w8a8_scale", None)
            if ws is None:
                return stock_apply(self, layer, x, bias)
            shp = x.shape
            flat=x.reshape(-1,shp[-1])
            if getattr(layer,'_e110_packed_gu',False):y=op_packed(flat,layer.weight,ws)
            elif getattr(layer,'_e110_h128_down',False):y=op_down(flat,layer.weight,ws)
            else:y=op(flat,layer.weight,ws,layer.w8a8_rot)
            y = y.reshape(*shp[:-1], y.shape[-1])
            return y if bias is None else y + bias

        apply._iluvatar_stock = stock_apply
        UL.process_weights_after_loading = process_weights_after_loading
        UL.apply = apply
        logger.info("BI-V150 W8A8: self-check passed (max rel err %.2e); qkv/o/gate_up/down will be int8, "
                    "down_proj H128 candidate basis (baseline self-check block %d)", err, ROT_BLOCK)
    except Exception as e:
        logger.warning("patch_linear_w8a8_for_iluvatar: %s", e)

