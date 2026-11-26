"""vLLM wiring for the accepted FlagGems MiniCPM operators. No compute kernels."""
from __future__ import annotations
from flag_gems.fused.minicpm import fused_layers as _gems,_group_a8_mlp as _group
from flag_gems.fused.minicpm.fused_layers import (
    I4,
    W,
    _MARKED,
    _MAXM,
    _big_ok,
    _fused_ok,
    _gemm,
    _gemm_big,
    _gemm_silu_kernel,
    _mark,
    _norm_mlp_fake,
    _norm_qkv_fake,
    _norm_ref,
    _nq,
    _nq_big,
    _nq_kernel,
    _rope_rot3_kernel,
    _rope_rot_fake,
    _silu_mul_kernel,
    logger,
    logging,
    norm_mlp,
    norm_qkv,
    rope_rot,
    tl,
    torch,
    triton
)

def _rope_ok(at) -> bool:
    re = getattr(at, "rotary_emb", None)
    cs = getattr(re, "cos_sin_cache", None)
    conds = {"int4_impl": bool(getattr(getattr(at.attn, "impl", None), "_e040_int4", False)), "rotary_emb": re is not None, "cache": cs is not None,
             "neox": bool(getattr(re, "is_neox_style", False)), "rotary_dim==head_size==128": bool(re is not None and re.rotary_dim == re.head_size == I4.D),
             "cache_fp32_or_bf16": bool(cs is not None and cs.dtype in (torch.float32, torch.bfloat16)), "cache_shape": bool(cs is not None and cs.dim() == 2 and cs.shape[1] == I4.D),
             "head_dim": bool(at.head_dim == I4.D)}
    return all(conds.values())     # no logging here: this runs inside the dynamo-traced forward (a logger call with list args broke the compile)


def _ok_layer(layer) -> bool:
    for mod in (layer.self_attn.qkv_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj):
        if getattr(mod, "w8a8_scale", None) is None:
            return False
    return layer.mlp.gate_up_proj.weight.shape[1] == layer.input_layernorm.weight.shape[0]


def patch_decoder_for_iluvatar() -> None:
    """Replace LlamaDecoderLayer.forward by the fused-halves version for layers whose qkv / gate_up / down are W8A8. Idempotent;
    call after patch_linear_w8a8_for_iluvatar()."""
    try:
        from vllm.model_executor.models import llama as LM
        from vllm.utils.torch_utils import direct_register_custom_op

        DL = LM.LlamaDecoderLayer
        if getattr(DL.forward, "_e043_stock", None) is not None:
            return
        direct_register_custom_op("iluvatar_norm_qkv", norm_qkv, fake_impl=_norm_qkv_fake)
        direct_register_custom_op("iluvatar_norm_mlp", _group.norm_mlp, fake_impl=_norm_mlp_fake)
        direct_register_custom_op("iluvatar_rope_rot", rope_rot, fake_impl=_rope_rot_fake)
        op_rope = torch.ops.vllm.iluvatar_rope_rot
        op_qkv = torch.ops.vllm.iluvatar_norm_qkv
        op_mlp = torch.ops.vllm.iluvatar_norm_mlp
        stock = DL.forward
        direct_register_custom_op('iluvatar_packed_mlp',_group.activation_mlp,fake_impl=_group.activation_mlp_fake)
        op_plain_mlp=torch.ops.vllm.iluvatar_packed_mlp
        MP=LM.LlamaMLP
        stock_mlp=MP.forward
        def mlp_forward(self,x):
            gu,dn=self.gate_up_proj,self.down_proj
            if (getattr(gu,'_e110_packed_gu',False) and getattr(dn,'_e110_h128_down',False)
                    and getattr(gu,'bias',None)is None and getattr(dn,'bias',None)is None
                    and getattr(gu,'tp_size',1)==1 and getattr(dn,'tp_size',1)==1):
                shp=x.shape
                y=op_plain_mlp(x.reshape(-1,shp[-1]),gu.weight,gu.w8a8_scale,dn.weight,dn.w8a8_scale)
                return y.reshape(*shp[:-1],y.shape[-1])
            return stock_mlp(self,x)
        mlp_forward._e110_stock=stock_mlp
        MP.forward=mlp_forward

        def forward(self, positions, hidden_states, residual):
            if residual is None or not _ok_layer(self):
                return stock(self, positions, hidden_states, residual)
            at = self.self_attn
            ln1, ln2 = self.input_layernorm, self.post_attention_layernorm
            qkv, residual = op_qkv(hidden_states, residual, ln1.weight, ln1.variance_epsilon, at.qkv_proj.weight, at.qkv_proj.w8a8_scale)
            if _rope_ok(at):
                qr, kr, vr = op_rope(qkv, positions, at.rotary_emb.cos_sin_cache, at.num_heads, at.num_kv_heads)
                attn_output = at.attn(qr, kr, vr)               # int4 impl sees fp32 key = already rotated
            else:
                q, k, v = qkv.split([at.q_size, at.kv_size, at.kv_size], dim=-1)
                q, k = at.rotary_emb(positions, q, k)
                attn_output = at.attn(q, k, v)
            attn_output, _ = at.o_proj(attn_output)
            mlp = self.mlp
            hidden_states, residual = op_mlp(attn_output, residual, ln2.weight, ln2.variance_epsilon, mlp.gate_up_proj.weight,
                                             mlp.gate_up_proj.w8a8_scale, mlp.down_proj.weight, mlp.down_proj.w8a8_scale)
            return hidden_states, residual

        forward._e043_stock = stock
        DL.forward = forward
        logger.warning("E043 decoder-layer fusion patch installed (norm+quant+qkv, norm+quant+gate_up(silu)+down)")
    except Exception as e:
        logger.warning("patch_decoder_for_iluvatar: %s", e)

