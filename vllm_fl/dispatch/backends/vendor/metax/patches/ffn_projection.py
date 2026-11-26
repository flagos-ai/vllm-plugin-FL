# SPDX-License-Identifier: Apache-2.0
"""Select row-scaled C500 kernels and cross-layer fused decoder chains for compatible unquantized projections."""

import os
import torch
from torch.nn import functional as F
from vllm import envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    vllm_lib,
)

from flag_gems.fused.metax_kcontig_mm import try_metax_kcontig_nt_mm
from flag_gems.fused.metax_post_attn_mlp import (
    input_norm_pack_forward,
    input_norm_qkv_forward,
    post_attn_mlp_final_norm_forward,
    post_attn_mlp_forward,
    post_attn_mlp_next_norm_pack_forward,
)
from flag_gems.fused.metax_rowwise_int8 import project, quantize_rows


logger = init_logger(__name__)

_USE_I8_LM_HEAD = True
_USE_I8_ATTN_PREFILL = True
_LINEAR_HOOK_INSTALLED = False
_HOOKS_INSTALLED = False


def c500_ffn_project(x: torch.Tensor, weight: torch.Tensor,
                     codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0]
    minimum = 16 if weight.shape[0] == 12288 else 512
    if x.stride(1) != 1 or x.stride(0) < x.shape[1] or not 1 <= rows <= 4096:
        return F.linear(x, weight)
    if rows < minimum:
        out = x.new_empty((rows, weight.shape[0]))
        if try_metax_kcontig_nt_mm(x, weight, out):
            return out
        return F.linear(x, weight)
    return project(x, codes, scales)


def _fake(x, weight, codes, scales):
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op("c500_ffn_project", c500_ffn_project, fake_impl=_fake,
                          dispatch_key="CUDA")


def c500_input_norm_pack(
    hidden_states: torch.Tensor,
    w_inorm: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return input_norm_pack_forward(hidden_states, w_inorm, eps=eps)


def _fake_input_norm_pack(
    hidden_states: torch.Tensor,
    w_inorm: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = hidden_states.shape[0]
    return (
        hidden_states.new_empty((rows, 2048)),
        hidden_states.new_empty((rows,), dtype=torch.float32),
        hidden_states.new_empty((rows, 2048)),
    )


direct_register_custom_op(
    "c500_input_norm_pack",
    c500_input_norm_pack,
    fake_impl=_fake_input_norm_pack,
    dispatch_key="CUDA",
)


def c500_post_attn_mlp_next_norm_pack(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps_pnorm: float,
    w_next_inorm: torch.Tensor,
    eps_next_inorm: float,
    c_o: torch.Tensor | None = None,
    s_o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return post_attn_mlp_next_norm_pack_forward(
        attn_out,
        residual,
        w_o,
        c_o,
        s_o,
        w_pnorm,
        w_gu,
        c_gu,
        s_gu,
        w_down,
        c_down,
        s_down,
        eps_pnorm=eps_pnorm,
        w_next_inorm=w_next_inorm,
        eps_next_inorm=eps_next_inorm,
        use_i8_o_prefill=_USE_I8_ATTN_PREFILL,
    )


def _fake_post_attn_mlp_next_norm_pack(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps_pnorm: float,
    w_next_inorm: torch.Tensor,
    eps_next_inorm: float,
    c_o: torch.Tensor | None = None,
    s_o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = attn_out.shape[0]
    return (
        attn_out.new_empty((rows, 2048)),
        attn_out.new_empty((rows,), dtype=torch.float32),
        residual.new_empty((rows, 2048)),
    )


direct_register_custom_op(
    "c500_post_attn_mlp_next_norm_pack",
    c500_post_attn_mlp_next_norm_pack,
    fake_impl=_fake_post_attn_mlp_next_norm_pack,
    dispatch_key="CUDA",
)


def c500_post_attn_mlp_final_norm(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps_pnorm: float,
    w_final_norm: torch.Tensor,
    eps_final_norm: float,
    c_o: torch.Tensor | None = None,
    s_o: torch.Tensor | None = None,
) -> torch.Tensor:
    return post_attn_mlp_final_norm_forward(
        attn_out,
        residual,
        w_o,
        c_o,
        s_o,
        w_pnorm,
        w_gu,
        c_gu,
        s_gu,
        w_down,
        c_down,
        s_down,
        eps_pnorm=eps_pnorm,
        w_final_norm=w_final_norm,
        eps_final_norm=eps_final_norm,
        use_i8_o_prefill=_USE_I8_ATTN_PREFILL,
    )


def _fake_post_attn_mlp_final_norm(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps_pnorm: float,
    w_final_norm: torch.Tensor,
    eps_final_norm: float,
    c_o: torch.Tensor | None = None,
    s_o: torch.Tensor | None = None,
) -> torch.Tensor:
    return attn_out.new_empty((attn_out.shape[0], 2048))


direct_register_custom_op(
    "c500_post_attn_mlp_final_norm",
    c500_post_attn_mlp_final_norm,
    fake_impl=_fake_post_attn_mlp_final_norm,
    dispatch_key="CUDA",
)


def c500_post_attn_mlp(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps: float,
    c_o: torch.Tensor | None = None,
    s_o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return post_attn_mlp_forward(
        attn_out,
        residual,
        w_o,
        c_o,
        s_o,
        w_pnorm,
        w_gu,
        c_gu,
        s_gu,
        w_down,
        c_down,
        s_down,
        eps=eps,
        use_i8_o_prefill=_USE_I8_ATTN_PREFILL,
    )


def _fake_post_attn_mlp(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    w_o: torch.Tensor,
    w_pnorm: torch.Tensor,
    w_gu: torch.Tensor,
    c_gu: torch.Tensor,
    s_gu: torch.Tensor,
    w_down: torch.Tensor,
    c_down: torch.Tensor,
    s_down: torch.Tensor,
    eps: float,
    c_o: torch.Tensor | None = None,
    s_o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return attn_out.new_empty((attn_out.shape[0], 2048)), residual.new_empty((residual.shape[0], 2048))


direct_register_custom_op(
    "c500_post_attn_mlp",
    c500_post_attn_mlp,
    fake_impl=_fake_post_attn_mlp,
    dispatch_key="CUDA",
)


def c500_input_norm_qkv(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    w_inorm: torch.Tensor,
    w_qkv: torch.Tensor,
    eps: float,
    c_qkv: torch.Tensor | None = None,
    s_qkv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return input_norm_qkv_forward(
        hidden_states,
        residual,
        w_inorm,
        w_qkv,
        c_qkv,
        s_qkv,
        eps=eps,
        use_i8_qkv_prefill=_USE_I8_ATTN_PREFILL,
    )


def _fake_input_norm_qkv(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    w_inorm: torch.Tensor,
    w_qkv: torch.Tensor,
    eps: float,
    c_qkv: torch.Tensor | None = None,
    s_qkv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = hidden_states.shape[0]
    return hidden_states.new_empty((rows, 2560)), hidden_states.new_empty((rows, 2048))


direct_register_custom_op(
    "c500_input_norm_qkv",
    c500_input_norm_qkv,
    fake_impl=_fake_input_norm_qkv,
    dispatch_key="CUDA",
)


def c500_lm_head_project(
    x: torch.Tensor,
    weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    rows = x.shape[0]
    if x.stride(1) != 1 or x.stride(0) < x.shape[1] or not 1 <= rows <= 4096:
        return F.linear(x, weight)
    if rows < 8:
        out = x.new_empty((rows, weight.shape[0]))
        if try_metax_kcontig_nt_mm(x, weight, out):
            return out
        return F.linear(x, weight)
    return project(x, codes, scales)


direct_register_custom_op(
    "c500_lm_head_project",
    c500_lm_head_project,
    fake_impl=_fake,
    dispatch_key="CUDA",
)


def _bind_c500_decoder_layer_eager(layer) -> bool:
    ok = False
    try:
        sa = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        inorm = getattr(layer, "input_layernorm", None)
        pnorm = getattr(layer, "post_attention_layernorm", None)
        rot = getattr(sa, "rotary_emb", None) if sa is not None else None
        attn = getattr(sa, "attn", None) if sa is not None else None
        impl = getattr(attn, "impl", None) if attn is not None else None
        o_proj = getattr(sa, "o_proj", None) if sa is not None else None
        qkv_proj = getattr(sa, "qkv_proj", None) if sa is not None else None
        gu_proj = getattr(mlp, "gate_up_proj", None) if mlp is not None else None
        down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None

        if (
            not envs.VLLM_BATCH_INVARIANT
            and getattr(layer, "hidden_size", None) == 2048
            and sa is not None
            and (getattr(sa, "q_size", 0), getattr(sa, "kv_size", 0), getattr(sa, "head_dim", 0)) == (2048, 256, 128)
            and qkv_proj is not None
            and type(qkv_proj).__name__ == "QKVParallelLinear"
            and getattr(qkv_proj, "bias", None) is None
            and getattr(qkv_proj, "tp_size", 1) == 1
            and tuple(qkv_proj.weight.shape) == (2560, 2048)
            and qkv_proj.weight.dtype == torch.bfloat16
            and qkv_proj.weight.stride(1) == 1
            and o_proj is not None
            and type(o_proj).__name__ == "RowParallelLinear"
            and getattr(o_proj, "bias", None) is None
            and getattr(o_proj, "tp_size", 1) == 1
            and tuple(o_proj.weight.shape) == (2048, 2048)
            and o_proj.weight.is_cuda
            and o_proj.weight.dtype == torch.bfloat16
            and o_proj.weight.stride(1) == 1
            and "C500" in torch.cuda.get_device_name(o_proj.weight.device)
            and rot is not None
            and getattr(rot, "is_neox_style", False) is True
            and getattr(rot, "head_size", 0) == 128
            and getattr(rot, "rotary_dim", 0) == 128
            and getattr(getattr(rot, "cos_sin_cache", None), "shape", (0, 0))[-1] == 128
            and not getattr(getattr(rot, "apply_rotary_emb", None), "enable_fp32_compute", False)
            and attn is not None
            and not getattr(attn, "calculate_kv_scales", False)
            and getattr(attn, "query_quant", None) is None
            and not getattr(attn, "use_direct_call", False)
            and getattr(attn, "kv_sharing_target_layer_name", None) is None
            and impl is not None
            and getattr(impl, "_plain_c500_gqa", False) is True
            and inorm is not None
            and getattr(inorm, "has_weight", False) is True
            and getattr(inorm, "variance_size_override", None) is None
            and tuple(inorm.weight.shape) == (2048,)
            and inorm.weight.dtype == torch.bfloat16
            and pnorm is not None
            and getattr(pnorm, "has_weight", False) is True
            and getattr(pnorm, "variance_size_override", None) is None
            and tuple(pnorm.weight.shape) == (2048,)
            and pnorm.weight.dtype == torch.bfloat16
            and gu_proj is not None
            and type(gu_proj).__name__ == "MergedColumnParallelLinear"
            and hasattr(gu_proj, "_c500_weight_codes")
            and down_proj is not None
            and type(down_proj).__name__ == "RowParallelLinear"
            and hasattr(down_proj, "_c500_weight_codes")
            and hasattr(o_proj, "_c500_weight_codes")
            and hasattr(qkv_proj, "_c500_weight_codes")
        ):
            impl._c500_rotary_emb = rot
            impl._c500_qkv_proj = (
                qkv_proj.weight,
                qkv_proj._c500_weight_codes,
                qkv_proj._c500_weight_scales,
            )
            ok = True
    except Exception as exc:
        logger.warning("C500 decoder fusion eligibility check skipped: %s", exc)
        ok = False

    layer._c500_fused_ok = ok
    return ok


def _bind_c500_model_chain(model) -> None:
    from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaModel

    bound_count = 0
    for mod in model.modules():
        if isinstance(mod, LlamaDecoderLayer):
            if _bind_c500_decoder_layer_eager(mod):
                bound_count += 1

    chained_models = 0
    for mod in model.modules():
        if isinstance(mod, LlamaModel):
            layers = list(mod.layers[mod.start_layer : mod.end_layer])
            norm = getattr(mod, "norm", None)
            chain_ok = (
                len(layers) >= 1
                and all(isinstance(l, LlamaDecoderLayer) and getattr(l, "_c500_fused_ok", False) for l in layers)
                and norm is not None
                and getattr(norm, "has_weight", False) is True
                and getattr(norm, "variance_size_override", None) is None
                and tuple(norm.weight.shape) == (2048,)
                and norm.weight.dtype == torch.bfloat16
                and not getattr(mod, "aux_hidden_state_layers", None)
            )
            mod._c500_chain_ok = chain_ok
            if chain_ok:
                chained_models += 1

    if bound_count:
        logger.info_once(
            "C500 bound cross-layer fused chain (%d models, %d decoder layers)",
            chained_models,
            bound_count,
        )


def _install_c500_model_hooks():
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    import vllm.model_executor.layers.attention.attention as attn_mod
    import vllm.model_executor.model_loader.utils as loader_utils
    from vllm.distributed import get_pp_group
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )
    from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaModel

    orig_pwal = loader_utils.process_weights_after_loading

    def _c500_process_weights_after_loading(model, model_config, target_device):
        orig_pwal(model, model_config, target_device)
        _bind_c500_model_chain(model)

    loader_utils.process_weights_after_loading = _c500_process_weights_after_loading
    try:
        import vllm.model_executor.model_loader.base_loader as base_loader
        if getattr(base_loader, "process_weights_after_loading", None) is orig_pwal:
            base_loader.process_weights_after_loading = _c500_process_weights_after_loading
    except Exception:
        pass

    orig_emb_load = UnquantizedEmbeddingMethod.process_weights_after_loading

    class _C500LMHeadMethod(UnquantizedEmbeddingMethod):
        def apply(self, layer, x, bias=None):
            if (
                not _USE_I8_LM_HEAD
                or bias is not None
                or x.dtype != torch.bfloat16
                or x.device != layer.weight.device
                or x.shape[-1] != 2048
                or torch.is_grad_enabled()
            ):
                return super().apply(layer, x, bias)
            batch_dims = x.shape[:-1]
            x_flat = x.reshape(-1, 2048)
            if not 1 <= x_flat.shape[0] <= 4096:
                return super().apply(layer, x, bias)
            out = torch.ops.vllm.c500_lm_head_project(
                x_flat, layer.weight, layer._c500_lm_codes, layer._c500_lm_scales
            )
            return out.view(*batch_dims, layer.weight.shape[0])

    def _load_emb(self, layer):
        orig_emb_load(self, layer)
        if type(self) not in (UnquantizedEmbeddingMethod, _C500LMHeadMethod):
            return
        weight = getattr(layer, "weight", None)
        if (
            not _USE_I8_LM_HEAD
            or not isinstance(layer, ParallelLMHead)
            or weight is None
            or tuple(weight.shape) != (130560, 2048)
            or not weight.is_cuda
            or weight.dtype != torch.bfloat16
            or weight.stride(1) != 1
            or weight.stride(0) < 2048
            or getattr(layer, "bias", None) is not None
            or getattr(layer, "tp_size", 1) != 1
            or envs.VLLM_BATCH_INVARIANT
            or "C500" not in torch.cuda.get_device_name(weight.device)
        ):
            for name in ("_c500_lm_codes", "_c500_lm_scales"):
                if name in layer._buffers:
                    delattr(layer, name)
            if type(self) is _C500LMHeadMethod:
                layer.quant_method = UnquantizedEmbeddingMethod()
            return

        with torch.cuda.device(weight.device), torch.no_grad():
            codes, scales = quantize_rows(weight)
        layer.register_buffer("_c500_lm_codes", codes, persistent=False)
        layer.register_buffer("_c500_lm_scales", scales, persistent=False)
        if type(self) is not _C500LMHeadMethod:
            layer.quant_method = _C500LMHeadMethod()
        logger.info_once("C500 row-scaled INT8 packed lm_head weights of shape %s", tuple(weight.shape))

    UnquantizedEmbeddingMethod.process_weights_after_loading = _load_emb

    get_attn_ctx = attn_mod.get_attention_context

    def _c500_unified_attention_with_output(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer_name,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        kv_cache_dummy_dep: torch.Tensor | None = None,
    ) -> None:
        resolved = _resolve_layer_name(layer_name)
        attn_metadata, attn_layer, kv_cache, _ = get_attn_ctx(resolved)
        if (
            kv_cache_dummy_dep is not None
            and kv_cache_dummy_dep.dtype in (torch.int32, torch.int64)
            and getattr(attn_layer.impl, "_c500_rotary_emb", None) is not None
        ):
            if (
                output_scale is not None
                and output_scale.ndim == 1
                and output_scale.dtype == torch.float32
                and output_scale.shape[0] == query.shape[0]
                and getattr(attn_layer.impl, "_c500_qkv_proj", None) is not None
            ):
                attn_layer.impl.forward(
                    attn_layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output=output,
                    output_scale=None,
                    output_block_scale=output_block_scale,
                    positions=kv_cache_dummy_dep,
                    qkv_pack_scales=output_scale,
                )
                return
            attn_layer.impl.forward(
                attn_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
                positions=kv_cache_dummy_dep,
            )
            return
        attn_layer.impl.forward(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    vllm_lib.impl(
        "unified_attention_with_output",
        _c500_unified_attention_with_output,
        dispatch_key="CUDA",
        allow_override=True,
    )
    attn_mod.unified_attention_with_output = _c500_unified_attention_with_output

    orig_layer_forward = LlamaDecoderLayer.forward

    def _c500_decoder_layer_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            torch.is_grad_enabled()
            or hidden_states.dtype != torch.bfloat16
            or not getattr(self, "_c500_fused_ok", False)
        ):
            return orig_layer_forward(self, positions, hidden_states, residual)

        sa = self.self_attn
        qkv_proj = sa.qkv_proj
        c_qkv = qkv_proj._c500_weight_codes if _USE_I8_ATTN_PREFILL else None
        s_qkv = qkv_proj._c500_weight_scales if _USE_I8_ATTN_PREFILL else None
        qkv, residual = torch.ops.vllm.c500_input_norm_qkv(
            hidden_states,
            residual,
            self.input_layernorm.weight,
            qkv_proj.weight,
            self.input_layernorm.variance_epsilon,
            c_qkv,
            s_qkv,
        )
        q, k, v = qkv.split([2048, 256, 256], dim=-1)
        n_tok = q.shape[0]
        attn_out = torch.empty((n_tok, 16, 128), dtype=q.dtype, device=q.device)
        encoded = _encode_layer_name(sa.attn.layer_name)
        torch.ops.vllm.unified_attention_with_output(
            q.view(n_tok, 16, 128),
            k.view(n_tok, 2, 128),
            v.view(n_tok, 2, 128),
            attn_out,
            encoded,
            kv_cache_dummy_dep=positions.flatten(),
        )

        o_proj = sa.o_proj
        c_o = o_proj._c500_weight_codes if _USE_I8_ATTN_PREFILL else None
        s_o = o_proj._c500_weight_scales if _USE_I8_ATTN_PREFILL else None
        gu_proj = self.mlp.gate_up_proj
        down_proj = self.mlp.down_proj
        hidden_states, residual = torch.ops.vllm.c500_post_attn_mlp(
            attn_out.view(n_tok, 2048),
            residual,
            o_proj.weight,
            self.post_attention_layernorm.weight,
            gu_proj.weight,
            gu_proj._c500_weight_codes,
            gu_proj._c500_weight_scales,
            down_proj.weight,
            down_proj._c500_weight_codes,
            down_proj._c500_weight_scales,
            self.post_attention_layernorm.variance_epsilon,
            c_o,
            s_o,
        )
        return hidden_states, residual

    LlamaDecoderLayer.forward = _c500_decoder_layer_forward

    orig_model_forward = LlamaModel.forward

    def _c500_llama_model_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ):
        if (
            torch.is_grad_enabled()
            or extra_layer_kwargs
            or not getattr(self, "_c500_chain_ok", False)
            or not get_pp_group().is_first_rank
            or not get_pp_group().is_last_rank
            or getattr(self, "aux_hidden_state_layers", None)
        ):
            return orig_model_forward(
                self, input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds, **extra_layer_kwargs
            )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        if hidden_states.dtype != torch.bfloat16:
            return orig_model_forward(
                self, input_ids, positions, intermediate_tensors, inputs_embeds=hidden_states, **extra_layer_kwargs
            )

        layers = self.layers[self.start_layer : self.end_layer]
        n_layers = len(layers)
        n_tok = hidden_states.shape[0]
        pos_flat = positions.flatten()

        layer0 = layers[0]
        x_pack, s_pack, residual = torch.ops.vllm.c500_input_norm_pack(
            hidden_states,
            layer0.input_layernorm.weight,
            layer0.input_layernorm.variance_epsilon,
        )

        for idx in range(n_layers):
            layer = layers[idx]
            sa = layer.self_attn
            attn_out = torch.empty((n_tok, 16, 128), dtype=torch.bfloat16, device=hidden_states.device)
            encoded = _encode_layer_name(sa.attn.layer_name)
            kv_dummy = x_pack[:, :256].view(n_tok, 2, 128)
            torch.ops.vllm.unified_attention_with_output(
                x_pack.view(n_tok, 16, 128),
                kv_dummy,
                kv_dummy,
                attn_out,
                encoded,
                output_scale=s_pack,
                kv_cache_dummy_dep=pos_flat,
            )

            o_proj = sa.o_proj
            gu_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj
            if idx + 1 < n_layers:
                next_layer = layers[idx + 1]
                x_pack, s_pack, residual = torch.ops.vllm.c500_post_attn_mlp_next_norm_pack(
                    attn_out.view(n_tok, 2048),
                    residual,
                    o_proj.weight,
                    layer.post_attention_layernorm.weight,
                    gu_proj.weight,
                    gu_proj._c500_weight_codes,
                    gu_proj._c500_weight_scales,
                    down_proj.weight,
                    down_proj._c500_weight_codes,
                    down_proj._c500_weight_scales,
                    layer.post_attention_layernorm.variance_epsilon,
                    next_layer.input_layernorm.weight,
                    next_layer.input_layernorm.variance_epsilon,
                    o_proj._c500_weight_codes,
                    o_proj._c500_weight_scales,
                )
            else:
                hidden_states = torch.ops.vllm.c500_post_attn_mlp_final_norm(
                    attn_out.view(n_tok, 2048),
                    residual,
                    o_proj.weight,
                    layer.post_attention_layernorm.weight,
                    gu_proj.weight,
                    gu_proj._c500_weight_codes,
                    gu_proj._c500_weight_scales,
                    down_proj.weight,
                    down_proj._c500_weight_codes,
                    down_proj._c500_weight_scales,
                    layer.post_attention_layernorm.variance_epsilon,
                    self.norm.weight,
                    self.norm.variance_epsilon,
                    o_proj._c500_weight_codes,
                    o_proj._c500_weight_scales,
                )
        return hidden_states

    LlamaModel.forward = _c500_llama_model_forward
    _HOOKS_INSTALLED = True
    logger.info_once("Installed C500 cross-layer fused QKV-RoPE-Cache + Post-Attn-MLP-NextNorm chain hooks")


def _ensure_linear_hook_installed():
    global _LINEAR_HOOK_INSTALLED
    if _LINEAR_HOOK_INSTALLED:
        return
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    orig_base_load_model = BaseModelLoader.load_model

    def _c500_base_load_model(self, *args, **kwargs):
        _install_c500_model_hooks()
        model = orig_base_load_model(self, *args, **kwargs)
        if _HOOKS_INSTALLED:
            _bind_c500_model_chain(model)
        return model

    BaseModelLoader.load_model = _c500_base_load_model

    original_load = UnquantizedLinearMethod.process_weights_after_loading

    class _C500FeedForwardMethod(UnquantizedLinearMethod):
        def apply(self, layer, x, bias=None):
            if (bias is not None or x.ndim != 2 or x.dtype != torch.bfloat16
                    or x.device != layer.weight.device or x.shape[-1] != layer.weight.shape[-1]
                    or torch.is_grad_enabled()):
                return super().apply(layer, x, bias)
            return torch.ops.vllm.c500_ffn_project(
                x, layer.weight, layer._c500_weight_codes, layer._c500_weight_scales
            )

    def _load(self, layer):
        original_load(self, layer)
        if type(self) not in (UnquantizedLinearMethod, _C500FeedForwardMethod):
            return
        weight = getattr(layer, "weight", None)
        role = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
        expected = {"gate_up_proj": (12288, 2048), "down_proj": (2048, 6144)}
        if _USE_I8_ATTN_PREFILL:
            expected["o_proj"] = (2048, 2048)
            expected["qkv_proj"] = (2560, 2048)

        if (weight is None or role not in expected
                or tuple(weight.shape) != expected[role] or not weight.is_cuda
                or weight.dtype != torch.bfloat16 or weight.stride(1) != 1
                or weight.stride(0) < weight.shape[1] or getattr(layer, "bias", None) is not None
                or getattr(layer, "tp_size", 1) != 1 or envs.VLLM_BATCH_INVARIANT
                or "C500" not in torch.cuda.get_device_name(weight.device)):
            for name in ("_c500_weight_codes", "_c500_weight_scales"):
                if name in layer._buffers:
                    delattr(layer, name)
            if type(self) is _C500FeedForwardMethod:
                layer.quant_method = UnquantizedLinearMethod()
            return

        with torch.cuda.device(weight.device), torch.no_grad():
            codes, scales = quantize_rows(weight)
        layer.register_buffer("_c500_weight_codes", codes, persistent=False)
        layer.register_buffer("_c500_weight_scales", scales, persistent=False)
        if role in ("gate_up_proj", "down_proj") and type(self) is not _C500FeedForwardMethod:
            layer.quant_method = _C500FeedForwardMethod()
        logger.info_once("C500 row-scaled INT8 packed %s weights of shape %s", role, tuple(weight.shape))
        _install_c500_model_hooks()

    UnquantizedLinearMethod.process_weights_after_loading = _load
    _LINEAR_HOOK_INSTALLED = True


try:
    _ensure_linear_hook_installed()
except ImportError:
    from vllm.model_executor import custom_op as _co

    _orig_co_init = _co.CustomOp.__init__

    def _deferred_co_init(self, *args, **kwargs):
        _ensure_linear_hook_installed()
        _co.CustomOp.__init__ = _orig_co_init
        return _orig_co_init(self, *args, **kwargs)

    _co.CustomOp.__init__ = _deferred_co_init
