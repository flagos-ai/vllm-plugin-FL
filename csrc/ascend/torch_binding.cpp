/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Reduced, FL-owned binding for the vLLM-Ascend 0.20.2 operators exercised by
// Qwen3.6.  Keeping this registration surface small avoids linking unrelated
// sparse-attention, quantization and MC2 kernels into the standalone plugin.

#include <ATen/ATen.h>
#include <torch/library.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h"
#include "moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe/moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h"

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm(
    const at::Tensor& x,
    const at::Tensor& gamma,
    double epsilon)
{
    const int64_t diff = x.dim() - gamma.dim();
    std::vector<int64_t> rstd_shape;
    if (diff > 0) {
        rstd_shape.reserve(x.dim());
        for (int64_t i = 0; i < diff; ++i) {
            rstd_shape.push_back(x.size(i));
        }
        for (int64_t i = 0; i < gamma.dim(); ++i) {
            rstd_shape.push_back(1);
        }
    } else {
        rstd_shape.assign(x.dim(), 1);
    }
    at::Tensor rstd = at::empty(rstd_shape, x.options().dtype(at::kFloat));
    at::Tensor y = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnGemmaRmsNorm, x, gamma, epsilon, y, rstd);
    return {y, rstd};
}

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef query_start_loc,
    at::IntArrayRef cache_indices,
    at::IntArrayRef initial_state_mode,
    at::IntArrayRef num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode)
{
    EXEC_NPU_CMD(
        aclnnCausalConv1d,
        x,
        weight,
        bias,
        conv_state,
        query_start_loc,
        cache_indices,
        initial_state_mode,
        num_accepted_tokens,
        activation_mode,
        pad_slot_id,
        run_mode,
        output);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_gated_delta_rule_fwd_h(
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& u,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const c10::optional<at::Tensor>& initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout)
{
    const bool output_final_state_value = output_final_state.value_or(false);
    const int64_t chunk_size_value = chunk_size.value_or(64);
    const at::Tensor& g_value = c10::value_or_else(
        g, [] { return at::Tensor(); });
    const at::Tensor& gk_value = c10::value_or_else(
        gk, [] { return at::Tensor(); });
    const at::Tensor& initial_state_value = c10::value_or_else(
        initial_state, [] { return at::Tensor(); });

    const int64_t B = k.size(0);
    const int64_t K = k.size(3);
    const int64_t T = k.size(2);
    const int64_t HV = u.size(1);
    const int64_t V = u.size(3);
    const int64_t NT = chunk_indices.has_value()
        ? static_cast<int64_t>(chunk_indices->size() / 2)
        : (T + chunk_size_value - 1) / chunk_size_value;

    at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options());
    at::Tensor v_new_out = at::zeros(u.sizes(), u.options());
    at::Tensor final_state_out;
    if (output_final_state_value) {
        const int64_t N = cu_seqlens.has_value()
            ? static_cast<int64_t>(cu_seqlens->size() - 1)
            : B;
        auto options = initial_state.has_value()
            ? initial_state->options()
            : h_out.options();
        final_state_out = at::empty({N, HV, K, V}, options);
    } else {
        final_state_out = at::empty({1}, k.options());
    }

    // EXEC_NPU_CMD's current adapter converts arguments through non-const
    // lvalue references. Keep optional defaults in named locals, matching
    // the vLLM-Ascend 0.20.2 binding contract.
    bool save_new_value_value = save_new_value.value_or(true);
    bool use_exp2_value = use_exp2.value_or(false);
    bool transpose_state_layout_value =
        transpose_state_layout.value_or(false);

    EXEC_NPU_CMD(
        aclnnChunkGatedDeltaRuleFwdH,
        k,
        w,
        u,
        g_value,
        gk_value,
        initial_state_value,
        output_final_state_value,
        chunk_size_value,
        save_new_value_value,
        cu_seqlens,
        chunk_indices,
        use_exp2_value,
        transpose_state_layout_value,
        h_out,
        v_new_out,
        final_state_out);

    return {
        h_out,
        v_new_out,
        output_final_state_value ? final_state_out : at::Tensor(),
    };
}

at::Tensor chunk_fwd_o(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& h,
    double scale,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& g_gamma,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> transpose_state_layout)
{
    at::Tensor output = at::zeros(v.sizes(), v.options());
    int64_t chunk_size_value = chunk_size.value_or(64);
    const at::Tensor& g_value = c10::value_or_else(
        g, [] { return at::Tensor(); });
    (void)g_gamma;
    (void)transpose_state_layout;
    EXEC_NPU_CMD(
        aclnnChunkFwdO,
        q,
        k,
        v,
        h,
        g_value,
        cu_seqlens,
        chunk_indices,
        scale,
        chunk_size_value,
        output);
    return output;
}

namespace meta {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_moe_init_routing_custom(
    const at::Tensor& x,
    const at::Tensor& expert_idx,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& offset,
    int64_t active_num,
    int64_t expert_capacity,
    int64_t expert_num,
    int64_t drop_pad_mode,
    int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag,
    int64_t quant_mode,
    at::IntArrayRef active_expert_range,
    int64_t row_idx_type)
{
    (void)scale;
    (void)offset;
    (void)expert_tokens_num_flag;
    (void)row_idx_type;
    const c10::SymInt batch = x.sym_size(0);
    const c10::SymInt hidden = x.sym_size(1);
    const c10::SymInt top_k = expert_idx.sym_size(1);
    const c10::SymInt expanded_tokens = active_num > 0
        ? c10::SymInt(active_num)
        : batch * top_k;
    c10::SymDimVector expanded_shape;
    if (drop_pad_mode == 1) {
        expanded_shape = {
            c10::SymInt(expert_num), c10::SymInt(expert_capacity), hidden};
    } else {
        expanded_shape = {expanded_tokens, hidden};
    }
    auto expanded_options = quant_mode == -1
        ? x.options()
        : x.options().dtype(at::kChar);
    const int64_t first_expert = active_expert_range.empty()
        ? 0
        : active_expert_range[0];
    const int64_t last_expert = active_expert_range.empty()
        ? expert_num
        : active_expert_range[1];
    c10::SymDimVector expert_tokens_shape = expert_tokens_num_type == 2
        ? c10::SymDimVector{
              c10::SymInt(expert_num), c10::SymInt(2)}
        : c10::SymDimVector{
              c10::SymInt(last_expert - first_expert)};
    return {
        at::empty_symint(expanded_shape, expanded_options),
        at::empty_symint({batch * top_k}, expert_idx.options()),
        at::empty_symint(
            expert_tokens_shape, x.options().dtype(at::kLong)),
        at::empty_symint(
            {drop_pad_mode == 1
                 ? c10::SymInt(expert_num * expert_capacity)
                 : expanded_tokens},
            x.options().dtype(at::kFloat)),
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k(
    const at::Tensor& x,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt)
{
    (void)k_group;
    (void)group_count;
    (void)group_select_mode;
    (void)renorm;
    (void)norm_type;
    (void)out_flag;
    (void)routed_scaling_factor;
    (void)eps;
    (void)bias_opt;
    const c10::SymInt rows = x.sym_size(0);
    const c10::SymInt experts = x.sym_size(1);
    return {
        at::empty_symint({rows, c10::SymInt(k)}, x.options()),
        at::empty_symint(
            {rows, c10::SymInt(k)}, x.options().dtype(at::kInt)),
        at::empty_symint(
            {rows, experts}, x.options().dtype(at::kFloat)),
    };
}

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm(
    const at::Tensor& x,
    const at::Tensor& gamma,
    double epsilon)
{
    (void)epsilon;
    c10::SymDimVector rstd_shape;
    const int64_t diff = x.dim() - gamma.dim();
    if (diff > 0) {
        rstd_shape.reserve(x.dim());
        for (int64_t i = 0; i < diff; ++i) {
            rstd_shape.push_back(x.sym_size(i));
        }
        for (int64_t i = 0; i < gamma.dim(); ++i) {
            rstd_shape.push_back(c10::SymInt(1));
        }
    } else {
        rstd_shape.assign(x.dim(), c10::SymInt(1));
    }
    return {
        at::empty_symint(x.sym_sizes(), x.options()),
        at::empty_symint(rstd_shape, x.options().dtype(at::kFloat)),
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_bias(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& beta,
    double epsilon)
{
    (void)x2;
    (void)beta;
    (void)epsilon;
    c10::SymDimVector rstd_shape;
    const int64_t diff = x1.dim() - gamma.dim();
    if (diff > 0) {
        rstd_shape.reserve(x1.dim());
        for (int64_t i = 0; i < diff; ++i) {
            rstd_shape.push_back(x1.sym_size(i));
        }
        for (int64_t i = 0; i < gamma.dim(); ++i) {
            rstd_shape.push_back(c10::SymInt(1));
        }
    } else {
        rstd_shape.assign(x1.dim(), c10::SymInt(1));
    }
    return {
        at::empty_symint(x1.sym_sizes(), x1.options()),
        at::empty_symint(rstd_shape, x1.options().dtype(at::kFloat)),
        at::empty_symint(x1.sym_sizes(), x1.options()),
    };
}

at::Tensor npu_recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    const c10::optional<double> scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk)
{
    (void)query;
    (void)key;
    (void)state;
    (void)beta;
    (void)scale;
    (void)actual_seq_lengths;
    (void)ssm_state_indices;
    (void)num_accepted_tokens;
    (void)g;
    (void)gk;
    return at::empty_symint(
        value.sym_sizes(), value.options().dtype(at::kBFloat16));
}

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef query_start_loc,
    at::IntArrayRef cache_indices,
    at::IntArrayRef initial_state_mode,
    at::IntArrayRef num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id,
    int64_t run_mode)
{
    (void)x;
    (void)weight;
    (void)conv_state;
    (void)bias;
    (void)query_start_loc;
    (void)cache_indices;
    (void)initial_state_mode;
    (void)num_accepted_tokens;
    (void)activation_mode;
    (void)pad_slot_id;
    (void)run_mode;
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
chunk_gated_delta_rule_fwd_h(
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& u,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const c10::optional<at::Tensor>& initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout)
{
    (void)w;
    (void)g;
    (void)gk;
    (void)save_new_value;
    (void)use_exp2;
    (void)transpose_state_layout;
    const int64_t B = k.size(0);
    const int64_t K = k.size(3);
    const int64_t T = k.size(2);
    const int64_t HV = u.size(1);
    const int64_t V = u.size(3);
    const int64_t chunk_size_value = chunk_size.value_or(64);
    const int64_t NT = chunk_indices.has_value()
        ? static_cast<int64_t>(chunk_indices->size() / 2)
        : (T + chunk_size_value - 1) / chunk_size_value;
    at::Tensor h = at::empty({B, HV, NT, K, V}, k.options());
    at::Tensor v_new = at::empty(u.sizes(), u.options());
    if (!output_final_state.value_or(false)) {
        return {h, v_new, at::Tensor()};
    }
    const int64_t N = cu_seqlens.has_value()
        ? static_cast<int64_t>(cu_seqlens->size() - 1)
        : B;
    auto options = initial_state.has_value()
        ? initial_state->options()
        : h.options();
    return {h, v_new, at::empty({N, HV, K, V}, options)};
}

at::Tensor chunk_fwd_o(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& h,
    double scale,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& g_gamma,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> transpose_state_layout)
{
    (void)q;
    (void)k;
    (void)h;
    (void)scale;
    (void)g;
    (void)g_gamma;
    (void)cu_seqlens;
    (void)chunk_indices;
    (void)chunk_size;
    (void)transpose_state_layout;
    return at::empty(v.sizes(), v.options());
}

}  // namespace meta
}  // namespace vllm_ascend

TORCH_LIBRARY(_C_ascend, ops) {
    ops.def(
        "npu_moe_init_routing_custom(Tensor x, Tensor expert_idx, *, "
        "Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
        "int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, "
        "int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, "
        "int quant_mode=0, int[2] active_expert_range=[], "
        "int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)");
    ops.def(
        "moe_gating_top_k(Tensor x, int k, int k_group, int group_count, "
        "int group_select_mode, int renorm, int norm_type, bool out_flag, "
        "float routed_scaling_factor, float eps, Tensor? bias_opt=None) "
        "-> (Tensor y, Tensor expert_idx, Tensor out)");
    ops.def(
        "npu_gemma_rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) "
        "-> (Tensor y, Tensor rstd)");
    ops.def(
        "npu_add_rms_norm_bias(Tensor x1, Tensor x2, Tensor gamma, "
        "Tensor? beta=None, float epsilon=1e-6) "
        "-> (Tensor y, Tensor rstd, Tensor x)");
    ops.def(
        "npu_recurrent_gated_delta_rule(Tensor query, Tensor key, "
        "Tensor value, Tensor(a!) state, *, Tensor? beta=None, "
        "float? scale=None, Tensor? actual_seq_lengths=None, "
        "Tensor? ssm_state_indices=None, Tensor? num_accepted_tokens=None, "
        "Tensor? g=None, Tensor? gk=None) -> Tensor");
    ops.def(
        "npu_causal_conv1d_custom(Tensor output, Tensor x, Tensor weight, "
        "Tensor conv_state, Tensor? bias_opt, int[] query_start_loc_opt, "
        "int[] cache_indices_opt, int[] initial_state_mode_opt, "
        "int[] num_accepted_tokens_opt, int activation_mode, "
        "int pad_slot_id, int run_mode) -> Tensor");
    ops.def(
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, "
        "Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, "
        "bool? output_final_state=False, int? chunk_size=None, "
        "bool? save_new_value=True, int[]? cu_seqlens=None, "
        "int[]? chunk_indices=None, bool? use_exp2=False, "
        "bool? transpose_state_layout=False) "
        "-> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)");
    ops.def(
        "chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, "
        "*, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, "
        "int[]? chunk_indices=None, int? chunk_size=None, "
        "bool? transpose_state_layout=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) {
    ops.impl(
        "npu_moe_init_routing_custom",
        &vllm_ascend::npu_moe_init_routing_custom);
    ops.impl("moe_gating_top_k", &vllm_ascend::moe_gating_top_k);
    ops.impl("npu_gemma_rms_norm", &vllm_ascend::npu_gemma_rms_norm);
    ops.impl("npu_add_rms_norm_bias", &vllm_ascend::npu_add_rms_norm_bias);
    ops.impl(
        "npu_recurrent_gated_delta_rule",
        &vllm_ascend::npu_recurrent_gated_delta_rule);
    ops.impl(
        "npu_causal_conv1d_custom",
        &vllm_ascend::npu_causal_conv1d_custom);
    ops.impl(
        "chunk_gated_delta_rule_fwd_h",
        &vllm_ascend::chunk_gated_delta_rule_fwd_h);
    ops.impl("chunk_fwd_o", &vllm_ascend::chunk_fwd_o);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
    ops.impl(
        "npu_moe_init_routing_custom",
        &vllm_ascend::meta::npu_moe_init_routing_custom);
    ops.impl("moe_gating_top_k", &vllm_ascend::meta::moe_gating_top_k);
    ops.impl("npu_gemma_rms_norm", &vllm_ascend::meta::npu_gemma_rms_norm);
    ops.impl(
        "npu_add_rms_norm_bias",
        &vllm_ascend::meta::npu_add_rms_norm_bias);
    ops.impl(
        "npu_recurrent_gated_delta_rule",
        &vllm_ascend::meta::npu_recurrent_gated_delta_rule);
    ops.impl(
        "npu_causal_conv1d_custom",
        &vllm_ascend::meta::npu_causal_conv1d_custom);
    ops.impl(
        "chunk_gated_delta_rule_fwd_h",
        &vllm_ascend::meta::chunk_gated_delta_rule_fwd_h);
    ops.impl("chunk_fwd_o", &vllm_ascend::meta::chunk_fwd_o);
}
