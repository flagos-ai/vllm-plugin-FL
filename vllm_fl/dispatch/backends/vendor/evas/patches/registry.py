# Copyright (c) 2026 BAAI. All rights reserved.
"""Declarative patch registry for evas/LPU monkey-patches.

Each ``PatchSpec`` (name / desc / apply_fn / gate / anchors) is data, so
adding a patch never touches ``patch.py``: ``apply_evas_patches()`` builds the
list via ``_build_registry``, honours ``VLLM_FL_LPU_DISABLE_PATCHES``, and
applies what passes its gate.  Patches stay LPU-gated, individually
disable-able and idempotent.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PatchSpec:
    """Metadata + apply function for a single LPU monkey-patch."""

    # Unique short name (used in VLLM_FL_LPU_DISABLE_PATCHES).
    name: str

    # Human-readable description.
    desc: str

    # The apply function. Should be idempotent (no-op after first call).
    # Returns False when it could not install because its target symbol is
    # missing (the caller then logs "Declined", never "Applied", and raises
    # under VLLM_FL_STRICT_UPSTREAM_REBIND=1). None means "installed" -- the
    # legacy contract -- so a patch that has nothing to report stays a plain
    # rebind. A blind setattr cannot fail, so a spec that wraps one of our own
    # classes must return False instead of guessing.
    apply_fn: Callable[[], bool | None]

    # Optional gate predicate: if it returns False, the patch is skipped.
    # Used by the DSV4 patches to wait for base vllm's DSV4 package.
    gate: Optional[Callable[[], bool]] = None

    # Symbols this patch rebinds, as greppable dotted paths -- usually upstream
    # (vllm/torch/torch_lpu), occasionally our own module when the patch wraps
    # plugin code (e.g. mamba_state_unpacked -> ModelRunnerFL).  This is the blast
    # radius: on a bump or refactor, grep these to see what moved.  Purely
    # declarative -- production code never resolves them; tests do (see
    # tests/unit_tests/dispatch/test_evas_upstream_anchors.py).
    upstream_anchors: tuple[str, ...] = ()

    # One line on how version-sensitive those anchors are (what to re-verify
    # on an upstream bump).  Empty when the anchors are stable torch-level names.
    version_note: str = ""

    # Lowest vllm core version whose anchors exist, for a patch that only has
    # a target from some core version onward.  None means the anchors must
    # resolve on any supported core.  Without the floor the anchor test cannot
    # tell a rename from a symbol the installed core never had, so it reports
    # the second as a failure and the whole check goes unread.
    min_vllm: Optional[tuple[int, int]] = None


# ---- Patch specs (registered in apply order) ----


def _dsv4_support_available() -> bool:
    """True once the base vllm DSV4 package is importable.

    DSV4 lives in base vllm's per-platform package (``vllm.models.deepseek_v4``);
    LPU logic lives in the patches, not in a model fork.  The DSV4 patches must
    not fire before the base package resolves (e.g. on a vllm without it), so
    non-DSV4 models are unaffected.
    """
    try:
        import vllm.models.deepseek_v4  # noqa: F401
        return True
    except ImportError:
        return False


def _packed_kv_config_available() -> bool:
    """True when this vllm still plans a packed KV cache.

    The whole patch is a rebind of that planner; on a vllm that predates it
    there is nothing to unbind, so the gate keeps the patch list honest.
    """
    try:
        from vllm.v1.core import kv_cache_utils
    except ImportError:
        return False
    return hasattr(kv_cache_utils, "_get_kv_cache_config_packed")


def _lmhead_fp8_requested() -> bool:
    """True when the opt-in fp8 head knob (or a logits dump) asked for it.

    The patch has nothing to do on a default run, so the gate makes that
    visible as a skip instead of counting an untouched head as patched.
    """
    from .lmhead_fp8 import requested

    return requested()


def _minimax_m3_available() -> bool:
    """True once the upstream MiniMax-M3 model module is importable.

    Same contract as ``_dsv4_support_available``: the patch touches only that
    model's own norm class, so it is inert for every other architecture.
    """
    try:
        import vllm.models.minimax_m3.nvidia.model  # noqa: F401
        return True
    except ImportError:
        return False


def _build_registry(disable_patches: set[str]) -> list[PatchSpec]:
    """Build the ordered list of patches, skipping disabled ones."""
    specs: list[PatchSpec] = [
        PatchSpec(
            name="device_properties",
            desc="Add multi_processor_count=1 to LPU _LPUDeviceProperties "
                 "(LPU driver only exposes name + total_memory)",
            apply_fn=_apply_device_properties,
            upstream_anchors=(
                "torch_lpu._C._LPUDeviceProperties.multi_processor_count",
            ),
            version_note=(
                 "torch_lpu ABI attribute add, not a vllm symbol: breaks if the " +
                 "properties class becomes immutable or the field moves."
            ),
        ),
        PatchSpec(
            name="sampler",
            desc="Bypass the Triton apply_top_k_top_p dispatcher on LPU; use "
                 "the upstream pure-PyTorch path (topk/sort verified clean on "
                 "the 151k vocab) + host-side temperature/argmax fixes",
            apply_fn=_apply_sampler,
            upstream_anchors=(
                "vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p",
                "vllm.v1.sample.ops.topk_topp_sampler.random_sample",
                "vllm.v1.sample.sampler.Sampler.apply_temperature",
                "vllm.v1.sample.sampler.Sampler.greedy_sample",
                "vllm.v1.sample.sampler.Sampler.sample",
            ),
            version_note=(
                 "Sampler method names track the installed vllm -- re-verify the " +
                 "four method names after any sampler refactor."
            ),
        ),
        PatchSpec(
            name="rejection_sampler",
            desc="Rebind the spec-decode rejection sampler's four Triton "
                 "kernels (expand / greedy / random / recovered-token) to "
                 "pure-torch stand-ins in the rejection_sampler module's own "
                 "namespace -- an evas-Triton compile dead-end that kills "
                 "engine init (profile_run -> _dummy_sampler_run) whenever "
                 "speculative decoding is on. Also rebinds that module's "
                 "module-scope apply_top_k_top_p alias to the upstream "
                 "pure-PyTorch implementation (the sampler patch cannot reach "
                 "it: from-import copy)",
            apply_fn=_apply_rejection_sampler,
            upstream_anchors=(
                "vllm.v1.sample.rejection_sampler.expand_kernel",
                "vllm.v1.sample.rejection_sampler.rejection_greedy_sample_kernel",
                "vllm.v1.sample.rejection_sampler.rejection_random_sample_kernel",
                "vllm.v1.sample.rejection_sampler.sample_recovered_tokens_kernel",
                "vllm.v1.sample.rejection_sampler.apply_top_k_top_p",
            ),
            version_note=(
                 "The four kernels are module-scope Triton templates in "
                 "rejection_sampler.py and are called as kernel[(grid,)](...); "
                 "the stand-in must stay subscriptable. The apply_top_k_top_p "
                 "rebind is a from-import alias in that module, so it needs its "
                 "own entry here even though the sampler patch already handles "
                 "topk_topp_sampler. Both are re-verified upstream; the stand-in "
                 "transcription pins the kernels' loop bodies, which a sampler "
                 "refactor can change."
            ),
        ),
        PatchSpec(
            name="spec_decode_proposer",
            desc="Rebind the drafter proposer's two padded-prep Triton kernels "
                 "(eagle_prepare_next_token_padded / eagle_prepare_inputs_padded) "
                 "to pure-torch stand-ins in the llm_base_proposer module's own "
                 "namespace -- an evas-Triton compile dead-end that kills the "
                 "first decode step of any padded-drafter spec-decode run "
                 "(MTP). The three sibling kernels in utils.py stay upstream: "
                 "two are unreachable at one draft token, and "
                 "copy_and_expand_eagle_inputs is behind the "
                 "needs_extra_input_slots gate, which is zero only for a "
                 "drafter fed hidden states (see the module docstring)",
            apply_fn=_apply_spec_decode_proposer,
            upstream_anchors=(
                "vllm.v1.spec_decode.llm_base_proposer"
                ".eagle_prepare_next_token_padded_kernel",
                "vllm.v1.spec_decode.llm_base_proposer"
                ".eagle_prepare_inputs_padded_kernel",
            ),
            version_note=(
                 "Both names are from-import copies in llm_base_proposer.py "
                 "(cpu_model_runner.py:78-104 rebinds the same ones there), so "
                 "the anchor is the proposer module, not the defining utils.py. "
                 "They are called as kernel[(grid,)](...); the stand-in must stay "
                 "subscriptable. The transcriptions pin the kernels' bodies, "
                 "including the num_draft>0 guard on the rejected count."
            ),
        ),
        PatchSpec(
            name="async_spec_decode_bookkeeping",
            desc="Rebind the async spec-decode bookkeeping "
                 "(update_num_computed_tokens_for_batch_change) to an int64-free "
                 "body in the model runners' own namespaces -- its opening "
                 "prev_positions.clamp(min=0) is int64, which LPU rejects "
                 "(Operator clamp received unsupported dtypes: [Long]), and it "
                 "is the next step after the padded-drafter prep on every "
                 "async-spec-decode decode step",
            apply_fn=_apply_async_spec_decode_bookkeeping,
            upstream_anchors=(
                "vllm.v1.spec_decode.utils"
                ".update_num_computed_tokens_for_batch_change",
                "vllm_fl.worker.model_runner"
                ".update_num_computed_tokens_for_batch_change",
            ),
            version_note=(
                 "A module-scope @torch.compile-decorated def in "
                 "spec_decode/utils.py, "
                 "from-imported by both vllm.v1.worker.gpu_model_runner and our "
                 "fork's worker/model_runner (a standalone class, so the fork "
                 "namespace is the runtime one -- the anchor pins that import "
                 "line). LPU runs it eagerly, so dropping the decorator is "
                 "semantics-free. The transcription pins clamp(min=0) as a "
                 "bounds guard, the three whole-buffer gathers (a shrunk batch "
                 "indexes past n), read-before-write ordering, the [:n] vs "
                 "whole-argument write split, and the self-reading keep-branch "
                 "of the second where -- re-verify all five on any spec-decode "
                 "bookkeeping refactor."
            ),
        ),
        PatchSpec(
            name="block_table",
            desc="Replace BlockTable.compute_slot_mapping Triton kernel with "
                 "pure-PyTorch",
            apply_fn=_apply_block_table,
            upstream_anchors=(
                "vllm.v1.worker.block_table.BlockTable.compute_slot_mapping",
                "vllm.v1.worker.block_table.BlockTable.commit_block_table",
            ),
            version_note=(
                 "compute_slot_mapping is a Triton kernel in the worker BlockTable; the " +
                 "dirty-flag wrap also pins the attr-setter block list. Confirm both " +
                 "after a block_table refactor."
            ),
        ),
        PatchSpec(
            name="vocab_embedding",
            desc="Replace VocabParallelEmbedding.get_masked_input_and_mask with a "
                 "CPU-mask version (TP>1 concurrent native lt/ge launches segfault "
                 "the LPU driver, hpe_runtime.c:2392)",
            apply_fn=_apply_vocab_embedding,
            upstream_anchors=(
                "vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask",
            ),
            version_note=(
                 "Module-level helper; renamed/moved in some vllm releases -- grep " +
                 "rather than trust the path."
            ),
        ),
        PatchSpec(
            name="bind_kv_cache",
            desc="Add LPU to the bind_kv_cache multi-attention-layer allowlist so "
                 "DSV4 (swa_cache + attn per layer) does not raise NotImplementedError. "
                 "MUST be before events patch: events imports model_runner which "
                 "captures bind_kv_cache at module level.",
            apply_fn=_apply_bind_kv_cache,
            upstream_anchors=(
                "vllm.v1.worker.utils.bind_kv_cache",
            ),
            version_note=(
                 "Allowlist of multi-attention-layer device types. MUST run before the " +
                 "events patch (events imports model_runner, which captures this name " +
                 "at module level)."
            ),
        ),
        PatchSpec(
            name="kv_config_unpacked",
            desc="Rebind the packed KV cache planner to emit one independent "
                 "buffer per (page_size, slot) instead of one pool-wide "
                 "allocation. On LPU any strict sub-interval access of a "
                 "window into a shared backing is staged as the whole backing, "
                 "so a packed pool makes every page read/write cost the pool "
                 "(~51 ms/GiB ⇒ seconds per token at 43 layers). Same block "
                 "count and total bytes; VLLM_FL_KV_CONFIG_PACKED=1 is the "
                 "packed A/B arm. The plan is computed in the EngineCore "
                 "process, so this rebind is also installed from the plugin's "
                 "register_model() (see vllm_fl/__init__.py).",
            apply_fn=_apply_kv_config_unpacked,
            gate=_packed_kv_config_available,
            upstream_anchors=(
                "vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed",
                "vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4",
                "vllm.v1.core.kv_cache_utils._bucket_layers_by_page_size",
            ),
            version_note=(
                "Anchor = the packed planner and the bucket helper it uses. If "
                "the planner is renamed or dropped the patch logs a warning "
                "and leaves the layout upstream; if _bucket_layers_by_page_size "
                "changes shape the plan is rejected by the shared_by/total-byte "
                "check instead of silently allocating the wrong bytes."
            ),
        ),
        PatchSpec(
            name="mamba_state_unpacked",
            desc="Give a small mamba state (conv_state) its own backing buffer "
                 "instead of a window into the packed page. kv_config_unpacked "
                 "fixes the attention side; a mamba page still packs conv_state "
                 "+ ssm_state into one allocation, so on LPU the GDN cache-row "
                 "helpers are staged as the whole 147 MB page per row. Wraps "
                 "ModelRunnerFL._reshape_kv_cache_tensors and re-homes the "
                 "states that pass three gates (block <= 1/4 page, backing/"
                 "payload >= 4x, block is one packed run). -3.7% generation "
                 "time on Qwen3.6-27B TP1 for +129.4 MiB; "
                 "VLLM_FL_MAMBA_STATE_UNPACKED=0 is the packed A/B arm.",
            apply_fn=_apply_mamba_state_unpacked,
            upstream_anchors=(
                "vllm_fl.worker.model_runner.ModelRunnerFL._reshape_kv_cache_tensors",
                "vllm_fl.worker.model_runner.ModelRunnerFL._kv_cache_spec_attn_group_iterator",
            ),
            version_note=(
                "Wraps our own ModelRunnerFL, not upstream GPUModelRunner, so the "
                "blast radius is an FL refactor rather than a vllm bump. A renamed "
                "method logs a warning and leaves the layout upstream (the cost "
                "returns, nothing breaks); the [mamba-state] log line is the proof "
                "the wrapped path was actually reached."
            ),
        ),
        PatchSpec(
            name="events",
            desc="Replace torch.Event with no-op stand-in in "
                 "AsyncGPUModelRunnerOutput",
            apply_fn=_apply_events,
            upstream_anchors=(
                "torch.Event",
                "torch.accelerator.synchronize",
                "torch.accelerator.empty_cache",
            ),
            version_note=(
                 "torch.Event is a C extension type (rebind the name, do not patch " +
                 "attrs). torch.accelerator.* is newer-torch only; verify presence on " +
                 "a torch bump."
            ),
        ),
        PatchSpec(
            name="gdn",
            desc="Replace GatedDeltaNet Triton ops with pure-PyTorch fork "
                 "(Qwen3.5-27B), and rebound mamba.ops.causal_conv1d to "
                 "pure-PyTorch at the operator level (the Triton conv kernels "
                 "fail to legalize under evas-triton-opt, and 7 modules import "
                 "from that one site)",
            apply_fn=_apply_gdn,
            upstream_anchors=(
                "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core",
                "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core_decode_non_spec",
                "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._warmup_prefill_kernels",
                "vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn",
                "vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update",
            ),
            version_note=(
                 "The except-ImportError skip hides a rename as a silent no-op -- check " +
                 "for it on a bump (Qwen3.5 would fall back to broken Triton GDN). " +
                 "causal_conv1d_* is rebound at the source module, so importers " +
                 "that `from ...ops.causal_conv1d import x` BEFORE this patch runs " +
                 "still hold the Triton binding -- kda.py redirects the KDA one. " +
                 "The conv fallbacks cover the query_start_loc / has_initial_state / " +
                 "cache_indices contract only. The paged-block group " +
                 "(block_idx_* / initial_state_idx / null_block_id) and the " +
                 "spec-decode group (num_accepted_tokens / query_start_loc) are " +
                 "rejected by name with TypeError on BOTH entry points -- " +
                 "causal_conv1d_fn gets that from its signature, " +
                 "causal_conv1d_update needs an explicit guard because those " +
                 "names exist upstream-side. validate_data is accepted and " +
                 "ignored (upstream only gates asserts on it; the decode call " +
                 "passes True). So Mamba-1/2 stay LPU-blocked on purpose, and " +
                 "GDN speculative decoding raises rather than convolving the " +
                 "wrong token count."
            ),
        ),
        PatchSpec(
            name="kda",
            desc="Redirect Kimi Delta Attention Triton ops (fla.ops.kda) to "
                 "pure-PyTorch on LPU: fused_kda_gate + fused_recurrent_kda "
                 "(decode) + chunk_kda_with_fused_gate (prefill) + "
                 "rms_norm_gated (FusedRMSNormGated's forward_cuda body, "
                 "since CustomOp has no evas dispatch branch). Sibling of the "
                 "gdn patch -- different op family (per-key-dim decay vs "
                 "per-head scalar); only causal_conv1d is shared.",
            apply_fn=_apply_kda,
            upstream_anchors=(
                "vllm.model_executor.layers.fla.ops.kda.fused_kda_gate",
                "vllm.model_executor.layers.fla.ops.kda.fused_recurrent_kda",
                "vllm.model_executor.layers.fla.ops.kda.chunk_kda_with_fused_gate",
                "vllm.model_executor.layers.fla.ops.kda.rms_norm_gated",
                "vllm.model_executor.layers.fla.ops.kda.FusedRMSNormGated.forward_cuda",
                "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn.causal_conv1d_fn",
                "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn.causal_conv1d_update",
            ),
            version_note=(
                 "The decode recurrence is shared with Qwen GDN "
                 "(fused_recurrent_gated_delta_rule_fwd_kernel) and differs only "
                 "in IS_KDA, i.e. exp(g) is a per-key-dim vector. Keep the two "
                 "ports in step if that kernel is refactored. The prefill path "
                 "runs the sequential recurrence instead of the 5-stage chunked "
                 "pipeline (a blocked reorganisation of the same math, kept "
                 "upstream for Triton parallelism) -- fine for short prompts, "
                 "would need the chunked form for long ones."
            ),
        ),
        PatchSpec(
            name="vit",
            desc="Force Qwen3-VL ViT pos_embed_interpolate to pure-PyTorch "
                 "fallback",
            apply_fn=_apply_vit,
            upstream_anchors=(
                "vllm.model_executor.models.qwen3_vl.Qwen3_VisionTransformer.fast_pos_embed_interpolate",
                "torch.nn.functional.scaled_dot_product_attention",
            ),
            version_note=(
                 "Upstream picks the Triton branch because HAS_TRITON is True on LPU; " +
                 "the F.sdpa global rebind is deliberate (head_size != 128 crashes " +
                 "native)."
            ),
        ),
        PatchSpec(
            name="model_compat",
            desc="LPU model-level compatibility patches (GLM-5.2 + DSV4), "
                 "merged into model_compat.apply_lpu_model_compat_patch: "
                 "GLM MLA/Indexer class rebind (EvasGLMMLAAttention / EvasGLMIndexer factory) + GLM load_weights filter; "
                 "DSV4 attention-class swap (EvasDSV4Attention), sequential "
                 "attention execute, embed-repeat, indexer dispatch, "
                 "decoder-layer mhc-import gate",
            apply_fn=_apply_model_compat,
            upstream_anchors=(
                "vllm.model_executor.models.deepseek_v2.DeepseekV2MLAAttention",
                "vllm.model_executor.models.deepseek_v2.Indexer",
                "vllm.model_executor.models.deepseek_v2.GlmMoeDsaForCausalLM.load_weights",
                "vllm.models.deepseek_v4.nvidia.model.DeepseekV4DecoderLayer.__init__",
                "vllm.models.deepseek_v4.nvidia.model._select_dsv4_attn_cls",
                "vllm.models.deepseek_v4.nvidia.model.mhc_pre_tilelang",
                "vllm.models.deepseek_v4.nvidia.model.mhc_fused_post_pre_tilelang",
                "vllm.models.deepseek_v4.nvidia.model.mhc_post_tilelang",
                "vllm.models.deepseek_v4.nvidia.model.hc_head_fused_kernel_tilelang",
                "vllm.models.deepseek_v4.nvidia.model.DeepseekV4Model.forward",
                "vllm.models.deepseek_v4.attention.execute_in_parallel",
                "vllm.models.deepseek_v4.attention.maybe_execute_in_parallel",
                "vllm.models.deepseek_v4.attention.fused_q_kv_rmsnorm",
                "vllm.models.deepseek_v4.attention.fused_indexer_q_rope_quant",
                "vllm.model_executor.layers.fused_moe.router."
                "fused_topk_bias_router.vllm_topk_softplus_sqrt",
                "vllm.model_executor.layers.fused_moe.router."
                "fused_topk_router.vllm_topk_sigmoid",
                "vllm.v1.attention.backends.fa_utils."
                "is_flash_attn_varlen_func_available",
            ),
            version_note=(
                 "Widest upstream surface (GLM deepseek_v2 + DSV4 " +
                 "nvidia.model/attention). nvidia.model.* is per-platform and " +
                 "reshuffles across vllm releases -- expect this to need the most " +
                 "rework on a bump."
            ),
        ),
        PatchSpec(
            name="deepseek_rope",
            desc="Replace DeepseekScalingRotaryEmbedding.forward_native to "
                 "avoid LPU-unsafe torch.repeat_interleave (OOB-writes into "
                 "adjacent seq_lens/cache_seqlens on LPU; deepseek-v3.1 MLA "
                 "decode crash). Uses torch.stack+reshape (identical result).",
            apply_fn=_apply_deepseek_rope,
            upstream_anchors=(
                "vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope.DeepseekScalingRotaryEmbedding.forward_native",
            ),
            version_note=(
                 "DeepseekScalingRotaryEmbedding lives under layers/rotary_embedding/. " +
                 "DeepseekV4ScalingRotaryEmbedding overrides forward_native itself and " +
                 "is intentionally NOT reached by this patch."
            ),
        ),
        PatchSpec(
            name="deepseek_v4_weights",
            desc="Route DSV4 load_weights through EvasBackend (FP8 weights stay "
                 "FP8 single-residency: uint8 E4M3 + fp32 block scales, no BF16 "
                 "dequant) and force the MTP drafter onto the same uint8 contract",
            apply_fn=_apply_dsv4_weights,
            upstream_anchors=(
                "vllm.models.deepseek_v4.DeepseekV4ForCausalLM.__init__",
                "vllm.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
                "vllm.models.deepseek_v4.DeepSeekV4MTP.__init__",
                "vllm.models.deepseek_v4.DeepSeekV4MTP.load_weights",
            ),
            version_note=(
                 "Both classes live in the per-platform ``vllm.models.deepseek_v4`` " +
                 "package. Gate is _dsv4_support_available for that reason."
            ),
            gate=_dsv4_support_available,
        ),
        PatchSpec(
            name="dsv4_mtp",
            desc="Route the DSV4 MTP draft forward's four module-scope kernels "
                 "to LPU stand-ins (nvidia/mtp.py binds them in its own "
                 "namespace, which the model_compat rebinds of nvidia/model.py "
                 "cannot reach): fused_mtp_input_rmsnorm / "
                 "mtp_shared_head_rmsnorm -> pure-PyTorch on flag_gems.rms_norm, "
                 "mhc_post_tilelang -> flag_gems.mhc_post, "
                 "hc_head_fused_kernel_tilelang -> the model_compat adapter",
            apply_fn=_apply_dsv4_mtp,
            upstream_anchors=(
                "vllm.models.deepseek_v4.nvidia.mtp.fused_mtp_input_rmsnorm",
                "vllm.models.deepseek_v4.nvidia.mtp.mtp_shared_head_rmsnorm",
                "vllm.models.deepseek_v4.nvidia.mtp.mhc_post_tilelang",
                "vllm.models.deepseek_v4.nvidia.mtp.hc_head_fused_kernel_tilelang",
            ),
            version_note=(
                 "nvidia/mtp.py is the per-platform drafter module, reached only "
                 "when speculative decoding instantiates the draft (the e2e script "
                 "sets that up under VLLM_FL_DSV4_MTP=1, which no framework code "
                 "reads). Its module-scope imports are the whole surface this patch "
                 "touches. The two MHC kernels also have torch.ops.vllm wrappers in "
                 "layers/mhc.py that no namespace rebind reaches -- the DSV4 path "
                 "does not go through them."
            ),
            gate=_dsv4_support_available,
        ),
        PatchSpec(
            name="dsv4_dspark",
            desc="Route the DSV4 DSpark drafter's two module-scope tilelang "
                 "kernels to LPU stand-ins (vllm >= 0.25 nvidia/dspark.py binds "
                 "them per module and calls them directly, bypassing the "
                 "layers/mhc.py CustomOp fallbacks): mhc_post_tilelang -> "
                 "flag_gems.mhc_post, hc_head_fused_kernel_tilelang -> the "
                 "model_compat adapter",
            apply_fn=_apply_dsv4_dspark,
            upstream_anchors=(
                "vllm.models.deepseek_v4.nvidia.dspark.mhc_post_tilelang",
                "vllm.models.deepseek_v4.nvidia.dspark.hc_head_fused_kernel_tilelang",
            ),
            version_note=(
                 "nvidia/dspark.py only exists from vllm 0.25; on 0.24 the "
                 "drafter is nvidia/mtp.py and this patch is a no-op. Both "
                 "stand-ins are the same ones dsv4_mtp uses, so the numerics are "
                 "already validated by the 0.24 MTP path."
            ),
            min_vllm=(0, 25),
            gate=_dsv4_support_available,
        ),
        PatchSpec(
            name="fp8_quant",
            desc="Monkey-patch the vLLM fp8 activation-quant entry points to the "
                 "LPU pure-PyTorch quantizers (uint8 E4M3 + fp32 scale) so the "
                 "upstream MoE A-quant flow (moe_kernel_quantize_input) yields "
                 "our uint8 contract, consumed by the triton_experts_fl fork: "
                 "scaled_fp8_quant -> per-token, and the fused_moe.utils copy of "
                 "per_token_group_quant_fp8 -> per-(token, 128) block",
            apply_fn=_apply_fp8_quant,
            upstream_anchors=(
                "vllm._custom_ops.scaled_fp8_quant",
                "vllm.model_executor.layers.fused_moe.utils.per_token_group_quant_fp8",
            ),
            version_note=(
                 "Upstream bodies call torch.ops._C.*_scaled_fp8_quant and the " +
                 "CUDA/Triton per_token_group_quant_fp8. The LPU stand-ins must " +
                 "keep emitting uint8 E4M3 + fp32 scale or the triton_experts_fl " +
                 "dtype contract breaks. The group-quant rebind is scoped to the " +
                 "fused_moe.utils module attribute only -- the other importers " +
                 "need the native fp8 contract."
            ),
        ),
        PatchSpec(
            name="triton_experts_fl",
            desc="Fork TritonExpertsFL.apply to accept uint8 E4M3 activations "
                 "(patched scaled_fp8_quant emits uint8 + fp32 scale, no native "
                 "fp8 dtype on LPU).  Verbatim upstream body + uint8 in the "
                 "dtype assert / compute_type chain; non-uint8 inputs unaffected.",
            apply_fn=_apply_triton_experts_fl,
            upstream_anchors=(
                # The class the patch actually rebinds lives in our fork module.
                "vllm_fl.ops.fused_moe.fused_moe_utils.TritonExpertsFL.apply",
                # Upstream body the fork was copied from: drift here silently
                # diverges the LPU copy from upstream behavior.
                "vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts.apply",
            ),
            version_note=(
                 "Verbatim upstream body + uint8 in the dtype assert/compute_type " +
                 "chain -- re-sync the copy on any fused_moe experts refactor."
            ),
        ),
        PatchSpec(
            name="gemm",
            desc="Route F.linear / torch.matmul / torch.mm through torch.einsum "
                 "(LPU 2D GEMM incorrect; applies even with FlagGems since "
                 "evas Triton mm kernel is also imprecise)",
            apply_fn=_apply_gemm,
            upstream_anchors=(
                "torch.nn.functional.linear",
                "torch.matmul",
                "torch.mm",
                "torch.bmm",
            ),
            version_note=(
                 "Wraps torch-level names (not aten ops) so every call path is " +
                 "covered. Bare `a @ b` resolves via __matmul__ and bypasses this -- " +
                 "see the GEMM correctness-layering memory."
            ),
            # Gate removed: FlagGems evas mm Triton kernel (cos=0.996) and
            # native LPU addmm are both numerically insufficient for 28-layer
            # accumulation. The einsum path is the only correct GEMM on LPU.
        ),
        PatchSpec(
            name="fpad",
            desc="Replace F.pad with pure-PyTorch constant pad (LPU pad kernel "
                 "corrupts output to zeros on derived tensors, even plain rand)",
            apply_fn=_apply_fpad,
            upstream_anchors=(
                "torch.nn.functional.pad",
            ),
            version_note=(
                 "Constant mode only. No LPU-safe upstream path until the firmware pad " +
                 "bug is fixed; the wrap is the whole workaround."
            ),
        ),
        PatchSpec(
            name="scatter",
            desc="Transparently cast scatter_ int64 indices to int32 on LPU "
                 "(LPU firmware rejects int64: 'scatter expected dtype int32 for index')",
            apply_fn=_apply_scatter,
            upstream_anchors=(
                "torch.Tensor.scatter_",
            ),
            version_note=(
                 "Method wrap (not aten) captures every caller. The int64->int32 cast " +
                 "is permanent while the firmware rejects int64 indices."
            ),
        ),
        PatchSpec(
            name="randint",
            desc="Wrap torch.randint to fall back to CPU on LPU "
                 "(torch_lpu lacks randint aten op)",
            apply_fn=_apply_randint,
            upstream_anchors=(
                "torch.randint",
            ),
            version_note=(
                 "Intercepts only when device is omitted and the default device is " +
                 "LPU; explicit-device callers are intentionally unaffected."
            ),
        ),
        PatchSpec(
            name="rotary_operands",
            desc="Wrap the rotary dispatch op with the two properties the evas "
                 "kernel needs and upstream forward_oot does not supply: an "
                 "int32 position buffer (the kernel narrows an int64 load at "
                 "the wrong address) and a rope operand whose double-die "
                 "descriptor matches its 3-D shape (a tail view of a 2-D "
                 "buffer unzips one rank short and the launch is rejected). "
                 "Keeps vllm_fl/ops/rotary_embedding.py byte-identical to "
                 "upstream/main",
            apply_fn=_apply_rotary,
            upstream_anchors=(
                "vllm_fl.ops.rotary_embedding.RotaryEmbeddingFL.forward_oot",
                "vllm_fl.ops.rotary_embedding._rotary_embedding",
            ),
            version_note=(
                 "Anchors on our own module, not vllm: the wrapper rebinds the "
                 "module-global CachedOp that forward_oot calls. Re-verify if "
                 "forward_oot starts resolving the dispatch op locally or "
                 "stops passing inplace=True."
            ),
        ),
        PatchSpec(
            name="rms_norm_operands",
            desc="Wrap the rms_norm dispatch op so both operands are "
                 "acceptable to the dual-die row kernel. (1) Both operands "
                 "carry a descriptor faithful to their own shape: a "
                 "multimodal model's residual stream is a row slice of the "
                 "persistent inputs_embeds buffer, so its descriptor is the "
                 "parent's and unzip hands the fused residual norm the parent "
                 "halves; the launch is rejected and the op latches away. "
                 "(2) A rank>2 call (Qwen3 q/k norms) is flattened to the "
                 "kernel's [M, N] row shape and the result reshaped back; "
                 "rank>2 otherwise raises and latches the op. Repairing at the "
                 "dispatch op covers the kernel and the pure arm and keeps "
                 "vllm_fl/ops/layernorm.py byte-identical to upstream/main",
            apply_fn=_apply_rms_norm_operands,
            upstream_anchors=(
                "vllm_fl.ops.layernorm.RMSNormFL.forward_oot",
                "vllm_fl.ops.layernorm._rms_norm",
            ),
            version_note=(
                 "Anchors on our own module, not vllm: the wrapper rebinds the "
                 "module-global CachedOp that forward_oot calls. Re-verify if "
                 "forward_oot starts resolving the dispatch op locally, if "
                 "the platform stops inheriting the parent descriptor (then "
                 "the repair is a no-op and can retire), or if the row kernel "
                 "starts accepting rank>2 directly (then so can the flatten)."
            ),
        ),
        PatchSpec(
            name="nonblocking_copy",
            desc="Force blocking CPU->LPU Tensor.copy_ (drop non_blocking): "
                 "H2D copies from temporary pinned sources corrupt data "
                 "(shifted/garbled device buffers) and wedge the copy engine "
                 "(QP dump, sq pidb N cidb 0) on LPU. TP4 e2e triage 09-16",
            apply_fn=_apply_nonblocking_copy,
            upstream_anchors=(
                "torch.Tensor.copy_",
                "vllm.v1.utils.CpuGpuBuffer.copy_to_gpu",
            ),
            version_note=(
                 "Tensor.copy_ is a hot C method -- keep the wrapper branch-cheap. " +
                 "CpuGpuBuffer lives in vllm.v1.utils and has moved before."
            ),
        ),
        PatchSpec(
            name="flashevo_ccl",
            desc="Route CommunicatorFL TP collectives through flashevo XCCL "
                 "on LPU (torch.distributed zccl crashes in the multiproc "
                 "worker env: LPUEvent.record -> evError 2181). Keeps "
                 "vllm_fl/distributed/communicator.py byte-identical to "
                 "upstream/main",
            apply_fn=_apply_flashevo_ccl,
            upstream_anchors=(
                "vllm_fl.distributed.communicator.CommunicatorFL.all_reduce",
                "vllm_fl.distributed.communicator.CommunicatorFL.reduce_scatter",
                "vllm_fl.distributed.communicator.CommunicatorFL.reduce_scatterv",
                "vllm_fl.distributed.communicator.CommunicatorFL.all_gather",
                "vllm_fl.distributed.communicator.CommunicatorFL.all_gatherv",
            ),
            version_note=(
                 "Wraps our CommunicatorFL (not a vllm symbol). The LPU bodies " +
                 "mirror the upstream shape preambles (movedim/chunk_size in " +
                 "reduce_scatter[v], sizes handling in all_gatherv) -- re-verify " +
                 "those on a communicator bump."
            ),
        ),
        PatchSpec(
            name="deepseek_v4_swa_metadata",
            desc="Install pure-PyTorch metadata-construction shims for the "
                 "upstream DeepseekSparseSWAMetadataBuilder on LPU: "
                 "get_mla_metadata (FlashMLA C-ext) -> evas pure-PyTorch; "
                 "_compute_swa_indices_and_lens_kernel / "
                 "_compute_prefill_metadata_kernel (Triton, evas-triton-opt "
                 "compile crash) -> decode_swa_indices/decode_swa_lens/"
                 "prefill_gather_lens; redirect the sparse-MLA warmup entry in "
                 "kernel_warmup (its from-import binds the name at module top).",
            apply_fn=_apply_deepseek_v4_swa_metadata,
            upstream_anchors=(
                "vllm.v1.attention.backends.mla.sparse_swa._compute_swa_indices_and_lens_kernel",
                "vllm.v1.attention.backends.mla.sparse_swa._compute_prefill_metadata_kernel",
                "vllm.v1.attention.backends.mla.sparse_swa.get_mla_metadata",
                "vllm.v1.attention.backends.mla.indexer._build_prefill_chunk_metadata_kernel",
                "vllm.v1.attention.ops.flashmla.get_mla_metadata",
                "vllm.model_executor.warmup.kernel_warmup."
                "deepseek_v4_sparse_mla_attention_warmup",
            ),
            version_note=(
                 "sparse_swa/indexer hold already-bound `from ... import` names, so " +
                 "BOTH the origin module and the importer alias must be rebound. The " +
                 "qsl/t2r recovery path also assumes CommonAttentionMetadata " +
                 "query_start_loc semantics (see verify_swa_indices_lens_lpu.py " +
                 "fixture contract)."
            ),
            gate=_dsv4_support_available,
        ),
        PatchSpec(
            name="minimax_m3",
            desc="Install the pure-torch MiniMax-M3 twins: MiniMAXGemmaRMSNorm "
                 "(upstream calls flashinfer and dies in torch._C._cuda_init), the "
                 "fused qknorm+partial-rope+paged-insert op (torch.ops._C, not "
                 "built for LPU), the lightning indexer and the block-sparse GQA "
                 "attend (both Triton, tl.dot / evas-triton-opt rejects "
                 "index_topk), and the MoE SwiGLU-OAI (ROCm Triton kernel). The "
                 "attend twin is dense-masked O(S^2): a correctness port, not a "
                 "performance one.",
            apply_fn=_apply_minimax_m3,
            upstream_anchors=(
                "vllm.models.minimax_m3.nvidia.model.MiniMAXGemmaRMSNorm.forward",
                "vllm._custom_ops.fused_minimax_m3_qknorm_rope_kv_insert",
                "vllm.models.minimax_m3.common.indexer.select_indexer_impl_cls",
                "vllm.models.minimax_m3.common.sparse_attention.select_main_impl_cls",
                "vllm.models.minimax_m3.amd.ops.swiglu_oai.swiglu_oai_split",
            ),
            version_note=(
                 "Pins one class method, one _custom_ops wrapper, one Triton "
                 "kernel, and the two impl-selection hooks (checked with "
                 "hasattr, since rebinding a module attribute is otherwise "
                 "silent)."
            ),
            gate=_minimax_m3_available,
        ),
        PatchSpec(
            name="dsv4_compressor",
            desc="Install the DSV4 compressor / C128A-metadata / compressed-"
                 "slot-mapping pure-PyTorch stand-ins in "
                 "patches/ops_fallback/dsv4_compressor.py. The upstream Triton "
                 "kernels here are evas-triton-opt compile dead-ends "
                 "(scf.for iter_arg type mismatch, tt.reduce) and the "
                 "compressed slot_mapping kernel silently never writes on LPU. "
                 "The attention math itself comes from EvasDSV4Attention, "
                 "swapped in by model_compat. Applied unconditionally: every "
                 "rebinding is inert unless the DSV4 attention path runs.",
            apply_fn=_apply_dsv4_compressor,
            upstream_anchors=(
                "vllm.models.deepseek_v4.sparse_mla.build_c128a_topk_metadata",
                "vllm.models.deepseek_v4.common.ops.save_partial_states.save_partial_states",
                "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache.compress_norm_rope_store_triton",
                "vllm.models.deepseek_v4.compressor.save_partial_states",
                "vllm.models.deepseek_v4.compressor.compress_norm_rope_store_triton",
                "vllm.v1.attention.backends.mla.compressor_utils.get_compressed_slot_mapping",
            ),
            version_note=(
                 "Compressor binds its two ops by `from ... import` at import time, so " +
                 "the ops modules AND the compressor aliases must both be rebound. Set " +
                 "VLLM_FL_STRICT_UPSTREAM_REBIND=1 to turn a leak-sweep miss into an " +
                 "error."
            ),
            gate=_dsv4_support_available,
        ),
        PatchSpec(
            name="lmhead_fp8",
            desc="Opt-in e4m3 W8A8 language-model head (VLLM_FL_LMHEAD_FP8): "
                 "rebind Qwen3.5 compute_logits so the head's quant_method "
                 "serves the dual-die fp8 block kernel instead of the 512-row "
                 "chunked bf16 einsum (the 2.37 GiB bf16 head misses the 1.5 GiB "
                 "transpose fast path and runs at 2.9 GB/s). Setting DUMP_LOGITS "
                 "alone also applies this patch with the projection left on bf16, "
                 "so the reference arm can dump the same call indices.",
            apply_fn=_apply_lmhead_fp8,
            gate=_lmhead_fp8_requested,
            upstream_anchors=(
                "vllm.model_executor.models.qwen3_5.Qwen3_5ForCausalLMBase.compute_logits",
                "vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits",
            ),
            version_note=(
                "The seam is LogitsProcessor receiving the head module as an "
                "argument and calling its quant_method.apply; if compute_logits "
                "starts reading self.lm_head directly, or _get_logits stops using "
                "quant_method.apply, the swap silently stops taking effect."
            ),
        ),
    ]

    if disable_patches:
        specs = [s for s in specs if s.name not in disable_patches]

    logger.info("Registered evas patches: %s", [s.name for s in specs])
    return specs


# ---- Internal apply helpers (deferred imports) ----


def _apply_device_properties() -> bool | None:
    """4 platform patches in one (device_properties, events, scatter, bind_kv_cache)."""
    from .platform_compat import apply_lpu_platform_compat_patch

    return apply_lpu_platform_compat_patch()


def _apply_lmhead_fp8() -> bool | None:
    from .lmhead_fp8 import apply_lpu_lmhead_fp8_patch

    return apply_lpu_lmhead_fp8_patch()


def _apply_sampler() -> bool | None:
    from .ops_fallback.sampler import apply_lpu_sampler_patch

    return apply_lpu_sampler_patch()


def _apply_rejection_sampler() -> bool | None:
    from .ops_fallback.rejection_sampler import apply_lpu_rejection_sampler_patch

    return apply_lpu_rejection_sampler_patch()


def _apply_spec_decode_proposer() -> bool | None:
    from .ops_fallback.spec_decode_proposer import (
        apply_lpu_spec_decode_proposer_patch,
    )

    return apply_lpu_spec_decode_proposer_patch()


def _apply_async_spec_decode_bookkeeping() -> bool | None:
    from .ops_fallback.async_spec_decode_bookkeeping import (
        apply_lpu_async_spec_decode_bookkeeping_patch,
    )

    return apply_lpu_async_spec_decode_bookkeeping_patch()


def _apply_block_table() -> bool | None:
    from .ops_fallback.block_table import apply_lpu_block_table_patch

    return apply_lpu_block_table_patch()


def _apply_vocab_embedding() -> bool | None:
    from .ops_fallback.vocab_embedding import apply_lpu_vocab_embedding_patch

    return apply_lpu_vocab_embedding_patch()


def _apply_events() -> bool | None:
    """Same merged function as _apply_device_properties; idempotent."""
    from .platform_compat import apply_lpu_platform_compat_patch

    return apply_lpu_platform_compat_patch()


def _apply_gdn() -> bool | None:
    from .ops_fallback.gdn import apply_lpu_gdn_patch

    return apply_lpu_gdn_patch()


def _apply_kda() -> bool | None:
    from .ops_fallback.kda import apply_lpu_kda_patch

    return apply_lpu_kda_patch()


def _apply_vit() -> bool | None:
    from .ops_fallback.vit import apply_lpu_vit_patch

    return apply_lpu_vit_patch()


def _apply_bind_kv_cache() -> bool | None:
    """Same merged function as _apply_device_properties; idempotent."""
    from .platform_compat import apply_lpu_platform_compat_patch

    return apply_lpu_platform_compat_patch()


def _apply_kv_config_unpacked() -> bool | None:
    from .kv_config import apply_kv_config_unpacked

    return apply_kv_config_unpacked()


def _apply_mamba_state_unpacked() -> bool | None:
    from .mamba_state import apply_mamba_state_unpacked

    return apply_mamba_state_unpacked()


def _apply_gemm() -> bool | None:
    from .ops_fallback.gemm import apply_lpu_gemm_patch

    return apply_lpu_gemm_patch()


def _apply_fpad() -> bool | None:
    from .ops_fallback.pad import apply_lpu_pad_patch

    return apply_lpu_pad_patch()


def _apply_scatter() -> bool | None:
    """Same merged function as _apply_device_properties; idempotent."""
    from .platform_compat import apply_lpu_platform_compat_patch

    return apply_lpu_platform_compat_patch()


def _apply_randint() -> bool | None:
    from .ops_fallback.randint import apply_lpu_randint_patch

    return apply_lpu_randint_patch()


def _apply_rotary() -> bool | None:
    from .ops_fallback.rotary import apply_lpu_rotary_patch

    return apply_lpu_rotary_patch()


def _apply_rms_norm_operands() -> bool | None:
    from .ops_fallback.rms_norm_operands import apply_lpu_rms_norm_operands_patch

    return apply_lpu_rms_norm_operands_patch()


def _apply_nonblocking_copy() -> bool | None:
    from .ops_fallback.nonblocking_copy import (
        apply_lpu_nonblocking_copy_patch,
        apply_lpu_verified_copy_to_gpu_patch,
    )

    copy_status = apply_lpu_nonblocking_copy_patch()
    to_gpu_status = apply_lpu_verified_copy_to_gpu_patch()
    # Two impls, one spec: the spec is declined when either half is, so the
    # registry cannot report a half-installed patch as applied.
    if copy_status is False or to_gpu_status is False:
        return False
    return None


def _apply_flashevo_ccl() -> bool | None:
    from .flashevo_ccl import apply_lpu_flashevo_ccl_patch

    return apply_lpu_flashevo_ccl_patch()


def _apply_deepseek_v4_swa_metadata() -> bool | None:
    from .ops_fallback.dsv4_swa_metadata import apply_lpu_dsv4_swa_metadata_patch

    return apply_lpu_dsv4_swa_metadata_patch()


def _apply_dsv4_compressor() -> bool | None:
    from .ops_fallback.dsv4_compressor import apply_lpu_dsv4_compressor_patch

    return apply_lpu_dsv4_compressor_patch()


def _apply_model_compat() -> bool | None:
    """Apply the merged LPU model-compat patch (GLM-5.2 MLA/Indexer/MoE +
    DSV4 decoder-layer/mega-MoE/MoE).  Idempotent inside the callee."""
    from .model_compat import apply_lpu_model_compat_patch

    return apply_lpu_model_compat_patch()


def _apply_deepseek_rope() -> bool | None:
    from .ops_fallback.deepseek_rope import apply_lpu_deepseek_rope_patch

    return apply_lpu_deepseek_rope_patch()


def _apply_dsv4_weights() -> bool | None:
    from .ops_fallback.dsv4_weights import apply_lpu_dsv4_weights_patch

    return apply_lpu_dsv4_weights_patch()


def _apply_dsv4_mtp() -> bool | None:
    from .ops_fallback.dsv4_mtp import apply_lpu_dsv4_mtp_patch

    return apply_lpu_dsv4_mtp_patch()


def _apply_dsv4_dspark() -> bool | None:
    from .ops_fallback.dsv4_dspark import apply_lpu_dsv4_dspark_patch

    return apply_lpu_dsv4_dspark_patch()


def _apply_fp8_quant() -> bool | None:
    from .ops_fallback.fp8_quant import apply_lpu_fp8_quant_patch

    return apply_lpu_fp8_quant_patch()


def _apply_minimax_m3() -> bool | None:
    from .ops_fallback.minimax_m3 import apply_lpu_minimax_m3_patch

    return apply_lpu_minimax_m3_patch()


def _apply_triton_experts_fl() -> bool | None:
    from .ops_fallback.triton_experts_fl import apply_lpu_triton_experts_fl_patch

    return apply_lpu_triton_experts_fl_patch()
