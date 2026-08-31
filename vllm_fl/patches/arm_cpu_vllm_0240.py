"""ARM CPU compatibility layer for the stock vLLM 0.24.0 package.

Keep model/runtime compatibility fixes in vllm-plugin-FL rather than carrying
a permanently modified vLLM source tree. The integration intentionally fails
closed on a different vLLM version because it patches private 0.24.0 APIs.
"""

from __future__ import annotations

import gc
import platform
from importlib import metadata

import torch

_INSTALLED = False


def _require_vllm_0240() -> None:
    installed = metadata.version("vllm")
    base = installed.split("+", 1)[0]
    if base != "0.24.0":
        raise RuntimeError(
            "The FL ARM CPU compatibility layer requires vLLM 0.24.0; "
            f"found {installed}"
        )


def _install_packed_w4a8() -> None:
    from compressed_tensors.config import CompressionFormat

    from vllm.logger import init_logger
    from vllm.model_executor.kernels.linear import (
        MPLinearLayerConfig,
        choose_mp_linear_kernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as config_module,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a8_int as scheme_module,
    )
    from vllm.model_executor.parameter import (
        BasevLLMParameter,
        ChannelQuantScaleParameter,
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    scheme_cls = scheme_module.CompressedTensorsW4A8Int
    config_cls = config_module.CompressedTensorsConfig
    if getattr(scheme_cls, "_vllm_fl_packed_w4a8", False):
        return

    logger = init_logger(__name__)
    original_init = scheme_cls.__init__
    original_create_weights = scheme_cls.create_weights
    original_get_scheme = config_cls._get_scheme_from_parts

    def scheme_init(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        is_static_input_scheme: bool = False,
        input_symmetric: bool = True,
        packed: bool = False,
    ) -> None:
        original_init(
            self,
            strategy=strategy,
            num_bits=num_bits,
            group_size=group_size,
            is_static_input_scheme=is_static_input_scheme,
            input_symmetric=input_symmetric,
        )
        self._vllm_fl_checkpoint_packed = packed
        self._vllm_fl_pack_factor = 32 // num_bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        if not getattr(self, "_vllm_fl_checkpoint_packed", False):
            return original_create_weights(
                self,
                layer,
                output_size,
                input_size,
                output_partition_sizes,
                input_size_per_partition,
                params_dtype,
                weight_loader,
                **kwargs,
            )

        output_size_per_partition = sum(output_partition_sizes)
        row_parallel = input_size != input_size_per_partition
        effective_group_size = (
            input_size_per_partition
            if self.group_size == -1 and row_parallel
            else input_size
            if self.group_size == -1
            else self.group_size
        )
        if input_size_per_partition % effective_group_size:
            raise ValueError(
                f"input partition {input_size_per_partition} is not divisible "
                f"by W4 group size {effective_group_size}"
            )

        kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=effective_group_size,
            zero_points=False,
            has_g_idx=False,
        )
        kernel_type = choose_mp_linear_kernel(kernel_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info(
                "Using %s for packed CompressedTensorsW4A8Int",
                kernel_type.__name__,
            )
            self._kernel_backends_being_used.add(kernel_type.__name__)

        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self._vllm_fl_pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self._vllm_fl_pack_factor,
            packed_dim=1,
        )
        layer.register_parameter("weight_packed", weight)

        scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                input_size_per_partition // effective_group_size,
                dtype=params_dtype,
            ),
        }
        if self.group_size == -1 and row_parallel:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **scale_args)
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **scale_args
            )
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter(
            "weight_shape",
            BasevLLMParameter(
                data=torch.empty(2, dtype=torch.int64),
                weight_loader=weight_loader,
            ),
        )
        self.kernel = kernel_type(
            kernel_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name=None,
            w_gidx_param_name=None,
        )

    def get_scheme_from_parts(
        self,
        weight_quant,
        input_quant,
        output_quant=None,
        format: str | None = None,
        layer_name: str | None = None,
    ):
        resolved_format = format if format is not None else self.quant_format
        if (
            resolved_format == CompressionFormat.pack_quantized.value
            and self._is_dynamic_token_w4a8_int(weight_quant, input_quant)
        ):
            return scheme_cls(
                num_bits=weight_quant.num_bits,
                strategy=weight_quant.strategy,
                group_size=weight_quant.group_size,
                is_static_input_scheme=False,
                input_symmetric=input_quant.symmetric,
                packed=True,
            )
        return original_get_scheme(
            self,
            weight_quant,
            input_quant,
            output_quant=output_quant,
            format=format,
            layer_name=layer_name,
        )

    scheme_cls.__init__ = scheme_init
    scheme_cls.create_weights = create_weights
    config_cls._get_scheme_from_parts = get_scheme_from_parts
    scheme_cls._vllm_fl_packed_w4a8 = True


def _install_cpu_gemm_guard() -> None:
    from vllm.model_executor.layers import utils as layer_utils

    original = layer_utils.dispatch_cpu_unquantized_gemm
    if getattr(original, "_vllm_fl_ndim_guard", False):
        return

    def guarded(layer: torch.nn.Module, remove_weight: bool) -> None:
        weight = getattr(layer, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim != 2:
            return
        return original(layer, remove_weight)

    guarded._vllm_fl_ndim_guard = True
    layer_utils.dispatch_cpu_unquantized_gemm = guarded


def _install_text_only_vision_guard() -> None:
    from contextvars import ContextVar

    from vllm.model_executor.models import qwen3_5 as qwen

    if getattr(qwen, "_vllm_fl_text_only_vision", False):
        return

    text_only_build = ContextVar("vllm_fl_qwen_text_only_build", default=False)
    original_vision_init = qwen.Qwen3_VisionTransformer.__init__

    def vision_init(self, *args, **kwargs) -> None:
        if text_only_build.get():
            kwargs["quant_config"] = None
        original_vision_init(self, *args, **kwargs)

    qwen.Qwen3_VisionTransformer.__init__ = vision_init

    for model_cls in (
        qwen.Qwen3_5ForConditionalGeneration,
        qwen.Qwen3_5MoeForConditionalGeneration,
    ):
        original_model_init = model_cls.__init__

        def model_init(
            self,
            *,
            vllm_config,
            prefix: str = "model",
            _original=original_model_init,
        ) -> None:
            language_only = bool(
                vllm_config.model_config.multimodal_config.language_model_only
            )
            token = text_only_build.set(language_only)
            try:
                _original(self, vllm_config=vllm_config, prefix=prefix)
            finally:
                text_only_build.reset(token)

        model_cls.__init__ = model_init

    qwen._vllm_fl_text_only_vision = True


def _install_cpu_attention_block_constraint() -> None:
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    if hasattr(CPUAttentionBackend, "get_supported_kernel_block_sizes"):
        return

    CPUAttentionBackend.get_supported_kernel_block_sizes = staticmethod(
        lambda: [MultipleOf(32)]
    )


def _install_cpu_cleanup_guard() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._cleanup_profiling_kv_cache
    if getattr(original, "_vllm_fl_cpu_guard", False):
        return

    def cleanup(self) -> None:
        if self.device.type != "cpu":
            return original(self)
        if hasattr(self, "kv_caches") and self.kv_caches:
            for index in range(len(self.kv_caches)):
                self.kv_caches[index] = None
            self.kv_caches.clear()
        if hasattr(self, "cross_layers_kv_cache"):
            self.cross_layers_kv_cache = None
            self.cross_layers_attn_backend = None
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None
        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                kv_cache = layer.kv_cache
                layer.kv_cache = (
                    torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
                )
            if hasattr(layer, "impl"):
                if hasattr(layer.impl, "_k_scale_cache"):
                    layer.impl._k_scale_cache = None
                if hasattr(layer.impl, "_v_scale_cache"):
                    layer.impl._v_scale_cache = None
        gc.collect()

    cleanup._vllm_fl_cpu_guard = True
    GPUModelRunner._cleanup_profiling_kv_cache = cleanup


def _zero_cpu_attention_blocks(self, block_ids: list[int]) -> None:
    """Clear recycled attention pages for vLLM's hybrid CPU cache.

    vLLM 0.24.0 asks the runner to clear newly allocated pages whenever a
    model has Mamba/GDN layers.  Its CPU runner discards that request, based
    on the assumption that invalid attention positions are always masked.
    With a hybrid page size larger than the CPU attention kernel block, stale
    K/V data can still affect a recycled physical page.  The generic zeroer
    already knows the hybrid page mapping and intentionally skips Mamba state;
    GDN handles fresh state through ``has_initial_state`` during prefill.
    """
    zeroer = getattr(self, "_kv_block_zeroer", None)
    if zeroer is not None:
        zeroer.zero_block_ids(block_ids)


def _install_cpu_hybrid_kv_zeroing() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import (
        SingleTypeKVCacheManager,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    from vllm.v1.worker.utils import KVBlockZeroer

    # Keep this compatibility in the plugin: a stock vLLM 0.24 zeroer only
    # records FullAttentionSpec storage and launches a Triton pointer kernel,
    # neither of which is correct for DFlash2's recycled host sliding-window
    # cache.  Retain checked tensor slices and clear them directly on the CPU.
    original_zeroer_init = KVBlockZeroer.__init__
    if not getattr(original_zeroer_init, "_vllm_fl_cpu_tensor_slices", False):
        original_zero_block_ids = KVBlockZeroer.zero_block_ids

        def zeroer_init(
            self,
            device,
            pin_memory,
            attn_groups_iter,
            kernel_block_sizes,
            cache_dtype,
            static_forward_context,
            runner_only_attn_layers=None,
        ) -> None:
            groups = list(attn_groups_iter)
            original_zeroer_init(
                self,
                device=device,
                pin_memory=pin_memory,
                attn_groups_iter=groups,
                kernel_block_sizes=kernel_block_sizes,
                cache_dtype=cache_dtype,
                static_forward_context=static_forward_context,
                runner_only_attn_layers=runner_only_attn_layers,
            )
            if device.type != "cpu":
                return

            excluded = runner_only_attn_layers or set()
            seen_ptrs: set[int] = set()
            slices: list[tuple[torch.Tensor, int, int]] = []
            for group in groups:
                spec = group.kv_cache_spec
                if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                    continue
                group_id = group.kv_cache_group_id
                if group_id >= len(kernel_block_sizes):
                    continue
                kernel_block_size = kernel_block_sizes[group_id]
                if spec.block_size % kernel_block_size:
                    raise ValueError(
                        f"KV block size {spec.block_size} is not divisible by "
                        f"kernel block size {kernel_block_size}"
                    )
                ratio = spec.block_size // kernel_block_size
                block_dim = group.backend.get_kv_cache_block_dim(
                    kernel_block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                    cache_dtype_str=cache_dtype,
                )
                for layer_name in group.layer_names:
                    if layer_name in excluded:
                        continue
                    cache = static_forward_context[layer_name].kv_cache
                    if not isinstance(cache, torch.Tensor):
                        continue
                    data_ptr = cache.data_ptr()
                    if data_ptr in seen_ptrs:
                        continue
                    seen_ptrs.add(data_ptr)
                    slices.append((cache, block_dim, ratio))
            self._vllm_fl_cpu_slices = slices

        def zero_block_ids(self, block_ids: list[int]) -> None:
            slices = getattr(self, "_vllm_fl_cpu_slices", None)
            if self.device.type != "cpu" or slices is None:
                return original_zero_block_ids(self, block_ids)
            for cache, block_dim, ratio in slices:
                for block_id in dict.fromkeys(block_ids):
                    start = block_id * ratio
                    if block_id < 0 or start + ratio > cache.shape[block_dim]:
                        raise IndexError(
                            f"KV block {block_id} is outside cache shape "
                            f"{tuple(cache.shape)} at block_dim={block_dim}"
                        )
                    cache.narrow(block_dim, start, ratio).zero_()

        zeroer_init._vllm_fl_cpu_tensor_slices = True
        zero_block_ids._vllm_fl_cpu_tensor_slices = True
        KVBlockZeroer.__init__ = zeroer_init
        KVBlockZeroer.zero_block_ids = zero_block_ids

    # Stock vLLM 0.24 does not report pages allocated by SlidingWindowManager
    # to the runner zeroer.  Wrap both allocation paths and add only IDs the
    # underlying implementation did not already report, so this is safe with a
    # source tree that has subsequently taken the same upstream fix.
    original_allocate = SingleTypeKVCacheManager.allocate_new_blocks
    if not getattr(original_allocate, "_vllm_fl_reports_sliding_pages", False):

        def allocate_new_blocks(
            self,
            request_id: str,
            num_tokens: int,
            num_tokens_main_model: int,
        ):
            report_start = len(self.new_block_ids)
            blocks = original_allocate(
                self,
                request_id,
                num_tokens,
                num_tokens_main_model,
            )
            if isinstance(self.kv_cache_spec, SlidingWindowSpec):
                reported = set(self.new_block_ids[report_start:])
                self.new_block_ids.extend(
                    block.block_id for block in blocks if block.block_id not in reported
                )
            return blocks

        allocate_new_blocks._vllm_fl_reports_sliding_pages = True
        SingleTypeKVCacheManager.allocate_new_blocks = allocate_new_blocks

    original_allocate_external = (
        SingleTypeKVCacheManager.allocate_external_computed_blocks
    )
    if not getattr(
        original_allocate_external,
        "_vllm_fl_reports_sliding_pages",
        False,
    ):

        def allocate_external_computed_blocks(
            self,
            request_id: str,
            num_local_computed_tokens: int,
            num_external_computed_tokens: int,
        ) -> None:
            block_start = len(self.req_to_blocks[request_id])
            report_start = len(self.new_block_ids)
            original_allocate_external(
                self,
                request_id,
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
            if isinstance(self.kv_cache_spec, SlidingWindowSpec):
                blocks = self.req_to_blocks[request_id][block_start:]
                reported = set(self.new_block_ids[report_start:])
                self.new_block_ids.extend(
                    block.block_id for block in blocks if block.block_id not in reported
                )

        allocate_external_computed_blocks._vllm_fl_reports_sliding_pages = True
        SingleTypeKVCacheManager.allocate_external_computed_blocks = (
            allocate_external_computed_blocks
        )

    current = CPUModelRunner._zero_block_ids
    if getattr(current, "_vllm_fl_hybrid_kv_zeroing", False):
        return
    _zero_cpu_attention_blocks._vllm_fl_hybrid_kv_zeroing = True
    CPUModelRunner._zero_block_ids = _zero_cpu_attention_blocks


def _cpu_attention_scheduler_rebuild_policy(
    scheduler: object,
    *,
    expected_isa: int | None,
    requested_split_kv: bool,
    apple_arm: bool,
) -> bool | None:
    """Return the split-KV setting for a required metadata rebuild.

    The vLLM 0.24 C++ metadata header starts at byte 64. Its first int32 is
    the dispatch ISA and its fourth int32 is ``reduction_split_num``. On
    Apple ARM, the split-KV reduction barrier can execute inside a serialized
    nested OpenMP region: the only active thread then waits forever for the
    logical worker count. Rebuilding without KV splitting preserves attention
    semantics and avoids that barrier.

    ``None`` means the existing scheduler is safe to reuse.
    """
    if expected_isa is None:
        return None

    scheduler_isa = None
    reduction_split_num = 0
    if (
        isinstance(scheduler, torch.Tensor)
        and scheduler.is_contiguous()
        and scheduler.numel() * scheduler.element_size() >= 80
    ):
        header = scheduler.view(torch.uint8)[64:80].clone().view(torch.int32)
        scheduler_isa = int(header[0])
        reduction_split_num = int(header[3])

    if apple_arm and reduction_split_num > 0:
        return False
    if scheduler_isa != expected_isa:
        return False if apple_arm else requested_split_kv
    return None


def _install_cpu_attention_scheduler_isa_guard() -> None:
    """Keep CPU attention scheduler metadata safe for the active runtime.

    vLLM 0.24 can reuse metadata built for another attention subgroup in a
    hybrid GDN/full-attention model.  The metadata embeds the C++ dispatch ISA;
    on Apple ARM we observed VEC16 metadata paired with a NEON/BFMMLA-packed KV
    cache. Apple ARM split-KV metadata is also rebuilt without splitting to
    avoid the vLLM reduction barrier deadlock under the Triton-CPU runtime.
    """
    from vllm import _custom_ops as ops, envs
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackendImpl

    original = CPUAttentionBackendImpl.forward
    if getattr(original, "_vllm_fl_scheduler_isa_guard", False):
        return

    isa_codes = {
        "amx": 0,
        "vec": 1,
        "vec16": 2,
        "neon": 3,
        "vxe": 4,
        "rvv": 5,
        "vsx": 6,
    }

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is not None:
            scheduler = getattr(attn_metadata, "scheduler_metadata", None)
            expected_isa = isa_codes.get(self.isa)
            requested_split_kv = envs.VLLM_CPU_ATTN_SPLIT_KV
            rebuild_split_kv = _cpu_attention_scheduler_rebuild_policy(
                scheduler,
                expected_isa=expected_isa,
                requested_split_kv=requested_split_kv,
                apple_arm=(
                    platform.system() == "Darwin"
                    and platform.machine().lower() in {"aarch64", "arm64"}
                ),
            )
            if rebuild_split_kv is not None:
                attn_metadata.scheduler_metadata = ops.cpu_attn_get_scheduler_metadata(
                    num_reqs=attn_metadata.query_start_loc.numel() - 1,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_size,
                    seq_lens=attn_metadata.seq_lens,
                    dtype=query.dtype,
                    query_start_loc=attn_metadata.query_start_loc,
                    causal=attn_metadata.causal,
                    sliding_window_size=self.sliding_window,
                    isa=self.isa,
                    enable_kv_split=rebuild_split_kv,
                    dynamic_causal=attn_metadata.dynamic_causal,
                )
        return original(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    forward._vllm_fl_scheduler_isa_guard = True
    CPUAttentionBackendImpl.forward = forward


def _install_cpu_spec_decode_kernels() -> None:
    """Use vLLM's native CPU helpers for speculative control-plane ops.

    vLLM 0.24 skips all of its C++ fallbacks when Triton-CPU is importable.
    That is appropriate for model kernels, but it also sends small EAGLE/MTP
    metadata and rejection-sampling operations through Triton.  In particular,
    ``sample_recovered_tokens_kernel`` produces multi-megabyte LLVM IR on ARM
    and can take minutes to compile.  Select only the official CPU wrappers
    here; quantized model and GDN kernels remain on Triton-CPU/FlagGems.
    """
    import vllm.utils.cpu_triton_utils as cpu_tl
    import vllm.v1.sample.rejection_sampler as rejection_sampler
    import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer
    import vllm.v1.spec_decode.utils as spec_utils
    import vllm.v1.worker.block_table as block_table

    recovered_kernel = cpu_tl.sample_recovered_tokens_kernel
    recovered_impl = recovered_kernel.func
    if not getattr(recovered_impl, "_vllm_fl_fp64_gumbel_abi", False):

        def recovered_compat(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            inv_q,
            vocab_size,
            block_size=None,
            *,
            NO_DRAFT_PROBS=False,
            USE_FP64_GUMBEL=False,
        ):
            if USE_FP64_GUMBEL:
                raise NotImplementedError(
                    "FP64 Gumbel speculative sampling is unsupported by "
                    "the vLLM 0.24 CPU C++ kernel"
                )
            return recovered_impl(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                inv_q,
                vocab_size,
                block_size,
                NO_DRAFT_PROBS=NO_DRAFT_PROBS,
            )

        recovered_compat._vllm_fl_fp64_gumbel_abi = True
        recovered_kernel.func = recovered_compat

    # vLLM's 0.24 CPU C++ expand helper reads every input as int64. Sampling
    # also uses this operation for floating-point temperature and top-p rows,
    # so routing those tensors through the C++ helper truncates their values
    # and leaves the floating-point output storage untouched. Expand the tiny
    # per-request control vectors with PyTorch on the host instead.
    expand_kernel = cpu_tl.expand_kernel
    expand_impl = expand_kernel.func
    if not getattr(expand_impl, "_vllm_fl_dtype_safe", False):

        def expand_compat(
            output,
            input_val,
            cu_num_tokens,
            replace_from,
            replace_to,
            MAX_NUM_TOKENS=None,
        ):
            del MAX_NUM_TOKENS
            values = input_val
            if replace_from != replace_to:
                values = torch.where(
                    values == replace_from,
                    values.new_tensor(replace_to),
                    values,
                )
            starts = torch.cat((cu_num_tokens.new_zeros(1), cu_num_tokens[:-1]))
            counts = (cu_num_tokens - starts).to(torch.int64)
            expanded = torch.repeat_interleave(values, counts)
            if expanded.numel() != output.numel():
                raise ValueError(
                    "expanded sampling metadata has "
                    f"{expanded.numel()} tokens, expected {output.numel()}"
                )
            output.copy_(expanded)

        expand_compat._vllm_fl_dtype_safe = True
        expand_kernel.func = expand_compat

    block_table._compute_slot_mapping_kernel = cpu_tl.compute_slot_mapping_kernel
    llm_base_proposer.eagle_prepare_inputs_padded_kernel = (
        cpu_tl.eagle_prepare_inputs_padded_kernel
    )
    llm_base_proposer.eagle_prepare_next_token_padded_kernel = (
        cpu_tl.eagle_prepare_next_token_padded_kernel
    )
    llm_base_proposer.copy_and_expand_eagle_inputs_kernel = (
        cpu_tl.copy_and_expand_eagle_inputs_kernel
    )
    spec_utils.eagle_step_slot_mapping_metadata_kernel = (
        cpu_tl.eagle_step_slot_mapping_metadata_kernel
    )
    rejection_sampler.rejection_greedy_sample_kernel = (
        cpu_tl.rejection_greedy_sample_kernel
    )
    rejection_sampler.rejection_random_sample_kernel = (
        cpu_tl.rejection_random_sample_kernel
    )
    rejection_sampler.expand_kernel = expand_kernel
    rejection_sampler.sample_recovered_tokens_kernel = recovered_kernel


def _install_cpu_dflash2_proposer() -> None:
    """Select the plugin DFlash2 proposer in vLLM's standalone CPU runner."""
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    original_init = CPUModelRunner.__init__
    if getattr(original_init, "_vllm_fl_dflash2_proposer", False):
        return

    def init_with_dflash2(self, vllm_config, device) -> None:
        original_init(self, vllm_config, device)
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or not speculative_config.use_dflash():
            return
        draft_architectures = (
            getattr(
                speculative_config.draft_model_config.hf_config,
                "architectures",
                (),
            )
            or ()
        )
        if "DFlash2DraftModel" not in draft_architectures:
            return

        from vllm_fl.spec_decode.dflash2 import DFlash2Proposer

        self.drafter = DFlash2Proposer(vllm_config, device, self)
        self.use_aux_hidden_state_outputs = True

    init_with_dflash2._vllm_fl_dflash2_proposer = True
    CPUModelRunner.__init__ = init_with_dflash2


def _apply_top_k_top_p_small_k_cpu(
    logits: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    *,
    max_fast_k: int = 256,
) -> torch.Tensor | None:
    """Filter a small top-k before applying top-p on CPU.

    vLLM's general Triton kernel scans the full vocabulary several times.  For
    the common DFlash2 evaluation setting (top-k=20), reducing to K candidates
    first is mathematically identical and substantially cheaper on CPU.
    ``None`` asks the caller to use vLLM's general implementation.
    """
    if (
        logits.device.type != "cpu"
        or logits.ndim != 2
        or top_k is None
        or top_p is None
        or top_k.numel() != logits.shape[0]
        or top_p.numel() != logits.shape[0]
    ):
        return None
    smallest_k = int(top_k.min())
    largest_k = int(top_k.max())
    if smallest_k <= 0 or largest_k > min(max_fast_k, logits.shape[1]):
        return None

    top_k_long = top_k.to(torch.long)
    probe_k = min(largest_k + 1, logits.shape[1])
    descending_logits, descending_ids = logits.topk(probe_k, dim=-1, sorted=True)
    has_outside_candidate = top_k_long < logits.shape[1]
    last_inside = descending_logits.gather(1, (top_k_long - 1).unsqueeze(1))
    first_outside = descending_logits.gather(
        1, top_k_long.clamp_max(probe_k - 1).unsqueeze(1)
    )
    # vLLM's Triton kernel has an explicit, deterministic rule for duplicates
    # crossing the K boundary. torch.topk may choose another equal-logit token,
    # which can change seeded output, so retain the general path for that rare
    # case instead of silently changing sampling semantics.
    unsafe_rows = has_outside_candidate & (
        last_inside.squeeze(1) == first_outside.squeeze(1)
    )

    candidate_logits = descending_logits[:, :largest_k].flip(1)
    candidate_ids = descending_ids[:, :largest_k].flip(1)

    # With per-row K, the final K entries in ascending order are active.
    ranks = torch.arange(largest_k, device=logits.device).unsqueeze(0)
    first_active = largest_k - top_k_long
    inactive = ranks < first_active.unsqueeze(1)
    candidate_logits.masked_fill_(inactive, -float("inf"))

    cumulative = candidate_logits.softmax(dim=-1)
    cumulative.cumsum_(dim=-1)
    top_p_mask = cumulative <= 1 - top_p.unsqueeze(1)
    # Preserve at least the maximum-logit candidate, matching vLLM.
    top_p_mask[:, -1] = False
    if largest_k > 1:
        # An equal-logit group is order-independent unless the top-p boundary
        # splits that group. In that one case defer to vLLM's deterministic
        # duplicate rule.
        pair_ranks = torch.arange(largest_k - 1, device=logits.device).unsqueeze(0)
        active_pair = pair_ranks >= first_active.unsqueeze(1)
        split_equal_pair = (
            (candidate_logits[:, 1:] == candidate_logits[:, :-1])
            & (top_p_mask[:, 1:] != top_p_mask[:, :-1])
            & active_pair
        )
        unsafe_rows |= split_equal_pair.any(dim=1)
    candidate_logits.masked_fill_(top_p_mask, -float("inf"))

    output = torch.full_like(logits, -float("inf")).scatter_(
        1, candidate_ids, candidate_logits
    )
    if torch.any(unsafe_rows):
        # Quantized logits can tie exactly. vLLM's combined Triton top-k/top-p
        # kernel has a specialized token-ID rule for a boundary tie, including
        # the p<1 case. Re-run only those rows through the original kernel;
        # ordinary rows still avoid its full-vocabulary scan.
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        output[unsafe_rows] = apply_top_k_top_p_triton(
            logits[unsafe_rows].clone(),
            top_k[unsafe_rows],
            top_p[unsafe_rows],
        )
    return output


def _install_cpu_small_top_k_top_p() -> None:
    """Route small-K CPU sampling around the full-vocabulary Triton scan."""
    import vllm.v1.sample.ops.topk_topp_sampler as sampler_ops

    original = sampler_ops.apply_top_k_top_p
    if getattr(original, "_vllm_fl_small_k_cpu", False):
        return

    def apply_top_k_top_p(logits, top_k, top_p):
        optimized = _apply_top_k_top_p_small_k_cpu(logits, top_k, top_p)
        if optimized is not None:
            return optimized
        return original(logits, top_k, top_p)

    apply_top_k_top_p._vllm_fl_small_k_cpu = True
    sampler_ops.apply_top_k_top_p = apply_top_k_top_p

    # rejection_sampler imported the function directly, so update its binding
    # as well as the sampler module's global used by TopKTopPSampler methods.
    import vllm.v1.sample.rejection_sampler as rejection_sampler

    rejection_sampler.apply_top_k_top_p = apply_top_k_top_p


def _install_cpu_spec_decode_compat() -> None:
    """Bridge a legacy CPU sampler to the current MTP call ABI.

    The generic rejection sampler passes the optional synthetic-decoding
    tensors used by the generic sampler. The legacy CPU implementation still
    implements the ordinary greedy/random algorithm and has the older
    signature.  Ignore those tensors for the ordinary path and fail closed if
    synthetic rejection sampling is explicitly requested.
    """
    import vllm.utils.cpu_triton_utils as cpu_tl

    greedy_kernel = cpu_tl.rejection_greedy_sample_kernel
    greedy_impl = greedy_kernel.func
    if getattr(greedy_impl, "_vllm_fl_spec_decode_abi", False):
        return

    random_kernel = cpu_tl.rejection_random_sample_kernel
    random_impl = random_kernel.func

    def greedy_compat(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
        uniform_probs=None,
        synthetic_conditional_rates=None,
        *,
        SYNTHETIC_MODE=False,
    ):
        del uniform_probs, synthetic_conditional_rates
        if SYNTHETIC_MODE:
            raise NotImplementedError(
                "synthetic speculative rejection sampling is unsupported on "
                "the legacy ARM CPU path"
            )
        return greedy_impl(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )

    def random_compat(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates=None,
        *,
        NO_DRAFT_PROBS=False,
        SYNTHETIC_MODE=False,
    ):
        del synthetic_conditional_rates
        if SYNTHETIC_MODE:
            raise NotImplementedError(
                "synthetic speculative rejection sampling is unsupported on "
                "the legacy ARM CPU path"
            )
        return random_impl(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            NO_DRAFT_PROBS=NO_DRAFT_PROBS,
        )

    greedy_compat._vllm_fl_spec_decode_abi = True
    random_compat._vllm_fl_spec_decode_abi = True
    greedy_kernel.func = greedy_compat
    random_kernel.func = random_compat


def install_arm_cpu_vllm_0240_compat() -> bool:
    """Install all Python-only compatibility hooks exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return False
    _require_vllm_0240()
    _install_packed_w4a8()
    _install_cpu_gemm_guard()
    _install_text_only_vision_guard()
    _install_cpu_attention_block_constraint()
    _install_cpu_cleanup_guard()
    _install_cpu_hybrid_kv_zeroing()
    _install_cpu_attention_scheduler_isa_guard()
    _install_cpu_spec_decode_kernels()
    _install_cpu_small_top_k_top_p()
    _install_cpu_dflash2_proposer()
    _INSTALLED = True
    return True


__all__ = ["install_arm_cpu_vllm_0240_compat"]
