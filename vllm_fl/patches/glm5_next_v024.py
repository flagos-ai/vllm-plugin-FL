# SPDX-License-Identifier: Apache-2.0
"""Register the plugin-owned GLM5-Next implementation on vLLM 0.24."""

import os
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

from vllm_fl.activation import (
    ActivationPlan,
    MoEDispatchDefaults,
    PendingPatch,
    bind_patches,
    merge_per_op_defaults,
    register_plan_provider,
)
from vllm_fl.dispatch.policy import SelectionPolicy
from vllm_fl.kernels.glm5_next.provider import (
    get_glm5_provider,
    use_nvidia_reference,
)
from vllm_fl.runtime.model_policy import (
    ModelPolicyError,
    ModelPolicyFactory,
    RuntimePlan,
    register_model_policy_factory,
)

logger = init_logger(__name__)

_CAUSAL_ARCH = "Glm5NextForCausalLM"
_CONDITIONAL_ARCH = "Glm5NextForConditionalGeneration"
_MHC_CUDA_MAX_TOKENS = 0

# Empty-build vLLM wheels do not provide ``vllm._C``/``_moe_C``.  GLM5's
# portable unquantized-MoE path still needs the dispatch-owned alignment and
# Triton-kernel entry points, but older launch recipes often whitelist only
# ``grouped_topk,moe_sum``.  Keep this list model-local: enabling these ops for
# every model would change the platform-wide CUDA policy, while enabling it
# when the GLM provider has selected FlagGems is required before the first
# profile-run/KV-cache probe.
_GLM5_PORTABLE_MOE_OPS = (
    "moe_align_block_size",
    "invoke_fused_moe_triton_kernel",
)


def glm5_portable_moe_defaults() -> MoEDispatchDefaults:
    """FlagOS dispatch defaults for the GLM5 portable MoE path.

    Returned to the activation plan instead of mutating ``os.environ``: a plain
    plugin import must never rewrite the process environment, and a non-GLM
    model must never inherit GLM's dispatch policy.  The validated
    NVIDIA/DeepGEMM provider needs no override at all.

    ``required_impls`` makes an explicit user order that cannot satisfy the
    portable path (for example ``vendor.cuda`` on an empty build) abort at
    startup instead of being silently rewritten.
    """
    if use_nvidia_reference():
        return MoEDispatchDefaults()

    portable_ops = tuple(_GLM5_PORTABLE_MOE_OPS)
    portable_order = tuple(("flagos", "reference") for _ in portable_ops)
    return MoEDispatchDefaults(
        whitelist_ops=portable_ops,
        per_op_order=tuple(
            (op_name, order) for op_name, order in zip(portable_ops, portable_order)
        ),
        required_impls=tuple(
            (op_name, order) for op_name, order in zip(portable_ops, portable_order)
        ),
    )


def is_vllm_024() -> bool:
    """Return whether the installed vLLM belongs to the 0.24 ABI line.

    Keep this probe local to the model adapter: the current plugin main branch
    has no generic ``patches._version`` module, and importing a historical
    helper would make an otherwise valid 0.24 install fail at plugin startup.
    """
    try:
        release = version("vllm").split("+", 1)[0].split(".")
    except PackageNotFoundError:
        return False
    return len(release) >= 2 and release[:2] == ["0", "24"]


def _is_missing_cache_op(exc: AttributeError, op_name: str) -> bool:
    """Return whether vLLM failed because ``_C_cache_ops`` lacks an op."""
    message = str(exc)
    return "_C_cache_ops" in message and op_name in message


def _has_vllm_cache_op(op_name: str) -> bool:
    """Probe the extension ABI without invoking a device kernel."""
    try:
        getattr(torch.ops._C_cache_ops, op_name)
    except AttributeError:
        return False
    return True


def _concat_and_cache_mla_bf16_fallback(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    """Correctness fallback for a vendor MLA backend without its cache op.

    This is deliberately limited to BF16 cache semantics.  Quantized cache
    formats require the vendor or FlagGems implementation because ``scale``
    participates in the stored representation.
    """
    del scale
    if kv_cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError(
            "GLM5-Next portable concat_and_cache_mla only supports BF16 KV "
            f"cache, got {kv_cache_dtype!r}"
        )
    source = kv_c if k_pe.shape[-1] == 0 else torch.cat((kv_c, k_pe), dim=-1)
    slots = slot_mapping.flatten().to(torch.int64)
    valid = slots >= 0
    cache_flat = kv_cache.view(-1, kv_cache.shape[-1])
    cache_flat[slots[valid]] = source[valid]


def _mla_boundary_patches(custom_ops, fingerprint) -> list[PendingPatch]:
    """Keep vendor sparse MLA while filling its optional vLLM ABI edges.

    Vendor backends remain responsible for the actual sparse attention.  The
    wrappers below only cover GLM5-Next's zero-width RoPE query and the common
    BF16 cache-write ABI that some OOT vLLM builds do not provide.
    """
    if custom_ops is None:
        from vllm import _custom_ops as custom_ops

    patches = []

    def stage(attr, replacement):
        patches.append(
            PendingPatch(
                target=f"{custom_ops.__name__}.{attr}",
                owner=custom_ops,
                attr=attr,
                replacement=replacement,
                pristine=getattr(custom_ops, attr),
                fingerprint=fingerprint,
                phase="worker",
            )
        )

    strict_flaggems = get_glm5_provider() == "flaggems"

    concat_mla_q = custom_ops.concat_mla_q
    if not getattr(concat_mla_q, "_glm5_next_nope_fix", False):

        @wraps(concat_mla_q)
        def concat_mla_q_nope(ql_nope, q_pe, q_out):
            if strict_flaggems:
                raise RuntimeError(
                    "VLLM_FL_GLM5_PROVIDER=flaggems was requested, but the "
                    "stock/vendor sparse-MLA path called "
                    "vllm._custom_ops.concat_mla_q. The worker did not select "
                    "FlagGemsSparseMLABackend; check the worker environment, "
                    "Plugin FL patch/version, and vendor backend overrides."
                )
            if q_pe.shape[-1] == 0:
                q_out.copy_(ql_nope)
                return None
            return concat_mla_q(ql_nope, q_pe, q_out)

        concat_mla_q_nope._glm5_next_nope_fix = True
        concat_mla_q_nope._glm5_next_original = concat_mla_q
        stage("concat_mla_q", concat_mla_q_nope)

    concat_and_cache_mla = custom_ops.concat_and_cache_mla
    if not _has_vllm_cache_op("concat_and_cache_mla") and not getattr(
        concat_and_cache_mla, "_glm5_next_vendor_fallback", False
    ):
        native_cache_op_available = True
        flaggems_cache_writer = None
        flaggems_cache_writer_loaded = False

        @wraps(concat_and_cache_mla)
        def concat_and_cache_mla_vendor_first(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            scale,
        ):
            nonlocal native_cache_op_available
            nonlocal flaggems_cache_writer
            nonlocal flaggems_cache_writer_loaded

            if native_cache_op_available:
                try:
                    return concat_and_cache_mla(
                        kv_c,
                        k_pe,
                        kv_cache,
                        slot_mapping,
                        kv_cache_dtype,
                        scale,
                    )
                except AttributeError as exc:
                    if not _is_missing_cache_op(exc, "concat_and_cache_mla"):
                        raise
                    native_cache_op_available = False
                    logger.warning(
                        "Vendor vLLM has no _C_cache_ops.concat_and_cache_mla; "
                        "trying FlagGems and then the BF16 correctness fallback"
                    )

            if not flaggems_cache_writer_loaded:
                flaggems_cache_writer_loaded = True
                try:
                    from flag_gems.fused.concat_and_cache_mla import (
                        concat_and_cache_mla as flaggems_cache_writer_impl,
                    )

                    flaggems_cache_writer = flaggems_cache_writer_impl
                except (ImportError, AttributeError, OSError):
                    flaggems_cache_writer = None

            if flaggems_cache_writer is not None:
                # The writer is selected once when it is first loaded.  It may
                # already have written to the cache before raising, so an error
                # is propagated instead of retried against the Torch fallback
                # (which would double-write or corrupt the KV cache).
                flag_cache_dtype = (
                    "auto" if kv_cache_dtype == "bfloat16" else kv_cache_dtype
                )
                return flaggems_cache_writer(
                    kv_c,
                    k_pe,
                    kv_cache,
                    slot_mapping,
                    kv_cache_dtype=flag_cache_dtype,
                    scale=scale,
                )

            return _concat_and_cache_mla_bf16_fallback(
                kv_c,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                scale,
            )

        concat_and_cache_mla_vendor_first._glm5_next_vendor_fallback = True
        concat_and_cache_mla_vendor_first._glm5_next_original = concat_and_cache_mla
        stage("concat_and_cache_mla", concat_and_cache_mla_vendor_first)

    return patches


def _install_mla_boundary_compat_ops(custom_ops: ModuleType | None = None) -> bool:
    if custom_ops is None:
        from vllm import _custom_ops as custom_ops
    return bool(bind_patches(_mla_boundary_patches(custom_ops, _glm5_fingerprint())))


def _silu_and_mul_with_clamp_oot(self, x: torch.Tensor) -> torch.Tensor:
    """Use FlagGems for GLM's bounded SwiGLU when its exact op is present."""
    if self.alpha != 1.0 or self.beta != 0.0:
        return self.forward_native(x)
    dim = x.shape[-1] // 2
    try:
        from flag_gems.fused.silu_and_mul_with_clamp import (
            silu_and_mul_with_clamp,
        )

        return silu_and_mul_with_clamp(x[..., :dim], x[..., dim:], self.swiglu_limit)
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        return self.forward_native(x)


def _mhc_rms_norm(
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    """Apply the RMSNorm fused by the CUDA mHC reference kernels."""
    if norm_weight is None:
        return layer_input
    layer_input_fp32 = layer_input.float()
    inv_rms = torch.rsqrt(
        layer_input_fp32.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    return (layer_input_fp32 * inv_rms * norm_weight.float()).to(layer_input.dtype)


def _mhc_pre_oot_with_norm(
    self,
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=0.0,
):
    try:
        from flag_gems.fused.mhc import mhc_pre

        post_mix, comb_mix, layer_input = mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        post_mix, comb_mix, layer_input = self.forward_native(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )
    return (
        post_mix,
        comb_mix,
        _mhc_rms_norm(layer_input, norm_weight, norm_eps),
    )


def _mhc_post_oot_flaggems(self, x, residual, post_layer_mix, comb_res_mix):
    try:
        from flag_gems.fused.mhc import mhc_post

        return mhc_post(x, residual, post_layer_mix, comb_res_mix)
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        return self.forward_native(x, residual, post_layer_mix, comb_res_mix)


def _mhc_fused_post_pre_oot_with_norm(
    self,
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    try:
        from flag_gems.fused.mhc import mhc_post, mhc_pre

        residual_cur = mhc_post(x, residual, post_layer_mix, comb_res_mix)
        post_mix, comb_mix, layer_input = mhc_pre(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
    except (ImportError, OSError, NotImplementedError, RuntimeError):
        residual_cur, post_mix, comb_mix, layer_input = self.forward_native(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )
    return (
        residual_cur,
        post_mix,
        comb_mix,
        _mhc_rms_norm(layer_input, norm_weight, norm_eps),
    )


def _mhc_pre_oot_bounded_cuda(self, residual, *args, **kwargs):
    if residual.shape[0] <= _MHC_CUDA_MAX_TOKENS:
        return self.forward_cuda(residual, *args, **kwargs)
    return _mhc_pre_oot_with_norm(self, residual, *args, **kwargs)


def _mhc_post_oot_bounded_cuda(self, x, residual, *args, **kwargs):
    if x.shape[0] <= _MHC_CUDA_MAX_TOKENS:
        return self.forward_cuda(x, residual, *args, **kwargs)
    return self.forward_native(x, residual, *args, **kwargs)


def _mhc_fused_post_pre_oot_bounded_cuda(self, x, residual, *args, **kwargs):
    if x.shape[0] <= _MHC_CUDA_MAX_TOKENS:
        return self.forward_cuda(x, residual, *args, **kwargs)
    return _mhc_fused_post_pre_oot_with_norm(self, x, residual, *args, **kwargs)


class Glm5NextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Preserve VLM checkpoints and default bare text configs to CausalLM."""

    def is_deepseek_mla(self) -> bool:
        # vLLM v0.24 predates the glm5_next(_text) model-type entries in its
        # MLA allowlist. Keep the same capability check used by newer vLLM so
        # head_size and KV-cache/backend selection use kv_lora_rank + RoPE dim.
        return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

    def get_architectures(self) -> list[str]:
        architectures = super().get_architectures()
        if not architectures:
            architectures = [_CAUSAL_ARCH]
        self.hf_config.architectures = architectures.copy()
        return architectures


class Glm5NextForCausalLMConfig(HybridAttentionMambaModelConfig):
    """Compose v0.24 hybrid-cache and DSA validation."""

    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        validate_glm5_config(vllm_config)
        HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)

        # GLM5-Next's TP16 mHC/all-reduce path is not safe to capture with
        # vLLM 0.24's breakable CUDA-graph allocator when the generic O2/O3
        # ``fuse_allreduce_rms`` pass is enabled.  The failure happens during
        # the first graph capture (CachingHostAllocator use-count assertion),
        # before the server can become ready.  The reference GLM5 deployment
        # uses FULL_AND_PIECEWISE with this pass disabled.  Keep eager mode's
        # existing behavior intact while making graph mode safe by default;
        # an explicit ``--enforce-eager`` still leaves the fusion enabled.
        if not getattr(vllm_config.model_config, "enforce_eager", False):
            pass_config = vllm_config.compilation_config.pass_config
            if pass_config.fuse_allreduce_rms is not False:
                pass_config.fuse_allreduce_rms = False
                logger.info(
                    "GLM5-Next: disabled fuse_allreduce_rms for CUDA-graph "
                    "capture safety on the vLLM 0.24 ABI"
                )

        text_config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "bfloat16":
            cache_config.cache_dtype = "auto"
        if getattr(text_config, "index_kpool_compress", False):
            kpool = int(getattr(text_config, "index_kpool", 1))
            from vllm_fl.models.glm5_next_kpool import glm5_indexer_page_alignment

            page = glm5_indexer_page_alignment(current_platform.get_device_capability())
            required = kpool * page
            if cache_config.block_size % required:
                aligned = (
                    (cache_config.block_size + required - 1) // required
                ) * required
                logger.info(
                    "GLM5-Next kpool changes KV block_size from %d to %d "
                    "for a %d-entry compressed page",
                    cache_config.block_size,
                    aligned,
                    page,
                )
                cache_config.block_size = aligned


# Pristine (unpatched) attribute values, captured at module import -- i.e.
# before any activation runs -- so a patch installed by another owner between
# import and activation is detected as a conflict instead of being adopted as
# the original implementation.
_BASELINES: dict[str, object] = {}


def _capture_baselines() -> None:
    try:
        from vllm.model_executor.layers.mhc import (
            MHCFusedPostPreOp,
            MHCPostOp,
            MHCPreOp,
        )

        for cls in (MHCPreOp, MHCPostOp, MHCFusedPostPreOp):
            _BASELINES[_attr_target(cls, "forward_oot")] = getattr(cls, "forward_oot")
    except Exception:  # pragma: no cover - vLLM layout differences
        logger.debug("Could not capture mHC baselines", exc_info=True)
    try:
        from vllm.model_executor.layers.activation import SiluAndMulWithClamp

        _BASELINES[_attr_target(SiluAndMulWithClamp, "forward_oot")] = getattr(
            SiluAndMulWithClamp, "forward_oot"
        )
    except Exception:  # pragma: no cover
        logger.debug("Could not capture SiluAndMulWithClamp baseline", exc_info=True)


def _attr_target(owner, attr: str) -> str:
    return f"{owner.__module__}.{owner.__name__}.{attr}"


_capture_baselines()


def _pending_attr(
    owner,
    attr: str,
    value,
    fingerprint: str,
    expected_params: tuple[str, ...],
) -> PendingPatch:
    target = _attr_target(owner, attr)
    pristine = _BASELINES.get(target)
    if pristine is None:
        pristine = getattr(owner, attr)
        logger.warning(
            "No import-time baseline for %s; capturing at activation time",
            target,
        )
    return PendingPatch(
        target=target,
        owner=owner,
        attr=attr,
        replacement=value,
        fingerprint=fingerprint,
        pristine=pristine,
        expected_params=expected_params,
    )


def _glm5_fingerprint() -> str:
    try:
        release = version("vllm").split("+", 1)[0]
    except PackageNotFoundError:  # pragma: no cover - vLLM must be installed
        release = "unknown"
    return f"glm5_next_v024@vllm{release}:provider={get_glm5_provider()}"


def _is_glm5_model(vllm_config) -> bool:
    """Return whether ``vllm_config`` describes a GLM5-Next checkpoint."""
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    hf_config = getattr(model_config, "hf_config", None)
    candidates = (
        getattr(model_config, "hf_text_config", None),
        hf_config,
        model_config,
    )
    for cfg in candidates:
        if getattr(cfg, "model_type", None) in ("glm5_next", "glm5_next_text"):
            return True
    architectures = (
        getattr(model_config, "architectures", None)
        or getattr(hf_config, "architectures", None)
        or ()
    )
    return any(arch in (_CAUSAL_ARCH, _CONDITIONAL_ARCH) for arch in architectures)


def validate_glm5_config(vllm_config) -> None:
    """Reject modes whose loading/state contracts are not implemented on 0.24.

    Run both before hybrid config adaptation and after vLLM resolves the final
    configuration. Constructors also use this guard for direct model callers.
    """
    # Model selection has already matched GLM here; registration must never
    # validate a model-specific environment variable.
    get_glm5_provider()
    parallel = getattr(vllm_config, "parallel_config", None)
    if getattr(parallel, "pipeline_parallel_size", 1) != 1:
        raise ModelPolicyError(
            "GLM5-Next on vLLM 0.24 requires pipeline_parallel_size=1: "
            "mHC state transfer between pipeline stages is not implemented"
        )
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise ModelPolicyError(
            "GLM5-Next on vLLM 0.24 does not support speculative decoding: "
            "KDA rollback and KPool verify grouping are not implemented"
        )
    if getattr(parallel, "enable_eplb", False):
        raise ModelPolicyError(
            "GLM5-Next on vLLM 0.24 does not support enable_eplb: "
            "model-level expert remapping is not implemented"
        )
    model = getattr(vllm_config, "model_config", None)
    configs = (
        getattr(model, "hf_config", None),
        getattr(model, "hf_text_config", None),
    )
    if (
        getattr(model, "quantization", None) is not None
        or getattr(vllm_config, "quant_config", None) is not None
        or any(getattr(cfg, "quantization_config", None) for cfg in configs)
    ):
        raise ModelPolicyError(
            "GLM5-Next on vLLM 0.24 supports unquantized checkpoints only; "
            "FP8/mixed-precision attention projection loading is not implemented"
        )


def _glm5_attention_override(use_mla: bool, use_sparse: bool) -> str | None:
    """Attention-backend override contributed by an active GLM plan.

    Returns ``None`` when the generic dispatch should decide, so a non-GLM MLA
    model is never routed through the GLM portable backend.
    """
    if not use_mla:
        return None

    provider = get_glm5_provider()
    auto_portable = (
        provider == "auto"
        and current_platform.is_cuda()
        and getattr(current_platform, "vendor_name", None) == "nvidia"
        and not use_nvidia_reference()
    )
    if provider != "flaggems" and not auto_portable:
        return None

    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    flaggems_backend = FlagGemsBackend()
    if not flaggems_backend.is_available():
        if provider == "flaggems":
            raise RuntimeError("VLLM_FL_GLM5_PROVIDER=flaggems requires FlagGems")
        raise RuntimeError(
            "GLM5 auto provider selected a portable MLA backend because the "
            "NVIDIA ABI/DeepGEMM is unavailable, but FlagGems is not installed"
        )
    backend_path = (
        "vllm_fl.dispatch.backends.flaggems.impl.mla_sparse.FlagGemsSparseMLABackend"
        if use_sparse
        else "vllm_fl.dispatch.backends.flaggems.impl.mla.MLAFLBackend"
    )
    logger.info_once(
        "GLM5 plan selected attention backend: %s", backend_path, scope="local"
    )
    return backend_path


def _mhc_patches(fingerprint: str) -> list[PendingPatch]:
    from vllm.model_executor.layers.mhc import (
        MHCFusedPostPreOp,
        MHCPostOp,
        MHCPreOp,
    )

    if use_nvidia_reference():
        global _MHC_CUDA_MAX_TOKENS
        _MHC_CUDA_MAX_TOKENS = int(
            os.environ.get("VLLM_FL_GLM5_MHC_CUDA_MAX_TOKENS", "0")
        )
        if _MHC_CUDA_MAX_TOKENS > 0:
            specs = (
                (MHCPreOp, _mhc_pre_oot_bounded_cuda, ("self", "residual")),
                (MHCPostOp, _mhc_post_oot_bounded_cuda, ("self", "x", "residual")),
                (
                    MHCFusedPostPreOp,
                    _mhc_fused_post_pre_oot_bounded_cuda,
                    ("self", "x", "residual"),
                ),
            )
            logger.info(
                "Bound GLM5-Next mHC OOT dispatch to CUDA/TileLang for <=%d "
                "tokens and portable reference fallback above it",
                _MHC_CUDA_MAX_TOKENS,
            )
        else:
            specs = tuple(
                (op, op.forward_cuda, ("self",))
                for op in (MHCPreOp, MHCPostOp, MHCFusedPostPreOp)
            )
            logger.info(
                "Bound GLM5-Next mHC OOT dispatch to NVIDIA CUDA/TileLang kernels"
            )
    else:
        from vllm.model_executor.layers.activation import SiluAndMulWithClamp

        specs = (
            (
                MHCPreOp,
                _mhc_pre_oot_with_norm,
                ("self", "residual", "fn", "hc_scale", "hc_base", "rms_eps"),
            ),
            (
                MHCPostOp,
                _mhc_post_oot_flaggems,
                ("self", "x", "residual", "post_layer_mix", "comb_res_mix"),
            ),
            (
                MHCFusedPostPreOp,
                _mhc_fused_post_pre_oot_with_norm,
                ("self", "x", "residual", "post_layer_mix", "comb_res_mix"),
            ),
            (SiluAndMulWithClamp, _silu_and_mul_with_clamp_oot, ("self", "x")),
        )
        logger.info(
            "Prepared GLM5-Next portable mHC OOT fallback with RMSNorm and "
            "FlagGems bounded-SwiGLU dispatch"
        )
    return [
        _pending_attr(owner, "forward_oot", value, fingerprint, params)
        for owner, value, params in specs
    ]


def _apply_glm5_activation() -> None:
    """Model-scoped side effects, run before the GLM model is constructed.

    Everything here used to run at plugin registration for every model in the
    process.  Binding it to the GLM activation plan means a non-GLM model keeps
    the generic vLLM classes and dispatch untouched.  All patches are preflighted
    and applied as one group, including the MLA boundary wrappers, so a
    failure rolls back every side effect of this activation.
    """
    fingerprint = _glm5_fingerprint()

    patches: list[PendingPatch] = []
    # vLLM dispatches every CustomOp through ``forward_oot`` when an OOT
    # platform plugin is active.  Its default OOT implementation delegates to
    # ``forward_native``, which is not semantically interchangeable for mHC:
    # MHCPreOp and MHCFusedPostPreOp accept the fused RMSNorm weight and
    # epsilon, while their torch fallbacks ignore both.  Bind either the
    # NVIDIA reference kernels or the portable fallback before model
    # construction caches ``CustomOp._forward_method``.
    if current_platform.is_out_of_tree():
        patches.extend(_mhc_patches(fingerprint))

    # Worker-side kpool layout patches.  These used to be installed at plugin
    # registration for every model; the config-time and registry hooks stay
    # there because vLLM needs them before the worker exists, but the metadata
    # builder and KV-block zeroer hooks are runner-side and now follow the plan.
    from vllm_fl.patches.glm5_next_kpool_v024 import (
        glm5_next_kpool_runtime_patches,
    )

    patches.extend(glm5_next_kpool_runtime_patches(fingerprint))

    from vllm import _custom_ops as custom_ops

    patches.extend(_mla_boundary_patches(custom_ops, fingerprint))
    bind_patches(patches)


def _glm5_plan_provider(vllm_config) -> ActivationPlan | None:
    if not is_vllm_024() or not _is_glm5_model(vllm_config):
        return None
    return ActivationPlan(
        name="glm5_next_v024",
        fingerprint=_glm5_fingerprint(),
        apply=_apply_glm5_activation,
        attention_backend=_glm5_attention_override,
        moe_defaults=glm5_portable_moe_defaults(),
    )


def _glm5_runtime_plan(vllm_config, device_caps, user_policy):
    """Build GLM5's runtime selection (pure; see ``vllm_fl.runtime``)."""
    if not _is_glm5_model(vllm_config):
        return None

    defaults = glm5_portable_moe_defaults()
    base = user_policy if user_policy is not None else SelectionPolicy()
    policy = base
    if not defaults.is_empty():
        merged = merge_per_op_defaults(defaults, base.per_op_order_dict or None)
        policy = SelectionPolicy.from_dict(
            prefer=base.prefer,
            strict=base.strict,
            per_op_order=merged or None,
            deny_vendors=set(base.deny_vendors) or None,
            allow_vendors=set(base.allow_vendors) if base.allow_vendors else None,
        )

    # Publish the provider's defaults in the same policy consumed by private
    # indexer bindings. Explicit per-op user choices keep precedence.
    from vllm_fl.kernels.glm5_next.provider import INDEXER_OPERATORS

    order = (
        ("vendor:cuda", "flagos", "reference")
        if use_nvidia_reference()
        else ("flagos", "reference")
    )
    if policy.prefer != "flagos":
        order = policy.get_default_order()
    merged = {name: list(order) for name in INDEXER_OPERATORS}
    merged.update(policy.per_op_order_dict)
    policy = SelectionPolicy.from_dict(
        prefer=policy.prefer,
        strict=policy.strict,
        per_op_order=merged,
        deny_vendors=set(policy.deny_vendors),
        allow_vendors=set(policy.allow_vendors) if policy.allow_vendors else None,
    )

    # GLM5's attention backend depends on the per-layer selector context
    # (``use_mla`` / ``use_sparse``), so it stays on the activation plan's lazy
    # override; ``None`` here means "use the model-aware activation override".
    return RuntimePlan(selection_policy=policy)


def _register_glm5_next_registrations() -> None:
    """Idempotent config/model registration (safe at plugin import time)."""
    from vllm.model_executor.models import config as model_config
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import config as transformers_config
    from vllm.transformers_utils import model_arch_config_convertor

    from vllm_fl.configs.glm5_next import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Glm5NextVisionConfig,
    )
    from vllm_fl.patches.glm5_next_kpool_v024 import (
        install_glm5_next_kpool_v024,
    )

    install_glm5_next_kpool_v024()

    config_registry = transformers_config._CONFIG_REGISTRY
    config_registry["glm5_next"] = Glm5NextConfig
    config_registry["glm5_next_text"] = Glm5NextTextConfig
    config_registry["glm5_next_vision"] = Glm5NextVisionConfig

    convertors = model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS
    convertors["glm5_next"] = Glm5NextModelArchConfigConvertor
    convertors["glm5_next_text"] = Glm5NextModelArchConfigConvertor

    for architecture in (_CAUSAL_ARCH, _CONDITIONAL_ARCH):
        model_config.MODELS_CONFIG_MAP[architecture] = Glm5NextForCausalLMConfig

    model_registry._TEXT_GENERATION_MODELS.setdefault(
        _CAUSAL_ARCH, ("glm5_next", _CAUSAL_ARCH)
    )
    model_registry._VLLM_MODELS.setdefault(_CAUSAL_ARCH, ("glm5_next", _CAUSAL_ARCH))
    model_registry.ModelRegistry.register_model(
        _CAUSAL_ARCH,
        f"vllm_fl.models.glm5_next:{_CAUSAL_ARCH}",
    )

    # Keep the checkpoint's conditional architecture so vLLM constructs the
    # vision tower and enables --mm-encoder-tp-mode data instead of silently
    # reducing the model to its text-only runtime.
    model_registry._VLLM_MODELS.setdefault(
        _CONDITIONAL_ARCH, ("glm5_next", _CONDITIONAL_ARCH)
    )
    model_registry.ModelRegistry.register_model(
        _CONDITIONAL_ARCH,
        f"vllm_fl.models.glm5_next_multimodal:{_CONDITIONAL_ARCH}",
    )


def apply_glm5_next_v024_patches() -> bool:
    """Register GLM5-Next config/model entries and its activation plan.

    Registration only: no environment variable, FlagOS dispatch policy or
    class-level patch is applied here.  Those run from the worker via
    :func:`vllm_fl.activation.activate_for_model` when a GLM model is actually
    loaded, before its modules are constructed.
    """
    if not is_vllm_024():
        return False

    _register_glm5_next_registrations()
    register_plan_provider(_glm5_plan_provider)
    register_model_policy_factory(
        ModelPolicyFactory(
            name="glm5_next_v024",
            architectures=(_CAUSAL_ARCH, _CONDITIONAL_ARCH),
            model_types=("glm5_next", "glm5_next_text"),
            build=_glm5_runtime_plan,
            validate=validate_glm5_config,
        )
    )

    logger.info(
        "Registered vLLM 0.24 GLM5-Next text/VLM runtime with bounded KDA "
        "gate, kpool, and ViT data parallelism"
    )
    return True


__all__ = [
    "Glm5NextForCausalLMConfig",
    "Glm5NextModelArchConfigConvertor",
    "apply_glm5_next_v024_patches",
    "glm5_portable_moe_defaults",
]
