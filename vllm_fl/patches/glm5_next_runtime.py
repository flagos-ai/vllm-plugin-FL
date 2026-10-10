# SPDX-License-Identifier: Apache-2.0
"""Register and activate the plugin-owned GLM5-Next runtime components."""

from functools import lru_cache, wraps
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.config import HybridAttentionMambaModelConfig
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
from vllm_fl.kernels.glm5_next.provider import use_nvidia_reference
from vllm_fl.runtime.model_policy import (
    ModelPolicyError,
    ModelPolicyFactory,
    RuntimePlan,
    register_model_policy_factory,
)

logger = init_logger(__name__)

_CAUSAL_ARCH = "Glm5NextForCausalLM"
_CONDITIONAL_ARCH = "Glm5NextForConditionalGeneration"

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


def _has_vllm_cache_op(op_name: str) -> bool:
    """Probe the extension ABI without invoking a device kernel."""
    try:
        getattr(torch.ops._C_cache_ops, op_name)
    except AttributeError:
        return False
    return True


@lru_cache(maxsize=None)
def _concat_handles_zero_rope(concat) -> bool:
    device = current_platform.device_type
    q = torch.ones((1, 1, 512), dtype=torch.bfloat16, device=device)
    rope = torch.empty((1, 1, 0), dtype=q.dtype, device=device)
    out = torch.full_like(q, -2)
    try:
        concat(q, rope, out)
    except (AttributeError, NotImplementedError):
        return False
    except RuntimeError as exc:
        # The native ABI rejects zero RoPE before launching its cache kernel.
        # Recognize only this shape guard; device/execution failures propagate.
        message = str(exc).strip()
        guard = "rope_dim must be 64, got 0"
        if message == guard or (
            message.startswith("concat_mla_q, ") and message.endswith(", " + guard)
        ):
            return False
        raise
    return bool(torch.equal(q, out))


def _mla_boundary_patches(custom_ops, fingerprint) -> list[PendingPatch]:
    if custom_ops is None:
        from vllm import _custom_ops as custom_ops
    from vllm_fl.kernels.glm5_next.indexer_backend import _load_flaggems_op

    patches = []

    def stage(attr, replacement):
        original = getattr(custom_ops, attr)
        patches.append(
            PendingPatch(
                target=f"{custom_ops.__name__}.{attr}",
                owner=custom_ops,
                attr=attr,
                replacement=replacement,
                pristine=original,
                fingerprint=fingerprint,
                phase="worker",
            )
        )

    concat = custom_ops.concat_mla_q
    if not getattr(
        concat, "_glm5_next_nope_fix", False
    ) and not _concat_handles_zero_rope(concat):
        query_op = _load_flaggems_op("concat_mla_q", "concat_mla_q")
        if query_op is None:
            raise RuntimeError(
                "GLM zero-RoPE boundary requires FlagGems-vllm concat_mla_q"
            )

        @wraps(concat)
        def concat_nope(q_nope, q_pe, q_out):
            if q_pe.shape[-1] == 0:
                return query_op(q_nope, q_pe, q_out)
            return concat(q_nope, q_pe, q_out)

        concat_nope._glm5_next_nope_fix = True
        stage("concat_mla_q", concat_nope)

    cache = custom_ops.concat_and_cache_mla
    if not _has_vllm_cache_op("concat_and_cache_mla") and not getattr(
        cache, "_glm5_next_vendor_fallback", False
    ):
        writer = _load_flaggems_op("concat_and_cache_mla", "concat_and_cache_mla")
        if writer is None:
            raise RuntimeError(
                "GLM MLA cache boundary requires FlagGems-vllm concat_and_cache_mla"
            )

        @wraps(cache)
        def cache_write(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
            return writer(
                kv_c,
                k_pe,
                kv_cache,
                slot_mapping,
                "auto" if kv_cache_dtype == "bfloat16" else kv_cache_dtype,
                scale,
            )

        cache_write._glm5_next_vendor_fallback = True
        stage("concat_and_cache_mla", cache_write)
    return patches


def _install_mla_boundary_compat_ops(custom_ops: ModuleType | None = None) -> bool:
    if custom_ops is None:
        from vllm import _custom_ops as custom_ops
    return bool(bind_patches(_mla_boundary_patches(custom_ops, _glm5_fingerprint())))


def _silu_and_mul_with_clamp_oot(op, self, x):
    if self.alpha != 1.0 or self.beta != 0.0:
        raise NotImplementedError("FlagGems clamp requires alpha=1 and beta=0")
    dim = x.shape[-1] // 2
    return op(x[..., :dim], x[..., dim:], self.swiglu_limit)


def _bind_portable_forward(owner, name, flag, normalize=None):
    """Use the existing policy binding for portable custom-op dispatch."""
    from vllm_fl.dispatch.binding import OperatorBinding
    from vllm_fl.dispatch.manager import OpManager
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl
    from vllm_fl.utils import use_flaggems_op

    manager = OpManager()
    if flag is not None:
        flag._is_available = lambda: use_flaggems_op(name)
        manager.registry.register_impl(
            OpImpl(name, "glm5.flaggems", BackendImplKind.DEFAULT, flag)
        )

    def native(*args, **kwargs):
        return owner.forward_cuda(*args, **kwargs)

    native._is_available = use_nvidia_reference
    manager.registry.register_impl(
        OpImpl(name, "glm5.cuda", BackendImplKind.VENDOR, native, vendor="cuda")
    )

    def reference(*args, **kwargs):
        return owner.forward_native(*args, **kwargs)

    reference._is_available = lambda: current_platform.device_type == "cpu"
    manager.registry.register_impl(
        OpImpl(name, "glm5.torch", BackendImplKind.REFERENCE, reference)
    )
    binding = OperatorBinding(manager, name)

    @wraps(owner.forward_native)
    def forward(self, *args, **kwargs):
        if normalize is not None:
            return normalize(binding, self, *args, **kwargs)
        return binding(self, *args, **kwargs)

    return forward


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
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPostOp, MHCPreOp

    for cls in (MHCPreOp, MHCPostOp, MHCFusedPostPreOp, SiluAndMulWithClamp):
        _BASELINES[_attr_target(cls, "forward_oot")] = cls.forward_oot


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
    pristine = _BASELINES[target]
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
    return f"glm5_next_runtime@vllm{release}"


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

    from vllm_fl.dispatch.policy import get_policy

    policy = get_policy()
    order = (
        policy.get_per_op_order("flash_mla_sparse_fwd" if use_sparse else "attention")
        or policy.get_default_order()
    )
    native_allowed = "cuda" not in policy.deny_vendors and (
        not policy.allow_vendors or "cuda" in policy.allow_vendors
    )
    if use_nvidia_reference() and native_allowed and order[0] != "flagos":
        return None
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    flaggems_backend = FlagGemsBackend()
    if not flaggems_backend.is_available():
        raise RuntimeError("GLM portable MLA requires the paired FlagGems libraries")
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
    from functools import partial

    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPostOp, MHCPreOp

    from vllm_fl.kernels.glm5_next.indexer_backend import _load_flaggems_op

    def bind_library(op):
        def call(self, *args, **kwargs):
            return op(*args, **kwargs)

        return call

    specs = []
    for owner, name, api in (
        (MHCPreOp, "mhc_pre", "mhc_pre_with_norm"),
        (MHCPostOp, "mhc_post", "mhc_post"),
        (MHCFusedPostPreOp, "mhc_fused_post_pre", "mhc_fused_post_pre_with_norm"),
        (SiluAndMulWithClamp, "silu_and_mul_with_clamp", "silu_and_mul_with_clamp"),
    ):
        op = _load_flaggems_op(api, api)
        flag = (
            None
            if op is None
            else (
                partial(_silu_and_mul_with_clamp_oot, op)
                if name == "silu_and_mul_with_clamp"
                else bind_library(op)
            )
        )
        specs.append((owner, _bind_portable_forward(owner, name, flag)))
    return [
        _pending_attr(owner, "forward_oot", value, fingerprint, ("self",))
        for owner, value in specs
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
    from vllm_fl.patches.glm5_next_kpool import glm5_next_kpool_runtime_patches

    patches.extend(glm5_next_kpool_runtime_patches(fingerprint))

    from vllm import _custom_ops as custom_ops

    patches.extend(_mla_boundary_patches(custom_ops, fingerprint))
    bind_patches(patches)


def _glm5_plan_provider(vllm_config) -> ActivationPlan | None:
    if not _is_glm5_model(vllm_config):
        return None
    return ActivationPlan(
        name="glm5_next",
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
    from vllm_fl.patches.glm5_next_kpool import install_glm5_next_kpool

    install_glm5_next_kpool()

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


def register_glm5_next_support() -> bool:
    """Register GLM5-Next config/model entries and its activation plan.

    Registers config/model entries and tracked early engine/config hooks.
    Worker patches and dispatch defaults are installed by activate_for_model
    only after GLM model matching, before its modules are constructed.
    """

    _register_glm5_next_registrations()
    register_plan_provider(_glm5_plan_provider)
    register_model_policy_factory(
        ModelPolicyFactory(
            name="glm5_next",
            architectures=(_CAUSAL_ARCH, _CONDITIONAL_ARCH),
            model_types=("glm5_next", "glm5_next_text"),
            build=_glm5_runtime_plan,
            validate=validate_glm5_config,
        )
    )

    logger.info(
        "Registered GLM5-Next text/VLM runtime with bounded KDA "
        "gate, kpool, and ViT data parallelism"
    )
    return True


__all__ = [
    "Glm5NextForCausalLMConfig",
    "Glm5NextModelArchConfigConvertor",
    "register_glm5_next_support",
    "glm5_portable_moe_defaults",
]
