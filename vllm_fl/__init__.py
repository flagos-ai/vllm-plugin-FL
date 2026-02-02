# Copyright (c) 2025 BAAI. All rights reserved.


import os
import logging

logger = logging.getLogger(__name__)


def _register_fp8_kernel_mappings():
    """Register FP8 kernel mappings for OOT (Out-Of-Tree) platform.

    vLLM's FP8 quantization kernel selection uses a platform-specific mapping:
        _POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[FP8ScaledMMLinearKernel]]

    This mapping only includes CUDA, ROCM, and CPU platforms by default.
    When using PlatformFL (which uses PlatformEnum.OOT), vLLM raises KeyError.

    This function patches the global mapping to include OOT platform,
    using CUDA kernels when the OOT platform is CUDA-alike.

    Call Path Analysis:
    - vllm/model_executor/layers/quantization/kernels/scaled_mm/__init__.py:152
    - choose_scaled_mm_linear_kernel() iterates possible_kernels[current_platform._enum]
    - KeyError raised when _enum is PlatformEnum.OOT

    Fix: Add PlatformEnum.OOT -> CUDA kernels mapping for CUDA-alike OOT platforms.
    """
    try:
        from vllm.platforms import PlatformEnum, current_platform

        # Only register if we're on OOT platform
        if not current_platform.is_out_of_tree():
            logger.debug("Not OOT platform, skipping FP8 kernel mapping registration")
            return

        # Import the kernel mappings module
        from vllm.model_executor.layers.quantization.kernels import scaled_mm

        # Check if OOT is already registered (avoid duplicate registration)
        if PlatformEnum.OOT in scaled_mm._POSSIBLE_FP8_KERNELS:
            logger.debug("OOT platform FP8 kernels already registered")
            return

        # Determine which kernels to use based on device type
        if current_platform.is_cuda_alike():
            # Use CUDA FP8 kernels for CUDA-alike OOT platforms
            if PlatformEnum.CUDA in scaled_mm._POSSIBLE_FP8_KERNELS:
                scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = \
                    scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA].copy()
                logger.info(
                    "Registered OOT platform FP8 kernels (using CUDA kernels): %s",
                    [k.__name__ for k in scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT]]
                )
            else:
                logger.warning("CUDA FP8 kernels not available, OOT FP8 may fail")
        else:
            # For non-CUDA OOT platforms, use CPU fallback if available
            if PlatformEnum.CPU in scaled_mm._POSSIBLE_FP8_KERNELS:
                scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = \
                    scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.CPU].copy()
                logger.info("Registered OOT platform FP8 kernels (using CPU fallback)")
            else:
                # Empty list - will cause clear error message
                scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = []
                logger.warning("No FP8 kernels available for non-CUDA OOT platform")

        # Also register INT8 kernels if not already present
        if PlatformEnum.OOT not in scaled_mm._POSSIBLE_INT8_KERNELS:
            if current_platform.is_cuda_alike() and PlatformEnum.CUDA in scaled_mm._POSSIBLE_INT8_KERNELS:
                scaled_mm._POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = \
                    scaled_mm._POSSIBLE_INT8_KERNELS[PlatformEnum.CUDA].copy()
                logger.debug("Registered OOT platform INT8 kernels (using CUDA kernels)")
            elif PlatformEnum.CPU in scaled_mm._POSSIBLE_INT8_KERNELS:
                scaled_mm._POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = \
                    scaled_mm._POSSIBLE_INT8_KERNELS[PlatformEnum.CPU].copy()

        # Also register mixed_precision kernels (W4A8, Marlin, etc.)
        try:
            from vllm.model_executor.layers.quantization.kernels import mixed_precision
            if PlatformEnum.OOT not in mixed_precision._POSSIBLE_KERNELS:
                if current_platform.is_cuda_alike() and PlatformEnum.CUDA in mixed_precision._POSSIBLE_KERNELS:
                    mixed_precision._POSSIBLE_KERNELS[PlatformEnum.OOT] = \
                        mixed_precision._POSSIBLE_KERNELS[PlatformEnum.CUDA].copy()
                    logger.debug("Registered OOT platform mixed_precision kernels (using CUDA)")
                elif PlatformEnum.CPU in mixed_precision._POSSIBLE_KERNELS:
                    mixed_precision._POSSIBLE_KERNELS[PlatformEnum.OOT] = \
                        mixed_precision._POSSIBLE_KERNELS[PlatformEnum.CPU].copy()
        except ImportError:
            pass  # mixed_precision module may not be available

    except ImportError as e:
        logger.debug(f"FP8 kernel mapping registration skipped (import error): {e}")
    except Exception as e:
        logger.warning(f"Failed to register FP8 kernel mappings for OOT platform: {e}")


def register():
    """Register the FL platform."""

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return "vllm_fl.platform.PlatformFL"


def register_ops():
    """Register custom OOT operators for FL platform.

    This is called by vLLM's plugin system to register custom ops
    that override the default implementations when using FL platform.

    FlagGems Integration Modes:
    ---------------------------
    Controlled by GEMS_MODE environment variable:

    1. SELECTIVE (default): FlagGems via dispatch manager only
       - Preserves vLLM's optimized CUDA kernels
       - Only 1-3 FlagGems ops used (silu_and_mul, etc.)
       - Performance: ~93% of baseline

    2. TIERED: Context-aware dispatch with operator policies
       - Tier-0: Never replace (FlashAttention, bmm, softmax)
       - Tier-1: Conditional (mm, layer_norm - only for large prefill)
       - Tier-2: Safe (elementwise ops)
       - Tier-3: Experimental (opt-in)
       - Performance: ~95%+ of baseline

    3. GLOBAL (NOT RECOMMENDED): Replace ALL 307 aten operators
       - Causes ~4x throughput regression
       - Only use for debugging/testing FlagGems coverage
       - Performance: ~19% of baseline

    Environment Variables:
    - USE_FLAGGEMS=True/False: Enable FlagGems integration (default: True)
    - GEMS_MODE=SELECTIVE|TIERED|GLOBAL: Select mode (default: SELECTIVE)
    - GEMS_EXPERIMENTAL=True: Enable Tier-3 experimental ops
    - GEMS_PERF_TRACKING=True: Enable performance tracking
    """
    # Register FP8 kernel mappings for OOT platform FIRST
    # This fixes FP8 quantization compatibility issue where vLLM's
    # kernel selection code doesn't have OOT in its platform mappings
    _register_fp8_kernel_mappings()

    from vllm_fl.ops.custom_ops import register_oot_ops
    register_oot_ops()

    import vllm_fl.envs as fl_envs
    if fl_envs.USE_FLAGGEMS:
        # Get mode from environment
        gems_mode = os.environ.get("GEMS_MODE", "SELECTIVE").upper()

        # Backwards compatibility: GEMS_GLOBAL_ENABLE=True -> GLOBAL mode
        if os.environ.get("GEMS_GLOBAL_ENABLE", "False").lower() == "true":
            gems_mode = "GLOBAL"

        if gems_mode == "GLOBAL":
            # GLOBAL mode: Replace ALL aten operators (causes ~4x overhead)
            import flag_gems
            flag_gems.enable(record=False)
            logger.warning(
                "FlagGems GLOBAL mode enabled - ALL 307 aten ops replaced. "
                "This causes ~4x throughput regression! "
                "Use GEMS_MODE=SELECTIVE or TIERED for production."
            )

        elif gems_mode == "TIERED":
            # TIERED mode: Context-aware dispatch with operator policies
            try:
                from vllm_fl.dispatch.operator_policy import get_policy_manager
                from vllm_fl.dispatch.context_aware_dispatch import get_context_manager

                # Initialize policy manager
                policy_mgr = get_policy_manager()
                ctx_mgr = get_context_manager()

                logger.info(
                    "FlagGems TIERED mode - context-aware dispatch enabled. "
                    f"Policies: {len(policy_mgr._policies)} operators classified. "
                    "Set GEMS_PERF_TRACKING=1 to enable performance tracking."
                )
            except Exception as e:
                logger.warning(f"Failed to initialize TIERED mode: {e}. Falling back to SELECTIVE.")
                gems_mode = "SELECTIVE"

        if gems_mode == "SELECTIVE":
            # SELECTIVE mode (default): Only use FlagGems via dispatch manager
            # This preserves vLLM's optimized CUDA kernels for most operations
            logger.info(
                "FlagGems SELECTIVE mode - dispatch manager controls FlagGems use. "
                "Use GEMS_MODE=TIERED for context-aware dispatch."
            )


def register_model():
    """Register the FL model."""
    from vllm import ModelRegistry

    try:
        from vllm_fl.models.qwen3_next import Qwen3NextForCausalLM  # noqa: F401

        ModelRegistry.register_model(
            "Qwen3NextForCausalLM", "vllm_fl.models.qwen3_next:Qwen3NextForCausalLM"
        )
    except ImportError:
        logger.info(
            "From vllm_fl.models.qwen3_next cannot import Qwen3NextForCausalLM, skipped"
        )
    except Exception as e:
        logger.error(f"Register model error: {str(e)}")

    ModelRegistry.register_model(
        "MiniCPMO",
        "vllm_fl.models.minicpmo:MiniCPMO")

    # Register Kimi-K2.5 models (migrated from upstream vLLM)
    # These use the vLLM native implementations as the base,
    # with platform-specific dispatch handled by PlatformFL
    _register_kimi_k25_models(ModelRegistry)


def _register_kimi_k25_models(ModelRegistry):
    """Register Kimi-K2.5 related models.

    Kimi-K2.5 is a vision-language model that uses DeepseekV3 as its text backbone.
    We register both the full VLM and the text-only variant.

    Model Architecture:
    - KimiK25ForConditionalGeneration: Full VLM with vision tower (MoonViT) + text (DeepseekV3)
    - DeepseekV3ForCausalLM: Text-only backbone (already in vLLM registry)

    The platform dispatch (PlatformFL) handles attention backends:
    - MLA (Multi-head Latent Attention) for DeepseekV3
    - Standard attention for vision tower
    """
    try:
        # Import the Kimi-K2.5 config to ensure it's available
        from vllm.transformers_utils.configs import KimiK25Config  # noqa: F401

        # Register Kimi-K2.5 VLM (uses vLLM's native implementation)
        # The model is already registered in vLLM, but we ensure it's available
        # when using the FL platform by re-registering with explicit path
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm.model_executor.models.kimi_k25:KimiK25ForConditionalGeneration"
        )
        logger.info("Registered KimiK25ForConditionalGeneration model")

        # Also register KimiVL for older model variants
        ModelRegistry.register_model(
            "KimiVLForConditionalGeneration",
            "vllm.model_executor.models.kimi_vl:KimiVLForConditionalGeneration"
        )
        logger.info("Registered KimiVLForConditionalGeneration model")

    except ImportError as e:
        logger.warning(
            f"Could not register Kimi-K2.5 models (vLLM may not have support): {e}"
        )
    except Exception as e:
        logger.error(f"Error registering Kimi-K2.5 models: {e}")
