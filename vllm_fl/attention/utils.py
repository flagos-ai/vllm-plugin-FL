# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_mm_encoder_attention():
    """
    Patch vllm.attention.layers.mm_encoder_attention.maybe_get_vit_flash_attn_backend
    to support OOT platforms.

    The original implementation imports flash_attn_varlen_func from fa_utils,
    which may not have it defined for OOT platforms. This patch changes the
    FLASH_ATTN branch to import directly from vllm.vllm_flash_attn with a
    fallback to flash_attn.

    Patch (Hygon-only) — because PlatformFL._enum == OOT, the
       CustomOp dispatch routes to forward_oot → forward_native → SDPA,
       ignoring the FLASH_ATTN backend selection. On Hygon, SDPA has a known
       bug with small head counts (<6 heads). This patch routes to
       _forward_fa (Flash Attention) when is_flash_attn_backend is True.

    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # Part 1 (all platforms): patch maybe_get_vit_flash_attn_backend so that
    # OOT platforms can resolve flash_attn_varlen_func (vllm_flash_attn →
    # flash_attn fallback).  Wrapped in try/except to tolerate vLLM versions
    # that lack mm_encoder_attention (e.g. CUDA CI unit-test environments).
    try:
        import vllm.model_executor.layers.attention.mm_encoder_attention as mm_mod

        def _patched_maybe_get_vit_flash_attn_backend(attn_backend):
            if attn_backend == AttentionBackendEnum.FLASH_ATTN:
                try:
                    from vllm.vllm_flash_attn import flash_attn_varlen_func

                    logger.info_once(
                        "Using vllm.vllm_flash_attn for vit attention"
                    )
                except (ImportError, ModuleNotFoundError):
                    from flash_attn import flash_attn_varlen_func

                    logger.info_once("Using flash_attn for vit attention")
                return flash_attn_varlen_func
            elif attn_backend == AttentionBackendEnum.ROCM_AITER_FA:
                from aiter import flash_attn_varlen_func

                logger.info_once(
                    "Using aiter.flash_attn_varlen_func for vit attention"
                )
                return flash_attn_varlen_func
            else:
                return None

        mm_mod.maybe_get_vit_flash_attn_backend = (
            _patched_maybe_get_vit_flash_attn_backend
        )
        logger.info_once(
            "Patched maybe_get_vit_flash_attn_backend for OOT platforms"
        )
    except Exception:
        logger.info_once(
            "Skipped maybe_get_vit_flash_attn_backend patch: "
            "mm_encoder_attention module not available"
        )
        mm_mod = None

    # Part 2 (Hygon-only): route forward_native → _forward_fa to avoid SDPA
    # head-count bug, and inject flash_attn_varlen_func into fa_utils.
    from vllm.platforms import current_platform

    vendor = getattr(current_platform, "vendor_name", "")
    if vendor != "hygon":
        logger.info_once(
            "Skip Hygon FA patch: vendor=%r != 'hygon'", vendor
        )
        return

    if mm_mod is None:
        logger.info_once(
            "Skip Hygon FA patch: mm_encoder_attention not available"
        )
        return

    logger.info_once(
        "Hygon detected: patching forward_native → _forward_fa "
        "to avoid OOT dispatch routing to SDPA"
    )

    # Inject flash_attn_varlen_func into fa_utils (skipped because PlatformFL._enum==OOT)
    import vllm.v1.attention.backends.fa_utils as fa_utils
    from flash_attn import flash_attn_varlen_func

    fa_utils.flash_attn_varlen_func = flash_attn_varlen_func
    logger.info_once("Injected flash_attn.flash_attn_varlen_func into fa_utils")

    _orig_forward_native = mm_mod.MMEncoderAttention.forward_native

    def _patched_forward_native(
        self,
        query,
        key,
        value,
        cu_seqlens=None,
        max_seqlen=None,
        sequence_lengths=None,
    ):
        if self.is_flash_attn_backend:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        return _orig_forward_native(
            self, query, key, value, cu_seqlens, max_seqlen, sequence_lengths
        )

    mm_mod.MMEncoderAttention.forward_native = _patched_forward_native
