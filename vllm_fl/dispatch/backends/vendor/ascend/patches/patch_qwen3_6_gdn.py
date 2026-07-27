#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Copyright (c) 2026 BAAI. All rights reserved.
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
# mypy: ignore-errors

"""AscendC fused-op patch for Qwen3.5/Qwen3.6 GatedDeltaNet (GDN) layers.

Ports the vllm-ascend ``AscendGatedDeltaNetAttention`` integration
(``vllm_ascend/ops/gdn.py`` and ``vllm_ascend/ops/layernorm.py``) to the
FL plugin on vLLM 0.13:

* ``npu_causal_conv1d_custom`` replaces the Triton ``causal_conv1d_fn`` /
  ``causal_conv1d_update`` calls inside ``Qwen3NextGatedDeltaNet._forward_core``.
  Its ``(width, dim)`` weight layout is materialized once after checkpoint
  loading instead of being transposed implicitly on every forward.
* ``npu_fused_gdn_gating`` replaces the Triton ``fused_gdn_gating``.
* ``npu_recurrent_gated_delta_rule`` replaces ``fused_recurrent_gated_delta_rule``
  on the (speculative-)decode paths.  The chunked prefill path keeps the
  existing Triton ``chunk_gated_delta_rule`` (already patched to the Ascend
  implementation by ``patch_fla_ops``).
* ``npu_gemma_rms_norm`` / ``npu_add_rms_norm_bias`` back
  ``GemmaRMSNorm.forward_oot``.
* Non-speculative decode batches can run the q/k L2 norm + delta-rule state
  update as one fused Triton kernel (``fused_recurrent_delta_rule_update``,
  adapted from vllm-ascend's fused_sigmoid_gating_delta_rule_update) on top
  of the AscendC ``npu_fused_gdn_gating`` op, replacing the separate
  2x l2norm_fwd + npu_recurrent_gated_delta_rule calls. The in-kernel
  sigmoid-gating section of the upstream kernel is miscompiled by the
  Ascend Triton pipeline in this environment, so the gating stays on the
  AscendC op. The fused kernel is OFF by default: it was only verified on
  910B4-1 and faults on 910B3 with the CANN 8.5.0 bishengir pipeline
  (aivec MPU access error surfacing as aclrtSynchronizeEvent 507035).
  Set ``VLLM_FL_DISABLE_FUSED_DECODE_GDN=0`` to opt in; the default falls
  back to the AscendC recurrent op.
* ``RMSNormGated.forward_oot`` runs the fused Triton
  ``layer_norm_fwd_1pass`` kernel (ported from vllm-ascend) instead of the
  decomposed eager ``forward_native`` chain.
* Fresh (all-zero ``initial_state``) prefill batches run the fused
  PTO/Bisheng megakernel (vllm-ascend PR #8872 port,
  ``vllm_fl/ops/pto_chunk_gdn``): all six GDN stages in a single launch.
  The PTO-vs-Triton decision and the megakernel chunk counting are made
  from CPU-side flags attached to ``GDNAttentionMetadata`` by the builder
  wrap — no per-layer device→host syncs (the naive wrapper approach costs
  two syncs per GDN layer per prefill step, which regresses batch64
  serving). Other prefill batches keep the Triton chunk kernel.
  ``VLLM_FL_DISABLE_PTO_GDN=1`` disables the megakernel.

Layout notes (must stay consistent with the kernels):

* conv_state: vLLM 0.13 allocates the GDN conv cache as
  ``(state_len, conv_dim)`` per slot, which is exactly what the AscendC
  kernel expects, so the cache is passed through *without* the transpose
  used by the Triton path.
* ssm_state: the AscendC ``recurrent_gated_delta_rule`` kernel expects the
  state in ``(Hv, Dv, Dk)`` layout (see
  ``csrc/ascend/attention/recurrent_gated_delta_rule``), while vLLM 0.13
  allocates ``(Hv, Dk, Dv)``.  ``get_state_shape`` is therefore patched to
  swap the last two dims, and the chunked-prefill path transposes the
  initial/final state at the boundary.
* ``actual_seq_lengths`` of ``npu_recurrent_gated_delta_rule`` follows the
  cu_seqlens convention ``[0, len_1, ..., len_B]`` (batch = numel - 1).
* mamba KV-cache: upstream stores conv/ssm states interleaved inside one
  (padded) page per block; the AscendC kernels address the state caches
  assuming dense per-state tensors, so ``_reshape_kv_cache_tensors`` is
  wrapped to regroup the state views into dense per-state tensors over the
  same raw storage (transparent to the Triton fallback path, which uses
  explicit strides).

The patch bootstraps the CANN custom-op environment automatically
(``ASCEND_CUSTOM_OPP_PATH`` pointing at the packaged
``_cann_ops_custom/vendors/custom_transformer``) and is only skipped when
the ``_C_ascend`` bindings or the op package are unavailable; otherwise the
existing Triton path is kept. Set ``VLLM_FL_DISABLE_ASCENDC_GDN=1`` to
force the Triton path.
"""

import logging
import math
import os

import torch
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec

import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

from ..impl.fla.fused_recurrent import fused_recurrent_delta_rule_update
from ..impl.fla.l2norm import l2norm_fwd
from ..impl.linearnorm.layernorm_gated import rmsnorm_gated_oot

logger = logging.getLogger(__name__)

_CUSTOM_OPP_MARKER = "custom_transformer"
_REQUIRED_OPS = (
    "npu_causal_conv1d_custom",
    "npu_fused_gdn_gating",
    "npu_recurrent_gated_delta_rule",
    "npu_chunk_gated_delta_rule",
    "npu_gemma_rms_norm",
    "npu_add_rms_norm_bias",
)


def _bootstrap_custom_op_env() -> bool:
    """Make the packaged CANN custom-op package discoverable at runtime.

    Same idea as vllm-ascend's ``bootstrap_custom_op_env``: prepend the
    packaged ``_cann_ops_custom/vendors/custom_transformer`` dir to
    ``ASCEND_CUSTOM_OPP_PATH`` so users do not have to source
    ``set_env.bash`` before launching the server. The OPP path is scanned
    lazily by the AscendCL runtime at the first custom-op call, so setting
    it here (before any op invocation) is sufficient; the variable is also
    inherited by spawned worker processes.

    Additionally preload ``libcust_opapi.so`` by absolute path: the aclnn
    adapter resolves custom symbols via ``dlopen("libcust_opapi.so")`` by
    bare name, which only searches the *startup-time* ``LD_LIBRARY_PATH``
    (glibc caches it) — but finds the already-loaded library by SONAME
    after a preload. ``RTLD_LOCAL`` is used on purpose: ``RTLD_GLOBAL``
    leads to a double-free at process teardown.
    """
    try:
        import vllm_fl._C_ascend as _ext  # noqa: F401
    except Exception as e:
        logger.warning("Failed to import vllm_fl._C_ascend: %s", e)
        return False
    vendor_dir = os.path.join(
        os.path.dirname(_ext.__file__), "_cann_ops_custom", "vendors", _CUSTOM_OPP_MARKER
    )
    if not os.path.isdir(vendor_dir):
        logger.warning("CANN custom op package not found at %s", vendor_dir)
        return False
    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor_dir not in opp_path:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
            vendor_dir + (":" + opp_path if opp_path else "")
        )
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    lib_dir = os.path.join(vendor_dir, "op_api", "lib")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld_path if ld_path else "")
        try:
            import ctypes

            ctypes.CDLL(
                os.path.join(lib_dir, "libcust_opapi.so"), mode=ctypes.RTLD_LOCAL
            )
        except OSError as e:
            logger.warning("Failed to preload libcust_opapi.so: %s", e)
            return False
    return True


def _ascendc_ops_available() -> bool:
    """Check the CANN custom-op env and the ``_C_ascend`` torch bindings."""
    if os.environ.get("VLLM_FL_DISABLE_ASCENDC_GDN", "0") == "1":
        logger.info("VLLM_FL_DISABLE_ASCENDC_GDN=1, keep Triton GDN path")
        return False
    if _CUSTOM_OPP_MARKER not in os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""):
        if not _bootstrap_custom_op_env():
            logger.warning(
                "CANN custom op environment is not set and auto-bootstrap "
                "failed; keep Triton GDN path"
            )
            return False
    try:
        import vllm_fl._C_ascend  # noqa: F401
    except Exception as e:
        logger.warning("Failed to import vllm_fl._C_ascend: %s; keep Triton GDN path", e)
        return False
    missing = [name for name in _REQUIRED_OPS if not hasattr(torch.ops._C_ascend, name)]
    if missing:
        logger.warning("torch.ops._C_ascend missing ops %s; keep Triton GDN path", missing)
        return False
    return True


def _build_actual_seq_lengths(
    query_start_loc: torch.Tensor,
    num_sequences: int,
) -> torch.Tensor:
    """Build ``[0, len_1, ..., len_B]`` cu-seqlens style actual_seq_lengths."""
    actual_seq_lengths = torch.empty_like(query_start_loc[: num_sequences + 1])
    actual_seq_lengths[:1].copy_(query_start_loc[:1])
    torch.sub(
        query_start_loc[1 : num_sequences + 1],
        query_start_loc[:num_sequences],
        out=actual_seq_lengths[1:],
    )
    return actual_seq_lengths


def _fused_decode_gdn_enabled() -> bool:
    """Whether to use the fused Triton decode kernel (q/k L2 norm +
    recurrent delta-rule state update in a single launch, adapted from
    vllm-ascend) for non-speculative decode batches.

    Off by default: the kernel was verified on 910B4-1 but faults on 910B3
    (vector-core MPU access error, CANN 507035), so the AscendC recurrent
    op remains the safe default. Opt in with
    ``VLLM_FL_DISABLE_FUSED_DECODE_GDN=0``.
    """
    return os.environ.get("VLLM_FL_DISABLE_FUSED_DECODE_GDN", "1") != "1"


def _cache_conv1d_weight_transposed(layer: Qwen3NextGatedDeltaNet) -> None:
    """Materialize the AscendC conv weight layout once after weight loading."""
    weight = layer.conv1d.weight
    conv_weights = weight.detach().view(weight.size(0), weight.size(2))
    cached_weight = conv_weights.transpose(0, 1).contiguous()
    if "_ascendc_conv_weights_t" in layer._buffers:
        layer._ascendc_conv_weights_t = cached_weight
    elif hasattr(layer, "_ascendc_conv_weights_t"):
        layer._ascendc_conv_weights_t = cached_weight
    else:
        # Qwen3_5GatedDeltaNet deliberately skips the upstream GDN __init__,
        # so it does not receive the loader-time buffer registration below.
        layer.register_buffer(
            "_ascendc_conv_weights_t", cached_weight, persistent=False
        )


def _patch_gdn_conv_weight_loader() -> None:
    """Cache ``(width, dim)`` conv weights instead of transposing per token."""
    orig_init = Qwen3NextGatedDeltaNet.__init__
    if getattr(orig_init, "_vllm_fl_ascendc_conv_cache", False):
        return

    def init_with_conv_cache(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.register_buffer(
            "_ascendc_conv_weights_t", None, persistent=False
        )
        orig_weight_loader = self.conv1d.weight.weight_loader

        def weight_loader_with_conv_cache(param, *loader_args, **loader_kwargs):
            result = orig_weight_loader(param, *loader_args, **loader_kwargs)
            _cache_conv1d_weight_transposed(self)
            return result

        self.conv1d.weight.weight_loader = weight_loader_with_conv_cache

    init_with_conv_cache._vllm_fl_ascendc_conv_cache = True
    Qwen3NextGatedDeltaNet.__init__ = init_with_conv_cache


# ---------------------------------------------------------------------------
# PTO megakernel for the chunked-prefill path (vllm-ascend PR #8872 port)
# ---------------------------------------------------------------------------
_PTO_AVAILABLE: bool | None = None


def _pto_available() -> bool:
    """Whether the PTO megakernel can be used for fresh prefill batches."""
    global _PTO_AVAILABLE
    if _PTO_AVAILABLE is not None:
        return _PTO_AVAILABLE
    if os.environ.get("VLLM_FL_DISABLE_PTO_GDN", "0") == "1":
        logger.info("VLLM_FL_DISABLE_PTO_GDN=1, keep Triton chunk path")
        _PTO_AVAILABLE = False
        return False
    try:
        from vllm_fl.ops.pto_chunk_gdn.mega_kernel import run_mega_kernel  # noqa: F401

        _PTO_AVAILABLE = True
    except Exception as e:
        logger.warning("PTO chunk_gated_delta_rule unavailable: %s", e)
        _PTO_AVAILABLE = False
    return _PTO_AVAILABLE


def _pto_prefill_usable(attn_metadata) -> bool:
    """PTO applies only when *every* sequence in the batch starts from zero
    state (the megakernel cannot consume a non-zero initial_state). The flag
    is computed on CPU by the metadata builder, so this check is sync-free."""
    if not _pto_available():
        return False
    return getattr(attn_metadata, "any_initial_state_cpu", True) is False


def _patch_gdn_metadata_host_flags() -> None:
    """Attach CPU-side prefill flags to ``GDNAttentionMetadata``.

    ``any_initial_state_cpu`` and ``cu_seqlens_host`` let ``_forward_core``
    pick PTO vs Triton and size the megakernel workspaces without any
    device→host synchronization (the generic PTO wrapper previously paid one
    ``torch.any(initial_state != 0)`` plus one ``.cpu().tolist()`` sync per
    GDN layer per prefill step — a measurable regression at batch64).
    """
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    orig_build = GDNAttentionMetadataBuilder.build

    def build_with_host_flags(
        self,
        common_prefix_len,
        common_attn_metadata,
        *args,
        **kwargs,
    ):
        attn_metadata = orig_build(
            self, common_prefix_len, common_attn_metadata, *args, **kwargs
        )
        if attn_metadata.num_prefills > 0:
            context_lens_cpu = common_attn_metadata.num_computed_tokens_cpu
            attn_metadata.any_initial_state_cpu = bool((context_lens_cpu > 0).any())
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            attn_metadata.cu_seqlens_host = tuple(int(x) for x in qsl_cpu.tolist())
        return attn_metadata

    GDNAttentionMetadataBuilder.build = build_with_host_flags
    logger.info("Patched GDNAttentionMetadataBuilder with CPU prefill flags for PTO")


def _chunk_gdn_pto(q, k, v, g, beta, cu_seqlens, attn_metadata):
    """Run the PTO megakernel for a fresh (all-zero initial state) prefill batch."""
    from vllm_fl.ops.pto_chunk_gdn.mega_kernel import run_mega_kernel

    Hg, D = q.shape[2], q.shape[3]
    q16 = l2norm_fwd(q.to(torch.float16))
    k16 = l2norm_fwd(k.to(torch.float16))
    cu32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    lens_host = getattr(attn_metadata, "cu_seqlens_host", None)
    total_chunks = None
    if lens_host is not None:
        total_chunks = sum(
            (lens_host[i + 1] - lens_host[i] + 127) // 128
            for i in range(len(lens_host) - 1)
        )
    o, fs = run_mega_kernel(
        q16,
        k16,
        v.to(torch.float16),
        g.float(),
        beta.to(torch.float16),
        cu32.contiguous(),
        stream=torch.npu.current_stream()._as_parameter_,
        chunk_size=128,
        scale=D**-0.5,
        key_heads=Hg,
        return_final_state=True,
        total_chunks=total_chunks,
    )
    return o.to(q.dtype), fs.to(q.dtype)


def _chunk_gated_delta_rule_aclnn(
    query, key, value, g, beta, ssm_state, state_indices, has_initial_state, cu_seqlens
):
    """Run the aclnn npu_chunk_gated_delta_rule operator for prefill.

    This is a unified path for both fresh and non-fresh prefills, replacing
    the PTO megakernel (fresh only) and Triton chunk_gated_delta_rule (non-fresh).
    Matches the vllm-ascend PR #12607 implementation.

    Args:
        query: (1, T, Nk, Dk) - head_first=False
        key: (1, T, Nk, Dk)
        value: (1, T, Nv, Dv)
        g: (1, T, Nv) - log-gate values (fp32)
        beta: (1, T, Nv)
        ssm_state: full state cache tensor
        state_indices: indices into ssm_state for this batch
        has_initial_state: (B,) bool tensor indicating which sequences have non-zero initial state
        cu_seqlens: (B+1,) cumulative sequence lengths [0, s1, s1+s2, ...]

    Returns:
        out: (1, T, Nv, Dv) bf16
        final_state: (B, Nv, Dk, Dv) - transposed back to ssm_state layout
    """
    # Extract initial_state from ssm_state cache: (B, Nv, Dk, Dv) -> (B, Nv, Dv, Dk)
    initial_state = ssm_state[state_indices].transpose(-1, -2).contiguous()

    # Clear states for fresh sequences (equivalent to PR's clear_ssm_states)
    initial_state[~has_initial_state, ...] = 0

    # Convert to TND layout (drop batch dim, squeeze from [1, T, N, D] to [T, N, D])
    # The aclnn op does NOT l2-normalize q/k internally, so do it here
    q_tnd = l2norm_fwd(query.squeeze(0))   # (T, Nk, Dk)
    k_tnd = l2norm_fwd(key.squeeze(0))     # (T, Nk, Dk)
    v_tnd = value.squeeze(0)               # (T, Nv, Dv)
    beta_tnd = beta.squeeze(0)             # (T, Nv)
    g_tnd = g.squeeze(0)                   # (T, Nv), fp32 log-gate

    # Convert cu_seqlens [0, s1, s1+s2, ...] to per-sequence lengths (B,) int32
    actual_seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32).contiguous()

    # Compute scale
    scale = q_tnd.shape[-1] ** -0.5

    # Call the aclnn operator
    out_tnd, final_state = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        q_tnd,
        k_tnd,
        v_tnd,
        beta_tnd,
        initial_state,
        actual_seq_lengths,
        g_tnd,
        scale,
    )

    # Restore batch dim: (T, Nv, Dv) -> (1, T, Nv, Dv)
    out = out_tnd.unsqueeze(0)

    # Transpose final_state back to ssm_state layout: (B, Nv, Dv, Dk) -> (B, Nv, Dk, Dv)
    final_state = final_state.transpose(-1, -2).contiguous()

    return out, final_state


def _patch_mamba_cache_dense_layout() -> None:
    """Rebuild mamba KV-cache views as dense per-state tensors.

    Upstream vLLM lays out the conv/ssm states of a mamba block interleaved
    inside one (padded) page, so the per-state views have a first-dim stride
    larger than the dense block size. The AscendC kernels address the state
    cache assuming dense per-state tensors (verified: in-place state updates
    land at wrong offsets with the paged views), while the Triton kernels
    take explicit strides and work with either layout. Wrap
    ``ModelRunnerFL._reshape_kv_cache_tensors`` so that, after the original
    reshape, every MambaSpec layer's state views are rebuilt as dense,
    grouped per-state views over the same raw storage. This is semantically
    transparent for all other consumers (they index the views by block id).
    """
    from vllm.utils.torch_utils import get_dtype_size

    from vllm_fl.worker.model_runner import ModelRunnerFL

    orig_reshape = ModelRunnerFL._reshape_kv_cache_tensors

    def _reshape_kv_cache_tensors_dense_mamba(
        self,
        kv_cache_config,
        kv_cache_raw_tensors,
        kernel_block_sizes,
    ):
        kv_caches = orig_reshape(
            self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
        )
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, MambaSpec):
                continue
            if group.kv_cache_group_id == len(kernel_block_sizes):
                continue
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                state_tensors = []
                storage_offset_bytes = 0
                raw_u8 = raw_tensor.view(torch.uint8)
                for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                    dtype_size = get_dtype_size(dtype)
                    num_bytes = num_blocks * math.prod(shape) * dtype_size
                    tensor = (
                        raw_u8[storage_offset_bytes : storage_offset_bytes + num_bytes]
                        .view(dtype)
                        .view(num_blocks, *shape)
                    )
                    state_tensors.append(tensor)
                    storage_offset_bytes += num_bytes
                kv_caches[layer_name] = state_tensors
        return kv_caches

    ModelRunnerFL._reshape_kv_cache_tensors = _reshape_kv_cache_tensors_dense_mamba
    logger.info("Patched mamba KV-cache views to dense per-state layout for AscendC GDN ops")


class AscendCGatedDeltaNet(Qwen3NextGatedDeltaNet):
    """GDN layer backed by the AscendC fused kernels (eager mode)."""

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        conv_state_shape, temporal_state_shape = (
            MambaStateShapeCalculator.gated_delta_net_state_shape(
                self.tp_size,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.conv_kernel_size,
                self.num_spec,
            )
        )
        # The AscendC recurrent_gated_delta_rule kernel keeps the ssm state
        # in (Hv, Dv, Dk) layout; vLLM 0.13 uses (Hv, Dk, Dv).
        num_v_heads, head_k_dim, head_v_dim = temporal_state_shape
        return conv_state_shape, (num_v_heads, head_v_dim, head_k_dim)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        # conv cache is already (slot, state_len, conv_dim): pass through.
        conv_state = self_kv_cache[0]
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        if os.environ.get("VLLM_FL_DISABLE_CONV1D_PREPACK", "0") == "1":
            # Feature off: upstream cached transpose. The AscendC kernel
            # expects (width, dim). The loader normally creates this
            # contiguous layout once; the fallback also covers non-standard
            # loading paths that bypass the parameter's weight_loader.
            conv_weights_t = getattr(self, "_ascendc_conv_weights_t", None)
            if conv_weights_t is None:
                _cache_conv1d_weight_transposed(self)
                conv_weights_t = self._ascendc_conv_weights_t
        else:
            if not getattr(self, "_fl_conv1d_weight_packed", False):
                # Pack the conv weight once (dim, 1, width) -> (width, 1, dim):
                # the AscendC kernel consumes (width, dim), and packing here
                # avoids materializing a transpose view on every forward
                # (backport of vllm-ascend PR #7555's post-load packing, done
                # lazily because the vLLM 0.13 loader has no GDN post-load hook).
                w = self.conv1d.weight
                self.conv1d.weight.data = (
                    w.squeeze(1).transpose(0, 1).contiguous().unsqueeze(1))
                self._fl_conv1d_weight_packed = True
            # After packing this view is already (width, dim) and contiguous.
            conv_weights_t = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )
        activation_mode = 1 if self.activation else 0

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            spec_num_rows = spec_query_start_loc.size(0) - 1
            mixed_qkv_spec_out = torch.empty_like(mixed_qkv_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                mixed_qkv_spec_out,
                mixed_qkv_spec,
                conv_weights_t,
                conv_state,
                self.conv1d.bias,
                spec_query_start_loc,
                spec_state_indices_tensor[:spec_num_rows],
                None,  # initial_state_mode
                num_accepted_tokens,
                activation_mode,
                PAD_SLOT_ID,
                1,  # run_mode: decode/speculative update
            )
            mixed_qkv_spec = mixed_qkv_spec_out

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            non_spec_num_rows = non_spec_query_start_loc.size(0) - 1
            mixed_qkv_non_spec_out = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                mixed_qkv_non_spec_out,
                mixed_qkv_non_spec,
                conv_weights_t,
                conv_state,
                self.conv1d.bias,
                non_spec_query_start_loc,
                non_spec_state_indices_tensor[:non_spec_num_rows],
                has_initial_state,  # initial_state_mode
                None,  # num_accepted_tokens
                activation_mode,
                PAD_SLOT_ID,
                0,  # run_mode: varlen prefill
            )
            mixed_qkv_non_spec = mixed_qkv_non_spec_out
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec_out = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                mixed_qkv_non_spec_out,
                mixed_qkv_non_spec,
                conv_weights_t,
                conv_state,
                self.conv1d.bias,
                non_spec_query_start_loc,
                non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                None,  # initial_state_mode
                None,  # num_accepted_tokens
                activation_mode,
                PAD_SLOT_ID,
                1,  # run_mode: decode update
            )
            mixed_qkv_non_spec = mixed_qkv_non_spec_out
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        # 2. Recurrent attention
        g, beta = torch.ops._C_ascend.npu_fused_gdn_gating(
            self.A_log, a, b, self.dt_bias.to(self.A_log.dtype)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            actual_seq_lengths = _build_actual_seq_lengths(
                spec_query_start_loc, attn_metadata.num_spec_decodes
            )
            query_spec = l2norm_fwd(query_spec)
            key_spec = l2norm_fwd(key_spec)
            # The AscendC kernel does not apply the q/k L2 norm in-kernel,
            # and writes the updated state back in place.
            core_attn_out_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_spec.squeeze(0),
                key=key_spec.squeeze(0),
                value=value_spec.squeeze(0),
                g=g_spec.squeeze(0),
                beta=beta_spec.squeeze(0),
                state=ssm_state,
                scale=key_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=spec_state_indices_tensor.flatten(),
                num_accepted_tokens=num_accepted_tokens.to(torch.int32),
            ).unsqueeze(0)
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            # Environment variable control for aclnn chunk_gated_delta_rule integration:
            # VLLM_FL_USE_ACLNN_CHUNK_GDN=0 (default): use PTO + Triton (current behavior)
            # VLLM_FL_USE_ACLNN_CHUNK_GDN=1: use aclnn to replace both PTO and Triton
            use_aclnn = int(os.environ.get("VLLM_FL_USE_ACLNN_CHUNK_GDN", "0"))

            if use_aclnn == 1:
                # Use the new aclnn chunk_gated_delta_rule operator for all prefills
                # (both fresh and non-fresh), matching vllm-ascend PR #12607.
                core_attn_out_non_spec, last_recurrent_state = _chunk_gated_delta_rule_aclnn(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                    ssm_state,
                    non_spec_state_indices_tensor,
                    has_initial_state,
                    non_spec_query_start_loc,
                )
            elif _pto_prefill_usable(attn_metadata):
                # Fresh prefill batch: the fused PTO megakernel runs all six
                # GDN stages in a single launch. The decision is made from
                # CPU-side metadata (no device sync).
                core_attn_out_non_spec, last_recurrent_state = _chunk_gdn_pto(
                    query_non_spec,
                    key_non_spec,
                    value_non_spec,
                    g_non_spec,
                    beta_non_spec,
                    non_spec_query_start_loc,
                    attn_metadata,
                )
            else:
                # Chunked prefill stays on the (Ascend Triton) chunk kernel, which
                # uses the FLA (Hv, Dk, Dv) state layout: transpose at the boundary.
                initial_state = (
                    ssm_state[non_spec_state_indices_tensor].transpose(-1, -2).contiguous()
                )
                initial_state[~has_initial_state, ...] = 0
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = _qwen3_next_lib.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = (
                last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
            )
        elif attn_metadata.num_decodes > 0:
            if _fused_decode_gdn_enabled():
                # One fused Triton launch: q/k L2 norm + delta-rule state
                # update (state kept in the (Hv, Dv, Dk) AscendC layout,
                # which the kernel accesses in its native v-major
                # orientation). The sigmoid gating above stays on the
                # AscendC npu_fused_gdn_gating op.
                core_attn_out_non_spec = fused_recurrent_delta_rule_update(
                    q=query_non_spec.contiguous(),
                    k=key_non_spec.contiguous(),
                    v=value_non_spec.contiguous(),
                    g=g_non_spec.squeeze(0).contiguous(),
                    beta=beta_non_spec.squeeze(0).contiguous(),
                    initial_state_source=ssm_state,
                    initial_state_indices=non_spec_state_indices_tensor[
                        : attn_metadata.num_decodes
                    ],
                    cu_seqlens=non_spec_query_start_loc,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                actual_seq_lengths = _build_actual_seq_lengths(
                    non_spec_query_start_loc, attn_metadata.num_decodes
                )
                query_non_spec = l2norm_fwd(query_non_spec)
                key_non_spec = l2norm_fwd(key_non_spec)
                core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                    query=query_non_spec.squeeze(0),
                    key=key_non_spec.squeeze(0),
                    value=value_non_spec.squeeze(0),
                    g=g_non_spec.squeeze(0),
                    beta=beta_non_spec.squeeze(0),
                    state=ssm_state,
                    scale=key_non_spec.shape[-1] ** -0.5,
                    actual_seq_lengths=actual_seq_lengths,
                    ssm_state_indices=non_spec_state_indices_tensor[
                        : attn_metadata.num_decodes
                    ],
                ).unsqueeze(0)
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


class AscendCGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm backed by the AscendC ``npu_gemma_rms_norm`` kernel."""

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, 1.0 + self.weight, None, self.variance_epsilon
            )
            return x, residual

        # npu_gemma_rms_norm implements the Gemma (1 + weight) convention
        # internally, so the raw weight is passed (same as vllm-ascend).
        x, _ = torch.ops._C_ascend.npu_gemma_rms_norm(
            x, self.weight, self.variance_epsilon
        )
        return x


class AscendCRMSNormGated(RMSNormGated):
    """RMSNormGated backed by the fused Triton ``layer_norm_fwd_1pass`` kernel.

    The upstream OOT fallback (``forward_native``) decomposes the gated RMS
    norm into a long chain of eager ops (silu/pow/mean/rsqrt/mul), which is
    a major decode-stage cost for GDN layers; the fused kernel runs the
    whole norm (+ SiLU gating) in one launch, same as vllm-ascend's
    ``AscendRMSNormGated``.
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z is None:
            return RMSNormGated.forward_native(self, x, z)
        return rmsnorm_gated_oot(
            x,
            self.weight,
            z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


def patch_qwen3_6_gdn() -> bool:
    """Apply the AscendC GDN patch for Qwen3.5/Qwen3.6.

    Returns True when the AscendC kernels were wired in; otherwise the
    upstream/Triton implementations are kept.
    """
    if not _ascendc_ops_available():
        return False

    _patch_gdn_conv_weight_loader()
    Qwen3NextGatedDeltaNet.get_state_shape = AscendCGatedDeltaNet.get_state_shape
    Qwen3NextGatedDeltaNet._forward_core = AscendCGatedDeltaNet._forward_core
    _patch_mamba_cache_dense_layout()
    GemmaRMSNorm.forward_oot = AscendCGemmaRMSNorm.forward_oot
    RMSNormGated.forward_oot = AscendCRMSNormGated.forward_oot
    if _pto_available():
        _patch_gdn_metadata_host_flags()
    logger.info(
        "Patched Qwen3NextGatedDeltaNet and GemmaRMSNorm/RMSNormGated for Ascend "
        "(AscendC causal_conv1d / fused_gdn_gating / recurrent_gated_delta_rule "
        "/ gemma_rms_norm, fused Triton delta-rule decode update: %s, "
        "PTO megakernel for fresh prefill: %s)",
        "on" if _fused_decode_gdn_enabled() else "off",
        "on" if _pto_available() else "off",
    )
    return True
