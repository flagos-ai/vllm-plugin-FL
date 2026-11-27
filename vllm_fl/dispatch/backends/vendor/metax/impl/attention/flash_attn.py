# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

# --------------------------------------------------------------
# Note: use Maca's merge_attn_states to get cuda kernel invoked
# --------------------------------------------------------------
from .ops.merge_attn_states import merge_attn_states
from .utils.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from .utils.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
        flash_attn_with_kvcache,  # used for prefill decode split
    )
from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    _batch_invariant_MODE as _bi_mode,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    split_decodes_and_prefills,  # used for prefill decode split
    reshape_attn_output_for_spec_decode,  # used for prefill decode split with mtp
    reshape_query_for_spec_decode,  # used for prefill decode split with mtp
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import AttentionSpec

# --------------------------------------------------------------
# Note: used for prefill decode split with mtp on maca
# --------------------------------------------------------------
from .mla.common import QueryLenSupport

# Fused RoPE + KV-cache write (R6/G1)
from .. import mx_glue as _mx_glue

logger = init_logger(__name__)


# ┌------------------------  Metax Modification -------------------------┐
# Sync-free prefill cu_seqlens.
#
# The MACA prefill path needs cu_seqlens_k, the exclusive prefix sum of the
# prefill rows' context lengths.  It used to be rebuilt inside forward(), i.e.
# once per decoder layer:
#
#     torch.tensor([0] + attn_metadata.prefill_seq_lens.tolist(),
#                  device=..., dtype=torch.int32).cumsum(0, dtype=torch.int32)
#
# `.tolist()` on a device tensor is a blocking Device->Pageable copy: it drains
# the stream, so the host could never run ahead and every layer paid a full
# launch latency.  A 16k mixed step has 42 of them.  Measured with the torch
# profiler (logs/prof/metax_p16k, 60 steps): the 42 windows from the D2H copy
# to the prefill attention kernel span 29.5 ms per step but contain only
# 2.2 ms of GPU work, i.e. 27.3 ms of the 33.0 ms total GPU idle in a 214.9 ms
# step (82.7%).
#
# All the inputs are known on the host before the step starts: the model runner
# keeps seq_lens on the CPU (CommonAttentionMetadata._seq_lens_cpu, fed from
# ModelRunnerFL.optimistic_seq_lens_cpu, which is pinned) and also publishes
# query_start_loc_cpu.  So build both prefill arrays once per step in the
# metadata builder and push them to the device with a single async H2D copy.
# This mirrors what the MACA MLA builder already does for its chunked-context
# cu_seq_lens (impl/attention/mla/common.py:826-831).
#
# Set _PREFILL_META_ON_HOST = False to fall back to a device-side cumsum once
# per step; that is still sync-free, it just costs two extra kernel launches.
_PREFILL_META_ON_HOST = True

# Host staging slots for the per-step H2D copy.  A slot is only rewritten once
# the copy that read it has completed (guarded by a per-slot event), so two
# slots cover one step of host run-ahead.
_PREFILL_META_SLOTS = 2


def _cu_seqlens_from_lens(lens: torch.Tensor) -> torch.Tensor:
    """``[0, cumsum(lens)]`` as int32 on ``lens``' device, with no H<->D sync.

    Exact replacement for
    ``torch.tensor([0] + lens.tolist(), device=lens.device,
    dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)``: same values, same
    dtype, same device, but computed entirely on the device.
    """
    return torch.nn.functional.pad(
        torch.cumsum(lens, dim=0, dtype=torch.int32), (1, 0)
    )
# └------------------------- Metax Modification -------------------------┘


# ┌------------------------  Metax Modification -------------------------┐
# Calling-side attention policy (ops/mx_prefill_attn.py; measured with
# work/R4/T4-mx-prefill-attn/bench_t4.py, policy P1).  Same vendor kernels,
# called through their C entry points with out= (no slice copy), plus:
#   * one prefill row with k_len >= 4096: gather its pages into a contiguous
#     buffer and use the non-paged varlen kernel;
#   * 2-4 prefill rows with max k_len >= 10241: one KV-split fwd_kvcache per
#     row over a block table cut to the pages in use.
# Prefill rows only ever run eagerly (>512-token steps have no graph; smaller
# mixed steps run attention outside the PIECEWISE graphs).  Decode rows keep
# the exact vendor call (heuristic splits, full-width table) and only gain
# out=, which is also what FULL decode graphs capture.
# The metadata builder self-tests every call form once at start-up
# (_mxa.setup); if anything fails the original calls below are used.
# A/B experiments only: VLLM_FL_MX_ATTN=0 restores the original calls,
# VLLM_FL_MX_ATTN=prefill keeps the original decode-row call (slice copy).
# Mixed-step overlap (R5/P1, ops/mx_prefill_attn.py "mixed-step overlap"):
# in a step with both decode and prefill rows, the decode rows run on a side
# stream next to the prefill kernel (same vendor kernels and arguments; the
# outputs are bitwise identical).  A/B experiments only: VLLM_FL_MX_OVERLAP=0
# runs them one after the other as before.
import os as _os

from .ops import mx_prefill_attn as _mxa

_MXA_MODE = _os.environ.get("VLLM_FL_MX_ATTN", "1")
_MXA_REQUESTED = _MXA_MODE != "0"
_MXA_DECODE = _MXA_MODE != "prefill"
# Overlap ON by default.  It used to crash the engine under real concurrent
# benchmark load (EngineCore died, no Python traceback) because the side-stream
# decode tensors were not record_stream()'d, so the caching allocator recycled
# their blocks under load while the side stream was still running -- a
# use-after-free.  That is now fixed (record_stream in the decode() closure
# below), so the overlap is safe again; VLLM_FL_MX_OVERLAP=0 restores the
# sequential path (bitwise identical) if ever needed.
_MXA_OVERLAP = _os.environ.get("VLLM_FL_MX_OVERLAP", "1") != "0"
# └------------------------- Metax Modification -------------------------┘


# ┌------------------------  Metax Modification -------------------------┐
# Own native decode attention (ops/mx_dec_native.py, R6/N1 + R7/N4; measured
# with work/R6/N1-mx-dec-attn-native/bench_n1.py and
# work/R7/N4-mx-graph-dec-attn/bench_n4.py).  The q_len-1 decode rows run a MACA
# C++ kernel of our own design on the MMA unit: K tiles straight into MMA
# fragments, the P.V operand transposed inside the lane, fp32 online softmax,
# split segments merged by the last-arriving CTA (no combine launch).  The work
# schedule is rebuilt inside the kernel from seq_lens in device memory, in two
# flavours:
#   * overlapped mixed step (P1 side stream, next to the prefill kernel):
#     a persistent grid CLAIMS (row, kv head, chunk) items, so CTAs take the APs
#     as the prefill kernel frees them (R6/N1, unchanged);
#   * decode-only step (this is also the path the FULL decode graphs capture):
#     a STATIC stream-K partition of tile space, equal to +-1 tile per CTA, no
#     atomic claiming and <= 2 partials per CTA (R7/N4).  Nothing in the launch
#     configuration depends on the context, so one capture is valid for every
#     replay, and the per-row context, the page ids, the split and the partial
#     count are all read from device memory by the kernel itself.
# The vendor call stays the fallback: VLLM_FL_MX_DEC_NATIVE=0, an unsupported
# shape / cache layout, a row count (or, outside a graph, a context) outside the
# policy, or a failed build / start-up self-test (which captures and replays the
# kernel in a CUDA graph).  Nothing here changes the cudagraph mode or sizes.
from .ops import mx_dec_native as _mxn
# └------------------------- Metax Modification -------------------------┘


# ┌------------------------  Metax Modification -------------------------┐
# Self-developed INT8 KV cache (ops/mx_kv_int8.py, R8/Q1).  DEFAULT OFF
# (VLLM_FL_MX_INT8_KV=1 to request it).  The engine keeps allocating the same
# bf16 pages and the same num_blocks; when the feature latches on, our own
# kernels reinterpret each half page as int8 data plus one fp32 scale per
# (token, kv head) -- 528 B per token per layer instead of 1024 -- and nothing
# else is allowed to see those pages: the quantising write replaces
# reshape_and_cache_flash, the decode read is the native kernel's Q8 = 1
# configs, and the prefill read dequantises inside the gather the T4 path
# already does, so the vendor prefill kernel still sees ordinary bf16.
# The format is decided here, never by --kv-cache-dtype or any other serve
# flag (cache_dtype must stay "auto"/bfloat16 or the feature refuses to start).
from .ops import mx_kv_int8 as _kv8
# └------------------------- Metax Modification -------------------------┘


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class MacaFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        raise NotImplementedError(
            "FP8 dtype is not supported for FlashAttention on Maca."
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # /------------------------  Metax Modification -------------------------\
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    decode_query_start_loc: torch.Tensor
    decode_max_seq_len: int
    decode_seq_lens: torch.Tensor
    decode_block_table: torch.Tensor

    num_prefills: int
    num_prefill_tokens: int
    prefill_query_start_loc: torch.Tensor
    prefill_max_seq_len: int
    prefill_seq_lens: torch.Tensor
    prefill_block_table: torch.Tensor
    # \------------------------- Metax Modification -------------------------/

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # /------------------------  Metax Modification -------------------------\
    # cu_seqlens_k for the prefill rows ([0, cumsum(prefill_seq_lens)], int32,
    # length num_prefills + 1), built once per step by the metadata builder so
    # that forward() does not have to rebuild it per layer out of a device
    # tensor.  None when the batch has no prefill rows, or when some other
    # builder produced this metadata; forward() then derives it itself, still
    # without any H<->D sync.
    prefill_cu_seqlens_k: torch.Tensor | None = None
    # Attention policy (_mxa): query / kv lengths of the prefill rows as host
    # ints (from the CPU copies the builder already reads), and their block
    # table cut to the pages in use when they go out as per-row KV-split
    # launches.  None when the policy is off or there are no prefill rows.
    prefill_q_lens_host: tuple | None = None
    prefill_k_lens_host: tuple | None = None
    prefill_bt_narrow: torch.Tensor | None = None
    # Mixed-step overlap: run the decode rows next to the prefill rows
    # (decided once per step from the host lengths; _mxa.overlap_wanted).
    mx_overlap: bool = False
    # R9/F, INT8 KV only: this step's decode rows carry more than one query token
    # each (a prompt of 2..128 tokens is classified as a decode row, see
    # reorder_batch_threshold below), so the native int8 kernel -- one query token
    # per row -- cannot serve them and they go through the gather reader instead
    # (_kv8.decode_varlen).  The three tensors/tuples below are that reader's
    # arguments for the decode region, built here once per step exactly as the
    # prefill ones are, and only for the steps that need them.
    kv8_dec_varlen: bool = False
    kv8_dec_cu_seqlens_k: torch.Tensor | None = None
    kv8_dec_q_lens_host: tuple | None = None
    kv8_dec_k_lens_host: tuple | None = None
    # Token positions of this step ([>= num_actual_tokens] int64, the model
    # runner's persistent buffer).  The fused RoPE + KV write (mx_glue) rotates
    # q / k inside forward() at these positions.
    positions: torch.Tensor | None = None
    # \------------------------- Metax Modification -------------------------/

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # /------------------------  Metax Modification -------------------------\
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    # Defines the level of query length support for this backend.
    # - SINGLE_ONLY: Only single-token queries (no spec decode support)
    # - UNIFORM: Supports uniform multi-token queries (spec decode with uniform lengths)
    # - VARLEN: Supports variable-length queries (spec decode with mixed lengths)
    # If set to UNIFORM or VARLEN, this will increase `reorder_batch_threshold` when
    # speculative decoding is enabled.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    # The threshold for reordering the batch into decode and prefill requests.
    # If > 1, the batch will be reordered such that requests with
    # query length <= threshold are classified as decode requests.
    # Use `query_len_support` (above) to set this automatically
    # when speculative decoding is enabled.
    reorder_batch_threshold: int = 128  # process small prefills with decode pathway
    # \------------------------- Metax Modification -------------------------/

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        # /------------------------  Metax Modification -------------------------\
        supports_spec_decode = self.query_len_support != QueryLenSupport.SINGLE_ONLY
        self._init_reorder_batch_threshold(
            self.reorder_batch_threshold, supports_spec_decode
        )

        # Validate consistency between query_len_support and reorder_batch_threshold
        if self.query_len_support == QueryLenSupport.SINGLE_ONLY:
            assert self.reorder_batch_threshold == 1, (
                f"reorder_batch_threshold must be 1 when query_len_support is "
                f"SINGLE_ONLY, got {self.reorder_batch_threshold}"
            )
        # \------------------------- Metax Modification -------------------------/

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

        # /------------------------  Metax Modification -------------------------\
        # Buffers for the sync-free prefill cu_seqlens (see
        # _PREFILL_META_ON_HOST at the top of this file).  One device buffer
        # holds [cu_seqlens_k | cu_seqlens_q] for the prefill rows, each
        # max_num_seqs + 1 long, refreshed by a single async H2D copy per step.
        self._pf_stride = vllm_config.scheduler_config.max_num_seqs + 1
        self._pf_gpu = torch.zeros(
            2 * self._pf_stride, dtype=torch.int32, device=self.device
        )
        self._pf_host: list[torch.Tensor] = []
        self._pf_event: list[torch.cuda.Event] = []
        self._pf_slot = 0
        if _PREFILL_META_ON_HOST:
            try:
                self._pf_host = [
                    torch.zeros(
                        2 * self._pf_stride, dtype=torch.int32, pin_memory=True
                    )
                    for _ in range(_PREFILL_META_SLOTS)
                ]
            except RuntimeError:
                # No pinned memory on this device; use the device-side cumsum.
                logger.warning_once(
                    "MacaFlashAttention: pinned host memory unavailable, "
                    "building the prefill cu_seqlens with a device-side "
                    "cumsum instead."
                )
                self._pf_host = []
            else:
                # One reusable event per slot.  Synchronizing an event that has
                # never been recorded returns immediately, so no first-use
                # special case is needed.
                self._pf_event = [
                    torch.cuda.Event() for _ in range(_PREFILL_META_SLOTS)
                ]

        # Attention policy: self-test the call forms once per process (also
        # compiles the gather kernel before the first served step).
        if _MXA_REQUESTED and is_flash_attn_varlen_func_available():
            status = _mxa.setup(
                device=self.device,
                num_heads_q=self.num_heads_q,
                num_heads_kv=self.num_heads_kv,
                head_size=self.headdim,
                block_size=self.block_size,
                dtype=self.kv_cache_dtype,
                bt_width=cdiv(self.model_config.max_model_len, self.block_size),
                ref_varlen=flash_attn_varlen_func,
                ref_kvcache=flash_attn_with_kvcache,
            )
            if _mxa.ENABLED:
                logger.info_once("MacaFlashAttention: attention policy %s", status)
            else:
                logger.warning_once(
                    "MacaFlashAttention: attention policy %s; using the "
                    "original vendor calls",
                    status,
                )
        # Own native decode kernel: library, workspaces, self-test (once).  It
        # serves the decode-only steps (eager and inside the FULL decode graphs)
        # and, when the mixed-step overlap is on, the decode rows of an
        # overlapped mixed step, so it is set up whenever decode rows may reach
        # it at all.
        if _mxn.REQUESTED and _MXA_DECODE:
            mxn_status = _mxn.setup(
                device=self.device,
                num_heads_q=self.num_heads_q,
                num_heads_kv=self.num_heads_kv,
                head_size=self.headdim,
                block_size=self.block_size,
                dtype=self.kv_cache_dtype,
                max_rows=vllm_config.scheduler_config.max_num_seqs,
                ref_kvcache=(
                    flash_attn_with_kvcache
                    if is_flash_attn_varlen_func_available()
                    else None
                ),
            )
            if _mxn.ENABLED:
                logger.info_once(
                    "MacaFlashAttention: native decode kernel %s", mxn_status
                )
            else:
                logger.warning_once(
                    "MacaFlashAttention: native decode kernel %s; decode rows "
                    "use the vendor call",
                    mxn_status,
                )
        elif _mxn.REQUESTED:
            logger.info_once(
                "MacaFlashAttention: native decode kernel not set up "
                "(VLLM_FL_MX_ATTN=%s turns the decode path off)",
                _MXA_MODE,
            )
        # Self-developed INT8 KV cache (R8/Q1): latch the storage format ONCE,
        # here.  The builder is constructed before the first forward and before
        # CUDA-graph capture, so the cache can never end up holding a mix of
        # formats.  setup() runs the quantising write, both decode sites and
        # the prefill gather against torch references on a private little cache
        # and refuses the whole feature if anything is off or any gate fails;
        # a refusal leaves the deployed bf16 stack exactly as it was.  Default
        # off, and never keyed on a serve flag -- cache_dtype must be the
        # engine's own "auto"/bfloat16 or the feature declines to start.
        if _kv8.REQUESTED:
            kv8_status = _kv8.setup(
                self.device,
                mxn=_mxn,
                mxa=_mxa,
                num_heads_q=self.num_heads_q,
                num_heads_kv=self.num_heads_kv,
                head_size=self.headdim,
                block_size=self.block_size,
                dtype=self.model_config.dtype,
                max_rows=vllm_config.scheduler_config.max_num_seqs,
                max_model_len=self.model_config.max_model_len,
                cache_dtype_str=self.cache_config.cache_dtype,
                dcp_world_size=self.dcp_world_size,
                # R9/F: the resolved state of this model and this process, not a
                # literal.  Both gates used to be passed as False, so neither could
                # ever fire and both were assumptions rather than checks.
                kv_sharing=self._any_kv_sharing(vllm_config),
                cascade_enabled=self._cascade_possible(),
                # R10/F: with the glue's RoPE fusion installed, this impl owns the
                # rotation, and the INT8 path needs it WITHOUT the cache write.  Only
                # our own kernel can do that inside a FULL decode graph, so without it
                # the feature must decline here rather than abort the capture.
                rope_capture_safe=_mx_glue.rope_capture_safe(),
            )
            if _kv8.ENABLED:
                logger.info_once("MacaFlashAttention: INT8 KV cache %s", kv8_status)
            else:
                logger.warning_once(
                    "MacaFlashAttention: INT8 KV cache %s; the cache stays bf16 "
                    "and every vendor path is unchanged",
                    kv8_status,
                )
        # \------------------------- Metax Modification -------------------------/

    # /------------------------  Metax Modification -------------------------\
    @staticmethod
    def _any_kv_sharing(vllm_config: VllmConfig) -> bool:
        """R9/F: does ANY attention layer of this model read another layer's KV cache?

        The INT8 cache is decided per process but taken per layer (FlashAttentionImpl
        ._kv8_ok excludes a sharing layer), and ops/mx_prefill_attn.py used to send
        every prefill row to the int8 gather on the global flag alone -- so on a model
        with KV sharing a bf16-cached layer could have reached the int8 reader.  Both
        ends are fixed: the gather is keyed per layer now, and the feature refuses to
        start at all when any layer shares, because the owner decides the format.
        Returns True when it cannot be determined, so the gate errs towards refusing.
        """
        try:
            layers = get_layers_from_vllm_config(vllm_config, Attention)
        except Exception:  # noqa: BLE001
            return True
        return any(
            getattr(layer, "kv_sharing_target_layer_name", None) is not None
            for layer in layers.values()
        )

    def _cascade_possible(self) -> bool:
        """R9/F: can a cascade step actually reach a vendor reader in THIS process?

        Cascade attention reads the paged cache with vendor kernels, which cannot be
        int8.  The answer is not the model config's switch: this class overrides
        use_cascade_attention() to return False for every step once the INT8 cache is
        requested (and the MetaX platform disables cascade anyway).  So ask the
        override itself instead of asserting a literal -- if some other class's method
        is in force, or it answers anything but False, the gate refuses.
        """
        try:
            return bool(type(self).use_cascade_attention(self))
        except Exception:  # noqa: BLE001
            return True

    def _build_prefill_cu_seqlens(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_decodes: int,
        num_reqs: int,
        num_prefills: int,
        prefill_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cu_seqlens_k and rebased cu_seqlens_q for the prefill rows.

        Returns two device int32 tensors of length ``num_prefills + 1``:

        * ``cu_seqlens_k == [0, cumsum(seq_lens[num_decodes:num_reqs])]``
        * ``cu_seqlens_q == query_start_loc[num_decodes:num_reqs + 1]
          - query_start_loc[num_decodes]``

        which is exactly what forward() used to rebuild per layer and what
        build() used to compute with a device-side subtraction.  Nothing here
        reads a device tensor on the host, so no H<->D sync is added.
        """
        seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        if self._pf_host and seq_lens_cpu is not None:
            # Host path: seq_lens and query_start_loc are already on the CPU.
            n = num_prefills
            stride = self._pf_stride
            slot = self._pf_slot
            event = self._pf_event[slot]
            # Wait for the H2D copy that last read this slot before
            # overwriting it.  ModelRunnerFL.synchronize_input_prep()
            # (worker/model_runner.py:3824-3836, wrapped around the
            # _build_attention_metadata call at :4161-4164) already guarantees
            # this under async scheduling; the event keeps the builder correct
            # on its own.  With two slots it is a completed event by the time
            # we get back to it, so this does not block.
            event.synchronize()
            host = self._pf_host[slot]
            # host[0] and host[stride] stay 0 from the initial torch.zeros.
            torch.cumsum(
                seq_lens_cpu[num_decodes:num_reqs],
                dim=0,
                out=host[1 : n + 1],
                dtype=torch.int32,
            )
            torch.sub(
                query_start_loc_cpu[num_decodes : num_reqs + 1],
                query_start_loc_cpu[num_decodes],
                out=host[stride : stride + n + 1],
            )
            self._pf_gpu.copy_(host, non_blocking=True)
            event.record()
            self._pf_slot = (slot + 1) % len(self._pf_host)
            return (
                self._pf_gpu[: n + 1],
                self._pf_gpu[stride : stride + n + 1],
            )

        # Device path: used when pinned memory is unavailable or when the
        # runner declares the GPU tensors authoritative (async spec decode
        # sets CommonAttentionMetadata._seq_lens_cpu to None,
        # worker/model_runner.py:2371-2374).  Still sync-free, and still once
        # per step instead of once per layer.
        cu_seqlens_k = _cu_seqlens_from_lens(prefill_seq_lens)
        cu_seqlens_q = (
            common_attn_metadata.query_start_loc[num_decodes : num_reqs + 1]
            - common_attn_metadata.query_start_loc[num_decodes]
        )
        return cu_seqlens_k, cu_seqlens_q

    # \------------------------- Metax Modification -------------------------/

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # /------------------------  Metax Modification -------------------------\
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
            )
        )

        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens
        assert num_decodes + num_prefills == num_reqs
        # \------------------------- Metax Modification -------------------------/

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if self.use_full_cuda_graph and num_actual_tokens <= self.max_cudagraph_size:
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        # /------------------------  Metax Modification -------------------------\
        # For handling prefill decode split
        if num_decodes > 0:
            decode_max_seq_len = int(
                common_attn_metadata.seq_lens_cpu[:num_decodes].max()
            )
            decode_query_start_loc = common_attn_metadata.query_start_loc[
                : num_decodes + 1
            ]
            decode_seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            decode_block_table_tensor = common_attn_metadata.block_table_tensor[
                :num_decodes
            ]
        else:
            decode_max_seq_len = 0
            decode_query_start_loc = None
            decode_seq_lens = None
            decode_block_table_tensor = None

        if num_prefills > 0:
            prefill_max_seq_len = int(
                common_attn_metadata.seq_lens_cpu[num_decodes:num_reqs].max()
            )
            prefill_seq_lens = common_attn_metadata.seq_lens[num_decodes:num_reqs]
            prefill_block_table_tensor = common_attn_metadata.block_table_tensor[
                num_decodes:num_reqs
            ]
            # cu_seqlens_q / cu_seqlens_k for the prefill rows, built once here
            # instead of once per layer in forward() (which forced a blocking
            # Device->Pageable copy per layer).
            prefill_cu_seqlens_k, prefill_query_start_loc = (
                self._build_prefill_cu_seqlens(
                    common_attn_metadata,
                    num_decodes,
                    num_reqs,
                    num_prefills,
                    prefill_seq_lens,
                )
            )
        else:
            prefill_max_seq_len = 0
            prefill_query_start_loc = None
            prefill_seq_lens = None
            prefill_block_table_tensor = None
            prefill_cu_seqlens_k = None

        prefill_q_lens_host = prefill_k_lens_host = prefill_bt_narrow = None
        mx_overlap = False
        if _mxa.ENABLED and num_prefills > 0:
            prefill_q_lens_host, prefill_k_lens_host = _mxa.host_lens(
                common_attn_metadata.query_start_loc_cpu,
                common_attn_metadata.seq_lens_cpu,
                num_decodes,
                num_reqs,
            )
            if _mxa.tail_rows_split(prefill_k_lens_host):
                prefill_bt_narrow = prefill_block_table_tensor[
                    :, : cdiv(prefill_max_seq_len, self.block_size)
                ].contiguous()
            mx_overlap = (
                _MXA_OVERLAP
                and _MXA_DECODE
                and num_decodes > 0
                and prefill_cu_seqlens_k is not None
                and _mxa.overlap_wanted(
                    prefill_q_lens_host, prefill_k_lens_host, num_decodes
                )
            )

        # R9/F: the INT8 cache's own per-step decisions.  Both are feature-driven and
        # both cost nothing on the steps the contest measures: at concurrency 64 over
        # 4k/16k prompts every decode row carries one token and every mixed step has
        # 64 of them, so kv8_dec_varlen stays False and side_ok() stays True.
        kv8_dec_varlen = False
        kv8_dec_cu_seqlens_k = None
        kv8_dec_q_lens_host = kv8_dec_k_lens_host = None
        if _kv8.ENABLED and num_decodes > 0:
            # One pure function of the step's shape (mx_kv_int8.step_policy): the
            # int8 decode kernel loses to the vendor bf16 one on the side stream
            # below SIDE_MIN_ROWS rows and the gather reader has no side-stream site
            # at all, so such a step runs prefill and decode in sequence instead,
            # with the main-stream cfg 42 that is the better int8 config at every
            # measured shape.
            kv8_dec_varlen, mx_overlap = _kv8.step_policy(
                num_decodes, num_decode_tokens, mx_overlap
            )
            if kv8_dec_varlen:
                # multi-token decode rows: the gather reader serves them (the native
                # int8 kernel reads one query token per row).  Same host lengths and
                # cu_seqlens the prefill rows get, over rows [0, num_decodes).
                kv8_dec_q_lens_host, kv8_dec_k_lens_host = _mxa.host_lens(
                    common_attn_metadata.query_start_loc_cpu,
                    common_attn_metadata.seq_lens_cpu,
                    0,
                    num_decodes,
                )
                kv8_dec_cu_seqlens_k = _cu_seqlens_from_lens(decode_seq_lens)
        # \------------------------- Metax Modification -------------------------/

        if _bi_mode:
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = MacaFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            dcp_context_kv_lens = seq_lens - query_kv_lens

            dcp_context_kv_lens = get_dcp_local_seq_lens(
                dcp_context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            # /------------------------  Metax Modification -------------------------\
            # For handling prefill decode split
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            decode_query_start_loc=decode_query_start_loc,
            decode_max_seq_len=decode_max_seq_len,
            decode_seq_lens=decode_seq_lens,
            decode_block_table=decode_block_table_tensor,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_max_seq_len=prefill_max_seq_len,
            prefill_seq_lens=prefill_seq_lens,
            prefill_block_table=prefill_block_table_tensor,
            prefill_cu_seqlens_k=prefill_cu_seqlens_k,
            prefill_q_lens_host=prefill_q_lens_host,
            prefill_k_lens_host=prefill_k_lens_host,
            prefill_bt_narrow=prefill_bt_narrow,
            mx_overlap=mx_overlap,
            kv8_dec_varlen=kv8_dec_varlen,
            kv8_dec_cu_seqlens_k=kv8_dec_cu_seqlens_k,
            kv8_dec_q_lens_host=kv8_dec_q_lens_host,
            kv8_dec_k_lens_host=kv8_dec_k_lens_host,
            positions=common_attn_metadata.positions,
            # \------------------------- Metax Modification -------------------------/
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # ┌------------------------  Metax Modification -------------------------┐
        # R8/Q1: a cascade step would read the paged cache with vendor kernels,
        # which cannot be int8.  Keyed on REQUESTED rather than ENABLED so the
        # answer is the same for every step of the process, including the ones
        # the builder asks about before setup() has latched.
        if _kv8.REQUESTED:
            return False
        # └------------------------- Metax Modification -------------------------┘
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = _bi_mode

        if get_kv_quant_mode(self.kv_cache_dtype) != KVQuantMode.NONE and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        # ┌------------------------  Metax Modification -------------------------┐
        # Layers the attention policy (_mxa) may serve: exactly the vendor call
        # forms it was measured and self-tested with (no sinks / alibi /
        # window / softcap, unquantised cache, the vendor's GQA decode layout).
        self._mxa_ok = (
            _MXA_REQUESTED
            and attn_type == AttentionType.DECODER
            and self.sinks is None
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and not self.logits_soft_cap
            and get_kv_quant_mode(self.kv_cache_dtype) == KVQuantMode.NONE
            and not self.kv_cache_dtype.startswith("fp8")
            and head_size % 8 == 0
            and num_heads % num_kv_heads == 0
        )
        # Layers the native decode kernel may serve: the same plain call form,
        # plus the shapes the library is built for; the cache layout is checked
        # once on the first decode call.
        self._mxn_ok = (
            _mxn.REQUESTED
            and attn_type == AttentionType.DECODER
            and self.sinks is None
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and not self.logits_soft_cap
            and get_kv_quant_mode(self.kv_cache_dtype) == KVQuantMode.NONE
            and not self.kv_cache_dtype.startswith("fp8")
            and _mxn.supports(num_heads, num_kv_heads, head_size)
        )
        self._mxn_layout_ok: bool | None = None
        # One log line per call site (False: decode-only step, True: the decode
        # rows of an overlapped mixed step).
        self._mxn_logged = {False: False, True: False}
        # Layers the INT8 KV cache (R8/Q1) may serve.  The native decode kernel
        # is the ONLY reader of the int8 pages, so a layer the kernel cannot
        # serve cannot hold them either; a layer that shares another layer's
        # cache is out because the owner decides the format.  _kv8.setup()
        # re-checks the same conditions globally before it latches the feature,
        # so this flag only says which layer takes the int8 route at run time.
        self._kv8_ok = (
            _kv8.REQUESTED
            and self._mxn_ok
            and kv_sharing_target_layer_name is None
            and getattr(self, "dcp_world_size", 1) == 1
        )
        self._kv8_logged = False
        # Fused RoPE + KV-cache write (R6/G1, impl/mx_glue.py): the Llama
        # attention patch sets mx_rope to the layer's rotary embedding when it
        # hands this impl UNROTATED q / k; forward() then rotates both and
        # writes the cache in one FlagGems kernel instead of
        # reshape_and_cache_flash.
        self.mx_rope = None
        # Not with decode context parallelism: _forward_with_dcp attends the
        # step's own (here unrotated) key / value directly.
        #
        # R9/F: this used to carry "and not _kv8.REQUESTED", because the glue
        # kernel writes bf16 KV and the quantising fused variant of it is not
        # built.  That was a trap: the fusion is decided HERE, at model build
        # time, while _kv8.ENABLED is only latched later by the metadata
        # builder's setup(), so an INT8 self-test that then refused left the
        # RoPE fusion off for the whole process -- a silent <= 0.74 s per 16k
        # run (N6 measured 30.2 vs 47.1 us/layer on a 2048-token step) with the
        # feature not even active.  The conflict is resolved at the only place
        # that knows the answer, forward(): with the INT8 cache live the fused
        # call is replaced by _mx_glue.rope_inplace() plus our own quantising
        # write, which is the same pair of kernels the R8/Q1 path measured.  So
        # the fusion is now installed whenever it is possible and the int8
        # decision costs nothing when it refuses.  The rest of the N6 glue
        # (add_rms_norm, SiLU*mul, GLU GEMM) is untouched either way.
        self.mx_rope_supported = (
            attn_type == AttentionType.DECODER
            and kv_sharing_target_layer_name is None
            and get_kv_quant_mode(self.kv_cache_dtype) == KVQuantMode.NONE
            and not self.kv_cache_dtype.startswith("fp8")
            and getattr(self, "dcp_world_size", 1) == 1
        )
        # └------------------------- Metax Modification -------------------------┘

    def supports_quant_query_input(self) -> bool:
        return False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        kv8_live = self._kv8_ok and _kv8.ENABLED
        if self.mx_rope is not None and not kv8_live:
            # ┌------------------------  Metax Modification -------------------------┐
            # q / k arrive unrotated (Llama attention patch, mx_glue): rotate
            # both and write k / v into the cache in one kernel; the rotated q
            # replaces query for every attention call below.
            query = _mx_glue.attn_rope_kv(
                self.mx_rope, query, key, value, key_cache, value_cache, attn_metadata
            )
            # └------------------------- Metax Modification -------------------------┘
        elif (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # ┌------------------------  Metax Modification -------------------------┐
            # R9/F: q / k still arrive unrotated when the Llama patch installed the
            # fusion, but that kernel writes bf16 KV and this layer's pages are int8,
            # so the rotation runs on its own and the quantising write below takes
            # the cache part.  Same two kernels the R8/Q1 path measured; the point of
            # doing it here is that the fusion no longer has to be refused at model
            # build time just because the INT8 cache was requested.
            if self.mx_rope is not None:
                query, key = _mx_glue.rope_inplace(
                    self.mx_rope, query, key, attn_metadata
                )
            # └------------------------- Metax Modification -------------------------┘
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            # ┌------------------------  Metax Modification -------------------------┐
            # R8/Q1: our own quantising write into the same pages.  Same
            # contract as reshape_and_cache_flash (token count from
            # slot_mapping, slots < 0 skipped, no allocation, no host sync), so
            # a FULL decode graph captures it exactly as it captured the vendor
            # op.  No fallback: once the cache holds int8 the vendor writer
            # would corrupt it, which is why the format is latched at setup.
            if kv8_live:
                _kv8.write_kv_int8(
                    key,
                    value,
                    _kv8.view_of_halves(key_cache, value_cache),
                    attn_metadata.slot_mapping,
                )
            else:
                # └----------------------- Metax Modification ----------------------┘
                reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        if self.kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            dtype = MacaFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                # ┌------------------------  Metax Modification -------------------------┐
                # For handling prefill decode split
                num_decode_tokens = attn_metadata.num_decode_tokens
                mxa = self._mxa_ok and _mxa.ENABLED
                if (
                    mxa
                    and attn_metadata.mx_overlap
                    and attn_metadata.causal is True
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    # Mixed-step overlap: the builder set mx_overlap only with
                    # decode rows, prefill rows, host lengths and cu_seqlens_k.
                    # Same two calls as below; the decode rows run on a side
                    # stream next to the prefill kernel and the current
                    # stream waits for both before returning.
                    self._mixed_overlapped(
                        query, key_cache, value_cache, output, attn_metadata
                    )
                    return output
                if attn_metadata.num_prefills > 0:
                    # Built once per step by the metadata builder.  Rebuilding
                    # it here used to cost a blocking Device->Pageable copy per
                    # layer, which drained the stream and stopped the host from
                    # running ahead (see _PREFILL_META_ON_HOST at the top of
                    # this file).
                    cu_prefix_kv_lens = attn_metadata.prefill_cu_seqlens_k
                    if cu_prefix_kv_lens is None:
                        cu_prefix_kv_lens = _cu_seqlens_from_lens(
                            attn_metadata.prefill_seq_lens
                        )
                    if (
                        mxa
                        and attn_metadata.prefill_q_lens_host is not None
                        and attn_metadata.causal is True
                        and not torch.cuda.is_current_stream_capturing()
                    ):
                        _mxa.prefill_attention(
                            query[num_decode_tokens:num_actual_tokens],
                            key_cache,
                            value_cache,
                            output[num_decode_tokens:num_actual_tokens],
                            attn_metadata.prefill_block_table,
                            attn_metadata.prefill_query_start_loc,
                            cu_prefix_kv_lens,
                            attn_metadata.prefill_seq_lens,
                            attn_metadata.prefill_q_lens_host,
                            attn_metadata.prefill_k_lens_host,
                            self.scale,
                            attn_metadata.prefill_bt_narrow,
                            kv8=self._kv8_ok and _kv8.ENABLED,
                        )
                    elif self._kv8_ok and _kv8.ENABLED:
                        # ┌---------------  Metax Modification ---------------┐
                        # R8/Q1: the vendor paged kernel below would read the
                        # int8 pages as bf16.  Every prefill step must reach
                        # _mxa.prefill_attention (which hands it to our gather),
                        # so landing here is a configuration error, not a
                        # fallback.  The conditions are step features the
                        # builder controls, and setup() refused the feature
                        # unless the attention policy is live.
                        raise RuntimeError(
                            "MacaFlashAttention: INT8 KV cache reached the "
                            "vendor paged prefill call (policy "
                            f"{'on' if mxa else 'off'}, host lens "
                            f"{attn_metadata.prefill_q_lens_host is not None}, "
                            f"causal {attn_metadata.causal}, capturing "
                            f"{torch.cuda.is_current_stream_capturing()})"
                        )
                        # └-------------- Metax Modification ----------------┘
                    else:
                        output[num_decode_tokens:num_actual_tokens] = (
                            flash_attn_varlen_func(
                                q=query[num_decode_tokens:num_actual_tokens],
                                k=key_cache,
                                v=value_cache,
                                block_table=attn_metadata.prefill_block_table,
                                cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                                cu_seqlens_k=cu_prefix_kv_lens,
                                max_seqlen_q=attn_metadata.max_query_len,
                                max_seqlen_k=attn_metadata.prefill_max_seq_len,
                                softmax_scale=self.scale,
                                causal=attn_metadata.causal,
                                window_size=self.sliding_window,
                                alibi_slopes=self.alibi_slopes,
                                softcap=self.logits_soft_cap,
                                s_aux=self.sinks,
                            )
                        )
                if attn_metadata.num_decodes > 0:
                    # ┌--------------  Metax Modification --------------┐
                    # R9/F: decode rows carrying more than one query token.
                    # The backend classifies every request with query_len <=
                    # reorder_batch_threshold (128) as a decode row, so a
                    # short prompt -- every MATH-500 prompt, and the 14-token
                    # smoke request that made this path raise in R8 -- lands
                    # here with q_len > 1.  The native int8 kernel reads one
                    # query token per row; these rows are prefill rows in
                    # everything a reader cares about, so they take the same
                    # gather reader, which is the only int8 reader for q_len
                    # > 1.  The builder decided this once for the step.
                    if self._kv8_ok and _kv8.ENABLED and attn_metadata.kv8_dec_varlen:
                        _kv8.decode_varlen(
                            _mxa,
                            query[:num_decode_tokens],
                            _kv8.view_of_halves(key_cache, value_cache),
                            output[:num_decode_tokens],
                            attn_metadata.decode_block_table,
                            attn_metadata.decode_query_start_loc,
                            attn_metadata.kv8_dec_cu_seqlens_k,
                            attn_metadata.decode_seq_lens,
                            attn_metadata.kv8_dec_q_lens_host,
                            attn_metadata.kv8_dec_k_lens_host,
                            self.scale,
                        )
                        return output
                    # └-------------- Metax Modification --------------┘
                    # Use flash_attn_with_kvcache for normal decoding.
                    decode_query = query[:num_decode_tokens]
                    decode_query = reshape_query_for_spec_decode(
                        decode_query, attn_metadata.num_decodes
                    )
                    # R7/N4: own stream-K kernel for the decode-only rows.  This
                    # is the call site the FULL decode graphs capture, so the
                    # decision below may only use features that are constant for
                    # a captured batch size (see mx_dec_native.wanted_decode).
                    if self._decode_rows_native(
                        query[:num_decode_tokens],
                        key_cache,
                        value_cache,
                        output[:num_decode_tokens],
                        attn_metadata,
                        side=False,
                    ):
                        return output
                    if mxa and _MXA_DECODE:
                        # Same vendor call, written straight into output
                        # (graph-safe: FULL decode graphs capture this too).
                        _mxa.decode_attention(
                            decode_query,
                            key_cache,
                            value_cache,
                            output[:num_decode_tokens],
                            attn_metadata.decode_block_table,
                            attn_metadata.decode_seq_lens,
                            self.scale,
                        )
                        return output
                    output_unreshape = flash_attn_with_kvcache(
                        q=decode_query,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        block_table=attn_metadata.decode_block_table,
                        cache_seqlens=attn_metadata.decode_seq_lens,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.sliding_window,
                        alibi_slopes=self.alibi_slopes,
                        softcap=self.logits_soft_cap,
                        s_aux=self.sinks,
                    )
                    output[:num_decode_tokens] = reshape_attn_output_for_spec_decode(
                        output_unreshape
                    )
                return output
            # └------------------------- Metax Modification -------------------------┘

        # ┌------------------------  Metax Modification -------------------------┐
        # R8/Q1: cascade attention reads the paged cache with vendor kernels, so
        # it cannot see int8 pages.  use_cascade_attention() already returns
        # False for the whole process once the feature is requested (and the
        # MetaX platform disables it anyway); this is the belt to that braces.
        if self._kv8_ok and _kv8.ENABLED:
            raise RuntimeError(
                "MacaFlashAttention: INT8 KV cache reached cascade attention, "
                "which has no int8 reader"
            )
        # └------------------------- Metax Modification -------------------------┘
        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    # ┌------------------------  Metax Modification -------------------------┐
    def _mixed_overlapped(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> None:
        nd = attn_metadata.num_decode_tokens
        na = attn_metadata.num_actual_tokens
        decode_query = reshape_query_for_spec_decode(
            query[:nd], attn_metadata.num_decodes
        )

        def prefill(go):
            _mxa.prefill_attention(
                query[nd:na],
                key_cache,
                value_cache,
                output[nd:na],
                attn_metadata.prefill_block_table,
                attn_metadata.prefill_query_start_loc,
                attn_metadata.prefill_cu_seqlens_k,
                attn_metadata.prefill_seq_lens,
                attn_metadata.prefill_q_lens_host,
                attn_metadata.prefill_k_lens_host,
                self.scale,
                attn_metadata.prefill_bt_narrow,
                go,
                kv8=self._kv8_ok and _kv8.ENABLED,
            )

        def decode():
            # Runs on the side stream next to the prefill kernel: the own native
            # kernel (side workspace) when this layer / shape allows it.
            # record_stream the tensors this side-stream kernel reads / writes so
            # the caching allocator does not recycle their blocks (which are
            # freed / reused on the main stream under real concurrent load)
            # before the side stream has finished with them.  Without this the
            # engine died with a silent native crash under the benchmark load;
            # the earlier "no record_stream() is needed" assumption was wrong.
            _side = getattr(_mxa, "_OVL", None)
            _side = _side.stream if _side is not None else None
            if _side is not None:
                for _t in (query, key_cache, value_cache, output):
                    try:
                        _t.record_stream(_side)
                    except Exception:  # noqa: BLE001
                        pass
            if self._decode_rows_native(
                query[:nd], key_cache, value_cache, output[:nd], attn_metadata,
                side=True,
            ):
                return
            _mxa.decode_attention(
                decode_query,
                key_cache,
                value_cache,
                output[:nd],
                attn_metadata.decode_block_table,
                attn_metadata.decode_seq_lens,
                self.scale,
            )

        _mxa.mixed_overlapped(prefill, decode)

    def _decode_rows_native(
        self,
        query_dec: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        out_dec: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        side: bool,
    ) -> bool:
        """Decode rows (q_len 1) through the native kernel if this layer, the cache
        layout and the row policy allow it.  side=True: the rows of an overlapped
        mixed step, queued on the side stream next to the prefill kernel (claimed
        items).  side=False: a decode-only step, eager or being captured into a
        FULL cuda graph (stream-K).  Returns False when the caller should issue the
        vendor call instead -- under capture that decision is frozen into the graph,
        which is why wanted_decode() only looks at capture-time constants."""
        # ┌------------------------  Metax Modification -------------------------┐
        # R8/Q1: with the INT8 cache this is not a choice.  The vendor kernel
        # cannot read int8 pages, so the native Q8 = 1 configs are the only
        # reader and a rejection is a configuration error, not a fallback --
        # hence the raise.  It cannot appear only at run time: setup() ran the
        # same call for both sites over eight contexts before the cache held
        # anything.  Both sites come through here, so the site choice (stream-K
        # CFG_DEC_I8 for the decode-only step that owns the GPU, atomic-claimed
        # CFG_SIDE_I8 for the side stream of an overlapped mixed step) is just the
        # `side` flag, exactly as on the bf16 path.  R9/R+G moved that pair from
        # the bf16-magic cfgs 28 / 38 to the fp16-magic cfgs 42 / 47 (same
        # schedules, 1.33-1.78x the vendor instead of 1.10-1.25x); the ids live
        # in mx_dec_native, this call site does not know the difference.
        if self._kv8_ok and _kv8.ENABLED:
            nd8 = attn_metadata.num_decodes
            if not _kv8.decode_supported(nd8, attn_metadata.num_decode_tokens):
                # R9/F: reachable only if the builder failed to mark the step, since
                # forward() routes a multi-token decode region to _kv8.decode_varlen
                # before it gets here.  Keep the raise as the backstop it is.
                raise RuntimeError(
                    "MacaFlashAttention: INT8 KV cache cannot serve decode rows "
                    f"with q_len > 1 ({attn_metadata.num_decode_tokens} tokens "
                    f"over {nd8} rows) on the native kernel, and the step was not "
                    "marked kv8_dec_varlen by the metadata builder"
                )
            if side and not _kv8.side_ok(nd8):
                # The builder refuses the overlap below this row count, so the rows
                # should have gone out on the main stream; never silently run the
                # losing side-stream config instead.
                raise RuntimeError(
                    "MacaFlashAttention: INT8 KV decode rows reached the side "
                    f"stream with {nd8} rows, below the measured threshold "
                    f"{_kv8.SIDE_MIN_ROWS}; the step should not have been overlapped"
                )
            if _mxn.decode_attention_int8(
                query_dec,
                _kv8.kv_of_halves(key_cache, value_cache),
                _kv8.view_of_halves(key_cache, value_cache),
                attn_metadata.decode_block_table,
                attn_metadata.decode_seq_lens,
                self.scale,
                side=side,
                out=out_dec,
            ) is None:
                raise RuntimeError(
                    "MacaFlashAttention: the native INT8 decode kernel rejected "
                    f"{nd8} rows at the {'side' if side else 'decode-only'} site "
                    f"(library checks {_mxn.rejects()}); the INT8 cache has no "
                    "vendor fallback"
                )
            if not self._kv8_logged:
                self._kv8_logged = True
                logger.info_once(
                    "MacaFlashAttention: INT8 KV decode rows on the native "
                    "kernel (cfg %d decode-only / %d side)",
                    _mxn.CFG_DEC_I8,
                    _mxn.CFG_SIDE_I8,
                )
            return True
        # └------------------------- Metax Modification -------------------------┘
        if not (self._mxn_ok and _mxn.ENABLED):
            return False
        nd = attn_metadata.num_decodes
        if attn_metadata.num_decode_tokens != nd:
            return False
        if side:
            if not _mxn.wanted(nd):
                return False
        elif not _mxn.wanted_decode(nd, attn_metadata.decode_max_seq_len):
            return False
        if self._mxn_layout_ok is None:
            self._mxn_layout_ok = _mxn.layout_ok(key_cache, value_cache)
        if not self._mxn_layout_ok:
            return False
        done = _mxn.decode_attention(
            query_dec,
            key_cache,
            value_cache,
            out_dec,
            attn_metadata.decode_block_table,
            attn_metadata.decode_seq_lens,
            self.scale,
            side=side,
        )
        if not self._mxn_logged[side]:
            # first call of this layer at this site (logged once per process)
            self._mxn_logged[side] = True
            where = "overlapped mixed step" if side else "decode-only step"
            if done:
                logger.info_once(
                    "MacaFlashAttention: %s: decode rows on the native kernel "
                    "(capturing=%s)",
                    where,
                    torch.cuda.is_current_stream_capturing(),
                )
            else:
                logger.warning_once(
                    "MacaFlashAttention: native decode kernel rejected a %s "
                    "(library checks %s); vendor call",
                    where,
                    str(sorted(_mxn._REJECTS.items())),
                )
        return done

    # └------------------------- Metax Modification -------------------------┘

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        # /------------------------  Metax Modification -------------------------\
        assert attn_metadata.dcp_context_kv_lens is not None
        cu_seqlens_k = _cu_seqlens_from_lens(attn_metadata.dcp_context_kv_lens)

        context_attn_out, context_lse, _ = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_attn_probs=True,
            # scheduler_metadata=attn_metadata.scheduler_metadata,
            # fa_version=self.vllm_flash_attn_version,
            # q_descale=q_descale,
            # k_descale=k_descale,
            # v_descale=v_descale,
            s_aux=self.sinks,
        )
        # \------------------------- Metax Modification -------------------------/
        # FA returns LSE in shape [ H, B ] but cp_lse_ag_out_rs wants [ B, H ]
        context_attn_out_cor, context_lse_cor = cp_lse_ag_out_rs(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        # /------------------------  Metax Modification -------------------------\
        query_attn_out, query_lse, _ = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            return_attn_probs=True,
            # fa_version=self.vllm_flash_attn_version,
            # q_descale=q_descale,
            # k_descale=k_descale,
            # v_descale=v_descale,
            s_aux=self.sinks,
        )
        # \------------------------- Metax Modification -------------------------/

        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        num_actual_tokens = attn_metadata.num_actual_tokens
        # Call flash attention directly on Q, K, V tensors
        output[:num_actual_tokens] = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            # fa_version=self.vllm_flash_attn_version,
            # q_descale=layer._q_scale.expand(descale_shape),
            # k_descale=layer._k_scale.expand(descale_shape),
            # v_descale=layer._v_scale.expand(descale_shape),
            s_aux=self.sinks,
            # num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # /------------------------  Metax Modification -------------------------\
    cu_prefix_kv_lens = _cu_seqlens_from_lens(prefix_kv_lens)
    # \------------------------  Metax Modification -------------------------/

    # Process shared prefix.
    # /------------------------  Metax Modification -------------------------\
    prefix_output, prefix_lse, _ = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        cu_seqlens_k=cu_prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        s_aux=s_aux,
    )
    # \------------------------- Metax Modification -------------------------/

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])
    # /------------------------  Metax Modification -------------------------\
    cu_suffix_kv_lens = _cu_seqlens_from_lens(suffix_kv_lens)
    # \------------------------  Metax Modification -------------------------/

    # Process suffix per query.
    suffix_output, suffix_lse, _ = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_attn_probs=True,
        s_aux=s_aux,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
