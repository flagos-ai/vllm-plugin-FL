# SPDX-License-Identifier: Apache-2.0
"""Private scheduler-to-reference cache plans; original-token coordinates."""

from dataclasses import dataclass

import torch

from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def validate_cache_spec(spec):
    """Validate the plain cache layouts accepted by this attention backend."""
    if not isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
        raise TypeError("DeepSeek V4 requires an MLA cache spec")
    if (
        spec.num_kv_heads != 1
        or spec.head_size <= 0
        or spec.compress_ratio not in (1, 4, 128)
        or spec.block_size <= 0
        or spec.block_size % spec.compress_ratio
    ):
        raise ValueError("Invalid DeepSeek V4 cache dimensions or compression ratio")
    if spec.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.int8):
        raise NotImplementedError("Only plain BF16/FP16/FP32/INT8 caches are supported")
    if spec.cache_dtype_str not in (
        None,
        "auto",
        "float16",
        "bfloat16",
        "float32",
        "int8",
    ):
        raise NotImplementedError(
            "Packed or quantized MLA cache layouts are not supported"
        )


@dataclass
class DeepseekV4CacheMetadata:
    block_table: torch.Tensor
    request_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    output_request_ids: torch.Tensor
    output_positions: torch.Tensor
    output_slots: torch.Tensor
    seq_lens: torch.Tensor
    key_lengths: torch.Tensor
    num_completed: int


class DeepseekV4CacheBackend(AttentionBackend):
    # Tensor providers own capture/replay; there is no external attention task.
    requires_graph_task_updates = False

    @classmethod
    def get_preferred_block_size(cls, default_block_size):
        # This is the full-cache logical block, not the smaller state page.
        return 128

    @staticmethod
    def get_name():
        return "FL_DEEPSEEK_V4_CACHE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("Cache-only leaves execute through model dispatch")

    @staticmethod
    def get_builder_cls():
        return DeepseekV4MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        if num_kv_heads != 1:
            raise ValueError("DeepSeek V4 cache leaves require one KV head")
        return num_blocks, block_size, head_size

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return tuple(range(4 if include_num_layers_dimension else 3))


class DeepseekV4MetadataBuilder(AttentionMetadataBuilder[DeepseekV4CacheMetadata]):
    """Keep captured addresses stable without modifying common metadata.

    Exact CPU lengths are required: synchronizing device lengths here both
    hides stale speculative counts and introduces a per-layer device barrier.
    The runner's input-preparation event guards reuse of pinned host plans.
    """

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        validate_cache_spec(kv_cache_spec)
        parallel = vllm_config.parallel_config
        if vllm_config.speculative_config is not None or (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise NotImplementedError(
                "V4 reference cache plans require non-speculative CP1"
            )
        self.ratio = kv_cache_spec.compress_ratio
        self.logical_block = kv_cache_spec.block_size
        self.token_cap = vllm_config.scheduler_config.max_num_batched_tokens
        self.request_cap = vllm_config.scheduler_config.max_num_seqs
        pin = is_pin_memory_available()
        self.host = {}
        self.buffers = {}
        for name in (
            "request_ids",
            "positions",
            "output_request_ids",
            "output_positions",
            "seq_lens",
        ):
            size = self.request_cap if name == "seq_lens" else self.token_cap
            self.host[name] = torch.empty(size, dtype=torch.int64, pin_memory=pin)
            self.buffers[name] = torch.empty(size, dtype=torch.int64, device=device)
        self.slots = torch.empty(self.token_cap, dtype=torch.int64, device=device)
        self.output_slots = torch.empty_like(self.slots)
        self.key_lengths = torch.empty(
            self.request_cap, dtype=torch.int64, device=device
        )
        self.block_table = None

    @staticmethod
    def _cpu(tensor, name):
        if tensor is None or tensor.device.type != "cpu":
            raise ValueError(f"Exact CPU {name} metadata is required")
        return tensor

    def _slots(self, requests, positions):
        table = self.block_table
        column = positions.clamp_min(0) // self.logical_block
        pages = table[
            requests.clamp(0, self.request_cap - 1),
            column.clamp_max(table.shape[1] - 1),
        ].long()
        valid = (
            (requests >= 0) & (positions >= 0) & (column < table.shape[1]) & (pages > 0)
        )
        block = self.logical_block // self.ratio
        slots = pages * block + (positions.clamp_min(0) // self.ratio) % block
        return torch.where(valid, slots, -1)

    def _prepare_table(self, common):
        nreq, ntok = common.num_reqs, common.num_actual_tokens
        if not 0 < nreq <= self.request_cap or not 0 <= ntok <= self.token_cap:
            raise ValueError("Scheduler batch exceeds the private cache plan capacity")
        public = common.block_table_tensor
        if public.ndim != 2 or public.shape[0] < nreq or public.shape[1] == 0:
            raise ValueError("Missing scheduler block-table rows")
        if self.block_table is None:
            self.block_table = torch.empty(
                (self.request_cap, public.shape[1]),
                dtype=public.dtype,
                device=self.device,
            )
        if public.shape[1] > self.block_table.shape[1]:
            raise ValueError(
                "Block-table capacity cannot grow after plan initialization"
            )
        if public.dtype != self.block_table.dtype:
            raise ValueError(
                "Block-table dtype cannot change after plan initialization"
            )
        self.block_table.fill_(-1)
        return public

    def _metadata(self, ntok, completed):
        return DeepseekV4CacheMetadata(
            self.block_table,
            self.buffers["request_ids"][:ntok],
            self.buffers["positions"][:ntok],
            self.slots[:ntok],
            self.buffers["output_request_ids"][:ntok],
            self.buffers["output_positions"][:ntok],
            self.output_slots[:ntok],
            self.buffers["seq_lens"],
            self.key_lengths,
            completed,
        )

    def build_for_dummy(self, common_attn_metadata):
        """Reuse real-plan addresses without reading synthetic CPU lengths/pages."""
        common = common_attn_metadata
        self._prepare_table(common)
        for name, buffer in self.buffers.items():
            buffer.fill_(0 if name == "seq_lens" else -1)
        self.slots.fill_(-1)
        self.output_slots.fill_(-1)
        self.key_lengths.zero_()
        return self._metadata(common.num_actual_tokens, 0)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build_for_dummy(common_attn_metadata)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        public = self._prepare_table(common)
        starts = self._cpu(common.query_start_loc_cpu, "query_start_loc")
        ends = self._cpu(common._seq_lens_cpu, "seq_lens")
        nreq, ntok = common.num_reqs, common.num_actual_tokens
        if starts.numel() < nreq + 1 or ends.numel() < nreq:
            raise ValueError("Incomplete CPU scheduler metadata")
        starts, ends = starts[: nreq + 1].tolist(), ends[:nreq].tolist()
        if starts[0] != 0 or starts[-1] > ntok:
            raise ValueError("CPU query offsets exceed the scheduled token count")
        computed = common._num_computed_tokens_cpu
        if computed is not None:
            computed = self._cpu(computed, "num_computed_tokens")[:nreq].tolist()
            if len(computed) != nreq:
                raise ValueError("Incomplete CPU computed-token metadata")
        for name, host in self.host.items():
            host.fill_(0 if name == "seq_lens" else -1)
        completed = 0
        for request, end in enumerate(ends):
            begin, stop = starts[request : request + 2]
            count = stop - begin
            # The runner pads FULL decode with zero-length request rows. Their
            # query spans/computed counters can be synthetic or stale; never
            # derive live cache slots from them. Active rows retain strict checks.
            if end == 0 and 0 <= count <= 1:
                continue
            position = end - count
            if (
                count < 0
                or position < 0
                or (computed is not None and computed[request] != position)
            ):
                raise ValueError("Inconsistent exact CPU scheduler lengths")
            self.host["request_ids"][begin:stop] = request
            self.host["positions"][begin:stop] = torch.arange(position, end)
            self.host["seq_lens"][request] = end
            boundaries = torch.arange(position // self.ratio, end // self.ratio)
            size = boundaries.numel()
            self.host["output_request_ids"][completed : completed + size] = request
            self.host["output_positions"][completed : completed + size] = (
                boundaries + 1
            ) * self.ratio - 1
            completed += size
        # Scheduler page zero is a null placeholder, not readable cache storage.
        self.block_table[:nreq, : public.shape[1]].copy_(
            torch.where(public[:nreq] > 0, public[:nreq], -1)
        )
        for request, end in enumerate(ends):
            if end == 0:
                self.block_table[request].fill_(-1)
        for name, buffer in self.buffers.items():
            buffer.copy_(self.host[name], non_blocking=True)
        request_ids, positions = (
            self.buffers["request_ids"][:ntok],
            self.buffers["positions"][:ntok],
        )
        output_requests, output_positions = (
            self.buffers["output_request_ids"][:ntok],
            self.buffers["output_positions"][:ntok],
        )
        slots = self._slots(request_ids, positions)
        if self.ratio > 1:
            slots = torch.where((positions + 1) % self.ratio == 0, slots, -1)
        self.slots.fill_(-1)
        self.output_slots.fill_(-1)
        self.slots[:ntok].copy_(slots)
        self.output_slots[:ntok].copy_(self._slots(output_requests, output_positions))
        self.key_lengths.copy_(self.buffers["seq_lens"] // self.ratio)
        return self._metadata(ntok, completed)
