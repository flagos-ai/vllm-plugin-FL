# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

_flagcx_path = os.getenv("FLAGCX_PATH")
if _flagcx_path and os.path.isdir(_flagcx_path) and _flagcx_path not in sys.path:
    sys.path.append(_flagcx_path)

try:
    from plugin.interservice.flagcx_wrapper import (
        FLAGCXLibrary,
    )
except ImportError as e:
    raise ImportError(
        "Cannot import FlagCX wrapper. Set FLAGCX_PATH to the FlagCX repo "
        "root (containing plugin/interservice/flagcx_wrapper.py)."
    ) from e

EngineId = str
ReqId = str

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"

logger = init_logger(__name__)


@dataclass(frozen=True)
class TransferRegion:
    """One transferable KV region (one registered tensor, or one K/V half
    after blocks-first split), tagged with its layer and KV-cache group."""

    layer_name: str
    layer_index: int
    base_addr: int
    block_len: int
    kv_block_len: int
    group_index: int = 0


class FlagCXAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,
):
    """Sent from Decode → Prefill over ZMQ to request a KV transfer.

    Carries the Decode side's own registered-region metadata so the Prefill
    side can reconstruct the destination regions and compute absolute write
    addresses per KV-cache group (heterogeneous-TP aware)."""

    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    # req_id -> per KV-cache-group block ids on the Decode (receiver) side.
    req_blocks: dict[ReqId, list[list[int]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    registered_layer_names: list[str]
    registered_layer_indices: list[int]
    registered_group_indices: list[int]


@dataclass
class RecvReqMeta:
    local_block_ids: list[list[int]]
    remote_host: str
    remote_port: int
    remote_tp_size: int = 0


@dataclass
class SendBlockMeta:
    local_block_ids: list[list[int]]
    ready: threading.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0


@dataclass
class SendReqMeta:
    reqs: dict[ReqId, SendBlockMeta]
    lock: threading.Lock


@dataclass
class FinishedSendReqSet:
    set: set[ReqId]
    lock: threading.Lock


@dataclass
class FinishedReceiveReqSet:
    set: set[ReqId]
    lock: asyncio.Lock


class FlagCXConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, RecvReqMeta] = {}
        # Per-request, per KV-cache-group block ids.
        self.reqs_to_send: dict[ReqId, list[list[int]]] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        if load_remote_cache:
            self.reqs_to_recv[request_id] = RecvReqMeta(
                local_block_ids=local_block_ids,
                remote_host=kv_transfer_params["remote_host"],
                remote_port=kv_transfer_params["remote_port"],
                remote_tp_size=kv_transfer_params.get("remote_tp_size", 0),
            )
        else:
            self.reqs_to_send[request_id] = local_block_ids


class FlagCXConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            assert kv_cache_config is not None, (
                "kv_cache_config is required for SCHEDULER role"
            )
            self.connector_scheduler: FlagCXConnectorScheduler | None = (
                FlagCXConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: FlagCXConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = FlagCXConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        if vllm_config.model_config is None:
            # Mostly for unit tests that instantiate without a full model config.
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        if vllm_config.model_config.use_mla:
            return None
        logger.info_once(
            "FlagCXConnector setting KV cache layout to HND for "
            "heterogeneous TP-safe KV transfer."
        )
        return "HND"

    # ---- Scheduler-side ----
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    # ---- Worker-side ----
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, FlagCXConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.connector_worker is not None:
            self.connector_worker.wait_for_layer_load()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        pass


class FlagCXConnectorScheduler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = _get_side_channel_port(vllm_config)

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        logger.info("FlagCX Connector Scheduler init: %s", engine_id)

        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        # GDN is represented as a MambaSpec in vLLM.
        self._has_mamba = kv_cache_config.has_mamba_layers

        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

        self._reqs_need_recv: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        self._reqs_need_send: dict[ReqId, list[list[int]]] = {}

    def get_sw_clipped_blocks(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
    ) -> list[list[int]]:
        """Clip per-group block ids to the sliding window size."""
        if len(block_ids) == 0 or not self._is_hma_required:
            return list(block_ids)
        return [
            blocks[-self.blocks_per_sw[i] :] if self.blocks_per_sw[i] > 0 else blocks
            for i, blocks in enumerate(block_ids)
        ]

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba/GDN models because the decoder
        always recomputes the last prompt token locally to obtain the first
        logits and advance the recurrent state from h(N-2) to h(N-1)."""
        if self._has_mamba and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-2) instead of h(N-1). The decoder recomputes the last token to
        derive the first logits and advance to h(N-1).

        Guarded by ``_p_side_truncated`` so a preempted+rescheduled request is
        not truncated twice."""
        params = request.kv_transfer_params
        if (
            params is not None
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            count = (
                self._get_remote_prefill_token_count(len(token_ids))
                - num_computed_tokens
            )
            if count > 0:
                return count, True

        if (
            params is not None
            and params.get("do_remote_decode")
            and self._has_mamba
        ):
            self._truncate_mamba_request_for_prefill(request)

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        params = request.kv_transfer_params
        if not params:
            return

        if params.get("do_remote_prefill"):
            assert self.kv_role != "kv_producer"
            if all(p in params for p in ("remote_host", "remote_port")):
                unhashed_block_ids = (
                    blocks.get_unhashed_block_ids_all_groups()
                    if num_external_tokens > 0
                    else []
                )
                local_block_ids = self.get_sw_clipped_blocks(unhashed_block_ids)
                self._reqs_need_recv[request.request_id] = (
                    request,
                    local_block_ids,
                )
            else:
                logger.warning("Invalid KVTransferParams: %s", params)
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            self._reqs_need_send[request.request_id] = []

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = FlagCXConnectorMetadata()

        if self.kv_role != "kv_producer":
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if self.kv_role != "kv_consumer":
            for req_id, block_ids in self._reqs_need_send.items():
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params={},
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()

        return meta

    def request_finished(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            assert self.kv_role != "kv_producer"
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (
            not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        assert self.kv_role != "kv_consumer"
        delay_free_blocks = any(len(group) > 0 for group in block_ids)

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = self.get_sw_clipped_blocks(
                block_ids
            )

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        )


class FlagCXConnectorWorker:
    """Worker-side logic for FlagCX PD disaggregation.

    This avoids the previous Decode-initiated async comm init that could
    deadlock and leave requests stuck in WAITING_FOR_REMOTE_KVS.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        logger.info("FlagCX Connector Worker init: %s", engine_id)

        assert kv_cache_config is not None, (
            "kv_cache_config is required for the FlagCX WORKER role"
        )
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.engine_id: EngineId = engine_id
        self.hostname = get_ip()

        # ---- FlagCX library ----
        library_path = os.getenv("FLAGCX_LIB_PATH")
        if library_path is None:
            flagcx_path = os.getenv("FLAGCX_PATH", "")
            library_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
        self.flagcx = FLAGCXLibrary(library_path)
        self.kv_cache_device: torch.device | None = None

        # ---- P2P engine (one-sided RDMA + RPC control plane) ----
        # Replaces the old per-pair comm model. The engine owns the
        # handshake/RPC port; transfers are addressed by absolute VA.
        self.engine = self.flagcx.flagcxP2pEngineCreate()
        self.rpc_port: int = 0

        # ---- Cached connections, keyed by "host:rpc_port" ----
        # The engine caches connections internally; this memoizes the
        # opaque handle so we call get_conn once per remote session.
        self.session_conns: dict[str, Any] = {}
        self.conn_lock = threading.Lock()

        # ---- Side-channel ZMQ port ----
        self.side_channel_port: int = _get_side_channel_port(vllm_config)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_workers = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: SendReqMeta = SendReqMeta(reqs={}, lock=threading.Lock())

        # ---- Prefill (sender) background threads ----
        if self.kv_role != "kv_consumer":
            self._sender_t: threading.Thread | None = None
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="vllm-flagcx-sender",
            )

        # ---- Decode (receiver) background threads ----
        if self.kv_role != "kv_producer":
            # Async event loop for _receive_kv coroutines (sends request-level
            # metadata to Prefill over the ZMQ side channel and awaits the
            # transfer-done reply). No comm-init listener is needed anymore;
            # the engine's RPC server (started in register_kv_caches) handles
            # the data-plane handshake.
            self.receiver_loop = asyncio.new_event_loop()
            self._receiver_t = threading.Thread(
                target=self._receiver_loop_fn,
                args=(self.receiver_loop,),
                daemon=True,
            )
            self._receiver_t.start()

        self.finished_sending_reqs = FinishedSendReqSet(set(), threading.Lock())
        self.finished_recving_reqs = FinishedReceiveReqSet(set(), asyncio.Lock())
        self._pull_pending: dict[ReqId, int] = {}

        self._abort_request_timeout = int(
            os.getenv("FLAGCX_CONNECTOR_ABORT_REQUEST_TIMEOUT", "480")
        )

        # ---- Attention backend detection ----
        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla
        # Physical(kernel)-block fan-out per logical block. >1 only under
        # spec-decode/hybrid kernel-block mismatch (see stage 4); 1 otherwise.
        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()

        backends = get_current_attn_backends(vllm_config)
        backend = backends[0]
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()

        # Layer -> KV cache spec and layer -> KV-cache group index, used to
        # tag registered regions per (layer, group).
        self._layer_specs: dict[str, KVCacheSpec] = {}
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            specs_by_layer = getattr(group_spec, "kv_cache_specs", {})
            for layer_name in group.layer_names:
                self._layer_specs[layer_name] = specs_by_layer.get(
                    layer_name, group_spec
                )
        self._layer_group_indices: dict[str, int] = {
            layer: group_index
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer in group.layer_names
        }

        self.kv_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            is_mamba=kv_cache_config.has_mamba_layers,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=backends,
        )

        # Per-registered-region parallel lists, populated by register_kv_caches.
        self.block_len_per_layer: list[int] = []
        self.kv_block_len_per_layer: list[int] = []
        self.registered_layer_names: list[str] = []
        self.registered_layer_indices: list[int] = []
        self.registered_group_indices: list[int] = []

        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(FlagCXAgentMetadata)

    def _sync_block_size_with_kernel(self) -> None:
        # When the user logical block size differs from the physical kernel
        # block size (e.g. spec-decode / hybrid SSM models), pick the common
        # (smallest) block size so registration and transfer use kernel units.
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match physical "
                "kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size

    def _get_conn(self, session: str) -> Any:
        """Get (or lazily open) the engine connection to a remote session
        string "host:rpc_port". The first call performs the QP + desc-table
        handshake; subsequent calls return the cached handle."""
        with self.conn_lock:
            conn = self.session_conns.get(session)
            if conn is None:
                conn = self.flagcx.flagcxP2pGetConn(self.engine, session)
                self.session_conns[session] = conn
                logger.info("Opened P2P connection to %s", session)
            return conn

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches. use_mla: %s", self.use_mla)

        kv_data_ptrs: list[int] = []
        kv_data_lens: list[int] = []
        region_base_addresses: list[int] = []
        seen_storage_ptrs: set[int] = set()

        for layer_name, cache_or_caches in kv_caches.items():
            layer_index = extract_layer_index(layer_name)
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s: no KV cache spec present.", layer_name
                )
                continue
            # Dispatch on the spec: Mamba/GDN → [conv] (ssm dropped, recomputed
            # on D via the N-1 trick); split_k_and_v → [K, V]; else → [cache].
            cache_list = self.kv_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )

            for cache in cache_list:
                if self.kv_cache_device is None:
                    self.kv_cache_device = cache.device
                else:
                    assert self.kv_cache_device == cache.device
                base_addr = cache.data_ptr()
                block_len = cache.stride(0) * cache.element_size()

                if isinstance(layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                    kv_block_len = layer_spec.page_size_bytes
                elif self.kv_topo.is_kv_layout_blocks_first and not isinstance(
                    layer_spec, MambaSpec
                ):
                    # Blocks-first packs K and V in one block; each is half.
                    kv_block_len = block_len // 2
                else:
                    kv_block_len = block_len

                # One parallel-list entry per registered region view.
                region_base_addresses.append(base_addr)
                self.block_len_per_layer.append(block_len)
                self.kv_block_len_per_layer.append(kv_block_len)
                self.registered_layer_names.append(layer_name)
                self.registered_layer_indices.append(layer_index)
                self.registered_group_indices.append(
                    self._layer_group_indices[layer_name]
                )

                storage = cache.untyped_storage()
                storage_addr = storage.data_ptr()
                if storage_addr not in seen_storage_ptrs:
                    seen_storage_ptrs.add(storage_addr)
                    kv_data_ptrs.append(storage_addr)
                    kv_data_lens.append(storage.nbytes())

        self.kv_caches_base_addr = region_base_addresses
        self.device_kv_caches = kv_caches

        if not kv_data_ptrs:
            raise RuntimeError("No KV cache tensors were registered with FlagCX.")

        for base_addr, size in zip(kv_data_ptrs, kv_data_lens):
            self.flagcx.flagcxP2pRegister(self.engine, base_addr, size)

        # The engine's handshake port becomes our session identity; peers
        # connect to "{hostname}:{rpc_port}" for the data-plane transfer.
        self.rpc_port = self.flagcx.flagcxP2pGetRpcPort(self.engine)

        logger.info(
            "KV cache registered: %d regions across %d MRs, rpc_port=%d.",
            len(region_base_addresses),
            len(kv_data_ptrs),
            self.rpc_port,
        )

        # Decode (write target) starts the RPC accept daemon so Prefill can
        # connect and RDMA-write into our registered KV regions.
        if self.kv_role != "kv_producer":
            self.flagcx.flagcxP2pStartRpcServer(self.engine)
            logger.info("FlagCX P2P RPC server started on port %d", self.rpc_port)

        # Launch Prefill sender thread (ROUTER socket for Decode requests)
        if self.kv_role != "kv_consumer":
            ready_event = threading.Event()
            self._sender_t = threading.Thread(
                target=self._sender_thread,
                args=(ready_event, self.side_channel_port, self.tp_rank),
                daemon=True,
                name="flagcx_sender",
            )
            self._sender_t.start()
            ready_event.wait()

    def _sender_thread(
        self, ready_event: threading.Event, base_port: int, tp_rank: int
    ):
        """Prefill sender: ROUTER socket receives requests from Decode,
        dispatches to thread pool, and relays transfer status."""
        frontend_path = make_zmq_path("tcp", self.hostname, base_port + tp_rank)
        frontend = make_zmq_socket(self.zmq_ctx, frontend_path, zmq.ROUTER)

        backend_path = make_zmq_path("inproc", str(uuid.uuid4()))
        backend = make_zmq_socket(self.zmq_ctx, backend_path, zmq.PULL)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        ready_event.set()

        try:
            while True:
                sockets = dict(poller.poll())

                if frontend in sockets:
                    identity, _, metadata_bytes = frontend.recv_multipart()
                    self._sender_executor.submit(
                        self._sender_worker,
                        identity,
                        metadata_bytes,
                        backend_path,
                    )

                if backend in sockets:
                    identity, status = backend.recv_multipart()
                    frontend.send_multipart((identity, b"", status))

        except zmq.ContextTerminated:
            pass
        except Exception as e:
            logger.error("FlagCX sender thread error: %s", e)
        finally:
            frontend.close()
            backend.close()

    def _sender_worker(
        self,
        identity: bytes,
        metadata_bytes: bytes,
        worker_channel_path: str,
    ):
        status = TRANS_ERROR

        try:
            metadata = self._decoder.decode(metadata_bytes)
            self._send_kv_to_decode(metadata)
            status = TRANS_DONE
        except Exception as e:
            logger.error("FlagCX sender worker error: %s", e)
        finally:
            pusher = make_zmq_socket(self.zmq_ctx, worker_channel_path, zmq.PUSH)
            try:
                pusher.send_multipart((identity, status))
            except zmq.ZMQError as e:
                logger.warning(
                    "Internal error, maybe the server is shutting down. Error: %s",
                    e,
                )
            finally:
                pusher.close()

    def _producer_cache_is_replicated(self) -> bool:
        return self.kv_topo.local_replicates_kv_cache

    def _get_transfer_regions(
        self,
        base_addrs: list[int],
        block_lens: list[int],
        kv_block_lens: list[int],
        layer_names: list[str],
        layer_indices: list[int],
        group_indices: list[int],
    ) -> list[TransferRegion]:
        split_kv_regions = None
        if self.kv_topo.is_kv_layout_blocks_first:
            split_kv_regions = [
                not isinstance(
                    self._layer_specs[layer_name],
                    (MambaSpec, MLAAttentionSpec, SlidingWindowMLASpec),
                )
                for layer_name in layer_names
            ]
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            kv_block_lens=kv_block_lens,
            layer_names=layer_names,
            layer_indices=layer_indices,
            is_kv_layout_blocks_first=self.kv_topo.is_kv_layout_blocks_first,
            group_indices=group_indices,
            split_kv_regions=split_kv_regions,
        )

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )

    def _logical_to_kernel_block_ids(
        self, block_ids: list[list[int]]
    ) -> list[list[int]]:
        """Expand attention groups' logical block ids to kernel-physical block
        ids; Mamba/GDN state groups stay in the logical/page-id space."""
        if self._physical_blocks_per_logical_kv_block == 1:
            return block_ids
        block_arange = np.arange(self._physical_blocks_per_logical_kv_block).reshape(
            1, -1
        )
        group_specs = self.kv_cache_config.kv_cache_groups
        return [
            BlockTable.map_to_kernel_blocks(
                np.array(group),
                self._physical_blocks_per_logical_kv_block,
                block_arange,
            ).tolist()
            if not isinstance(group_specs[i].kv_cache_spec, MambaSpec)
            else group
            for i, group in enumerate(block_ids)
        ]

    def _send_kv_to_decode(self, meta: FlagCXAgentMetadata) -> None:
        target_d_ranks = self.kv_topo.handshake_target_ranks(meta.remote_tp_size)
        need = len(target_d_ranks)
        if meta.remote_tp_rank not in target_d_ranks:
            logger.warning(
                "D rank %d is not a valid transfer pair for P rank %d "
                "(targets=%s); proceeding anyway.",
                meta.remote_tp_rank,
                self.tp_rank,
                target_d_ranks,
            )

        ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
        with self.reqs_need_send.lock:
            for req_id in meta.req_blocks:
                send_meta = self.reqs_need_send.reqs.get(req_id)
                if send_meta is None:
                    logger.warning("Request %s not found in reqs_need_send", req_id)
                    return
                # Mark it as not expired. We will send it now.
                send_meta.expire_time = float("inf")
                send_meta.need_send = need
                ready_reqs.append((req_id, send_meta))

        # Wait until the scheduler has committed each request's blocks.
        for _, send_meta in ready_reqs:
            send_meta.ready.wait()

        remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
        conn = self._get_conn(remote_session)
        start_time = time.perf_counter()

        # Reconstruct both sides' regions (tagged per layer/group), align them
        # by (layer_name, occurrence), and validate TP-ratio length relations.
        local_regions = self._get_transfer_regions(
            self.kv_caches_base_addr,
            self.block_len_per_layer,
            self.kv_block_len_per_layer,
            self.registered_layer_names,
            self.registered_layer_indices,
            self.registered_group_indices,
        )
        remote_regions = self._get_transfer_regions(
            meta.kv_caches_base_addr,
            meta.block_lens,
            meta.kv_block_lens,
            meta.registered_layer_names,
            meta.registered_layer_indices,
            meta.registered_group_indices,
        )
        local_regions, remote_regions, err = _align_transfer_regions(
            local_regions, remote_regions
        )
        if err is not None:
            raise RuntimeError(err)
        err = _validate_asymmetric_region_lengths(
            local_regions,
            remote_regions,
            self.tp_size,
            meta.remote_tp_size,
            self._producer_cache_is_replicated(),
        )
        if err is not None:
            raise RuntimeError(err)

        src_vas, dst_vas, sizes = self._build_transfer_params(
            ready_reqs, meta, local_regions, remote_regions
        )

        if sizes:
            self.flagcx.flagcxP2pBatchWriteSync(conn, src_vas, dst_vas, sizes)

        finished: list[ReqId] = []
        with self.reqs_need_send.lock:
            for req_id, send_meta in ready_reqs:
                send_meta.sent += 1
                if send_meta.sent >= max(send_meta.need_send, 1):
                    self.reqs_need_send.reqs.pop(req_id, None)
                    finished.append(req_id)
        if finished:
            with self.finished_sending_reqs.lock:
                self.finished_sending_reqs.set.update(finished)

        logger.debug(
            "Sending to %s done (%d xfers), took %s",
            remote_session,
            len(sizes),
            time.perf_counter() - start_time,
        )

    def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: FlagCXAgentMetadata,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> tuple[list[int], list[int], list[int]]:
        """Build absolute-VA (src, dst, size) write lists, mapping each region
        to its own KV-cache group's block ids. Never flattens across groups."""
        src_ptrs: list[int] = []
        dst_ptrs: list[int] = []
        lengths: list[int] = []
        group_specs = self.kv_cache_config.kv_cache_groups

        for d_req_id, send_meta in ready_reqs:
            remote_block_ids_per_group = agent_meta.req_blocks[d_req_id]
            if not remote_block_ids_per_group or all(
                len(g) == 0 for g in remote_block_ids_per_group
            ):
                continue
            if len(send_meta.local_block_ids) != len(remote_block_ids_per_group):
                logger.error(
                    "req %s: KV group count mismatch: local=%d, remote=%d",
                    d_req_id,
                    len(send_meta.local_block_ids),
                    len(remote_block_ids_per_group),
                )
                continue

            # Per-group: strip mamba null placeholders, trim partial-hit tail.
            local_by_group: list[list[int]] = []
            remote_by_group: list[list[int]] = []
            has_error = False
            for group_index, (local_group, remote_group) in enumerate(
                zip(send_meta.local_block_ids, remote_block_ids_per_group)
            ):
                if isinstance(group_specs[group_index].kv_cache_spec, MambaSpec):
                    local_group = [b for b in local_group if b != NULL_BLOCK_ID]
                    remote_group = [b for b in remote_group if b != NULL_BLOCK_ID]
                n_local = len(local_group)
                n_remote = len(remote_group)
                if n_local < n_remote:
                    logger.error(
                        "req %s: local blocks(%d) < remote blocks(%d)",
                        d_req_id,
                        n_local,
                        n_remote,
                    )
                    has_error = True
                    break
                if n_local > n_remote:
                    local_group = local_group[-n_remote:] if n_remote > 0 else []
                local_by_group.append(local_group)
                remote_by_group.append(remote_group)
            if has_error or not any(local_by_group):
                continue

            local_by_group = self._logical_to_kernel_block_ids(local_by_group)
            remote_by_group = self._logical_to_kernel_block_ids(remote_by_group)

            for local_region, remote_region in zip(local_regions, remote_regions):
                group_index = local_region.group_index
                if group_index >= len(local_by_group):
                    continue
                local_block_ids = local_by_group[group_index]
                remote_block_ids = remote_by_group[group_index]
                if not local_block_ids:
                    continue

                should_transfer, src_off, dst_off, transfer_len = (
                    self._get_sender_transfer_plan(
                        local_kv_block_len=local_region.kv_block_len,
                        remote_kv_block_len=remote_region.kv_block_len,
                        remote_tp_rank=agent_meta.remote_tp_rank,
                        remote_tp_size=agent_meta.remote_tp_size,
                    )
                )
                if not should_transfer:
                    # Replicated KV cache: only one producer rank sends.
                    continue
                assert src_off + transfer_len <= local_region.kv_block_len, (
                    "Source transfer region exceeds local KV block size."
                )
                assert dst_off + transfer_len <= remote_region.kv_block_len, (
                    "Destination transfer region exceeds remote KV block size."
                )

                grp_local, grp_remote = group_concurrent_contiguous(
                    local_block_ids, remote_block_ids
                )
                can_coalesce = _can_coalesce_block_transfers(
                    local_region.block_len,
                    remote_region.block_len,
                    src_off,
                    dst_off,
                    transfer_len,
                )
                for gl, gr in zip(grp_local, grp_remote):
                    if can_coalesce:
                        src_ptrs.append(
                            local_region.base_addr
                            + gl[0] * local_region.block_len
                            + src_off
                        )
                        dst_ptrs.append(
                            remote_region.base_addr
                            + gr[0] * remote_region.block_len
                            + dst_off
                        )
                        lengths.append(transfer_len * len(gl))
                    else:
                        for lb, rb in zip(gl, gr):
                            src_ptrs.append(
                                local_region.base_addr
                                + lb * local_region.block_len
                                + src_off
                            )
                            dst_ptrs.append(
                                remote_region.base_addr
                                + rb * remote_region.block_len
                                + dst_off
                            )
                            lengths.append(transfer_len)

        return src_ptrs, dst_ptrs, lengths

    def _receiver_loop_fn(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _receive_kv(
        self,
        path: str,
        req_blocks: dict[ReqId, list[list[int]]],
    ):
        req_ids = list(req_blocks.keys())

        metadata = FlagCXAgentMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks=req_blocks,
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            kv_block_lens=self.kv_block_len_per_layer,
            registered_layer_names=self.registered_layer_names,
            registered_layer_indices=self.registered_layer_indices,
            registered_group_indices=self.registered_group_indices,
        )

        encoded_data = self._encoder.encode(metadata)

        sock: zmq.asyncio.Socket = make_zmq_socket(
            self.async_zmq_ctx, path, zmq.REQ, bind=False, linger=0
        )
        sock.setsockopt(zmq.RCVTIMEO, 60000)

        try:
            await sock.send(encoded_data)
            ret_msg = await sock.recv()
            if ret_msg != TRANS_DONE:
                logger.error(
                    "Error happens during transferring kvcache for %s, "
                    "see logs in prefiller.",
                    req_ids,
                )
                return

        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting FlagCX receiver thread.")
        except Exception as e:
            logger.error("FlagCXAgentMetadata transfer failed for %s: %s", req_ids, e)
            return
        finally:
            sock.close()

        async with self.finished_recving_reqs.lock:
            for req_id in req_ids:
                remaining = self._pull_pending.get(req_id, 1) - 1
                if remaining <= 0:
                    self._pull_pending.pop(req_id, None)
                    self.finished_recving_reqs.set.add(req_id)
                else:
                    self._pull_pending[req_id] = remaining

        logger.debug("pulling kv_caches for %s finished (path=%s)", req_ids, path)

    def start_load_kv(self, metadata: FlagCXConnectorMetadata):
        if self.kv_role != "kv_producer" and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._group_kv_pull(metadata.reqs_to_recv),
                self.receiver_loop,
            )

        if self.kv_role != "kv_consumer":
            with self.reqs_need_send.lock:
                for req_id, block_ids in metadata.reqs_to_send.items():
                    send_meta = self.reqs_need_send.reqs.get(req_id)
                    if send_meta is None:
                        send_meta = SendBlockMeta(
                            local_block_ids=[], ready=threading.Event()
                        )
                        self.reqs_need_send.reqs[req_id] = send_meta
                    # Non-empty means request_finished() has committed the
                    # per-group block ids; arm the send.
                    if block_ids:
                        send_meta.local_block_ids = block_ids
                        send_meta.ready.set()
                        send_meta.expire_time = (
                            time.perf_counter() + self._abort_request_timeout
                        )

    def wait_for_layer_load(self) -> None:
        return

    async def _group_kv_pull(
        self, reqs_to_recv: dict[ReqId, RecvReqMeta]
    ) -> None:
        """Fan out each request to every Prefill TP rank it must pull from.

        With D_TP > P_TP one Decode rank gathers its slice from several Prefill
        ranks (``handshake_target_ranks``); it finishes recving only after all
        of them respond (tracked via ``_pull_pending``). Runs on the receiver
        loop; populates ``_pull_pending`` and launches the per-peer pulls
        without an intervening await, so it is atomic w.r.t. other coroutines.
        """
        kv_pulls: dict[str, dict[ReqId, list[list[int]]]] = defaultdict(dict)
        for req_id, meta in reqs_to_recv.items():
            remote_tp_size = meta.remote_tp_size or self.tp_size
            target_p_ranks = self.kv_topo.handshake_target_ranks(remote_tp_size)
            self._pull_pending[req_id] = len(target_p_ranks)
            for p_rank in target_p_ranks:
                path = make_zmq_path(
                    "tcp", meta.remote_host, meta.remote_port + p_rank
                )
                kv_pulls[path][req_id] = meta.local_block_ids
        for path, req_blocks in kv_pulls.items():
            asyncio.ensure_future(self._receive_kv(path, req_blocks))

    async def _fetch_finished_recving(self) -> set[ReqId]:
        async with self.finished_recving_reqs.lock:
            result = self.finished_recving_reqs.set
            self.finished_recving_reqs.set = set()
        return result

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        fut = None
        if self.kv_role != "kv_producer":
            fut = asyncio.run_coroutine_threadsafe(
                self._fetch_finished_recving(), self.receiver_loop
            )

        if self.kv_role != "kv_consumer":
            with self.finished_sending_reqs.lock:
                finished_sending = self.finished_sending_reqs.set
                self.finished_sending_reqs.set = set()
        else:
            finished_sending = set()

        finished_recving = fut.result() if fut else set()

        now = time.perf_counter()
        with self.reqs_need_send.lock:
            expired = [
                rid
                for rid, sm in self.reqs_need_send.reqs.items()
                if sm.expire_time < now
            ]
            for rid in expired:
                logger.warning("Request %s send timed out, freeing blocks", rid)
                del self.reqs_need_send.reqs[rid]
            if expired:
                finished_sending.update(expired)

        return finished_sending or None, finished_recving or None

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self.zmq_ctx.term()
        self.async_zmq_ctx.term()
        if self.kv_role != "kv_consumer":
            self._sender_executor.shutdown(wait=False)
            if self._sender_t:
                self._sender_t.join(timeout=2)
        if (
            self.kv_role != "kv_producer"
            and hasattr(self, "receiver_loop")
            and self.receiver_loop.is_running()
        ):
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._receiver_t.join(timeout=2)
        # Tear down the engine (stops the RPC server, closes connections and
        # deregisters MRs).
        engine = getattr(self, "engine", None)
        if engine is not None:
            try:
                self.flagcx.flagcxP2pEngineDestroy(engine)
            except Exception as e:
                logger.warning("flagcxP2pEngineDestroy failed: %s", e)
            self.engine = None


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Group runs where both src and dst block ids are contiguous (diff==1),
    so a run can be emitted as a single larger transfer descriptor."""
    if len(src_indices) == 0:
        return [], []
    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = [g.tolist() for g in np.split(src_indices, brk)]
    dst_groups = [g.tolist() for g in np.split(dst_indices, brk)]
    return src_groups, dst_groups


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """TP ratio for heterogeneous-TP transfer planning.

    Positive: one local rank maps into a larger remote KV region.
    Negative: one local rank gathers from multiple remote KV regions.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local TP size {local_tp_size} not divisible by "
            f"remote TP size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size
    assert remote_tp_size % local_tp_size == 0, (
        f"Remote TP size {remote_tp_size} not divisible by "
        f"local TP size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    kv_block_lens: list[int],
    layer_names: list[str],
    layer_indices: list[int],
    is_kv_layout_blocks_first: bool,
    group_indices: list[int] | None = None,
    split_kv_regions: list[bool] | None = None,
) -> list[TransferRegion]:
    """Expand registered KV tensors into transferable regions.

    For blocks-first layouts (K and V packed in one block), a ``split_kv_region``
    entry yields two regions: K at ``base_addr`` and V at ``base_addr +
    kv_block_len``.
    """
    assert (
        len(base_addrs)
        == len(block_lens)
        == len(kv_block_lens)
        == len(layer_names)
        == len(layer_indices)
    ), "FlagCX transfer regions require matching metadata lengths."
    if group_indices is None:
        group_indices = [0] * len(layer_names)
    assert len(group_indices) == len(layer_names)
    if split_kv_regions is None:
        split_kv_regions = [is_kv_layout_blocks_first] * len(layer_names)
    assert len(split_kv_regions) == len(layer_names)

    regions: list[TransferRegion] = []
    for (
        base_addr,
        block_len,
        kv_block_len,
        layer_name,
        layer_index,
        group_index,
        split_kv_region,
    ) in zip(
        base_addrs,
        block_lens,
        kv_block_lens,
        layer_names,
        layer_indices,
        group_indices,
        split_kv_regions,
    ):
        regions.append(
            TransferRegion(
                layer_name=layer_name,
                layer_index=layer_index,
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
                group_index=group_index,
            )
        )
        if split_kv_region:
            regions.append(
                TransferRegion(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                    group_index=group_index,
                )
            )
    return regions


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank → one consumer-rank copy for heterogeneous TP.

    Returns ``(should_transfer, src_offset, dst_offset, transfer_len)``.
    """
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    """Whether a contiguous block group can be emitted as one larger copy."""
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    if len(local_regions) != len(remote_regions):
        return "FlagCX asymmetric TP requires matching KV region counts."

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    f"FlagCX KV region length mismatch (homogeneous TP) at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    f"FlagCX destination KV region length mismatch at region {idx}: "
                    f"local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    f"FlagCX source KV region length mismatch at region {idx}: "
                    f"local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
    return None


def _align_transfer_regions(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
) -> tuple[list[TransferRegion], list[TransferRegion], str | None]:
    """Align local/remote regions by (layer_name, occurrence) so PP shards with
    different layer subsets pair correctly; validate layer/group indices."""

    def keyed_regions(
        regions: list[TransferRegion],
    ) -> list[tuple[tuple[str, int], TransferRegion]]:
        counts: dict[str, int] = defaultdict(int)
        keyed: list[tuple[tuple[str, int], TransferRegion]] = []
        for region in regions:
            occurrence = counts[region.layer_name]
            counts[region.layer_name] += 1
            keyed.append(((region.layer_name, occurrence), region))
        return keyed

    remote_by_key = dict(keyed_regions(remote_regions))
    aligned_local: list[TransferRegion] = []
    aligned_remote: list[TransferRegion] = []
    for key, local_region in keyed_regions(local_regions):
        remote_region = remote_by_key.get(key)
        if remote_region is None:
            return [], [], (
                f"FlagCX producer layer has no matching consumer occurrence: "
                f"{key[0]} occurrence {key[1]}."
            )
        if local_region.layer_index != remote_region.layer_index:
            return [], [], (
                f"FlagCX registered layer index mismatch for "
                f"{local_region.layer_name}."
            )
        if local_region.group_index != remote_region.group_index:
            return [], [], (
                f"FlagCX registered group index mismatch for "
                f"{local_region.layer_name}."
            )
        aligned_local.append(local_region)
        aligned_remote.append(remote_region)
    return aligned_local, aligned_remote, None


def _get_side_channel_port(vllm_config: VllmConfig) -> int:
    base_port = int(os.getenv("FLAGCX_BOOTSTRAP_PORT", "8998"))
    return (
        base_port
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )
