# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import ctypes
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

_flagcx_path = os.getenv("FLAGCX_PATH")
if _flagcx_path and os.path.isdir(_flagcx_path):
    if _flagcx_path not in sys.path:
        sys.path.append(_flagcx_path)

try:
    from plugin.interservice.flagcx_wrapper import (
        FLAGCXLibrary,
        buffer_type,
        flagcxComm_t,
        flagcxStream_t,
        flagcxUniqueId,
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

class FlagCXAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,
):
    """Sent from Decode → Prefill over ZMQ to request a KV transfer."""
    remote_hostname: str
    remote_port: int
    request_ids: list[ReqId]
    kv_caches_base_addr: list[int]
    block_ids: list[list[int]]
    uid_bytes: Optional[bytes] = None  # set on first contact to init pair comm

@dataclass
class RecvReqMeta:
    local_block_ids: list[int]
    remote_host: str
    remote_port: int

@dataclass
class SendBlockMeta:
    local_block_ids: list[int]
    ready: threading.Event
    expire_time: float = float("inf")

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

@dataclass
class PendingSignalWait:
    req_ids: list[ReqId]
    comm: Any = None
    peer_rank: int = -1
    signal_value: int = 0
    ready: threading.Event = field(default_factory=threading.Event)

@dataclass
class PairCommInfo:
    """Per-pair comm state.  Each (Prefill TP rank, Decode TP rank) pair
    gets its own flagcxComm, signal buffer, and monotonic signal counter
    so that RDMA signals from different pairs never interfere."""
    comm: Any
    my_rank: int
    signal_counter: int = 0
    # GPU signal buffer – per-pair to avoid cross-pair RDMA write races.
    signal_buffer: Optional[torch.Tensor] = None
    # Serialises counter-increment + PutSignal enqueue so that signal
    # values always land in monotonic order on the remote signal buffer.
    send_lock: threading.Lock = field(default_factory=threading.Lock)

class FlagCXConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, RecvReqMeta] = {}
        self.reqs_to_send: dict[ReqId, list[int]] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        if load_remote_cache:
            self.reqs_to_recv[request_id] = RecvReqMeta(
                local_block_ids=local_block_ids,
                remote_host=kv_transfer_params["remote_host"],
                remote_port=kv_transfer_params["remote_port"],
            )
        else:
            self.reqs_to_send[request_id] = local_block_ids

class FlagCXConnector(KVConnectorBase_V1):
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
            self.connector_scheduler: FlagCXConnectorScheduler | None = (
                FlagCXConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: FlagCXConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = FlagCXConnectorWorker(
                vllm_config, self.engine_id
            )

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
    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = _get_side_channel_port(vllm_config)

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        logger.info("FlagCX Connector Scheduler init: %s", engine_id)

        self._reqs_need_recv: dict[ReqId, tuple["Request", list[int]]] = {}
        self._reqs_need_send: dict[ReqId, list[int]] = {}

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True
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
            if all(
                p in params for p in ("remote_host", "remote_port")
            ):
                local_block_ids = (
                    blocks.get_unhashed_block_ids()
                    if num_external_tokens > 0
                    else []
                )
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
        self, request: "Request", block_ids: list[int]
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
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = block_ids

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
        )

class FlagCXConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        logger.info("FlagCX Connector Worker init: %s", engine_id)

        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        self.hostname = get_ip()

        # ---- FlagCX library ----
        library_path = os.getenv("FLAGCX_LIB_PATH")
        if library_path is None:
            flagcx_path = os.getenv("FLAGCX_PATH", "")
            library_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
        self.flagcx = FLAGCXLibrary(library_path)

        # ---- Per-pair comms (lazily created on first transfer with each peer) ----
        # key: remote ZMQ address "host:port+tp_rank", value: PairCommInfo
        self.pair_comms: dict[str, PairCommInfo] = {}
        self.pair_comms_lock = threading.Lock()
        # KV tensor metadata (base_addr, size) — collected in register_kv_caches,
        # used to call flagcxOneSideRegister once per pair comm creation.
        self.kv_tensors_meta: list[tuple[int, int]] = []

        # ---- Side-channel ZMQ port (for block-id metadata) ----
        self.side_channel_port: int = _get_side_channel_port(vllm_config)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.num_blocks = 0

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_workers = (
            vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "num_workers", 10
            )
        )

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: SendReqMeta = SendReqMeta(
            reqs={}, lock=threading.Lock()
        )

        # Background threads
        if self.kv_role != "kv_consumer":
            self._sender_t: threading.Thread | None = None
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="vllm-flagcx-sender",
            )
        if self.kv_role != "kv_producer":
            self.receiver_loop = asyncio.new_event_loop()
            self._receiver_t = threading.Thread(
                target=self._receiver_loop_fn,
                args=(self.receiver_loop,),
                daemon=True,
            )
            self._receiver_t.start()
            # Persistent PUSH sockets keyed by Prefill ZMQ path.
            # Avoids per-request socket creation + linger=0 message loss.
            self._push_sockets: dict[str, zmq.asyncio.Socket] = {}

        self.finished_sending_reqs = FinishedSendReqSet(
            set(), threading.Lock()
        )
        self.finished_recving_reqs = FinishedReceiveReqSet(
            set(), asyncio.Lock()
        )

        # Pending GPU-side signal waits for incoming RDMA transfers
        self._active_signal_waits: list[PendingSignalWait] = []
        self._active_signal_waits_lock = threading.Lock()

        # Attention backend detection
        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            use_mla=self.use_mla,
        )
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        self._block_size: dict[EngineId, int] = {
            self.engine_id: self.block_size
        }
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,
            remote_block_size=self._block_size,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )

        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(FlagCXAgentMetadata)

    def _register_kv_for_comm(self, comm: Any) -> torch.Tensor:
        """Register all KV tensors + a fresh per-pair signal buffer with a
        newly created pair comm.  Both sides call this right after
        flagcxCommInitRank; the internal AllGather in flagcxOneSideRegister
        ensures rendezvous.

        Returns the newly allocated per-pair GPU signal buffer (caller must
        keep a reference to prevent GC)."""
        for base_addr, size in self.kv_tensors_meta:
            self.flagcx.flagcxOneSideRegister(comm, base_addr, size)
        # Allocate a *per-pair* signal buffer so different Prefill peers
        # writing via RDMA never stomp on each other.
        signal_buffer = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.flagcx.flagcxOneSideSignalRegister(
            comm, signal_buffer.data_ptr(), signal_buffer.nbytes
        )
        logger.info(
            "Registered %d KV MRs + per-pair signal buffer for pair comm",
            len(self.kv_tensors_meta),
        )
        return signal_buffer

    def _init_pair_comm_responder(
        self, uid_bytes: bytes, remote_zmq_addr: str
    ) -> Any:
        """Prefill side: given uid_bytes from Decode, init pair comm as rank=1.
        Blocks until both sides have completed CommInitRank + OneSideRegister."""
        with self.pair_comms_lock:
            if remote_zmq_addr in self.pair_comms:
                return self.pair_comms[remote_zmq_addr].comm
        uid = self.flagcx.unique_id_from_bytes(uid_bytes)
        uid_ptr = ctypes.POINTER(flagcxUniqueId)(uid)
        comm = self.flagcx.flagcxCommInitRank(2, uid_ptr, 1)
        signal_buffer = self._register_kv_for_comm(comm)
        with self.pair_comms_lock:
            self.pair_comms[remote_zmq_addr] = PairCommInfo(
                comm=comm, my_rank=1, signal_buffer=signal_buffer,
            )
        logger.info("Pair comm ready (responder/rank=1) ↔ %s", remote_zmq_addr)
        return comm

    def _finalize_pair_comm_initiator(
        self, uid: Any, remote_zmq_addr: str
    ) -> Any:
        """Decode side: called in executor thread after uid has been sent to
        Prefill.  Blocks on CommInitRank(rank=0) + OneSideRegister so both
        sides rendezvous before any PutSignal is issued."""
        comm = self.flagcx.flagcxCommInitRank(2, uid, 0)
        signal_buffer = self._register_kv_for_comm(comm)
        with self.pair_comms_lock:
            self.pair_comms[remote_zmq_addr] = PairCommInfo(
                comm=comm, my_rank=0, signal_buffer=signal_buffer,
            )
        logger.info("Pair comm ready (initiator/rank=0) ↔ %s", remote_zmq_addr)
        return comm

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Collect KV cache tensor metadata for later per-pair registration.

        flagcxOneSideRegister is NOT called here — it is deferred to
        _register_kv_for_comm, which is called when each per-pair comm is
        established so the collective AllGather happens within that pair.
        """
        logger.info("Registering KV caches. use_mla: %s", self.use_mla)

        seen_base_addresses: list[int] = []
        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None

        for layer_name, cache_or_caches in kv_caches.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue
                seen_base_addresses.append(base_addr)
                curr_size = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_size
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_size
                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size

                # Defer collective MR registration to per-pair comm creation
                self.kv_tensors_meta.append((base_addr, curr_size))

        self.kv_caches_base_addr = seen_base_addresses

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches

        logger.info(
            "KV cache metadata collected: %d tensors, num_blocks=%d, "
            "block_len=%d. flagcxOneSideRegister deferred to pair comm init.",
            len(seen_base_addresses),
            self.num_blocks,
            self.block_len,
        )

        # Launch sender thread for P nodes
        if self.kv_role == "kv_consumer":
            return

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
        frontend_path = make_zmq_path("tcp", self.hostname, base_port + tp_rank)
        frontend = make_zmq_socket(self.zmq_ctx, frontend_path, zmq.PULL)

        ready_event.set()

        try:
            while True:
                metadata_bytes = frontend.recv()
                self._sender_executor.submit(
                    self._sender_worker, metadata_bytes,
                )
        except zmq.ContextTerminated:
            pass
        except Exception as e:
            logger.error("FlagCX sender thread error: %s", e)
        finally:
            frontend.close()

    def _sender_worker(self, metadata_bytes: bytes):
        try:
            metadata = self._decoder.decode(metadata_bytes)
            # First contact from this Decode TP rank: init pair comm before
            # transferring.  CommInitRank(rank=1) blocks until Decode (rank=0)
            # also enters it — they rendezvous naturally.
            if metadata.uid_bytes is not None:
                remote_zmq_addr = (
                    f"{metadata.remote_hostname}:"
                    f"{metadata.remote_port + self.tp_rank}"
                )
                self._init_pair_comm_responder(
                    metadata.uid_bytes, remote_zmq_addr
                )
            self._send_kv_to_decode(metadata)
        except Exception as e:
            logger.error("FlagCX sender worker error: %s", e)

    def _send_kv_to_decode(self, meta: FlagCXAgentMetadata):
        send_reqs: list[tuple[ReqId, SendBlockMeta]] = []
        # The Decode ZMQ request may arrive before the scheduler metadata
        # reaches start_load_kv(), so poll with a timeout instead of
        # returning immediately.
        deadline = time.perf_counter() + 30  # 30s timeout
        while True:
            with self.reqs_need_send.lock:
                all_found = True
                for req_id in meta.request_ids:
                    if req_id not in self.reqs_need_send.reqs:
                        all_found = False
                        break
                if all_found:
                    for req_id in meta.request_ids:
                        send_meta = self.reqs_need_send.reqs[req_id]
                        send_meta.expire_time = float("inf")
                        send_reqs.append((req_id, send_meta))
                    break
            if time.perf_counter() > deadline:
                logger.warning(
                    "Timed out waiting for reqs_need_send: %s",
                    meta.request_ids,
                )
                return 0
            time.sleep(0.01)

        self._send_blocks(send_reqs, meta)

        with self.reqs_need_send.lock:
            for req_id in meta.request_ids:
                del self.reqs_need_send.reqs[req_id]

        with self.finished_sending_reqs.lock:
            self.finished_sending_reqs.set.update(meta.request_ids)

    def _send_blocks(
        self,
        send_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: FlagCXAgentMetadata,
    ):
        """RDMA WRITE KV blocks to the remote Decode node using
        flagcxPutSignal.

        Each layer × block-group → one flagcxPutSignal call.
        The last call carries a signal increment so the receiver's
        flagcxWaitSignal unblocks.
        """
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len

        # Look up the per-pair comm for this Decode TP rank.
        remote_zmq_addr = (
            f"{agent_meta.remote_hostname}:"
            f"{agent_meta.remote_port + self.tp_rank}"
        )
        pair_info = self.pair_comms.get(remote_zmq_addr)
        if pair_info is None:
            raise RuntimeError(
                f"No pair comm for {remote_zmq_addr}; "
                "uid_bytes must be included in the first metadata message"
            )
        comm = pair_info.comm
        my_rank = pair_info.my_rank
        peer_rank = 1 - my_rank  # Prefill is rank=1 → peer (Decode) is rank=0

        # Collect (src_offset, dst_offset, size) tuples per layer
        xfer_list: list[tuple[int, int, int, int, int]] = []
        # (layer_local_base, layer_remote_base, local_block_start,
        #  remote_block_start, num_blocks)

        assert len(send_reqs) == len(agent_meta.block_ids)
        for (req_id, send_meta), remote_block_ids in zip(
            send_reqs, agent_meta.block_ids
        ):
            send_meta.ready.wait()

            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            num_local_blocks = len(local_block_ids)
            assert num_local_blocks >= num_remote_blocks
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            group_local, group_remote = _group_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_layer_addr, remote_layer_addr in zip(
                local_base_addr, remote_base_addr
            ):
                for grp_local, grp_remote in zip(group_local, group_remote):
                    xfer_list.append((
                        local_layer_addr,
                        remote_layer_addr,
                        grp_local[0],
                        grp_remote[0],
                        len(grp_local),
                    ))

        if not xfer_list:
            return 0

        # Serialise counter-increment AND all PutSignal enqueues under the
        # per-pair send_lock.  This guarantees that RDMA signal writes land
        # in monotonic order on the remote signal buffer — otherwise a
        # higher signal value could be overwritten by a lower one from a
        # concurrent thread, causing cuStreamWaitValue64(GEQ) to deadlock.
        with pair_info.send_lock:
            pair_info.signal_counter += 1
            expected_signal = pair_info.signal_counter

            start_time = time.perf_counter()
            for i, (
                local_layer_addr, remote_layer_addr,
                local_start, remote_start, n_blocks
            ) in enumerate(xfer_list):
                src_offset = local_start * block_len
                dst_offset = remote_start * block_len
                size = n_blocks * block_len
                is_last = (i == len(xfer_list) - 1)

                # signal_offset = 0: per-pair signal buffer has a single uint64 slot
                signal_offset = 0
                signal_value = expected_signal if is_last else 0

                # For PutSignal we need to express offsets relative to the
                # registered MR base. Since each layer is a separate MR slot,
                # we find its mr_index.
                src_mr_idx = local_base_addr.index(local_layer_addr)
                dst_mr_idx = remote_base_addr.index(remote_layer_addr)

                self.flagcx.flagcxPutSignal(
                    comm, peer_rank,
                    src_offset, dst_offset, size,
                    signal_offset, src_mr_idx, dst_mr_idx,
                    signal_value,
                )

        logger.debug(
            "Queued %d async xfers to rank %d (signal=%d), took %.4f s",
            len(xfer_list),
            peer_rank,
            expected_signal,
            time.perf_counter() - start_time,
        )

        return expected_signal

    def _receiver_loop_fn(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _receive_kv(
        self, path: str, req_blocks: list[tuple[str, list[int]]],
        pending_wait: PendingSignalWait,
    ):
        """Send metadata to Prefiller via ZMQ, requesting it to RDMA-write
        the KV blocks.  On first contact, includes uid_bytes so Prefill can
        init the per-pair comm; Decode concurrently calls CommInitRank in an
        executor so both sides rendezvous without blocking the event loop."""
        req_ids, block_ids = map(list, zip(*req_blocks))

        # Check if this is the first contact with this Prefill TP rank.
        with self.pair_comms_lock:
            need_init = path not in self.pair_comms

        uid: Any = None
        uid_bytes_to_send: Optional[bytes] = None
        if need_init:
            # Generate uid BEFORE sending — Prefill needs it to enter
            # CommInitRank.  The uid is embedded in the metadata message;
            # we then immediately kick off CommInitRank(rank=0) in an executor.
            uid = self.flagcx.flagcxGetUniqueId()
            uid_bytes_to_send = bytes(uid.contents.internal)

        metadata = FlagCXAgentMetadata(
            remote_hostname=self.hostname,
            remote_port=self.side_channel_port,
            request_ids=req_ids,
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_ids=block_ids,
            uid_bytes=uid_bytes_to_send,
        )

        encoded_data = self._encoder.encode(metadata)

        # Use a persistent PUSH socket per Prefill peer.  Creating a new
        # socket per request with linger=0 risks dropping messages if the
        # TCP connection hasn't been established before the socket is closed.
        sock = self._push_sockets.get(path)
        if sock is None:
            sock = make_zmq_socket(
                self.async_zmq_ctx, path, zmq.PUSH, bind=False
            )
            self._push_sockets[path] = sock

        comm_future = None
        try:
            await sock.send(encoded_data)
            if need_init:
                # Kick off CommInitRank(rank=0) in a thread pool so the event
                # loop stays responsive while Prefill's worker thread also
                # enters CommInitRank(rank=1) upon receiving the message above.
                loop = asyncio.get_event_loop()
                comm_future = loop.run_in_executor(
                    None,
                    self._finalize_pair_comm_initiator,
                    uid,
                    path,
                )
            if comm_future is not None:
                await comm_future  # ensure Decode side also finished registering
            # Decode tracks signal counter locally — no need to wait for
            # Prefill's ZMQ reply.  flagcxWaitSignal on the GPU side will
            # block until the RDMA PutSignal from Prefill lands.
            pair_info = self.pair_comms.get(path)
            if pair_info:
                pair_info.signal_counter += 1
                pending_wait.comm = pair_info.comm
                pending_wait.peer_rank = 1 - pair_info.my_rank
                pending_wait.signal_value = pair_info.signal_counter
        except zmq.ContextTerminated:
            return
        except Exception as e:
            logger.error("FlagCX receive_kv failed for %s: %s", req_ids, e)
            return
        finally:
            pending_wait.ready.set()

        async with self.finished_recving_reqs.lock:
            self.finished_recving_reqs.set.update(req_ids)

    def start_load_kv(self, metadata: FlagCXConnectorMetadata):
        if self.kv_role != "kv_producer":
            kv_pulls = self._group_kv_pull(metadata)
            for path, req_blocks in kv_pulls.items():
                pending_wait = PendingSignalWait(
                    req_ids=[rb[0] for rb in req_blocks],
                )
                with self._active_signal_waits_lock:
                    self._active_signal_waits.append(pending_wait)
                asyncio.run_coroutine_threadsafe(
                    self._receive_kv(path, req_blocks, pending_wait),
                    self.receiver_loop,
                )

        if self.kv_role != "kv_consumer":
            with self.reqs_need_send.lock:
                for req_id, block_ids in metadata.reqs_to_send.items():
                    if block_ids:
                        send_meta = self.reqs_need_send.reqs[req_id]
                        send_meta.local_block_ids = block_ids
                        send_meta.ready.set()
                        send_meta.expire_time = (
                            time.perf_counter()
                            + 480
                        )
                    else:
                        self.reqs_need_send.reqs[req_id] = SendBlockMeta(
                            local_block_ids=[], ready=threading.Event()
                        )

    def wait_for_layer_load(self) -> None:
        """Issue GPU-side WaitSignal for all pending RDMA receives.

        Called once per forward step from FlagCXConnector.wait_for_layer_load.
        Drains the active-wait list so subsequent layer calls are no-ops.
        """
        with self._active_signal_waits_lock:
            waits = self._active_signal_waits
            self._active_signal_waits = []
        if not waits:
            return
        torch_stream = torch.cuda.current_stream()
        flagcx_stream = self.flagcx.adaptor_stream_copy(torch_stream)
        try:
            for w in waits:
                w.ready.wait(timeout=60)
                if w.comm is not None and w.signal_value > 0:
                    self.flagcx.flagcxWaitSignal(
                        w.comm, w.peer_rank, 0,
                        w.signal_value, flagcx_stream,
                    )
        finally:
            self.flagcx.adaptor_stream_free(flagcx_stream)

    def _group_kv_pull(self, metadata: FlagCXConnectorMetadata):
        kv_pulls: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
        for req_id, meta in metadata.reqs_to_recv.items():
            path = make_zmq_path(
                "tcp", meta.remote_host, meta.remote_port + self.tp_rank
            )
            kv_pulls[path].append((req_id, meta.local_block_ids))
        return kv_pulls

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

        # Expire stale sends
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
        if self.kv_role != "kv_producer":
            for sock in getattr(self, '_push_sockets', {}).values():
                sock.close()
        self.zmq_ctx.term()
        self.async_zmq_ctx.term()
        if self.kv_role != "kv_consumer":
            self._sender_executor.shutdown(wait=False)
            if self._sender_t:
                self._sender_t.join(timeout=2)
        if self.kv_role != "kv_producer" and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._receiver_t.join(timeout=2)

def _group_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    if len(src_indices) == 0:
        return [], []
    brk = np.where(
        (np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1)
    )[0] + 1
    src_groups = [g.tolist() for g in np.split(src_indices, brk)]
    dst_groups = [g.tolist() for g in np.split(dst_indices, brk)]
    return src_groups, dst_groups


def _get_side_channel_port(vllm_config: VllmConfig) -> int:
    base_port = int(os.getenv("FLAGCX_BOOTSTRAP_PORT", "8998"))
    return (
        base_port
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )
