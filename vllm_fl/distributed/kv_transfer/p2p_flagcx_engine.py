# Copyright (c) 2026 BAAI. All rights reserved.

import ctypes
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from typing import Any

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
    SendQueueItem,
    set_p2p_nccl_context,
    DEFAULT_MEM_POOL_SIZE_GB,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (
    TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import current_stream

# Load FlagCX wrapper from FLAGCX_PATH
_flagcx_path = os.getenv('FLAGCX_PATH')
if _flagcx_path and os.path.isdir(_flagcx_path):
    if _flagcx_path not in sys.path:
        sys.path.append(_flagcx_path)

from plugin.interservice.flagcx_wrapper import (
    FLAGCXLibrary,
    buffer_type,
    flagcxComm_t,
    flagcxDataTypeEnum,
    flagcxUniqueId,
)

logger = logging.getLogger(__name__)


class P2pFlagcxEngine(P2pNcclEngine):
    """P2P engine using FlagCX for KV cache transfer instead of native NCCL.

    Subclasses P2pNcclEngine and overrides only the communication-backend
    specific methods (__init__, create_connect, listen_for_requests, send,
    recv). All backend-agnostic logic (ZMQ signaling, send/recv queues,
    memory pool, threading) is inherited unchanged.
    """

    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "",
        port_offset: int = 0,
        library_path: str | None = None,
    ) -> None:
        # NOTE: we intentionally do NOT call super().__init__() because it
        # loads NCCLLibrary and starts a listener thread bound to NCCL.
        # Instead, we replicate the init logic with FlagCX as the backend.

        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")

        # --- FlagCX instead of NCCL ---
        if library_path is None:
            flagcx_path = os.getenv('FLAGCX_PATH')
            if flagcx_path:
                library_path = os.path.join(
                    flagcx_path, "build/lib/libflagcx.so"
                )
        self.flagcx = FLAGCXLibrary(library_path)

        # --- rest is identical to P2pNcclEngine.__init__ ---
        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        self.zmq_address = f"{self._hostname}:{self._port}"

        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
            self.http_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port
            http_port = self.config.get_from_extra_config("http_port", None)
            if http_port is None:
                example_cfg = {
                    "kv_connector": "P2pNcclConnector",
                    "kv_connector_extra_config": {"http_port": 8000},
                }
                example = (
                    f"--port=8000 --kv-transfer-config="
                    f"'{json.dumps(example_cfg)}'"
                )
                raise ValueError(
                    "kv_connector_extra_config.http_port is required. "
                    f"Example: {example}"
                )
            self.http_address = f"{self._hostname}:{http_port}"

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        mem_pool_size_gb = float(
            self.config.get_from_extra_config(
                "mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB
            )
        )
        self.pool = TensorMemoryPool(
            max_block_size=int(mem_pool_size_gb * 1024**3)
        )

        self.send_type = self.config.get_from_extra_config(
            "send_type", "PUT_ASYNC"
        )
        if self.send_type == "GET":
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            self.send_queue: deque[SendQueueItem] = deque()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(
                    target=self.send_async, daemon=True
                )
                self._send_thread.start()

        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.socks: dict[str, Any] = {}
        self.comms: dict[str, Any] = {}

        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8"
        )

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True
        )
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(
                target=self.ping, daemon=True
            )
            self._ping_thread.start()

        logger.warning(
            "💯P2pFlagcxEngine init, rank:%d, local_rank:%d, "
            "http_address:%s, zmq_address:%s, proxy_address:%s, "
            "send_type:%s, buffer_size_threshold:%.2f, "
            "nccl_num_channels:%s",
            self.rank,
            self.local_rank,
            self.http_address,
            self.zmq_address,
            self.proxy_address,
            self.send_type,
            self.buffer_size_threshold,
            self.nccl_num_channels,
        )

    # ------------------------------------------------------------------
    # Connection establishment  (FlagCX unique-id / comm-init)
    # ------------------------------------------------------------------

    def create_connect(self, remote_address: str | None = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info(
                    "👋comm exists, remote_address:%s, comms:%s",
                    remote_address,
                    self.comms,
                )
                return sock, self.comms[remote_address]

            # FlagCX: get unique id and serialize for ZMQ
            unique_id_ptr = self.flagcx.flagcxGetUniqueId()
            unique_id = unique_id_ptr.contents
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.accelerator.device_index(self.device.index):
                rank = 0
                with set_p2p_nccl_context(self.nccl_num_channels):
                    comm: flagcxComm_t = self.flagcx.flagcxCommInitRank(
                        2, ctypes.byref(unique_id), rank
                    )
                self.comms[remote_address] = (comm, rank)
                logger.info(
                    "🤝flagcxCommInitRank Success, %s👉%s, MyRank:%s",
                    self.zmq_address,
                    remote_address,
                    rank,
                )

        return self.socks[remote_address], self.comms[remote_address]

    # ------------------------------------------------------------------
    # Listener thread  (FlagCX unique-id deserialization)
    # ------------------------------------------------------------------

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)

            if data["cmd"] == "NEW":
                # FlagCX: reconstruct unique id from bytes
                unique_id = self.flagcx.unique_id_from_bytes(
                    bytes(data["unique_id"])
                )
                with torch.accelerator.device_index(self.device.index):
                    rank = 1
                    with set_p2p_nccl_context(self.nccl_num_channels):
                        comm: flagcxComm_t = self.flagcx.flagcxCommInitRank(
                            2, ctypes.byref(unique_id), rank
                        )
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info(
                        "🤝flagcxCommInitRank Success, %s👈%s, MyRank:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        rank,
                    )

            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                try:
                    with torch.cuda.stream(self.recv_stream):
                        tensor = torch.empty(
                            data["shape"],
                            dtype=getattr(torch, data["dtype"]),
                            device=self.device,
                        )
                    self.router_socket.send_multipart(
                        [remote_address, b"0"]
                    )
                    comm, rank = self.comms[remote_address.decode()]
                    self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                    tensor_size = tensor.element_size() * tensor.numel()
                    if (
                        self.buffer_size + tensor_size
                        > self.buffer_size_threshold
                    ):
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Threshold, "
                            "%s👈%s, data:%s, addr:%d",
                            self.zmq_address,
                            remote_address.decode(),
                            data,
                            addr,
                        )
                    else:
                        self.buffer_size += tensor_size

                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart(
                        [remote_address, b"1"]
                    )
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, "
                        "%s👈%s, data:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                    )

                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()

            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                with self.send_store_cv:
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is not None:
                        data = {
                            "ret": 0,
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                        self.send_store[tensor_id] = tensor
                        self.have_sent_tensor_id(tensor_id)
                    else:
                        data = {"ret": 1}

                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps(data)]
                )

                if data["ret"] == 0:
                    comm, rank = self.comms[remote_address.decode()]
                    self.send(
                        comm,
                        tensor.to(self.device),
                        rank ^ 1,
                        self.send_stream,
                    )
            else:
                logger.warning(
                    "🚧Unexpected, Received message from %s, data:%s",
                    remote_address,
                    data,
                )

    # ------------------------------------------------------------------
    # Data transfer  (FlagCX send / recv with stream adaptation)
    # ------------------------------------------------------------------

    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this flagcx communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            flagcxDataTypeEnum.from_torch(tensor.dtype),
            dst,
            comm,
            flagcx_stream,
        )
        self.flagcx.adaptor_stream_free(flagcx_stream)
        stream.synchronize()

    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this flagcx communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            flagcxDataTypeEnum.from_torch(tensor.dtype),
            src,
            comm,
            flagcx_stream,
        )
        self.flagcx.adaptor_stream_free(flagcx_stream)
        stream.synchronize()
