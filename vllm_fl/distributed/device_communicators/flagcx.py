# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/distributed/device_communicators/pynccl.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import importlib.util
import os
import sys
import threading
from types import SimpleNamespace
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)
_TXDA_ABORT_TIMEOUT_SECONDS = 5.0

_flagcx_path = os.getenv('FLAGCX_PATH')
if _flagcx_path and os.path.isdir(_flagcx_path):
    sys.path.append(_flagcx_path)

def _load_configured_source_api():
    """Pair a new source-built library with its matching standalone wrapper."""
    if not _flagcx_path:
        return None
    source = os.path.join(_flagcx_path, "src", "flagcx", "api.py")
    if not os.path.isfile(source):
        return None
    name = "_vllm_fl_flagcx_source_api"
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == source:
        return cached
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    # dataclasses in the wrapper resolve their defining module through this.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        for symbol in (
            "FLAGCXLibrary", "buffer_type", "flagcxComm_t", "flagcxDataTypeEnum",
            "flagcxUniqueId", "flagcxRedOpTypeEnum",
        ):
            if not hasattr(module, symbol):
                raise ImportError(f"Configured FlagCX wrapper is missing {symbol}: {source}")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_source_api = _load_configured_source_api()
if _source_api is not None:
    FLAGCXLibrary = _source_api.FLAGCXLibrary
    buffer_type = _source_api.buffer_type
    flagcxComm_t = _source_api.flagcxComm_t
    flagcxDataTypeEnum = _source_api.flagcxDataTypeEnum
    flagcxUniqueId = _source_api.flagcxUniqueId
    flagcxRedOpTypeEnum = _source_api.flagcxRedOpTypeEnum
    _flagcx_available = True
else:
    try:
        from plugin.interservice.flagcx_wrapper import (
            FLAGCXLibrary,
            buffer_type,
            flagcxComm_t,
            flagcxDataTypeEnum,
            flagcxUniqueId,
            flagcxRedOpTypeEnum,
        )
        _flagcx_available = True
    except ImportError:
        try:
            # Wheel-packaged images do not contain the legacy source wrapper.
            from flagcx.api import (
                FLAGCXLibrary,
                buffer_type,
                flagcxComm_t,
                flagcxDataTypeEnum,
                flagcxUniqueId,
                flagcxRedOpTypeEnum,
            )
            _flagcx_available = True
        except ImportError:
            _flagcx_available = False
            FLAGCXLibrary = None
            buffer_type = None
            flagcxComm_t = None
            flagcxDataTypeEnum = None
            flagcxUniqueId = None
            flagcxRedOpTypeEnum = None


def _resolve_flagcx_library_path() -> str:
    """Resolve only an explicitly configured source checkout or wheel root."""
    root = os.getenv("FLAGCX_PATH")
    if not root:
        raise RuntimeError("FLAGCX_PATH must be set to use FlagCX")
    candidates = (
        os.path.join(root, "build", "lib", "libflagcx.so"),
        os.path.join(root, "lib", "libflagcx.so"),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"FlagCX library not found under FLAGCX_PATH={root!r}; "
        "expected build/lib/libflagcx.so or lib/libflagcx.so"
    )


class PyFlagcxCommunicator:
    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[str, torch.device],
        library_path: Optional[str] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bound to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        self.comm = None
        self._destroy_lock = threading.Lock()
        self._destroy_started = False
        self._destroy_thread = None

        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            ### TODO(lms)
            assert dist.get_backend(group) != dist.Backend.NCCL, (
                "PyNcclCommunicator should be attached to a non-NCCL group.")
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return
        try:
            if FLAGCXLibrary is None:
                raise RuntimeError("FlagCX Python bindings are not available")
            if library_path is None:
                library_path = _resolve_flagcx_library_path()
            self.flagcx = FLAGCXLibrary(library_path)
        except Exception as exc:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.init_error = exc
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        self._legacy_unique_id_api = (
            hasattr(self.flagcx, "handler")
            and not hasattr(self.flagcx, "devHandle")
        )

        if self.rank == 0:
            # get the unique id from NCCL
            if self._legacy_unique_id_api:
                self.unique_id = self.flagcx.flagcxGetUniqueId().contents
            else:
                self.unique_id = self.flagcx.flagcxGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = flagcxUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        # assert isinstance(device, str)
        if isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        self._sync_collectives = (
            device.type == "txda"
            and os.getenv("VLLM_TXDA_SYNC_COLLECTIVES", "0") == "1"
        )
        self._group_depth = 0
        self._suspended_group_depth = 0
        self._collective_phase_active = False
        # nccl communicator and stream will use this device
        # `torch.cuda.device` / `torch.musa.device` are context managers that
        # change the current device to the specified one
        if self.device.type == "musa":
            device_ctx = torch.musa.device(self.device)
        elif self.device.type == "npu":
            device_ctx = torch.npu.device(self.device)
        elif self.device.type == "ptpu":
            device_ctx = torch.device(self.device)
        elif self.device.type == "txda":
            # Tsingmicro Txda NPU: use the torch.txda device context manager.
            # The torch.cuda.device fallback below has no CUDA runtime on txda.
            device_ctx = torch.txda.device(self.device)
        else:
            device_ctx = torch.cuda.device(self.device)

        with device_ctx:
            if self._legacy_unique_id_api:
                self.comm = self.flagcx.flagcxCommInitRank(
                    self.world_size, ctypes.pointer(self.unique_id), self.rank)
            else:
                self.comm = self.flagcx.flagcxCommInitRank(
                    self.world_size, self.unique_id, self.rank)

            try:
                stream = self._current_stream()
                # A small all_reduce for warmup.
                data = torch.zeros(1, device=device)
                # all_reduce returns a new output allocation. Keep it alive
                # until the asynchronous native collective has completed.
                warmup_output = self.all_reduce(data)
                if not self._sync_collectives:
                    stream.synchronize()
                del data, warmup_output
            except Exception:
                # A returned native handle still needs cleanup if warmup fails.
                # Other vendors retain their existing initialization behavior.
                if self.device.type == "txda":
                    self.destroy()
                raise

    def destroy(self):
        """Stop a TXDA communicator without waiting indefinitely for peers.

        The caller must stop issuing collectives first. This is deliberately
        explicit, not a Python finalizer: ranks may tear down in any order.
        FlagCX Abort releases the underlying TCCL communicator, but the current
        vendor implementation does not free all FlagCX host bookkeeping.
        """
        if getattr(getattr(self, "device", None), "type", None) != "txda":
            return

        with self._destroy_lock:
            if self._destroy_started or self.comm is None:
                return
            self._destroy_started = True
            self.available = False
            self.disabled = True
            library, comm, device = self.flagcx, self.comm, self.device
            self.comm = None

            def abort():
                # Keep the wrapper, CDLL and handle alive even after the caller
                # drops this communicator or the bounded join times out.
                try:
                    # The shipped wrapper exposes Destroy but not Abort. CDLL
                    # indexing creates a separate function object, so binding
                    # its prototype does not mutate a shared cached function.
                    native_abort = library.lib["flagcxCommAbort"]
                    native_abort.argtypes = [flagcxComm_t]
                    native_abort.restype = ctypes.c_int
                    with torch.txda.device(device):
                        library.FLAGCX_CHECK(native_abort(comm))
                except Exception:
                    logger.exception(
                        "TXDA FlagCX communicator abort failed on %s; "
                        "cleanup is incomplete",
                        device,
                    )

            # Like upstream PyNccl, let main-thread teardown continue if native
            # cleanup waits on peers or retained graphs. Never follow Abort with
            # Destroy: the underlying TCCL handle may already have been freed.
            thread = threading.Thread(
                target=abort, name="txda-flagcx-abort", daemon=True
            )
            self._destroy_thread = thread
            try:
                thread.start()
            except Exception:
                logger.exception(
                    "Could not start TXDA FlagCX communicator abort on %s; "
                    "cleanup is incomplete",
                    device,
                )
                return

        thread.join(timeout=_TXDA_ABORT_TIMEOUT_SECONDS)
        if thread.is_alive():
            logger.warning(
                "TXDA FlagCX communicator abort on %s did not finish within "
                "%.1f seconds; cleanup is incomplete and continues in a daemon "
                "thread. Do not reinitialize this communicator in this process",
                device,
                _TXDA_ABORT_TIMEOUT_SECONDS,
            )

    def _current_stream(self):
        if self.device.type == "npu":
            return torch.npu.current_stream(self.device)
        if self.device.type == "txda":
            # The upstream helper may cache the default stream while TXDA
            # graph preparation selects a side stream for model execution.
            return torch.txda.current_stream(self.device)
        return current_stream()

    def _copy_flagcx_stream(self, stream):
        if self.device.type == "npu":
            # FlagCX v0.13.0's wrapper reads cuda_stream even for NPU streams.
            stream = SimpleNamespace(cuda_stream=stream.npu_stream)
        return self.flagcx.adaptor_stream_copy(stream)

    def _cpu_group_barrier(self):
        if isinstance(self.group, StatelessProcessGroup):
            self.group.barrier()
        else:
            dist.barrier(group=self.group)

    def _prepare_collective(self, stream):
        """Run TXDA full-group collectives outside native TCCL groups."""
        if not getattr(self, "_sync_collectives", False):
            return
        if self._collective_phase_active:
            raise RuntimeError("TXDA FlagCX collective phase failed; abort the communicator")
        self._collective_phase_active = True
        self._suspended_group_depth = self._group_depth
        while self._group_depth:
            # Submit preceding P2P work before synchronizing its input stream.
            self._native_group_end()
            self._group_depth -= 1
        stream.synchronize()
        self._cpu_group_barrier()

    def _synchronize_collective_streams(self, streams):
        """Close each TXDA collective phase using the CPU control group."""
        self._cpu_group_barrier()
        for stream in streams:
            stream.synchronize()
        self._cpu_group_barrier()

    def _finish_collective(self, flagcx_stream, stream, *tensors):
        """Retain tensors until opt-in TXDA synchronization has completed."""
        try:
            if getattr(self, "_sync_collectives", False):
                self._synchronize_collective_streams([stream])
        finally:
            # Free only FlagCX's borrowed-stream wrapper, never the native
            # TXDA stream. The arguments retain tensor storage through sync.
            self.flagcx.adaptor_stream_free(flagcx_stream)
        if getattr(self, "_sync_collectives", False):
            # Reopen only after the entire collective phase succeeds. A failed
            # phase leaves the groups suspended and requires communicator abort.
            for _ in range(self._suspended_group_depth):
                self._native_group_start()
                self._group_depth += 1
            self._suspended_group_depth = 0
            self._collective_phase_active = False

    def all_reduce(self,
                   in_tensor: torch.Tensor,
                   out_tensor: torch.Tensor = None,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None) -> torch.Tensor:
        if self.disabled:
            return None
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this flagcx communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}")

        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = self._current_stream()
        self._prepare_collective(stream)
        flagcx_stream = self._copy_flagcx_stream(stream)
        self.flagcx.flagcxAllReduce(buffer_type(in_tensor.data_ptr()),
                                buffer_type(out_tensor.data_ptr()),
                                in_tensor.numel(),
                                flagcxDataTypeEnum.from_torch(in_tensor.dtype),
                                flagcxRedOpTypeEnum.from_torch(op), self.comm,
                                flagcx_stream)
        self._finish_collective(flagcx_stream, stream, in_tensor, out_tensor)
        return out_tensor

    def all_gather(self,
                   output_tensor: torch.Tensor,
                   input_tensor: torch.Tensor,
                   stream=None):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = self._current_stream()
        self._prepare_collective(stream)
        flagcx_stream = self._copy_flagcx_stream(stream)
        self.flagcx.flagcxAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()), input_tensor.numel(),
            flagcxDataTypeEnum.from_torch(input_tensor.dtype), self.comm,
            flagcx_stream)
        self._finish_collective(flagcx_stream, stream, input_tensor, output_tensor)

    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = self._current_stream()
        assert output_tensor.shape[0] == sum(sizes)
        if self.device.type == "npu":
            assert input_tensor.shape[0] == sizes[self.rank]
        split_offset = 0
        sync_collectives = getattr(self, "_sync_collectives", False)
        if not sync_collectives:
            flagcx_stream = self._copy_flagcx_stream(stream)
            self.group_start()
        for root, split_size in enumerate(sizes):
            dst_slice = output_tensor[split_offset:split_offset + split_size]
            if sync_collectives:
                self._prepare_collective(stream)
                flagcx_stream = self._copy_flagcx_stream(stream)
            self.flagcx.flagcxBroadcast(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(dst_slice.data_ptr()),
                dst_slice.numel(),
                flagcxDataTypeEnum.from_torch(input_tensor.dtype),
                root,
                self.comm,
                flagcx_stream,
            )
            if sync_collectives:
                self._finish_collective(
                    flagcx_stream, stream, input_tensor, output_tensor)
            split_offset += split_size
        if not sync_collectives:
            self.group_end()
        if self.device.type == "npu":
            # FlagCX v0.13's Ascend broadcast copies only `count` bytes to the
            # root's distinct receive buffer, so fill its own slice here.
            own_offset = sum(sizes[:self.rank])
            with torch.npu.stream(stream):
                output_tensor[own_offset:own_offset + sizes[self.rank]].copy_(
                    input_tensor)
        if not sync_collectives:
            self._finish_collective(flagcx_stream, stream, input_tensor, output_tensor)

    def reduce_scatter(self,
                       output_tensor: torch.Tensor,
                       input_tensor: torch.Tensor,
                       op: ReduceOp = ReduceOp.SUM,
                       stream=None):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = self._current_stream()
        self._prepare_collective(stream)
        flagcx_stream = self._copy_flagcx_stream(stream)
        self.flagcx.flagcxReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()), output_tensor.numel(),
            flagcxDataTypeEnum.from_torch(input_tensor.dtype),
            flagcxRedOpTypeEnum.from_torch(op), self.comm,
            flagcx_stream)
        self._finish_collective(flagcx_stream, stream, input_tensor, output_tensor)

    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = self._current_stream()

        split_offset = 0
        sync_collectives = getattr(self, "_sync_collectives", False)
        if not sync_collectives:
            flagcx_stream = self._copy_flagcx_stream(stream)
            self.group_start()
        for root, split_size in enumerate(sizes):
            chunk = input_tensor[split_offset:split_offset + split_size, ...]
            if sync_collectives:
                self._prepare_collective(stream)
                flagcx_stream = self._copy_flagcx_stream(stream)
            self.flagcx.flagcxReduce(
                buffer_type(chunk.data_ptr()),
                buffer_type(output_tensor.data_ptr()), chunk.numel(),
                flagcxDataTypeEnum.from_torch(input_tensor.dtype),
                flagcxRedOpTypeEnum.from_torch(op), root, self.comm,
                flagcx_stream)
            if sync_collectives:
                self._finish_collective(
                    flagcx_stream, stream, input_tensor, output_tensor)
            split_offset += split_size
        if not sync_collectives:
            self.group_end()
            self._finish_collective(flagcx_stream, stream, input_tensor, output_tensor)

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        if (getattr(self, "_sync_collectives", False)
                and self._collective_phase_active):
            raise RuntimeError("TXDA FlagCX collective phase failed; abort the communicator")
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self._current_stream()
        flagcx_stream = self._copy_flagcx_stream(stream)
        self.flagcx.flagcxSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                           flagcxDataTypeEnum.from_torch(tensor.dtype), dst,
                           self.comm, flagcx_stream)
        self.flagcx.adaptor_stream_free(flagcx_stream)

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        if (getattr(self, "_sync_collectives", False)
                and self._collective_phase_active):
            raise RuntimeError("TXDA FlagCX collective phase failed; abort the communicator")
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self._current_stream()
        flagcx_stream = self._copy_flagcx_stream(stream)
        self.flagcx.flagcxRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                           flagcxDataTypeEnum.from_torch(tensor.dtype), src,
                           self.comm, flagcx_stream)
        self.flagcx.adaptor_stream_free(flagcx_stream)

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = self._current_stream()
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self._prepare_collective(stream)
        flagcx_stream = self._copy_flagcx_stream(stream)
        self.flagcx.flagcxBroadcast(sendbuff, recvbuff, tensor.numel(),
                                flagcxDataTypeEnum.from_torch(tensor.dtype), src,
                                self.comm, flagcx_stream)
        self._finish_collective(flagcx_stream, stream, tensor)

    def _native_group_start(self):
        # The packaged flagcx.api wrapper requires a communicator on every
        # backend. The legacy source wrapper only takes one for Ascend.
        if not self._legacy_unique_id_api or self.device.type == "npu":
            self.flagcx.flagcxGroupStart(self.comm)
        else:
            self.flagcx.flagcxGroupStart()

    def _native_group_end(self):
        if not self._legacy_unique_id_api or self.device.type == "npu":
            self.flagcx.flagcxGroupEnd(self.comm)
        else:
            self.flagcx.flagcxGroupEnd()

    def group_start(self):
        if self.disabled:
            return
        if (getattr(self, "_sync_collectives", False)
                and self._collective_phase_active):
            raise RuntimeError("TXDA FlagCX collective phase failed; abort the communicator")
        self._native_group_start()
        if getattr(self, "_sync_collectives", False):
            self._group_depth += 1

    def group_end(self):
        if self.disabled:
            return
        sync_collectives = getattr(self, "_sync_collectives", False)
        if sync_collectives and self._collective_phase_active:
            raise RuntimeError("TXDA FlagCX collective phase failed; abort the communicator")
        if sync_collectives and not self._group_depth:
            raise RuntimeError("TXDA FlagCX group_end called without group_start")
        self._native_group_end()
        if sync_collectives:
            self._group_depth -= 1
