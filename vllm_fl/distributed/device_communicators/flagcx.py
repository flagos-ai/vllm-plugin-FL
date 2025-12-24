# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/distributed/device_communicators/pynccl.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional, Tuple, Union
import ctypes

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils import current_stream

import os
import sys
sys.path.append(os.getenv('FLAGCX_PATH'))
from plugin.interservice.flagcx_wrapper import (
    FLAGCXLibrary,
    buffer_type,
    flagcxComm_t,
    flagcxDataTypeEnum,
    flagcxUniqueId,
    flagcxRedOpTypeEnum,
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
            ### TODO(lms): simplify it
            if library_path is None:
                flagcx_path = os.getenv('FLAGCX_PATH')
                library_path=os.path.join(flagcx_path, "build/lib/libflagcx.so")
                self.flagcx = FLAGCXLibrary(library_path)
            else:
                self.flagcx = FLAGCXLibrary(library_path)
        except Exception:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.flagcx.flagcxGetUniqueId().contents
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
        # nccl communicator and stream will use this device
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        with torch.cuda.device(device):
            self.comm = self.flagcx.flagcxCommInitRank(
                self.world_size, ctypes.byref(self.unique_id), self.rank)

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data
            
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
            stream = current_stream()
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxAllReduce(buffer_type(in_tensor.data_ptr()),
                                buffer_type(out_tensor.data_ptr()),
                                in_tensor.numel(),
                                flagcxDataTypeEnum.from_torch(in_tensor.dtype),
                                flagcxRedOpTypeEnum.from_torch(op), self.comm,
                                flagcx_stream)
        self.flagcx.adaptor_stream_free(flagcx_stream)
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
            stream = current_stream()
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()), input_tensor.numel(),
            flagcxDataTypeEnum.from_torch(input_tensor.dtype), self.comm,
            flagcx_stream)
        self.flagcx.adaptor_stream_free(flagcx_stream)

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
            stream = current_stream()
        assert output_tensor.shape[0] == sum(sizes)
        split_offset = 0
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxGroupStart()
        for root, split_size in enumerate(sizes):
            dst_slice = output_tensor[split_offset:split_offset + split_size]
            self.flagcx.flagcxBroadcast(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(dst_slice.data_ptr()),
                dst_slice.numel(),
                flagcxDataTypeEnum.from_torch(input_tensor.dtype),
                root,
                self.comm,
                flagcx_stream,
            )
            split_offset += split_size
        self.flagcx.flagcxGroupEnd()
        self.flagcx.adaptor_stream_free(flagcx_stream)

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
            stream = current_stream()
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()), output_tensor.numel(),
            flagcxDataTypeEnum.from_torch(input_tensor.dtype),
            flagcxRedOpTypeEnum.from_torch(op), self.comm,
            flagcx_stream)
        self.flagcx.adaptor_stream_free(flagcx_stream)

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
            stream = current_stream()

        split_offset = 0
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxGroupStart()
        for root, split_size in enumerate(sizes):
            chunk = input_tensor[split_offset:split_offset + split_size, ...]
            self.flagcx.flagcxReduce(
                buffer_type(chunk.data_ptr()),
                buffer_type(output_tensor.data_ptr()), chunk.numel(),
                flagcxDataTypeEnum.from_torch(input_tensor.dtype),
                flagcxRedOpTypeEnum.from_torch(op), root, self.comm,
                flagcx_stream)
            split_offset += split_size
        self.flagcx.flagcxGroupEnd()
        self.flagcx.adaptor_stream_free(flagcx_stream)

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                           flagcxDataTypeEnum.from_torch(tensor.dtype), dst,
                           self.comm, flagcx_stream)
        self.flagcx.adaptor_stream_free(flagcx_stream)

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream = current_stream()
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
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
            stream = current_stream()
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        flagcx_stream = self.flagcx.adaptor_stream_copy(stream)
        self.flagcx.flagcxBroadcast(sendbuff, recvbuff, tensor.numel(),
                                flagcxDataTypeEnum.from_torch(tensor.dtype), src,
                                self.comm, flagcx_stream)
        self.flagcx.adaptor_stream_free(flagcx_stream)

    def group_start(self):
        self.flagcx.flagcxGroupStart()

    def group_end(self):
        self.flagcx.flagcxGroupEnd()
