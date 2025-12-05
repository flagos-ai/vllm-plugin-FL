from typing import List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase
from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

class CommunicatorFL(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.pyflagcx_comm: Optional[PyFlagcxCommunicator] = None
        if self.world_size > 1:
            self.pyflagcx_comm = PyFlagcxCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        if self.use_all2all:
            from .all2all import NaiveAll2AllManager
            ### naive all2all is device communicator all2all
            self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
            logger.info("Using naive all2all manager.")

    def all_reduce(self, input_):
        assert self.pyflagcx_comm is not None
        out = self.pyflagcx_comm.all_reduce(input_)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
        return out

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size
        pyflagcx_comm = self.pyflagcx_comm
        assert pyflagcx_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output = torch.empty(output_shape,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        pyflagcx_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(self,
                        input_: torch.Tensor,
                        dim: int = -1,
                        sizes: Optional[list[int]] = None):
        world_size = self.world_size
        pyflagcx_comm = self.pyflagcx_comm
        assert pyflagcx_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output = torch.empty(output_shape,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        if sizes is not None:
            pyflagcx_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
        else:
            pyflagcx_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pyflagcx_comm = self.pyflagcx_comm
        if pyflagcx_comm is not None and not pyflagcx_comm.disabled:
            pyflagcx_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pyflagcx_comm = self.pyflagcx_comm
        if pyflagcx_comm is not None and not pyflagcx_comm.disabled:
            pyflagcx_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        if self.pyflagcx_comm is not None:
            self.pyflagcx_comm = None
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None
    
    def all_gatherv(self,
                    input_: Union[torch.Tensor, list[torch.Tensor]],
                    dim: int = 0,
                    sizes: Optional[list[int]] = None):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        pyflagcx_comm = self.pyflagcx_comm
        assert pyflagcx_comm is not None and not pyflagcx_comm.disabled

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor,
                               sizes: Optional[list[int]] = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}")
                output_size = (sum(sizes), ) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size, ) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(output_size,
                                        dtype=input_.dtype,
                                        device=input_.device)
            if sizes is not None:
                pyflagcx_comm.all_gatherv(output_tensor, input_, sizes=sizes)
            else:
                pyflagcx_comm.all_gather(output_tensor, input_)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        pyflagcx_comm.group_start()
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        pyflagcx_comm.group_end()

        return output_list

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.all2all_manager is not None
        hidden_states, router_logits = self.all2all_manager.dispatch(
            hidden_states, router_logits, is_sequence_parallel)
        return hidden_states, router_logits

    def combine(self,
                hidden_states: torch.Tensor,
                is_sequence_parallel: bool = False) -> torch.Tensor:
        assert self.all2all_manager is not None
        hidden_states = self.all2all_manager.combine(hidden_states,
                                                     is_sequence_parallel)
        return hidden_states




