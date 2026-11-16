"""Instance-local PP transport for Iluvatar's Gloo/FlagCX split.

vLLM's asynchronous tensor-dict API uses the process group directly. Gloo
cannot send device pointers. Keep metadata/CPU tensors on Gloo and route
GPU payloads through FlagCX. This correctness fallback synchronizes device
transfers; it does not claim communication/computation overlap.
"""

from types import MethodType

import torch

from vllm.distributed.parallel_state import TensorMetadata, _split_tensor_dict


class _Completed:
    def wait(self):
        return True

    def is_completed(self):
        return True


def _send(group, tensor_dict, dst=None, all_gather_group=None, all_gather_tensors=None):
    if group.world_size <= 1:
        return []
    if dst is None:
        dst = (group.rank_in_group + 1) % group.world_size
    assert 0 <= dst < group.world_size and dst != group.rank_in_group
    metadata, tensors = _split_tensor_dict(tensor_dict)
    keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
    group.send_object(metadata, dst=dst)
    for key, tensor in zip(keys, tensors):
        if tensor.numel() == 0:
            continue
        if group._should_use_all_gather(
            key, tensor.numel(), all_gather_group, all_gather_tensors
        ):
            tensor = tensor.reshape(all_gather_group.world_size, -1)[
                all_gather_group.rank_in_group
            ]
        if tensor.is_cpu:
            torch.distributed.send(tensor, dst=group.ranks[dst], group=group.cpu_group)
        else:
            group.send(tensor, dst=dst)
            torch.cuda.synchronize(tensor.device)
    # Worker reserves handles[0] for metadata. All transfers are complete.
    return [_Completed()]


def _receive(group, src=None, all_gather_group=None, all_gather_tensors=None):
    if group.world_size <= 1:
        return None, [], []
    if src is None:
        src = (group.rank_in_group - 1) % group.world_size
    assert 0 <= src < group.world_size and src != group.rank_in_group
    result = {}
    postprocess = []
    for key, value in group.recv_object(src=src):
        if not isinstance(value, TensorMetadata):
            result[key] = value
            continue
        full = torch.empty(value.size, dtype=value.dtype, device=value.device)
        result[key] = full
        if full.numel() == 0:
            continue
        split = group._should_use_all_gather(
            key, full.numel(), all_gather_group, all_gather_tensors
        )
        target = full
        if split:
            target = full.reshape(all_gather_group.world_size, -1)[
                all_gather_group.rank_in_group
            ]
        if target.is_cpu:
            torch.distributed.recv(target, src=group.ranks[src], group=group.cpu_group)
        else:
            received = group.recv(target.size(), target.dtype, src)
            target.copy_(received)
            torch.cuda.synchronize(target.device)
        result[key] = target
        if split:

            def gather(key=key, target=target, shape=full.shape):
                result[key] = all_gather_group.all_gather(target, dim=0).reshape(shape)

            postprocess.append(gather)
    return result, [], postprocess


def install_pipeline_transport(group):
    """Bind only the Iluvatar PP instance; leave shared/NVIDIA classes alone."""
    from .communicator import IluvatarCommunicatorFL

    if group.world_size <= 1 or not isinstance(
        group.device_communicator, IluvatarCommunicatorFL
    ):
        return
    if getattr(group, "_iluvatar_pipeline_transport", False):
        return
    group.isend_tensor_dict = MethodType(_send, group)
    group.irecv_tensor_dict = MethodType(_receive, group)
    group._iluvatar_pipeline_transport = True
