"""Exercise the P800 FlagCX TP4 communicator startup used by the serving gate."""

import os

import torch
import torch.distributed as dist

from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    device = torch.device("cuda", local_rank)
    pci_bus = torch.cuda.get_device_properties(device).pci_bus_id
    print(f"warmup start rank={local_rank} pci_bus={pci_bus}", flush=True)

    # The constructor submits and synchronizes the same one-element FP32 SUM
    # used by serving worker startup before weights are loaded.
    comm = PyFlagcxCommunicator(dist.group.WORLD, device)
    assert not comm.disabled, (local_rank, comm.init_error)
    print(f"warmup passed rank={local_rank} pci_bus={pci_bus}", flush=True)

    source = torch.tensor([local_rank + 1], dtype=torch.float32, device=device)
    result = comm.all_reduce(source)
    assert result is not None
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), torch.tensor([10.0]))
    print(f"post-warmup sum passed rank={local_rank}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
