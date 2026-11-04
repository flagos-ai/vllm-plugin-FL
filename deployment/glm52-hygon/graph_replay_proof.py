"""Opt-in worker proof: count actual HIP Graph replay, without tensor sampling.

Same bounded logging on both A/B routes; does not enable external RPC endpoints.
"""
import os
import json
from pathlib import Path
import torch

# Workers must load the selected source trees, not merely inherit their paths.
import flag_gems
import vllm_fl
import flagos_compressor
_plugin = Path(__file__).resolve().parents[2]
_sources = {
    "plugin": (vllm_fl, _plugin),
    "flaggems": (flag_gems, Path(os.environ["GLM52_FLAGGEMS_ROOT"])),
    "compressor": (flagos_compressor, Path(os.environ["GLM52_COMPRESSOR_ROOT"])),
}
for _name, (_module, _expected) in _sources.items():
    if not Path(_module.__file__).resolve().is_relative_to(_expected.resolve()):
        raise RuntimeError(f"Unexpected GLM source for {_name}: {_module.__file__}")
print("GLM_RUNTIME_SOURCES " + json.dumps({
    "pid": os.getpid(), **{k: v[0].__file__ for k, v in _sources.items()}
}), flush=True)


class ReplayProofExtension:
    pass


_replay = torch.cuda.CUDAGraph.replay
_count = 0


def replay(graph):
    global _count
    result = _replay(graph)
    _count += 1
    if _count == 1 and os.environ.get('VLLM_FL_HYGON_SPLIT_GRAPH_COMM') == '1':
        from vllm.distributed.parallel_state import get_tp_group
        from vllm_fl.distributed.hygon_split_graph_comm import (
            CaptureSeparatedPyNccl, HygonSplitGraphCommunicator,
        )
        device_comm = get_tp_group().device_communicator
        assert isinstance(device_comm, HygonSplitGraphCommunicator)
        pair = device_comm.pynccl_comm
        assert isinstance(pair, CaptureSeparatedPyNccl)
        assert pair.eager.comm.value != pair.captured.comm.value
        print(f'GLM_SPLIT_COMM_PROOF pid={os.getpid()} world_size={pair.world_size} '
              f'eager={pair.eager.comm.value} captured={pair.captured.comm.value}', flush=True)
    if _count in (1, 128, 256):
        print(f"GLM_GRAPH_REPLAY pid={os.getpid()} count={_count} "
              f"kernel={os.environ.get('VLLM_FL_HYGON_GLM_MLA_KERNEL', 'flaggems')}", flush=True)
    return result


torch.cuda.CUDAGraph.replay = replay
