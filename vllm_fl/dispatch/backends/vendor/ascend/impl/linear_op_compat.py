"""Default TP linear-op selection for the copied current Ascend layers.

The full vLLM-Ascend selector contains optional SP, FlashComm and split-TP
branches.  Qwen3.6's validated TP=2 baseline has all of those disabled, for
which the current selector delegates communication to vLLM's standard linear
classes and uses the ordinary TP group.
"""

from __future__ import annotations

from vllm.distributed.parallel_state import get_tp_group


def get_parallel_op(disable_tp, prefix, layer, direct):
    del prefix, layer, direct
    if disable_tp:
        return None, 0, 1
    group = get_tp_group()
    return None, group.rank_in_group, group.world_size


def get_replicated_op(disable_tp, prefix, layer):
    del disable_tp, prefix, layer
    return None, None, None
