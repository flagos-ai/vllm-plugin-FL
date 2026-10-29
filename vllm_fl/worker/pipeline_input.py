"""Owned PP receive storage with explicit synchronization and consumption."""

from collections.abc import Callable, Iterable

import torch
from vllm.distributed.parallel_state import Handle
from vllm.sequence import IntermediateTensors


class AsyncIntermediateTensors(IntermediateTensors):
    """Own received tensors until the runner copies them into persistent inputs.

    The producer transfers ownership of the receive dictionary to this object.
    Work/callback lists are copied, so waiting never mutates the producer lists.
    Consumption is explicit and occurs only after copies on the consuming stream.
    """

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        self._comm_handles = tuple(comm_handles or ())
        self._comm_postprocess = tuple(comm_postprocess or ())
        self._comm_waited = False
        super().__init__(tensors)

    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        for handle in self._comm_handles:
            handle.wait()
        for fn in self._comm_postprocess:
            fn()
        self._comm_handles = ()
        self._comm_postprocess = ()
        self._comm_waited = True

    def release_after_copy(self, copied_keys: Iterable[str]) -> None:
        """Release only copied receive entries; preserve any unconsumed entries."""
        if not self._comm_waited:
            raise RuntimeError("Receive communication must complete before consumption")
        tensors = object.__getattribute__(self, "tensors")
        keys = tuple(copied_keys)
        if len(set(keys)) != len(keys) or any(key not in tensors for key in keys):
            raise ValueError("Copied keys must identify unconsumed receive tensors")
        for key in keys:
            del tensors[key]

    def __getattribute__(self, name: str):
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)
