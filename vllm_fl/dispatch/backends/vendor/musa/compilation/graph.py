# Copyright (c) 2026 BAAI. All rights reserved.
"""MUSA graph input ownership, borrowed outputs, and bounded native admission."""

import dataclasses
import os
import weakref
from contextlib import nullcontext, suppress
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from vllm.config import CUDAGraphMode
from vllm.logger import init_logger

from vllm_fl.compilation.graph import GraphEntry, GraphWrapper
from vllm_fl.dispatch.backends.vendor.musa.compilation.musa_graph_admission import (
    _musa_admission_budget,
)


def _weak_ref_musa_tensors(value: Any) -> Any:
    """Borrow captured outputs while their native graph pool owns the memory.

    These descriptors do not own allocations. Ordinary captured inputs remain
    strongly held by GraphEntry; this helper must not weaken those inputs.
    TorchMUSA 2.9 uses the same storage/metadata APIs in musagraph_trees.py.
    """
    tensors = {}
    storages = {}

    def convert(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "musa":
                return item
            if (
                item.layout != torch.strided
                or item.is_quantized
                or item.requires_grad
                or item.is_conj()
                or item.is_neg()
            ):
                raise ValueError(
                    "MUSA graph outputs require plain strided inference tensors"
                )
            identity = id(item)
            if identity in tensors:
                return tensors[identity]
            source = item.untyped_storage()
            if source.nbytes() == 0:
                return item
            if source.data_ptr() == 0:
                raise ValueError("Nonempty MUSA graph storage has a null address")
            import torch_musa

            native = getattr(
                getattr(torch_musa, "_MUSAC", None),
                "_construct_MUSA_Tensor_From_Storage_And_Metadata",
                None,
            )
            if not callable(native):
                raise RuntimeError(
                    "MUSA graph outputs require TorchMUSA's metadata tensor constructor"
                )
            key = source._cdata
            if key not in storages:
                storages[key] = torch._C._construct_storage_from_data_pointer(
                    source.data_ptr(), item.device, source.nbytes()
                )
            metadata = {
                "size": item.size(),
                "stride": item.stride(),
                "dtype": item.dtype,
                "device": item.device,
                "storage_offset": item.storage_offset(),
            }
            result = native(metadata, storages[key])
            tensors[identity] = result
            return result
        if isinstance(item, list):
            return [convert(child) for child in item]
        if isinstance(item, tuple):
            return tuple(convert(child) for child in item)
        if isinstance(item, dict):
            return {key: convert(child) for key, child in item.items()}

        from vllm.sequence import IntermediateTensors

        if isinstance(item, IntermediateTensors):
            return IntermediateTensors(
                {key: convert(child) for key, child in item.tensors.items()}
            )
        return item

    return convert(value)


@dataclasses.dataclass
class MUSAGraphEntry(GraphEntry):
    # Keep MUSA PIECEWISE capture inputs alive for stable-buffer replay.
    musa_input_leaves: list[Any] | None = None
    musa_input_spec: Any | None = None
    musa_input_metadata: list[Any] | None = None
    musa_input_addresses: list[int | None] | None = None
    musa_last_input_tensor_refs: list[Any] | None = None

    musa_eager_only: bool = False

    musa_capture_device: int | None = None


class MUSAGraphWrapper(GraphWrapper):
    """Replay admitted native graphs and run excess entries with original inputs."""

    entry_type = MUSAGraphEntry
    _graph_type = torch.musa.MUSAGraph

    @staticmethod
    def _musa_tensor_metadata(tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.storage_offset(),
            tensor.requires_grad,
        )

    @staticmethod
    def _musa_tensors_alias(left, right):
        if torch._C._is_alias_of(left, right):
            return True
        if left.device != right.device or left.device.type != "musa":
            return False
        # Borrowed descriptors have distinct StorageImpls. Compare complete
        # physical storage spans as well, conservatively retaining the existing
        # prohibition on staging writes across overlapping input allocations.
        left_storage = left.untyped_storage()
        right_storage = right.untyped_storage()
        left_start = left_storage.data_ptr()
        right_start = right_storage.data_ptr()
        left_size = left_storage.nbytes()
        right_size = right_storage.nbytes()
        return (
            left_size > 0
            and right_size > 0
            and left_start < right_start + right_size
            and right_start < left_start + left_size
        )

    def _stage_musa_inputs(self, entry, args, kwargs):
        """Validate the call and report whether captured buffers can be replayed.

        An arbitrary runnable may mutate inputs or return aliases of them.
        Replaying after copying a fresh allocation into captured storage would
        hide those mutations from the caller and return aliases of the wrong
        allocation. Without a functionalization contract, changed addresses
        therefore execute the runnable with its original incoming arguments.
        """

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        leaves, spec = tree_flatten((args, kwargs))
        require(spec == entry.musa_input_spec, "MUSA graph input structure changed")
        require(
            len(leaves) == len(entry.musa_input_leaves),
            "MUSA graph input count changed",
        )
        stable_inputs = True
        distinct_empty_input = False
        copies = []
        captured_tensors = []
        incoming_tensors = []
        incoming_by_destination = {}
        for captured, incoming, metadata, address, previous_ref in zip(
            entry.musa_input_leaves,
            leaves,
            entry.musa_input_metadata,
            entry.musa_input_addresses,
            entry.musa_last_input_tensor_refs,
        ):
            if isinstance(captured, torch.Tensor):
                require(
                    isinstance(incoming, torch.Tensor),
                    "MUSA graph tensor input type changed",
                )
                require(
                    self._musa_tensor_metadata(captured) == metadata,
                    "Captured MUSA graph input metadata was mutated",
                )
                if incoming is not captured:
                    require(
                        self._musa_tensor_metadata(incoming) == metadata,
                        "MUSA graph input tensor metadata changed",
                    )
                captured_address = captured.data_ptr()
                incoming_address = incoming.data_ptr()
                require(
                    captured_address == address,
                    "Captured MUSA graph input address changed",
                )
                # Empty tensors all report address zero, even when they have
                # unrelated StorageImpls. Only the original captured object has
                # a known identity/alias contract in that case.
                distinct_empty = incoming is not captured and incoming_address == 0
                distinct_empty_input = distinct_empty_input or distinct_empty
                stable_inputs = (
                    stable_inputs
                    and not distinct_empty
                    and incoming_address == address
                    and (
                        incoming is captured
                        or (previous_ref is not None and previous_ref() is incoming)
                    )
                )
                captured_tensors.append(captured)
                incoming_tensors.append(incoming)
                previous = incoming_by_destination.setdefault(id(captured), incoming)
                require(
                    previous.data_ptr() == incoming.data_ptr(),
                    "A repeated MUSA graph input changed its alias relationship",
                )
                if captured_address != incoming_address:
                    require(
                        not self._musa_tensors_alias(captured, incoming),
                        "MUSA graph staging source overlaps its destination",
                    )
                    require(
                        captured.device.type == "musa",
                        "A captured host tensor address changed",
                    )
                    copies.append((captured, incoming))
            else:
                require(
                    type(incoming) is type(captured)
                    and (incoming is captured or incoming == captured),
                    "MUSA graph non-tensor input changed",
                )

        # Unchanged objects, addresses and metadata preserve the alias facts
        # already checked at capture or the last stable-buffer call. Keep this
        # normal path linear, without storage-span queries or pairwise scans.
        if stable_inputs:
            return True

        # Validate the complete changed call before selecting graph or eager.
        for index, captured in enumerate(captured_tensors):
            for other_index in range(index):
                require(
                    self._musa_tensors_alias(captured, captured_tensors[other_index])
                    == self._musa_tensors_alias(
                        incoming_tensors[index], incoming_tensors[other_index]
                    ),
                    "MUSA graph input tensor alias relationship changed",
                )
        # Distinct views or cross-input aliases require a separate staging policy.
        for captured, incoming in copies:
            for other in captured_tensors:
                if other is captured:
                    continue
                require(
                    not self._musa_tensors_alias(captured, other),
                    "MUSA graph changed input aliases another captured input",
                )
                require(
                    not self._musa_tensors_alias(incoming, other),
                    "MUSA graph input aliases another staging destination",
                )
        # Keep all metadata and alias checks above active even for calls that
        # cannot replay. Do not stage or copy mutations back across allocations.
        if copies or distinct_empty_input:
            return False
        # A different descriptor can still refer to the same captured buffer.
        # Remember validated identities weakly, without retaining new owners.
        entry.musa_last_input_tensor_refs = [
            weakref.ref(x) if isinstance(x, torch.Tensor) else None for x in leaves
        ]
        return True

    def _capture_inputs(self, entry, args, kwargs):
        if self.runtime_mode != CUDAGraphMode.PIECEWISE:
            return None
        leaves, spec = tree_flatten((args, kwargs))
        metadata = [
            self._musa_tensor_metadata(x) if isinstance(x, torch.Tensor) else None
            for x in leaves
        ]
        addresses = [
            x.data_ptr() if isinstance(x, torch.Tensor) else None for x in leaves
        ]
        return leaves, spec, metadata, addresses

    def _finish_capture_inputs(self, entry, inputs):
        if inputs is None:
            return
        (
            entry.musa_input_leaves,
            entry.musa_input_spec,
            entry.musa_input_metadata,
            entry.musa_input_addresses,
        ) = inputs
        entry.musa_last_input_tensor_refs = [
            weakref.ref(x) if isinstance(x, torch.Tensor) else None
            for x in entry.musa_input_leaves
        ]

    def _weak_ref_output(self, output):
        return _weak_ref_musa_tensors(output)

    def _admission_log(self, event):
        state = _musa_admission_budget.snapshot()
        init_logger(__name__).info(
            "MUSA graph admission: event=%s class=%s.%s pid=%d "
            "live=%d pending=%d cap=%d total_admitted=%d epoch=%d",
            event,
            type(self).__module__,
            type(self).__qualname__,
            os.getpid(),
            state["live"],
            state["pending"],
            state["cap"],
            state["total_admitted"],
            state["epoch"],
        )

    def _capture_context(self, graph):
        return nullcontext()

    def _capture_graph(self, entry, args, kwargs):
        if entry.musa_eager_only:
            with _musa_admission_budget.eager():
                return self.runnable(*args, **kwargs)
        with _musa_admission_budget.capture() as admitted:
            if admitted:
                entry.musa_capture_device = torch.musa.current_device()
                output = super()._capture_graph(entry, args, kwargs)
                for event in _musa_admission_budget.register(entry.graph):
                    self._admission_log(event)
                return output
            entry.musa_eager_only = True
            if _musa_admission_budget.rejected():
                self._admission_log("first_admission_rejected")
        with _musa_admission_budget.eager():
            return self.runnable(*args, **kwargs)

    def _replay_graph(self, entry, args, kwargs):
        replayed = False
        with (
            _musa_admission_budget.replay(),
            torch.musa.device(entry.musa_capture_device),
        ):
            if entry.musa_input_leaves is None:
                self._validate_replay_addresses(entry, args)
                stable = True
            else:
                stable = self._stage_musa_inputs(entry, args, kwargs)
            if stable:
                with suppress(ImportError, RuntimeError):
                    from vllm.model_executor.offloader.base import get_offloader

                    get_offloader().sync_prev_onload()
                entry.graph.replay()
                replayed = True
        if replayed:
            return entry.output
        with _musa_admission_budget.eager():
            return self.runnable(*args, **kwargs)

    def clear_graphs(self):
        with _musa_admission_budget.clearing():
            super().clear_graphs()
            if _musa_admission_budget.cleared_to_zero():
                self._admission_log("cleared_to_zero")
