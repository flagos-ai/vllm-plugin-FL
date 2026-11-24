# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the Iluvatar graph output weak references.

Covers ``vllm_fl/dispatch/backends/vendor/iluvatar/graph.py``:

* CPU tensors pass through ``_weak_ref_iluvatar_tensors`` unchanged through
  the full recursion (list / tuple / dict / IntermediateTensors / scalars),
  mirroring tests/unit_tests/compilation/test_graph.py.
* A tensor whose device reports "cuda" enters the constructor branch; when
  the required PyTorch CUDA construction primitives are unavailable the
  function raises RuntimeError instead of silently retaining the output.

CPU-only (the fake cuda tensor is a subclass whose device property lies).
Vendor-gated to the iluvatar platform.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]


class _FakeCudaTensor(torch.Tensor):
    """A CPU tensor that claims to live on cuda.

    ``_weak_ref_iluvatar_tensor`` only inspects ``value.device.type`` before
    touching the storage/tensor construction primitives, so this subclass
    reliably reaches the cuda branch without any real device.
    """

    @property
    def device(self):
        return torch.device("cuda")


def _fake_cuda(payload: torch.Tensor) -> _FakeCudaTensor:
    return payload.as_subclass(_FakeCudaTensor)


class TestCpuPassthrough:
    def test_plain_cpu_tensor_identity(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        t = torch.arange(4.0)
        assert _weak_ref_iluvatar_tensors(t) is t

    def test_list_tuple_dict_recursion_preserves_identity(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        first = torch.ones(1)
        second = torch.ones(2)
        third = torch.ones(3)
        empty = torch.ones(8)[:0]
        value = {
            "nested": [first, (second, empty)],
            "other": {"deep": third},
        }
        result = _weak_ref_iluvatar_tensors(value)
        assert result["nested"][0] is first
        assert result["nested"][1][0] is second
        assert result["nested"][1][1] is empty
        assert result["other"]["deep"] is third

    def test_intermediate_tensors_recursion(self):
        from vllm.sequence import IntermediateTensors

        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        hidden = torch.zeros(2, 2)
        it = IntermediateTensors({"hidden": hidden})
        result = _weak_ref_iluvatar_tensors(it)
        assert isinstance(result, IntermediateTensors)
        assert result.tensors["hidden"] is hidden
        assert set(result.tensors.keys()) == {"hidden"}

    def test_scalars_and_none_pass_through(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        sentinel = object()
        value = [None, 3, "text", 3.5, True, sentinel, (), []]
        result = _weak_ref_iluvatar_tensors(value)
        assert result[0] is None
        assert result[1] == 3
        assert result[2] == "text"
        assert result[3] == 3.5
        assert result[4] is True
        assert result[5] is sentinel
        assert result[6] == ()
        assert result[7] == []

    def test_nested_empty_containers(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        result = _weak_ref_iluvatar_tensors(
            {"a": [], "b": ({},), "c": {"d": ()}}
        )
        assert result == {"a": [], "b": ({},), "c": {"d": ()}}
        assert isinstance(result["b"], tuple)

    def test_mixed_tree_roundtrip_structure(self):
        from vllm.sequence import IntermediateTensors

        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        t = torch.ones(2)
        value = [t, (t, {"k": t}), IntermediateTensors({"h": t}), None, "x"]
        out = _weak_ref_iluvatar_tensors(value)
        assert isinstance(out, list) and len(out) == 5
        assert out[0] is t and out[1][0] is t and out[1][1]["k"] is t
        assert out[2].tensors["h"] is t
        assert out[3] is None and out[4] == "x"


class TestCudaBranch:
    def test_cuda_tensor_without_primitives_raises(self, monkeypatch):
        """No _construct_CUDA_Tensor_From_Storage_And_Metadata -> RuntimeError."""
        import torch as torch_mod

        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensor,
        )

        fake = _fake_cuda(torch.arange(4.0))
        # this environment lacks the tensor-construction primitive; the
        # storage constructor may exist, the tensor one does not.
        # Force BOTH to be absent so the guard is deterministic.
        saved = {}
        for name in (
            "_construct_storage_from_data_pointer",
            "_construct_CUDA_Tensor_From_Storage_And_Metadata",
        ):
            saved[name] = getattr(torch_mod._C, name, None)
            if saved[name] is not None:
                monkeypatch.delattr(torch_mod._C, name, raising=True)
            else:
                monkeypatch.setattr(
                    torch_mod._C, name, None, raising=False
                )
        with pytest.raises(RuntimeError, match="Iluvatar graph weak references"):
            _weak_ref_iluvatar_tensor(fake)

    def test_non_callable_constructor_raises(self, monkeypatch):
        """Present-but-non-callable constructors also raise RuntimeError."""
        import torch as torch_mod

        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensor,
        )

        fake = _fake_cuda(torch.arange(4.0))
        saved = {}
        for name in (
            "_construct_storage_from_data_pointer",
            "_construct_CUDA_Tensor_From_Storage_And_Metadata",
        ):
            saved[name] = getattr(torch_mod._C, name, None)
            monkeypatch.setattr(torch_mod._C, name, "not-callable", raising=False)
        try:
            with pytest.raises(RuntimeError, match="Iluvatar graph weak references"):
                _weak_ref_iluvatar_tensor(fake)
        finally:
            for name, previous in saved.items():
                if previous is None:
                    monkeypatch.delattr(torch_mod._C, name, raising=False)
                else:
                    setattr(torch_mod._C, name, previous)

    def test_cuda_tensor_in_nested_tree_raises(self):
        """The cuda branch is reached through the recursive entry point too."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        tree = [torch.ones(2), {"gpu": _fake_cuda(torch.arange(3.0))}]
        with pytest.raises(RuntimeError, match="Iluvatar graph weak references"):
            _weak_ref_iluvatar_tensors(tree)
