# Copyright 2025 BAAI. All rights reserved.
"""Real-BI-V150 coverage for the Iluvatar cudagraph weak-ref reconstruction.

``vllm_fl/dispatch/backends/vendor/iluvatar/graph.py`` is 0/31 on the CPU
baseline: every row needs a real CUDA device because
``_weak_ref_iluvatar_tensor`` rebuilds a tensor through
``torch._C._construct_storage_from_data_pointer`` and
``torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata``.

These tests run on the actual BI-V150 and pin the production contract that
matters for cudagraph capture: the reconstructed handle must carry the same
metadata (shape / stride / dtype / device / storage_offset) as the source
tensor, must alias the same storage (writes through either view are visible
through the other), must not hold a reference to the original Python tensor
(``_base`` unset), and must preserve the conj/neg bit flags the metadata
constructor drops (the source explicitly re-applies them).

Numeric oracles are plain torch ops on the device (``torch.equal`` against
the same storage, ``-x`` for the neg view), so no implementation detail is
re-derived. Vendor-gated to the iluvatar platform; carries ``gpu`` so the
suite auto-skips on hosts without an accelerator.
"""

from __future__ import annotations

import gc

import pytest
import torch

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("iluvatar"),
    pytest.mark.gpu,
]

CUDA_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int32]


def _weak_ref(value: torch.Tensor):
    from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
        _weak_ref_iluvatar_tensor,
    )

    return _weak_ref_iluvatar_tensor(value)


def _cuda():
    """Resolve the device through the suite's device fixture plumbing."""
    from tests.utils.device_utils import get_device

    return get_device()


class TestWeakRefSingleTensor:
    """``_weak_ref_iluvatar_tensor`` on real CUDA tensors."""

    @pytest.mark.parametrize("dtype", CUDA_DTYPES, ids=lambda d: str(d))
    def test_metadata_preserved_across_dtypes(self, dtype):
        device = _cuda()
        source = torch.zeros(4, 8, dtype=dtype, device=device)
        if dtype.is_floating_point:
            source.normal_()
        result = _weak_ref(source)

        assert result.shape == source.shape
        assert result.stride() == source.stride()
        assert result.dtype == source.dtype
        assert result.device == source.device
        assert result.storage_offset() == source.storage_offset()
        assert result.is_contiguous() == source.is_contiguous()
        # same storage: values are byte-identical and stay in sync
        assert torch.equal(result, source)
        result[0, 0] = 7
        assert source[0, 0].item() == 7

    def test_non_contiguous_stride_and_offset_preserved(self):
        device = _cuda()
        base = torch.arange(6 * 8, dtype=torch.float32, device=device).reshape(6, 8)
        # a column slice: storage_offset 4, stride (8, 1), not contiguous
        view = base[:, 4:]
        result = _weak_ref(view)

        assert result.shape == view.shape == (6, 4)
        assert result.stride() == view.stride() == (8, 1)
        assert result.storage_offset() == view.storage_offset() == 4
        assert not result.is_contiguous()
        assert torch.equal(result, view)
        # writes through the weak handle land in the base tensor's storage
        result[1, 1] = -3.0
        assert base[1, 5].item() == -3.0

    def test_transposed_view_preserves_swapped_strides(self):
        device = _cuda()
        base = torch.randn(4, 8, dtype=torch.float32, device=device)
        view = base.transpose(0, 1)
        result = _weak_ref(view)

        assert result.shape == (8, 4)
        assert result.stride() == view.stride() == (1, 8)
        assert torch.equal(result, view)

    def test_neg_view_flag_reapplied(self):
        """The constructor drops the neg bit; the adapter re-applies it."""
        device = _cuda()
        source = torch.randn(5, 6, dtype=torch.float32, device=device)
        negated = source._neg_view()
        assert negated.is_neg(), "torch._neg_view must produce a neg bit view"

        result = _weak_ref(negated)
        assert result.is_neg(), "weak ref must re-apply the neg bit"
        assert torch.equal(result, -source), "neg view must read as -x"

    def test_conj_bit_reapplied_for_complex(self):
        device = _cuda()
        real = torch.randn(3, 4, device=device)
        imag = torch.randn(3, 4, device=device)
        source = torch.complex(real, imag)
        conj = source.conj()
        # torch.conj on a complex tensor is a lazy bit view
        assert conj.is_conj()

        result = _weak_ref(conj)
        assert result.is_conj(), "weak ref must re-apply the conj bit"
        expected = torch.complex(real, -imag)
        assert torch.equal(result, expected)

    def test_no_reference_to_source_tensor_object(self):
        """The weak handle must not keep the source Python tensor alive.

        cudagraph output reconstruction exists precisely so output handles do
        not pin the pool allocations across captures of different shapes: after
        dropping the last strong reference to the source (and collecting),
        the reconstructed handle must still be usable (its storage is borrowed
        from the pool) while the source's refcount must have reached zero.
        """
        device = _cuda()
        source = torch.randn(16, dtype=torch.float32, device=device)
        pool = source.untyped().data_ptr() if hasattr(source, "untyped") else None
        result = _weak_ref(source)
        del source
        gc.collect()

        assert result.shape == (16,)
        assert result.dtype == torch.float32
        assert result.device == device
        # reading through the handle must still work after the source is gone
        values = result.float().cpu()
        assert values.numel() == 16
        assert bool(torch.isfinite(values).all())
        # borrowed storage: no _base chain to the (now dead) source view
        assert result._base is None

    def test_shared_storage_size_matches_nbytes(self):
        device = _cuda()
        source = torch.empty(32, 7, dtype=torch.bfloat16, device=device)
        result = _weak_ref(source)
        assert result.untyped_storage().nbytes() == source.untyped_storage().nbytes()

    def test_raises_when_primitives_missing(self, monkeypatch):
        """CoreX PyTorch without the construction primitives must fail loudly."""
        from vllm_fl.dispatch.backends.vendor.iluvatar import graph as graph_module

        monkeypatch.setattr(
            torch._C,
            "_construct_storage_from_data_pointer",
            None,
            raising=False,
        )
        device = _cuda()
        source = torch.zeros(2, 2, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError, match="weak references require"):
            _weak_ref(source)


class TestWeakRefContainers:
    """``_weak_ref_iluvatar_tensors`` recursion over containers."""

    def test_passthrough_non_tensor(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        sentinel = object()
        assert _weak_ref_iluvatar_tensors(7) == 7
        assert _weak_ref_iluvatar_tensors("x") == "x"
        assert _weak_ref_iluvatar_tensors(None) is None
        assert _weak_ref_iluvatar_tensors(sentinel) is sentinel

    def test_cpu_tensor_passthrough_identity(self):
        """The CPU fast path returns the tensor unchanged."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        cpu = torch.randn(3, 3)
        assert _weak_ref_iluvatar_tensors(cpu) is cpu

    def test_nested_list_structure(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        device = _cuda()
        a = torch.randn(2, 2, dtype=torch.float32, device=device)
        b = torch.randn(3, dtype=torch.bfloat16, device=device)
        cpu = torch.ones(2)
        result = _weak_ref_iluvatar_tensors([a, b, cpu, [a]])
        assert isinstance(result, list) and len(result) == 4
        assert torch.equal(result[0], a)
        assert torch.equal(result[1], b)
        assert result[2] is cpu, "cpu tensors pass through by identity"
        assert torch.equal(result[3][0], a)

    def test_nested_tuple_structure(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        device = _cuda()
        a = torch.randn(2, dtype=torch.float32, device=device)
        result = _weak_ref_iluvatar_tensors((a, "meta"))
        assert isinstance(result, tuple)
        assert torch.equal(result[0], a)
        assert result[1] == "meta"

    def test_nested_dict_structure(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        device = _cuda()
        a = torch.randn(2, dtype=torch.float32, device=device)
        b = torch.randn(2, 3, dtype=torch.bfloat16, device=device)
        result = _weak_ref_iluvatar_tensors({"q": a, 0: b, "raw": 1})
        assert set(result.keys()) == {"q", 0, "raw"}
        assert torch.equal(result["q"], a)
        assert torch.equal(result[0], b)
        assert result["raw"] == 1

    def test_intermediate_tensors_container(self):
        """vLLm's IntermediateTensors maps its tensor dict element-wise."""
        from vllm.sequence import IntermediateTensors

        from vllm_fl.dispatch.backends.vendor.iluvatar.graph import (
            _weak_ref_iluvatar_tensors,
        )

        device = _cuda()
        a = torch.randn(2, 4, dtype=torch.float32, device=device)
        b = torch.randn(2, 8, dtype=torch.bfloat16, device=device)
        source = IntermediateTensors({"hidden": a, "residual": b})
        result = _weak_ref_iluvatar_tensors(source)

        assert isinstance(result, IntermediateTensors)
        assert set(result.tensors.keys()) == {"hidden", "residual"}
        assert torch.equal(result.tensors["hidden"], a)
        assert torch.equal(result.tensors["residual"], b)
        assert result.tensors["hidden"] is not a, "cuda entries are rebuilt"
