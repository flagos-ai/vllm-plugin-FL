# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capability gating and RoPE composition for the NVIDIA fast paths.

``nvidia_fast_paths`` must never let a plugin schema stub masquerade as an
executable CUDA kernel. The tests below inject platform/capability answers and
assert which branch is selected; the RoPE tests compare against an explicit
axis-selection reference derived from the MRoPE section contract.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import nvidia_fast_paths as nf
from vllm_fl.models.qwen3_8_flash_next.gpu.indexer_qsa import apply_qsa_rope


def _platform(*, cuda=True, rocm=False, cap90=True, family120=False):
    return SimpleNamespace(
        is_cuda=lambda: cuda,
        is_rocm=lambda: rocm,
        has_device_capability=lambda value: cap90,
        is_device_capability_family=lambda value: family120,
    )


class TestPlatformGates:
    def test_nvidia_platform_excludes_rocm_and_failures(self, monkeypatch):
        """Excludes ROCm, non-CUDA and failing platform probes from NVIDIA."""
        monkeypatch.setattr(nf, "current_platform", _platform())
        assert nf.is_nvidia_platform() is True
        monkeypatch.setattr(nf, "current_platform", _platform(rocm=True))
        assert nf.is_nvidia_platform() is False
        monkeypatch.setattr(nf, "current_platform", _platform(cuda=False))
        assert nf.is_nvidia_platform() is False

        def exploding():
            raise RuntimeError("no platform")

        monkeypatch.setattr(
            nf,
            "current_platform",
            SimpleNamespace(is_cuda=exploding, is_rocm=lambda: False),
        )
        assert nf.is_nvidia_platform() is False

    def test_cuda_kernel_probe_requires_nvidia_platform(self, monkeypatch):
        """Reports no CUDA kernel on a non-NVIDIA platform."""
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: False)
        assert nf._has_cuda_kernel("_C::anything") is False

    def test_cuda_kernel_probe_checks_dispatch_registration(self, monkeypatch):
        """Checks kernel availability through CUDA dispatch registration."""
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: True)
        monkeypatch.setattr(
            nf,
            "torch",
            SimpleNamespace(
                _C=SimpleNamespace(
                    _dispatch_has_kernel_for_dispatch_key=(
                        lambda op, key: op == "_C::good" and key == "CUDA"
                    )
                )
            ),
        )
        assert nf._has_cuda_kernel("_C::good") is True
        assert nf._has_cuda_kernel("_C::missing") is False

    def test_cuda_kernel_probe_swallows_runtime_errors(self, monkeypatch):
        """Treats dispatch probe runtime errors as kernel absence."""
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: True)

        def explode(op, key):
            raise RuntimeError("dispatch unavailable")

        monkeypatch.setattr(
            nf,
            "torch",
            SimpleNamespace(
                _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=explode)
            ),
        )
        assert nf._has_cuda_kernel("_C::op") is False

    def test_native_topk_requires_both_kernels(self, monkeypatch):
        """Requires both cooperative and persistent TopK kernels."""
        seen = []

        def fake(op):
            seen.append(op)
            return op == "_C::cooperative_topk"

        monkeypatch.setattr(nf, "_has_cuda_kernel", fake)
        assert nf.has_native_topk() is False
        assert seen == ["_C::cooperative_topk", "_C::persistent_topk"]

        monkeypatch.setattr(nf, "_has_cuda_kernel", lambda op: True)
        assert nf.has_native_topk() is True


class TestNativeTopk:
    def test_unavailable_kernels_raise(self, monkeypatch):
        """Raises when native QSA TopK is unavailable."""
        monkeypatch.setattr(nf, "has_native_topk", lambda: False)
        with pytest.raises(RuntimeError, match="native QSA TopK is unavailable"):
            nf.native_topk(
                torch.zeros(1, 8),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, 2, dtype=torch.int32),
                torch.zeros(4),
                2,
                8,
            )

    @pytest.mark.parametrize(
        ("rows", "width", "cap90", "family120", "expected"),
        [
            (4, 8, True, False, "cooperative"),
            (40, 8, True, False, "persistent"),
            (4, 3, True, False, "persistent"),
            (4, 8, False, False, "persistent"),
            (4, 8, True, True, "persistent"),
        ],
    )
    def test_dispatch_selection_uses_rows_stride_and_capability(
        self, monkeypatch, rows, width, cap90, family120, expected
    ):
        """Selects cooperative or persistent TopK from rows, stride and capability."""
        calls = []
        fake_ops = SimpleNamespace(
            _C=SimpleNamespace(
                cooperative_topk=lambda *args: calls.append(("cooperative", args)),
                persistent_topk=lambda *args: calls.append(("persistent", args)),
            )
        )
        monkeypatch.setattr(nf, "torch", SimpleNamespace(ops=fake_ops))
        monkeypatch.setattr(nf, "has_native_topk", lambda: True)
        monkeypatch.setattr(
            nf,
            "current_platform",
            _platform(cap90=cap90, family120=family120),
        )
        logits = torch.zeros(rows, width)
        lengths = torch.zeros(rows, dtype=torch.int32)
        output = torch.zeros(rows, 2, dtype=torch.int32)
        workspace = torch.zeros(8)
        nf.native_topk(logits, lengths, output, workspace, 2, 8)
        assert [name for name, _ in calls] == [expected]
        assert calls[0][1][4:] == (2, 8)


class TestNativeCacheUpdate:
    def test_unavailable_kernel_raises(self, monkeypatch):
        """Raises when the fused QSA cache update kernel is unavailable."""
        monkeypatch.setattr(nf, "has_native_cache_update", lambda: False)
        with pytest.raises(RuntimeError, match="fused QSA cache update"):
            nf.native_cache_update(
                torch.zeros(1, 1),
                torch.zeros(1, 1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.int64),
                "auto",
                torch.ones(1),
                torch.ones(1),
            )

    def test_available_kernel_receives_full_abi(self, monkeypatch):
        """Passes the full cache-update ABI to the available kernel."""
        calls = []
        fake_ops = SimpleNamespace(
            _C_cache_ops=SimpleNamespace(
                reshape_and_cache_flash=lambda *args: calls.append(args)
            )
        )
        monkeypatch.setattr(nf, "torch", SimpleNamespace(ops=fake_ops))
        monkeypatch.setattr(nf, "has_native_cache_update", lambda: True)
        key = torch.zeros(2, 1, 4)
        value = torch.zeros(2, 1, 4)
        key_cache = torch.zeros(4, 2, 1, 4)
        value_cache = torch.zeros(4, 2, 1, 4)
        slots = torch.zeros(2, dtype=torch.int64)
        nf.native_cache_update(
            key, value, key_cache, value_cache, slots, "auto", key, value
        )
        assert len(calls) == 1
        assert calls[0][4] is slots
        assert calls[0][5] == "auto"


class FakeRotary(nn.Module):
    """Records the cos/sin composition handed to the native RoPE helper."""

    def __init__(self, cache, *, rotary_dim, sections, interleaved=False):
        super().__init__()
        self.cache = cache
        self.rotary_dim = rotary_dim
        self.mrope_section = sections
        self.mrope_interleaved = interleaved
        self.calls = []
        self.apply_rotary_emb = SimpleNamespace(forward_native=self._forward)

    def _match_cos_sin_cache_dtype(self, tensor):
        return self.cache

    def _forward(self, tensor, cos, sin):
        self.calls.append((cos.clone(), sin.clone()))
        return tensor.clone()


class TestApplyRope:
    """The fallback applies the exact 1-D/MRoPE section composition."""

    def _cache(self, rows=8, width=6):
        return torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)

    def _tensor(self, tokens, width=6):
        return torch.arange(tokens * width, dtype=torch.float32).reshape(
            tokens, 1, width
        )

    @pytest.mark.parametrize("interleaved", [False, True])
    def test_three_axis_sections_select_the_documented_channels(self, interleaved):
        """Selects the documented cos/sin channels for three-axis MRoPE sections."""
        cache = self._cache(width=6)
        rotary = FakeRotary(
            cache,
            rotary_dim=6,
            sections=(2, 1, 0),
            interleaved=interleaved,
        )
        tensor = self._tensor(2)
        positions = torch.tensor([[1, 2], [3, 4], [5, 6]])
        result = apply_qsa_rope(rotary, positions, tensor)
        cos, sin = rotary.calls[0]
        if interleaved:
            expected_cos = torch.tensor(
                [
                    [cache[1, 0], cache[3, 1], cache[1, 2]],
                    [cache[2, 0], cache[4, 1], cache[2, 2]],
                ]
            )
            expected_sin = torch.tensor(
                [
                    [cache[1, 3], cache[3, 4], cache[1, 5]],
                    [cache[2, 3], cache[4, 4], cache[2, 5]],
                ]
            )
        else:
            # Sections (2, 1, 0) take channels [0, 1] from axis 0 and
            # channel [2] from axis 1.
            expected_cos = torch.tensor(
                [
                    [cache[1, 0], cache[1, 1], cache[3, 2]],
                    [cache[2, 0], cache[2, 1], cache[4, 2]],
                ]
            )
            expected_sin = torch.tensor(
                [
                    [cache[1, 3], cache[1, 4], cache[3, 5]],
                    [cache[2, 3], cache[2, 4], cache[4, 5]],
                ]
            )
        torch.testing.assert_close(cos, expected_cos)
        torch.testing.assert_close(sin, expected_sin)
        torch.testing.assert_close(result, tensor)
        # One rotary call, and the fake native kernel echoes the input shape.
        assert len(rotary.calls) == 1
        assert result.shape == tensor.shape
        assert torch.equal(result, tensor)

    def test_one_dimensional_positions_bypass_section_recombination(self):
        """Uses plain cache halves for one-dimensional positions."""
        cache = self._cache(width=6)
        rotary = FakeRotary(cache, rotary_dim=6, sections=(3, 0, 0))
        tensor = self._tensor(2)
        positions = torch.tensor([1, 3])
        result = apply_qsa_rope(rotary, positions, tensor)
        cos, sin = rotary.calls[0]
        torch.testing.assert_close(cos, cache[positions][:, :3])
        torch.testing.assert_close(sin, cache[positions][:, 3:])
        # 1-D positions skip section recombination: cos/sin are the plain cache
        # halves for the two requested positions and the tensor passes through.
        assert cos.shape == (2, 3)
        assert sin.shape == (2, 3)
        assert result.shape == tensor.shape
        assert torch.equal(result, tensor)

    def test_rotated_prefix_is_concatenated_with_untouched_tail(self):
        """Concatenates the rotated prefix with the untouched tail."""
        cache = self._cache(width=6)
        rotary = FakeRotary(cache, rotary_dim=4, sections=(2, 0, 0))
        tensor = self._tensor(2, width=6)
        positions = torch.tensor([0, 1])
        result = apply_qsa_rope(rotary, positions, tensor)
        # forward_native echoes the 4-wide prefix; the 2-wide tail is appended.
        torch.testing.assert_close(result, tensor)
        assert result.shape == (2, 1, 6)
        assert torch.equal(result[..., :4], tensor[..., :4])
        assert torch.equal(result[..., 4:], tensor[..., 4:])


class TestFlashinferNorm:
    def setup_method(self):
        nf._flashinfer_gemma_rmsnorm.cache_clear()

    def teardown_method(self):
        nf._flashinfer_gemma_rmsnorm.cache_clear()

    def _fake_flashinfer(self, monkeypatch, function):
        package = types.ModuleType("flashinfer")
        norm = types.ModuleType("flashinfer.norm")
        norm.gemma_rmsnorm = function
        package.norm = norm
        monkeypatch.setitem(sys.modules, "flashinfer", package)
        monkeypatch.setitem(sys.modules, "flashinfer.norm", norm)

    def test_non_nvidia_host_never_imports_flashinfer(self, monkeypatch):
        """Skips importing FlashInfer norm on a non-NVIDIA host."""
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: False)
        assert nf._flashinfer_gemma_rmsnorm() is None

    def test_import_failure_disables_the_fast_path(self, monkeypatch):
        """Disables the FlashInfer fast path when the import fails."""
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: True)
        monkeypatch.setitem(sys.modules, "flashinfer", None)
        monkeypatch.setitem(sys.modules, "flashinfer.norm", None)
        assert nf._flashinfer_gemma_rmsnorm() is None

    def test_import_success_is_returned_and_cached(self, monkeypatch):
        """Returns and caches the FlashInfer Gemma RMSNorm function."""
        marker = object()

        def gemma_rmsnorm(*args, **kwargs):
            return marker

        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: True)
        self._fake_flashinfer(monkeypatch, gemma_rmsnorm)
        assert nf._flashinfer_gemma_rmsnorm() is gemma_rmsnorm
        assert nf._flashinfer_gemma_rmsnorm() is gemma_rmsnorm

    def test_cpu_tensor_falls_back_even_with_a_kernel(self, monkeypatch):
        """Falls back for CPU tensors even when a kernel is available."""
        monkeypatch.setattr(nf, "_flashinfer_gemma_rmsnorm", lambda: lambda *a: 1)
        tensor = torch.zeros(2, 4)
        assert nf.fast_gemma_rmsnorm(SimpleNamespace(), tensor) is None

    def test_missing_kernel_returns_none(self, monkeypatch):
        """Returns None when no FlashInfer kernel is available."""
        monkeypatch.setattr(nf, "_flashinfer_gemma_rmsnorm", lambda: None)
        assert nf.fast_gemma_rmsnorm(SimpleNamespace(), torch.zeros(2, 4)) is None


class TestMropeKernelProbe:
    def setup_method(self):
        nf._mrope_kernel.cache_clear()

    def teardown_method(self):
        nf._mrope_kernel.cache_clear()

    def _fake_mrope(self, monkeypatch, function):
        module = types.ModuleType("vllm.model_executor.layers.rotary_embedding.mrope")
        module.triton_mrope = function
        monkeypatch.setitem(
            sys.modules, "vllm.model_executor.layers.rotary_embedding.mrope", module
        )

    def test_non_nvidia_host_returns_none(self, monkeypatch):
        """Returns no MRoPE kernel on a non-NVIDIA host."""
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: False)
        assert nf._mrope_kernel() == (None, False)

    def test_import_failure_returns_none(self, monkeypatch):
        """Returns no MRoPE kernel when the import fails."""
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: True)
        monkeypatch.setitem(
            sys.modules, "vllm.model_executor.layers.rotary_embedding.mrope", None
        )
        assert nf._mrope_kernel() == (None, False)

    def test_abi_probe_detects_is_neox_style(self, monkeypatch):
        """Detects whether the MRoPE kernel accepts is_neox_style."""

        def without_neox(*args):
            return None

        def with_neox(*args, is_neox_style=False):
            return None

        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: True)
        self._fake_mrope(monkeypatch, without_neox)
        assert nf._mrope_kernel() == (without_neox, False)
        nf._mrope_kernel.cache_clear()
        self._fake_mrope(monkeypatch, with_neox)
        assert nf._mrope_kernel() == (with_neox, True)


class TestFastQsaRopeGates:
    def test_non_nvidia_or_cpu_tensor_returns_none(self, monkeypatch):
        """Returns None for non-NVIDIA hosts or CPU tensors."""
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: False)
        assert (
            nf.fast_qsa_rope(
                SimpleNamespace(), torch.zeros(1), torch.zeros(1, 1, 1), None, None
            )
            is None
        )
        monkeypatch.setattr(nf, "is_nvidia_platform", lambda: True)
        assert (
            nf.fast_qsa_rope(
                SimpleNamespace(), torch.zeros(1), torch.zeros(1, 1, 4), None, None
            )
            is None
        )


@pytest.mark.gpu
class TestFastQsaRopeCuda:
    def test_one_dimensional_path_uses_forward_cuda(self, device):
        """Uses forward_cuda for one-dimensional RoPE positions."""
        tensor = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 1, 4)
        cos = torch.ones(2, 2, device=device)
        sin = torch.zeros(2, 2, device=device)
        seen = []

        def forward_cuda(prefix, c, s):
            seen.append(prefix)
            return prefix * 2

        rotary = SimpleNamespace(
            rotary_dim=4,
            apply_rotary_emb=SimpleNamespace(forward_cuda=forward_cuda),
        )
        result = nf.fast_qsa_rope(
            rotary, torch.tensor([0, 1], device=device), tensor, cos, sin
        )
        assert len(seen) == 1
        torch.testing.assert_close(result, tensor * 2)

    def test_two_dimensional_path_uses_probed_mrope_abi(self, monkeypatch, device):
        """Calls the probed MRoPE ABI with positional is_neox_style for 2-D positions."""
        tensor = torch.zeros(2, 1, 4, device=device)
        cos = torch.ones(3, 2, 2, device=device)
        sin = torch.zeros(3, 2, 2, device=device)
        calls = []

        # The probed eight-argument ABI is called with ``is_neox_style`` as a
        # ninth *positional* argument; mirror that exact signature.
        def triton_mrope(
            q,
            output,
            cos_arg,
            sin_arg,
            sections,
            head_dim,
            rotary_dim,
            interleaved,
            is_neox_style=False,
        ):
            calls.append(
                (
                    q,
                    output,
                    cos_arg,
                    sin_arg,
                    sections,
                    head_dim,
                    rotary_dim,
                    interleaved,
                    is_neox_style,
                )
            )
            return q * 3, None

        monkeypatch.setattr(nf, "_mrope_kernel", lambda: (triton_mrope, True))
        rotary = SimpleNamespace(
            rotary_dim=4,
            mrope_section=(1, 1, 0),
            mrope_interleaved=False,
            is_neox_style=True,
        )
        result = nf.fast_qsa_rope(
            rotary,
            torch.zeros(3, 2, dtype=torch.long, device=device),
            tensor,
            cos,
            sin,
        )
        assert len(calls) == 1
        (
            q,
            output,
            cos_arg,
            sin_arg,
            sections,
            head_dim,
            rotary_dim,
            interleaved,
            neox,
        ) = calls[0]
        assert neox is True
        assert q.shape == (2, 4)
        assert output.shape == (2, 4)
        assert cos_arg is cos
        assert sin_arg is sin
        assert sections == (1, 1, 0)
        assert (head_dim, rotary_dim) == (4, 4)
        assert interleaved is False
        torch.testing.assert_close(result, torch.zeros_like(tensor))
        assert result.shape == tensor.shape
