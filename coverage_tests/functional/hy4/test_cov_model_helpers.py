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

"""Numerical contracts of the HY4 model helpers and checkpoint plumbing.

The hyper-connection residual streams (``HYV4HyperConnection`` /
``HYV4HyperHead``) are compared against plain-PyTorch references for the
sigmoid-gated read/write math; the checkpoint helpers (``HY4LoadCoverage``,
expert slicing, packed-name mapping) are checked against independently
constructed expectations.
"""

import pytest
import torch
import torch.nn.functional as F

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.hy4.config import HYV4Config
from vllm_fl.models.hy4.coverage import HY4LoadCoverage
from vllm_fl.models.hy4.loader import _contiguous_runs, _read_local_experts
from vllm_fl.models.hy4.model import (
    HYV4ForCausalLM,
    HYV4HyperConnection,
    HYV4HyperHead,
    _hyv4_fp32_combine,
)

pytestmark = [pytest.mark.gpu]

RTOL = 1e-4
ATOL = 1e-4


def _tiny_config(**overrides):
    """A dimension-shrunk HYV4Config that still validates."""
    overrides.setdefault("vocab_size", 32)
    overrides.setdefault("hidden_size", 8)
    overrides.setdefault("num_hidden_layers", 2)
    overrides.setdefault("qk_nope_head_dim", 4)
    overrides.setdefault("qk_rope_head_dim", 4)
    overrides.setdefault("qk_head_dim", 8)
    overrides.setdefault("v_head_dim", 4)
    overrides.setdefault("q_lora_rank", 4)
    overrides.setdefault("kv_lora_rank", 6)
    overrides.setdefault("n_routed_experts", 2)
    overrides.setdefault("indexer_types", ["full", "full"])
    return HYV4Config(**overrides)


class TestHyperConnection:
    """Sigmoid-gated collapse of the four residual streams."""

    STREAMS = (3, 4, 8)

    def _module(self, device):
        torch.manual_seed(3)
        module = HYV4HyperConnection(_tiny_config()).to(device)
        with torch.no_grad():
            torch.nn.init.normal_(module.hc_fn, std=0.5)
            torch.nn.init.normal_(module.hc_base, std=0.5)
            torch.nn.init.normal_(module.hc_scale, std=0.5)
        return module

    def test_read_matches_reference(self, device):
        """Read matches reference."""
        module = self._module(device)
        streams = torch.randn(*self.STREAMS, device=device)
        collapsed, write = module.read(streams)

        flat = streams.flatten(1).float()
        norm = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + module.normalize_eps
        )
        gates = F.linear(flat, module.hc_fn) * norm
        read_g, write_g = gates.split(module.num_streams, dim=-1)
        read_ref = (
            torch.sigmoid(
                read_g * module.hc_scale[0] + module.hc_base[: module.num_streams]
            )
            + module.hc_eps
        )
        write_ref = (
            module.magnitude
            * torch.sigmoid(
                write_g * module.hc_scale[1] + module.hc_base[module.num_streams :]
            )
            + module.hc_eps
        )
        collapsed_ref = (read_ref.unsqueeze(-1) * streams.float()).sum(dim=1)

        assert collapsed.shape == (3, 8)
        assert write.shape == (3, 4)
        torch.testing.assert_close(
            collapsed, collapsed_ref.to(streams.dtype), rtol=RTOL, atol=ATOL
        )
        torch.testing.assert_close(write, write_ref, rtol=RTOL, atol=ATOL)

    def test_write_matches_reference_and_preserves_dtype(self, device):
        """Write matches reference and preserves dtype."""
        module = self._module(device)
        streams = torch.randn(*self.STREAMS, device=device)
        delta = torch.randn(3, 8, device=device, dtype=torch.bfloat16)
        _, write = module.read(streams)
        out = HYV4HyperConnection.write(streams, delta, write)

        reference = (
            streams.float() + write.float().unsqueeze(-1) * delta.float().unsqueeze(1)
        ).to(delta.dtype)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), reference.float(), rtol=1e-2, atol=1e-2)

    def test_read_write_round_trip_broadcasts_over_streams(self, device):
        # With zeroed scales/biases every lane gate collapses to
        # sigmoid(0)+eps, so the closed form of read/write is exact.
        """Read write round trip broadcasts over streams."""
        module = self._module(device)
        with torch.no_grad():
            module.hc_base.zero_()
            module.hc_scale.fill_(0.0)
        streams = torch.randn(*self.STREAMS, device=device)
        collapsed, write = module.read(streams)

        gate = torch.sigmoid(torch.zeros(())) + module.hc_eps
        expected_collapse = (gate * streams.float()).sum(dim=1)
        torch.testing.assert_close(
            collapsed.float(), expected_collapse, rtol=RTOL, atol=ATOL
        )

        delta = torch.ones(3, 8, device=device)
        out = HYV4HyperConnection.write(streams, delta, write)
        # write gate = magnitude * sigmoid(0) + eps, added once per lane.
        expected_out = (
            streams.float()
            + (module.magnitude * torch.sigmoid(torch.zeros(())) + module.hc_eps).item()
        )
        torch.testing.assert_close(out.float(), expected_out, rtol=RTOL, atol=ATOL)
        # Closed-form shapes/dtypes: one collapsed lane per token, one write
        # gate per stream, and the stream dtype is preserved.
        assert collapsed.shape == (3, 8)
        assert write.shape == (3, 4)
        assert out.shape == streams.shape
        assert out.dtype == streams.dtype


class TestHyperHead:
    """Final collapse of the residual streams before the last RMSNorm."""

    def test_forward_matches_reference(self, device):
        """Compare the model helper output with the independently computed result."""
        torch.manual_seed(7)
        config = _tiny_config()
        head = HYV4HyperHead(config).to(device)
        with torch.no_grad():
            torch.nn.init.normal_(head.hc_head_fn, std=0.5)
            torch.nn.init.normal_(head.hc_head_base, std=0.5)
            head.hc_head_scale.fill_(0.7)
        streams = torch.randn(3, config.hc_mult, config.hidden_size, device=device)

        flat = streams.flatten(1).float()
        norm = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + head.normalize_eps
        )
        read_ref = (
            torch.sigmoid(
                F.linear(flat, head.hc_head_fn) * norm * head.hc_head_scale
                + head.hc_head_base
            )
            + head.hc_eps
        )
        reference = (read_ref.unsqueeze(-1) * streams.float()).sum(dim=1)

        out = head(streams)
        assert out.shape == (3, config.hidden_size)
        torch.testing.assert_close(out, reference.to(out.dtype), rtol=RTOL, atol=ATOL)


class TestFp32Combine:
    """The MoE runner output combination stays in FP32."""

    def test_sum_of_routed_and_shared(self, device):
        """Return the sum of routed experts and shared expert outputs."""
        routed = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        shared = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        combined = _hyv4_fp32_combine(routed, shared)
        assert combined.dtype == torch.float32
        torch.testing.assert_close(
            combined, routed.float() + shared.float(), rtol=1e-2, atol=1e-2
        )


class TestLoadCoverage:
    """Checkpoint component tracking for fused parameters."""

    def _model(self, expert_map):
        class Owner:
            pass

        Owner.expert_map = expert_map

        class Model:
            # ``getattr``'s default (``range(model.config.n_routed_experts)``)
            # is evaluated even when _hy4_local_expert_ids exists.
            config = type("Cfg", (), {"n_routed_experts": 8})()

            def named_modules(self):
                return [("model.layers.0.mlp.experts.routed_experts", Owner())]

        return Model()

    def _params(self):
        return {
            "model.layers.0.mlp.experts.routed_experts.w13_weight": None,
            "model.layers.0.mlp.experts.routed_experts.w2_weight": None,
            "model.layers.0.mlp.gate_up_proj.weight": None,
            "model.layers.0.self_attn.indexer.wk_weights_proj.weight": None,
            "model.layers.0.self_attn.q_b_proj.weight": None,
        }

    def test_expected_components_from_expert_map(self, device):
        # Only rank-local experts (map >= 0) appear; w13 splits into w1/w3.
        """Expected components from expert map."""
        expert_map = torch.tensor([-1, 0, 1, -1, 2], device=device)
        coverage = HY4LoadCoverage(self._model(expert_map), self._params())
        assert coverage.expected == {
            ("model.layers.0.mlp.experts.routed_experts.w13_weight", 1, "w1"),
            ("model.layers.0.mlp.experts.routed_experts.w13_weight", 2, "w1"),
            ("model.layers.0.mlp.experts.routed_experts.w13_weight", 4, "w1"),
            ("model.layers.0.mlp.experts.routed_experts.w13_weight", 1, "w3"),
            ("model.layers.0.mlp.experts.routed_experts.w13_weight", 2, "w3"),
            ("model.layers.0.mlp.experts.routed_experts.w13_weight", 4, "w3"),
            ("model.layers.0.mlp.experts.routed_experts.w2_weight", 1, "w2"),
            ("model.layers.0.mlp.experts.routed_experts.w2_weight", 2, "w2"),
            ("model.layers.0.mlp.experts.routed_experts.w2_weight", 4, "w2"),
            ("model.layers.0.mlp.gate_up_proj.weight", None, 0),
            ("model.layers.0.mlp.gate_up_proj.weight", None, 1),
            ("model.layers.0.self_attn.indexer.wk_weights_proj.weight", None, 0),
            ("model.layers.0.self_attn.indexer.wk_weights_proj.weight", None, 1),
        }

    def test_expected_components_fall_back_to_local_ids(self, device):
        """Expected components fall back to local ids."""
        model = self._model(None)
        model._hy4_local_expert_ids = [0, 2]
        coverage = HY4LoadCoverage(
            model, {"model.layers.0.mlp.experts.routed_experts.w2_weight": None}
        )
        assert coverage.expected == {
            ("model.layers.0.mlp.experts.routed_experts.w2_weight", 0, "w2"),
            ("model.layers.0.mlp.experts.routed_experts.w2_weight", 2, "w2"),
        }

    def test_record_rejects_duplicates_and_finish_reports_missing(self, device):
        """Record rejects duplicates and finish reports missing."""
        coverage = HY4LoadCoverage(self._model(None), self._params())
        coverage.record("model.layers.0.mlp.gate_up_proj.weight", 0)
        with pytest.raises(ValueError, match="duplicate checkpoint component"):
            coverage.record("model.layers.0.mlp.gate_up_proj.weight", 0)
        with pytest.raises(ValueError, match="coverage failed"):
            coverage.finish()

    class _Param:
        shape = (8, 4)
        dtype = torch.bfloat16
        output_dim = 0
        tp_size = 2
        is_sharded_weight = False
        weight_loader = None

    def test_full_tensor_components_shape_uses_tp_scaling(self, device):
        """Infer component sizes from the tensor-parallel shard."""
        coverage = HY4LoadCoverage(self._model(None), self._params())
        param = self._Param()
        # output_dim 0 with tp_size 2: the fused tensor is twice as tall.
        tensor = torch.zeros(16, 4, device=device)
        components = coverage.full_tensor_components(
            "model.layers.0.mlp.gate_up_proj.weight", param, tensor, 1
        )
        assert components == {
            ("model.layers.0.mlp.gate_up_proj.weight", None, 0),
            ("model.layers.0.mlp.gate_up_proj.weight", None, 1),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            coverage.full_tensor_components(
                "model.layers.0.mlp.gate_up_proj.weight",
                param,
                torch.zeros(8, 4, device=device),
                1,
            )

    def test_full_tensor_components_rejects_fp8_values(self, device):
        """Full-tensor components rejects FP8 values."""
        coverage = HY4LoadCoverage(self._model(None), self._params())
        param = self._Param()
        with pytest.raises(ValueError, match="requires dequantized values"):
            coverage.full_tensor_components(
                "model.layers.0.mlp.gate_up_proj.weight",
                param,
                torch.zeros(16, 4, device=device, dtype=torch.float8_e4m3fn),
                1,
            )

    def test_full_tensor_components_rejects_duplicate_full_tensor(self, device):
        """Full-tensor components rejects duplicate full-tensor."""
        coverage = HY4LoadCoverage(self._model(None), self._params())
        coverage.record("model.layers.0.mlp.gate_up_proj.weight", 0)
        with pytest.raises(ValueError, match="duplicate checkpoint component"):
            coverage.full_tensor_components(
                "model.layers.0.mlp.gate_up_proj.weight",
                self._Param(),
                torch.zeros(16, 4, device=device),
                1,
            )

    def test_full_tensor_components_rejects_direct_routed_tensor(self, device):
        """Full-tensor components rejects direct routed tensor."""
        expert_map = torch.tensor([0], device=device)
        model = self._model(expert_map)
        params = {"model.layers.0.mlp.experts.routed_experts.w13_weight": None}
        coverage = HY4LoadCoverage(model, params)
        with pytest.raises(ValueError, match="direct routed-expert tensor"):
            coverage.full_tensor_components(
                "model.layers.0.mlp.experts.routed_experts.w13_weight",
                self._Param(),
                torch.zeros(16, 4, device=device),
                1,
            )

    def test_full_tensor_components_ignores_untracked_names(self, device):
        """Full-tensor components ignores untracked names."""
        coverage = HY4LoadCoverage(self._model(None), self._params())
        assert (
            coverage.full_tensor_components(
                "model.layers.0.self_attn.q_b_proj.weight",
                self._Param(),
                torch.zeros(8, 4, device=device),
                1,
            )
            == set()
        )


class TestPackedExpertTarget:
    """Checkpoint-name mapping for packed expert tensors."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            (
                "model.layers.3.mlp.experts.gate_up_proj.weight",
                (
                    "model.layers.3.mlp.experts.routed_experts.w13_weight.weight",
                    True,
                ),
            ),
            (
                "model.layers.3.mlp.experts.down_proj.weight",
                (
                    "model.layers.3.mlp.experts.routed_experts.w2_weight.weight",
                    False,
                ),
            ),
            ("model.layers.3.mlp.gate_up_proj.weight", None),
            ("model.layers.3.mlp.experts.gate.weight", None),
        ],
    )
    def test_names_map_to_routed_expert_parameters(self, name, expected):
        """Names map to routed expert parameters."""
        assert HYV4ForCausalLM._packed_expert_target(name) == expected


class TestExpertSlicing:
    """Local-expert selection over safetensors slices."""

    class _Slice:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, sl):
            return self.data[sl]

    @pytest.mark.parametrize(
        "ids,runs",
        [
            ([], []),
            ([3], [(3, 4)]),
            ([1, 2, 3], [(1, 4)]),
            ([0, 2, 3, 5, 6, 7], [(0, 1), (2, 4), (5, 8)]),
        ],
    )
    def test_contiguous_runs(self, ids, runs):
        """Group consecutive expert IDs into contiguous reads."""
        assert _contiguous_runs(ids) == runs

    def test_read_local_experts_concatenates_runs(self, device):
        """Read local experts concatenates runs."""
        data = torch.arange(24, device=device, dtype=torch.float32).view(8, 3)
        out = _read_local_experts(self._Slice(data), [0, 1, 2, 4, 5])
        torch.testing.assert_close(
            out, torch.cat([data[0:3], data[4:6]]), rtol=0, atol=0
        )
        # Runs [0..3) and [4..6) concatenate in expert order: five rows.
        assert out.shape == (5, 3)
        assert out.tolist() == torch.cat([data[0:3], data[4:6]]).tolist()

    def test_read_local_experts_single_run_avoids_concat(self, device):
        """Read local experts single run avoids concat."""
        data = torch.arange(24, device=device, dtype=torch.float32).view(8, 3)
        out = _read_local_experts(self._Slice(data), [2, 3, 4])
        # A single run must return the slice itself, not a copy of a copy.
        assert out.data_ptr() == data[2:5].data_ptr()

    def test_read_local_experts_requires_experts(self, device):
        """Read local experts requires experts."""
        data = torch.zeros(8, 3, device=device)
        with pytest.raises(ValueError, match="requires local experts"):
            _read_local_experts(self._Slice(data), [])
