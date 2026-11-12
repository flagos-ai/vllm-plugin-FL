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

"""Top-k sharing decisions and indexer preprocessing of HY4 attention.

``compute_skip_topk_layers`` decides which backbone layers reuse a previous
layer's top-k indices (from ``indexer_types`` or the frequency/pattern
fallback), ``is_skip_topk_indexer_weight`` maps that onto checkpoint names,
and ``Indexer.prepare_inputs`` builds the quantized query, key and per-head
weights the lightning indexer consumes. The helpers are checked against
independently computed expectations; ``prepare_inputs`` is checked against a
plain-PyTorch reference (GEMMs, interleaved RoPE, LayerNorm and UE8M0
per-128-group quantization) on the GPU.
"""

import pytest
import torch
import torch.nn.functional as F

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.hy4.attention import (
    Indexer,
    compute_skip_topk_layers,
    is_skip_topk_indexer_weight,
    validate_hy4_parallel_config,
)

pytestmark = [pytest.mark.gpu]

FP8_MAX = 448.0


def _config(**overrides):
    """A sparse-config namespace; defaults describe 4 sparse layers."""
    base = dict(
        index_topk=2048,
        num_hidden_layers=4,
        layer_types=["deepseek_sparse_attention"] * 4,
    )
    base.update(overrides)
    return type("Config", (), base)()


class TestComputeSkipTopkLayers:
    """Which layers reuse a previous layer's top-k indices."""

    def test_no_index_topk_means_no_sharing(self):
        # Configs without the sparse indexer knob (e.g. dense checkpoints
        # converted into the HY4 loader) share no top-k indices.
        """No index top-k means no sharing."""
        config = _config()
        delattr(type(config), "index_topk")
        assert compute_skip_topk_layers(config) == set()

    def test_indexer_types_shared_entries_are_selected(self):
        """Indexer types shared entries are selected."""
        config = _config(indexer_types=["full", "shared", "full", "shared"])
        assert compute_skip_topk_layers(config) == {1, 3}

    def test_indexer_types_wrong_length_is_rejected(self):
        """Indexer types wrong length is rejected."""
        config = _config(indexer_types=["full", "shared"])
        with pytest.raises(ValueError, match="one entry per hidden layer"):
            compute_skip_topk_layers(config)

    @pytest.mark.parametrize("bad", ["FULL", "share", "dense", "FULL ", ""])
    def test_indexer_types_unknown_entry_is_rejected(self, bad):
        """Indexer types unknown entry is rejected."""
        config = _config(indexer_types=["full", bad, "full", "full"])
        with pytest.raises(ValueError, match="only supports 'full' and 'shared'"):
            compute_skip_topk_layers(config)

    def test_indexer_types_shared_before_any_producer_is_rejected(self):
        """Indexer types shared before any producer is rejected."""
        config = _config(indexer_types=["shared", "full", "full", "full"])
        with pytest.raises(ValueError, match="no preceding full sparse producer"):
            compute_skip_topk_layers(config)

    def test_indexer_types_shared_dense_layer_is_allowed(self):
        # A shared layer that is NOT sparse does not need a producer: the
        # producer check only walks sparse layers.
        """Indexer types shared dense layer is allowed."""
        config = _config(
            indexer_types=["shared", "full", "full", "full"],
            layer_types=["dense", "deepseek_sparse_attention", "sparse", "sparse"],
        )
        assert compute_skip_topk_layers(config) == {0}

    def test_frequency_rule_skips_non_multiples(self):
        # offset defaults to 2: layer_idx skips when
        # max(layer_idx - offset + 1, 0) % freq != 0.
        """Frequency rule skips non multiples."""
        config = _config(indexer_types=None)
        assert config.indexer_types is None
        expected = {i for i in range(4) if max(i - 2 + 1, 0) % 1 != 0}
        assert expected == set()
        assert compute_skip_topk_layers(config) == set()

    def test_frequency_with_freq_two_skips_alternating_layers(self):
        """Frequency with freq two skips alternating layers."""
        config = _config(indexer_types=None, index_topk_freq=2)
        # layer 0: max(-1,0)=0 % 2 == 0 -> kept (producer)
        # layer 1: max(0,0)=0 % 2 == 0 -> kept
        # layer 2: 1 % 2 != 0 -> skipped; layer 3: 2 % 2 == 0 -> kept
        assert compute_skip_topk_layers(config) == {2}

    def test_frequency_with_offset_zero(self):
        # offset 0, freq 2, all-sparse: layer 0 is skipped before any
        # producer exists, so the configuration is rejected outright.
        """Frequency with offset zero."""
        config = _config(
            indexer_types=None, index_topk_freq=2, index_skip_topk_offset=0
        )
        with pytest.raises(ValueError, match="no preceding full sparse producer"):
            compute_skip_topk_layers(config)

    def test_frequency_rule_marks_dense_first_layer_sharable(self):
        # offset 0, freq 2 with a dense layer 0: the skipped backbone layer
        # 0 is dense, so the producer walk ignores it and layer 1 produces.
        """Frequency rule marks dense first layer sharable."""
        config = _config(
            indexer_types=None,
            index_topk_freq=2,
            index_skip_topk_offset=0,
            layer_types=["dense", "sparse", "deepseek_sparse_attention", "sparse"],
        )
        assert compute_skip_topk_layers(config) == {0, 2}

    @pytest.mark.parametrize("freq", [0, -1, 1.5, "2", None])
    def test_non_positive_or_non_int_frequency_is_rejected(self, freq):
        """Nonpositive or non integer frequency is rejected."""
        config = _config(indexer_types=None, index_topk_freq=freq)
        with pytest.raises(ValueError, match="index_topk_freq must be a positive"):
            compute_skip_topk_layers(config)

    def test_pattern_marks_skips(self):
        """Pattern marks skips."""
        config = _config(indexer_types=None, index_topk_pattern=["F", "S", "S", "F"])
        assert compute_skip_topk_layers(config) == {1, 2}

    def test_pattern_shorter_than_layer_count_ignores_tail(self):
        # Layers beyond the pattern are simply never marked by it.
        """Pattern shorter than layer count ignores tail."""
        config = _config(indexer_types=None, index_topk_pattern=["F"])
        assert compute_skip_topk_layers(config) == set()

    def test_freq_path_with_large_offset_keeps_every_layer(self):
        # offset 5 makes layers 0..3 all evaluate max(i-4,0)=0, and freq 2
        # keeps 0 % 2 == 0, so nothing is skipped.
        """Freq path with large offset keeps every layer."""
        config = _config(
            indexer_types=None, index_topk_freq=2, index_skip_topk_offset=5
        )
        assert compute_skip_topk_layers(config) == set()


class TestValidateHy4ParallelConfig:
    """PP guard over the shared-top-k decision."""

    def test_pp1_with_shared_layers_is_allowed(self):
        """Allow shared top-k layers on one pipeline stage."""
        config = _config(indexer_types=["full", "shared", "full", "shared"])
        assert validate_hy4_parallel_config(config, 1) is None

    def test_pp_over_1_with_shared_layers_is_rejected(self):
        """PP over 1 with shared layers is rejected."""
        config = _config(indexer_types=["full", "shared", "full", "shared"])
        with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
            validate_hy4_parallel_config(config, 2)

    def test_pp_over_1_without_shared_layers_is_allowed(self):
        """Allow multiple pipeline stages for independent top-k layers."""
        config = _config(indexer_types=["full"] * 4)
        assert validate_hy4_parallel_config(config, 4) is None


class TestIsSkipTopkIndexerWeight:
    """Mapping the skip set onto checkpoint weight names."""

    SKIP = {1, 3}

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("model.layers.1.indexer.wq_b.weight", True),
            ("model.layers.3.indexer.wk_weights_proj.weight", True),
            ("model.layers.0.indexer.wq_b.weight", False),
            ("model.layers.2.indexer.k_cache", False),
            ("model.layers.1.self_attn.o_proj.weight", False),
            ("model.embed_tokens.weight", False),
        ],
    )
    def test_names_are_classified_by_layer_and_suffix(self, name, expected):
        """Names are classified by layer and suffix."""
        assert is_skip_topk_indexer_weight(name, self.SKIP) is expected

    def test_empty_skip_set_disables_the_filter(self):
        """Empty skip set disables the filter."""
        assert (
            is_skip_topk_indexer_weight("model.layers.1.indexer.wq_b.weight", set())
            is False
        )

    def test_names_without_indexer_component_are_ignored(self):
        """Names without indexer component are ignored."""
        assert (
            is_skip_topk_indexer_weight("model.layers.1.mlp.gate.weight", self.SKIP)
            is False
        )

    def test_nested_layer_prefixes_resolve_the_backbone_layer(self):
        # The regex must anchor on a ".layers.N." segment, not on any digit.
        """Nested layer prefixes resolve the backbone layer."""
        assert (
            is_skip_topk_indexer_weight(
                "model.layers.11.indexer.wq_b.weight", self.SKIP
            )
            is False
        )
        assert (
            is_skip_topk_indexer_weight("model.layers.1.indexer.wq_b.weight", self.SKIP)
            is True
        )


def _reference_ue8m0_quant(x: torch.Tensor, group_size: int):
    """Independent UE8M0 per-group FP8 quantization (power-of-two scales)."""
    flat = x.reshape(-1, group_size)
    absmax = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale_raw = absmax / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(scale_raw)))
    q = (flat / scale).clamp(-FP8_MAX, FP8_MAX)
    return q.to(torch.float8_e4m3fn), scale


def _reference_rope_interleaved(
    x: torch.Tensor, positions: torch.Tensor, base: float, rotary_dim: int
):
    """GPT-J-style (pair-interleaved) RoPE applied to the last dim.

    Mirrors ``ApplyRotaryEmb.forward_static`` with ``is_neox_style=False``:
    even/odd elements of each adjacent pair form (x1, x2) and are rotated
    with frequencies ``base**(-2i/rotary_dim)``.
    """
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
    inv_freq = base ** (
        -torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device)
        / rotary_dim
    )
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos, sin = torch.cos(angles), torch.sin(angles)
    expand = lambda t: t.reshape(t.shape[0], *([1] * (x.dim() - 2)), half)
    cos, sin = expand(cos), expand(sin)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    rotated = torch.stack([out1, out2], dim=-1).flatten(-2)
    return torch.cat([rotated, x_pass], dim=-1)


def _quantizer(x: torch.Tensor, group_size: int, use_ue8m0=None):
    """The production FP8 per-token-group quantizer resolved from vLLM."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    return per_token_group_quant_fp8(x, group_size, use_ue8m0=use_ue8m0)


class TestIndexerPrepareInputs:
    """Indexer preprocessing against a plain-PyTorch reference on the GPU."""

    N_HEAD = 4
    HEAD_DIM = 128
    ROPE_DIM = 64
    HIDDEN = 96
    Q_LORA = 32
    TOKENS = 5

    def _make_indexer(self, device):
        from vllm_fl.models.hy4.runtime import HY4RuntimePlan

        hidden_states_weight = torch.randn(
            self.HEAD_DIM + self.N_HEAD, self.HIDDEN, device=device
        )
        wq_weight = torch.randn(self.N_HEAD * self.HEAD_DIM, self.Q_LORA, device=device)
        k_norm_weight = torch.randn(self.HEAD_DIM, device=device)
        k_norm_bias = torch.randn(self.HEAD_DIM, device=device)

        indexer = object.__new__(Indexer)
        indexer.n_head = self.N_HEAD
        indexer.head_dim = self.HEAD_DIM
        indexer.rope_dim = self.ROPE_DIM
        indexer.q_lora_rank = self.Q_LORA
        indexer.softmax_scale = self.HEAD_DIM**-0.5
        indexer.quant_block_size = 128
        indexer.scale_fmt = "ue8m0"

        def wq_b(qr):
            out = F.linear(qr, wq_weight)
            return out, None

        def wk_weights_proj(hidden):
            out = F.linear(hidden, hidden_states_weight)
            return out, None

        indexer.wq_b = wq_b
        indexer.wk_weights_proj = wk_weights_proj
        indexer.k_norm = lambda k: F.layer_norm(
            k, (self.HEAD_DIM,), k_norm_weight, k_norm_bias, 1e-6
        )
        indexer.runtime_plan = HY4RuntimePlan(
            provider="flaggems",
            query_quantizer=_quantizer,
            operations={},
            prefill_backend=None,
            kv_cache_dtypes=("auto", "bfloat16"),
        )
        return indexer, wq_weight, hidden_states_weight, k_norm_weight, k_norm_bias

    def test_outputs_match_independent_reference(self, device):
        """Compare query quantization, rotated keys and weights with PyTorch."""
        torch.manual_seed(11)
        (indexer, wq_weight, wk_weight, k_norm_w, k_norm_b) = self._make_indexer(device)

        qr = torch.randn(self.TOKENS, self.Q_LORA, device=device)
        hidden = torch.randn(self.TOKENS, self.HIDDEN, device=device)
        positions = torch.arange(self.TOKENS, device=device, dtype=torch.int64)
        theta = 10_000_000.0

        def rope(pos, q_pe, k_pe):
            return (
                _reference_rope_interleaved(q_pe, pos, theta, self.ROPE_DIM),
                _reference_rope_interleaved(k_pe, pos, theta, self.ROPE_DIM),
            )

        _, q_fp8, k, weights = indexer.prepare_inputs(hidden, qr, positions, rope)

        # Reference -----------------------------------------------------
        q_ref = F.linear(qr, wq_weight).view(-1, self.N_HEAD, self.HEAD_DIM)
        q_nope, q_pe = torch.split(
            q_ref, [self.HEAD_DIM - self.ROPE_DIM, self.ROPE_DIM], dim=-1
        )
        kw = F.linear(hidden, wk_weight)
        k_ref, weights_raw = kw[:, : self.HEAD_DIM], kw[:, self.HEAD_DIM :]
        k_ref = F.layer_norm(k_ref, (self.HEAD_DIM,), k_norm_w, k_norm_b, 1e-6)
        k_nope, k_pe = torch.split(
            k_ref, [self.HEAD_DIM - self.ROPE_DIM, self.ROPE_DIM], dim=-1
        )
        q_pe_ref = _reference_rope_interleaved(q_pe, positions, theta, self.ROPE_DIM)
        k_pe_ref = _reference_rope_interleaved(k_pe, positions, theta, self.ROPE_DIM)

        q_cat = torch.cat([q_nope, q_pe_ref], dim=-1)
        k_cat = torch.cat([k_nope, k_pe_ref], dim=-1)

        q_flat = q_cat.reshape(-1, self.HEAD_DIM)
        q_q_ref, q_s_ref = _reference_ue8m0_quant(q_flat.float(), 128)
        expected_q = q_q_ref.view(-1, self.N_HEAD, self.HEAD_DIM)
        expected_s = q_s_ref.view(-1, self.N_HEAD, 1)
        expected_weights = (
            weights_raw.unsqueeze(-1)
            * expected_s
            * (self.HEAD_DIM**-0.5)
            * (self.N_HEAD**-0.5)
        ).squeeze(-1)

        # k layout: no_pe first, pe last, MQA (single head).
        assert k.shape == (self.TOKENS, self.HEAD_DIM)
        torch.testing.assert_close(k, k_cat, rtol=1e-4, atol=1e-4)
        assert q_fp8.dtype == torch.float8_e4m3fn
        assert q_fp8.shape == (self.TOKENS, self.N_HEAD, self.HEAD_DIM)
        torch.testing.assert_close(
            q_fp8.float(), expected_q.float(), rtol=1e-2, atol=1e-1
        )
        assert weights.shape == (self.TOKENS, self.N_HEAD)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-3, atol=1e-3)

    def test_quantizer_reference_itself_matches_kernel(self, device):
        # The UE8M0 reference used above is validated against the production
        # quantizer so a drift in either direction is caught.
        """Check the power-of-two scaling reference against the FP8 kernel."""
        torch.manual_seed(17)
        x = torch.randn(64, 128, device=device) * 3.0
        q_kernel, s_kernel = _quantizer(x, 128, use_ue8m0=True)
        q_ref, s_ref = _reference_ue8m0_quant(x, 128)
        torch.testing.assert_close(s_kernel, s_ref, rtol=0, atol=0)
        torch.testing.assert_close(
            q_kernel.float(), q_ref.float(), rtol=1e-2, atol=1e-1
        )
