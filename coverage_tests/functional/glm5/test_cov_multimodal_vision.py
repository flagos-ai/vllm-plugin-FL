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

"""GLM5-Next vision tower math and processor metadata.

Projection layers are replaced with explicit linear maps so the residual and
clamp algebra is checked against tensors computed in the test.  Position ids
and packed-sequence boundaries are rebuilt with plain loops independent of the
implementation, and processing-info budgets are supplied as plain numbers.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.multimodal.parse import ImageSize

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5 import multimodal as mm


class _FakeConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class TestPatchEmbed:
    def test_convolution_is_applied_per_temporal_patch(self, monkeypatch):
        """Verifies convolution is applied per temporal patch."""
        monkeypatch.setattr(mm, "Conv3dLayer", _FakeConv3d)
        embed = mm.Glm5NextVisionPatchEmbed(
            patch_size=2, temporal_patch_size=1, in_channels=3, hidden_size=5
        )
        with torch.no_grad():
            embed.proj.conv.weight.copy_(
                torch.arange(5 * 3 * 1 * 2 * 2, dtype=torch.float32).reshape(
                    5, 3, 1, 2, 2
                )
                / 10
            )
            embed.proj.conv.bias.zero_()
        token = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        out = embed(token)
        reference = torch.nn.functional.conv3d(
            token.view(2, 3, 1, 2, 2), embed.proj.conv.weight
        ).view(2, 5)
        assert out.shape == (2, 5)
        assert torch.allclose(out, reference)


class _Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t(), None


class _Activation(nn.Module):
    def __init__(self, swiglu_limit):
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return torch.nn.functional.silu(gate) * up


def _install_linear_doubles(monkeypatch):
    def merged(input_size, output_sizes, **kwargs):
        return _Linear(input_size, sum(output_sizes))

    def row(input_size, output_size, **kwargs):
        return _Linear(input_size, output_size)

    def column(input_size, output_size, **kwargs):
        return _Linear(input_size, output_size)

    monkeypatch.setattr(mm, "MergedColumnParallelLinear", merged)
    monkeypatch.setattr(mm, "RowParallelLinear", row)
    monkeypatch.setattr(mm, "ColumnParallelLinear", column)
    monkeypatch.setattr(mm, "SiluAndMulWithClamp", _Activation)


class TestVisionMlp:
    def test_gate_up_values_are_clamped_before_silu(self, monkeypatch):
        """Verifies gate up values are clamped before silu."""
        _install_linear_doubles(monkeypatch)
        mlp = mm.Glm5NextVisionMLP(
            in_features=4, hidden_features=3, swiglu_limit=2.0, bias=False
        )
        with torch.no_grad():
            mlp.gate_up_proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0, 0, 0],
                        [0, 1.0, 0, 0],
                        [0, 0, 1.0, 0],
                        [0, 0, 0, 1.0],
                        [2.0, 0, 0, 0],
                        [0, 2.0, 0, 0],
                    ]
                )
            )
            mlp.down_proj.weight.copy_(torch.eye(4, 3))
        x = torch.tensor([[10.0, -10.0, 1.0, -1.0], [0.5, 0.5, 0.5, 0.5]])
        out = mlp(x)
        gate_up = x @ mlp.gate_up_proj.weight.t()
        gate = gate_up[:, :3].clamp(max=2.0)
        up = gate_up[:, 3:].clamp(min=-2.0, max=2.0)
        expected = (torch.nn.functional.silu(gate) * up) @ mlp.down_proj.weight.t()
        assert torch.allclose(out, expected)


def _attention_layer(head_dim=2, heads=2):
    layer = object.__new__(mm.Glm5NextVisionAttention)
    nn.Module.__init__(layer)
    layer.head_dim = head_dim
    layer.num_attention_heads_per_partition = heads
    layer.hidden_size_per_attention_head = head_dim
    layer.q_norm = lambda x: x * 0.5
    layer.k_norm = lambda x: x + 1
    layer.apply_rotary_emb = lambda qk, cos, sin: qk + 1
    return layer


class TestVisionAttention:
    def test_split_qkv_restores_the_head_layout(self):
        """Verifies split qkv restores the head layout."""
        layer = _attention_layer()
        qkv = torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 12)
        q, k, v = layer.split_qkv(qkv)
        assert q.shape == (2, 1, 2, 2)
        assert k.shape == (2, 1, 2, 2)
        assert torch.equal(q, qkv[:, :, :4].view(2, 1, 2, 2))
        assert torch.equal(v, qkv[:, :, 8:].view(2, 1, 2, 2))

    def test_forward_normalizes_rotates_and_projects(self, monkeypatch):
        """Verifies forward normalizes rotates and projects."""
        monkeypatch.setattr(mm, "use_nvidia_reference", lambda: False)
        layer = _attention_layer()
        qkv = torch.arange(2 * 1 * 12, dtype=torch.float32).reshape(2, 1, 12)
        layer.qkv = lambda x: (qkv, None)
        records = {}

        class FakeAttention:
            def __call__(self, **kwargs):
                records.update(kwargs)
                return kwargs["query"] + kwargs["value"]

        layer.attn = FakeAttention()
        layer.proj = lambda x: ("projected", None)
        out = layer.forward(
            torch.zeros(2, 1, 4),
            cu_seqlens=torch.tensor([0, 2]),
            rotary_pos_emb_cos=torch.zeros(1),
            rotary_pos_emb_sin=torch.zeros(1),
        )
        assert out == "projected"
        assert records["query"].shape == (1, 2, 2, 2)
        assert records["key"].shape == (1, 2, 2, 2)
        assert records["value"].shape == (1, 2, 2, 2)
        expected_q, expected_k, expected_v = layer.split_qkv(qkv)
        expected_q = (expected_q.reshape(-1, 2) * 0.5).view_as(expected_q)
        expected_k = (expected_k.reshape(-1, 2) + 1).view_as(expected_k)
        expected_q = expected_q.transpose(0, 1).contiguous() + 1
        expected_k = expected_k.transpose(0, 1).contiguous() + 1
        assert torch.allclose(records["query"], expected_q)
        assert torch.allclose(records["key"], expected_k)
        assert torch.allclose(records["value"], expected_v.transpose(0, 1).contiguous())

    def test_missing_rotary_embeddings_skip_the_rotation(self, monkeypatch):
        """Verifies missing rotary embeddings skip the rotation."""
        monkeypatch.setattr(mm, "use_nvidia_reference", lambda: False)
        layer = _attention_layer()
        qkv = torch.ones(1, 1, 12)
        layer.qkv = lambda x: (qkv, None)
        calls = []

        class FakeAttention:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return kwargs["query"]

        layer.attn = FakeAttention()
        layer.proj = lambda x: (x, None)
        layer.apply_rotary_emb = lambda *args: pytest.fail("must not rotate")
        layer.forward(
            torch.zeros(1, 1, 4),
            cu_seqlens=torch.tensor([0, 1]),
            rotary_pos_emb_cos=None,
            rotary_pos_emb_sin=None,
        )
        assert len(calls) == 1


class TestVisionBlock:
    def test_residual_order_is_attention_then_mlp(self):
        """Verifies residual order is attention then mlp."""
        block = object.__new__(mm.Glm5NextVisionBlock)
        nn.Module.__init__(block)
        block.norm1 = lambda x: x
        block.norm2 = lambda x: x * 2

        class FixedAttention:
            def __call__(self, x, **kwargs):
                return torch.ones_like(x)

        class FixedMlp:
            def __call__(self, x):
                return torch.full_like(x, 3.0)

        block.attn = FixedAttention()
        block.mlp = FixedMlp()
        x = torch.tensor([[1.0, -1.0]])
        residual = x + 1
        expected = residual + torch.full_like(residual, 3.0)
        assert block(x, None, None, None).tolist() == expected.tolist()


class TestPatchMerger:
    def test_projection_norm_gelu_and_clamped_swiglu(self, monkeypatch):
        """Verifies projection norm gelu and clamped swiglu."""
        _install_linear_doubles(monkeypatch)
        merger = mm.Glm5NextPatchMerger(
            d_model=4, context_dim=3, swiglu_limit=2.0, bias=False
        )
        with torch.no_grad():
            merger.proj.weight.copy_(torch.eye(4))
            merger.gate_up_proj.weight.copy_(
                torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
            )
            merger.down_proj.weight.copy_(
                torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
            )
        x = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        out = merger(x)
        h = torch.nn.functional.gelu(merger.post_projection_norm(x))
        gate_up = h @ merger.gate_up_proj.weight.t()
        gate = gate_up[:, :3].clamp(max=2.0)
        up = gate_up[:, 3:].clamp(min=-2.0, max=2.0)
        expected = (torch.nn.functional.silu(gate) * up) @ merger.down_proj.weight.t()
        assert torch.allclose(out, expected)


class TestRotaryPositions:
    def test_two_grids_match_the_plain_python_layout(self):
        """Verifies two grids match the plain python layout."""
        layer = object.__new__(mm.Glm5NextVisionTransformer)
        nn.Module.__init__(layer)
        layer.spatial_merge_size = 2
        calls = []

        class FakeRope:
            def get_cos_sin(self, size):
                calls.append(size)
                cos = torch.arange(size * 4, dtype=torch.float32).reshape(size, 4)
                return cos, cos + 100

        layer.rotary_pos_emb = FakeRope()
        grid = [[1, 4, 4], [2, 2, 2]]
        cos, sin, pos_ids = layer.rot_pos_emb(grid)
        assert calls == [4]
        packed = []
        for repeat, height, width in grid:
            hpos = torch.arange(height).unsqueeze(1).expand(-1, width)
            wpos = torch.arange(width).unsqueeze(0).expand(height, -1)
            hpos = (
                hpos.reshape(height // 2, 2, width // 2, 2)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos = (
                wpos.reshape(height // 2, 2, width // 2, 2)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            packed.append(torch.stack([hpos, wpos], dim=-1).repeat(repeat, 1))
        reference_ids = torch.cat(packed, dim=0)
        assert torch.equal(pos_ids, reference_ids)
        assert cos.shape == (24, 8)
        assert torch.equal(cos, torch.arange(16).reshape(4, 4)[pos_ids].flatten(1))
        assert torch.equal(
            sin, (torch.arange(16).reshape(4, 4) + 100)[pos_ids].flatten(1)
        )


class TestAttentionMask:
    @pytest.mark.parametrize(
        ("backend", "expected"),
        [
            (mm.AttentionBackendEnum.FLASH_ATTN, 3),
            (mm.AttentionBackendEnum.TRITON_ATTN, 3),
            (mm.AttentionBackendEnum.ROCM_AITER_FA, 3),
            ("other", None),
        ],
    )
    def test_max_seqlen_only_for_length_aware_backends(self, backend, expected):
        """Verifies max seqlen only for length aware backends."""
        layer = object.__new__(mm.Glm5NextVisionTransformer)
        layer.attn_backend = backend
        got = layer.compute_attn_mask_seqlen(torch.tensor([0, 2, 5], dtype=torch.int32))
        if expected is None:
            assert got is None
        else:
            assert int(got) == expected


class TestPrepareEncoderMetadata:
    def _layer(self):
        layer = object.__new__(mm.Glm5NextVisionTransformer)
        nn.Module.__init__(layer)
        layer.hidden_size = 8
        layer.tp_size = 2
        layer.attn_backend = "backend"
        layer.rot_pos_emb = lambda grid: (
            torch.zeros(10, 4),
            torch.ones(10, 4),
            None,
        )
        return layer

    def test_sequence_boundaries_are_padded_to_the_capture_batch(self, monkeypatch):
        """Verifies sequence boundaries are padded to the capture batch."""
        recorded = {}

        class FakeEncoderAttention:
            @staticmethod
            def maybe_compute_seq_lens(backend, cu_seqlens, device):
                recorded["seq_backend"] = backend
                return torch.as_tensor(cu_seqlens, dtype=torch.int32).clone()

            @staticmethod
            def compute_max_seqlen(backend, cu_seqlens):
                return max(
                    int(cu_seqlens[index + 1] - cu_seqlens[index])
                    for index in range(len(cu_seqlens) - 1)
                )

            @staticmethod
            def maybe_recompute_cu_seqlens(backend, cu_seqlens, hidden, tp, device):
                recorded["recompute"] = (hidden, tp)
                return "cu"

        monkeypatch.setattr(mm, "MMEncoderAttention", FakeEncoderAttention)
        layer = self._layer()
        metadata = layer.prepare_encoder_metadata(
            [[1, 2, 2], [1, 1, 1]],
            max_batch_size=4,
            device=torch.device("cpu"),
        )
        assert metadata["sequence_lengths"].tolist() == [0, 4, 5, 5, 5]
        assert metadata["max_seqlen"].item() == 4
        assert metadata["cu_seqlens"] == "cu"
        assert metadata["rotary_pos_emb_cos"].shape == (10, 4)
        assert recorded["seq_backend"] == "backend"
        assert recorded["recompute"] == (8, 2)

    def test_max_seqlen_override_and_frame_padding(self, monkeypatch):
        """Verifies max seqlen override and frame padding."""

        class FakeEncoderAttention:
            @staticmethod
            def maybe_compute_seq_lens(*args):
                return None

            @staticmethod
            def compute_max_seqlen(*args):
                return 99

            @staticmethod
            def maybe_recompute_cu_seqlens(*args):
                return "cu"

        monkeypatch.setattr(mm, "MMEncoderAttention", FakeEncoderAttention)
        layer = self._layer()
        metadata = layer.prepare_encoder_metadata(
            [[2, 2, 2]],
            max_frames_per_batch=3,
            max_seqlen_override=7,
            device=torch.device("cpu"),
        )
        assert metadata["max_seqlen"].item() == 7
        # One frame yields four patches; two empty frames pad the boundary.
        assert metadata["cu_seqlens"] == "cu"

    def test_dtype_and_device_follow_the_patch_embedding(self):
        """Verifies dtype and device follow the patch embedding."""
        layer = object.__new__(mm.Glm5NextVisionTransformer)
        layer.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16))
        )
        assert layer.dtype is torch.bfloat16
        assert layer.device == torch.device("cpu")


class _PatchEmbedding:
    def __init__(self, dtype=torch.float32):
        self.proj = SimpleNamespace(weight=torch.zeros(1, dtype=dtype))
        self.calls = []

    def __call__(self, x):
        self.calls.append(x)
        return x


class _Downsample:
    def __init__(self, out_channels):
        self.out_channels = out_channels
        self.calls = []

    def __call__(self, x):
        self.calls.append(x)
        pooled = x.mean(dim=(2, 3), keepdim=True)
        return pooled[:, : self.out_channels]


class TestVisionTransformerForward:
    def _layer(self):
        layer = object.__new__(mm.Glm5NextVisionTransformer)
        nn.Module.__init__(layer)
        layer.patch_embed = _PatchEmbedding()
        layer.spatial_merge_size = 2
        layer.out_hidden_size = 4
        layer.post_layernorm = lambda x: x
        layer.downsample = _Downsample(out_channels=4)

        class Block:
            def __init__(self):
                self.calls = []

            def __call__(self, x, **kwargs):
                self.calls.append(kwargs)
                return x + 1

        layer.blocks = [Block(), Block()]
        layer.merger = lambda x: x * 2
        return layer

    def test_encoder_metadata_skips_the_cpu_rebuild(self):
        """Verifies encoder metadata skips the cpu rebuild."""
        layer = self._layer()
        metadata = {
            "rotary_pos_emb_cos": "cos",
            "rotary_pos_emb_sin": "sin",
            "cu_seqlens": "cu",
            "max_seqlen": "max",
        }
        x = torch.zeros(4, 12)
        out = layer.forward(x, [[1, 2, 2]], encoder_metadata=metadata)
        assert out.shape == (1, 4)
        assert [block.calls[0]["cu_seqlens"] for block in layer.blocks] == [
            "cu",
            "cu",
        ]
        assert [block.calls[0]["max_seqlen"] for block in layer.blocks] == [
            "max",
            "max",
        ]

    def test_eager_path_builds_boundaries_from_the_grid(self):
        """Verifies eager path builds boundaries from the grid."""
        layer = self._layer()
        layer.rot_pos_emb = lambda grid: ("cos", "sin", None)
        layer.compute_attn_mask_seqlen = lambda cu: "max"
        x = torch.zeros(4, 12)
        out = layer.forward(x, [[1, 2, 2]])
        assert out.shape == (1, 4)
        assert layer.blocks[0].calls[0]["cu_seqlens"].tolist() == [0, 4]
        assert layer.blocks[0].calls[0]["rotary_pos_emb_cos"] == "cos"


def _budget(**overrides):
    values = dict(
        max_pixels=28 * 28 * 2 * 100,
        min_pixels=1,
        max_tokens=512,
        largest_canvas=lambda frames=1: (112, 112),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _Processor:
    def __init__(self):
        self.image_processor = SimpleNamespace(patch_expand_factor=1, merge_size=2)
        self.video_processor = SimpleNamespace(patch_expand_factor=1, merge_size=2)
        self.serving_budgets = {
            "image": _budget(max_tokens=512),
            "video": _budget(max_tokens=64),
        }
        self.serving_options = {"videos_kwargs": {"max_frames": 4}}
        self.image_token_id = 99

    def resolve_serving_kwargs(self, kwargs):
        return {"videos_kwargs": {"fps_interval": 0.1}}


def _processing_info():
    info = object.__new__(mm.Glm5NextProcessingInfo)
    processor = _Processor()
    info.get_hf_processor = lambda **kwargs: processor
    info.get_hf_config = lambda: SimpleNamespace(
        vision_config=SimpleNamespace(
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        video_start_token_id=1,
        video_end_token_id=2,
        image_start_token_id=3,
        image_end_token_id=4,
        image_token_id=99,
    )
    info.get_tokenizer = lambda: SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [7, 8]
    )
    info.get_video_processor = lambda: processor.video_processor
    return info, processor


class TestProcessingInfoBudget:
    def test_budget_accessors_track_the_processor_config(self):
        """Verifies budget accessors track the processor config."""
        info, _ = _processing_info()
        assert info.get_max_image_tokens() == 512
        assert info._vision_budget("video").max_tokens == 64
        assert info._video_frame_cap() == 4
        size = info.get_image_size_with_most_features()
        assert isinstance(size, ImageSize)
        assert (size.width, size.height) == (112, 112)

    def test_vision_info_counts_patches_after_alignment(self):
        """Verifies vision info counts patches after alignment."""
        info, _ = _processing_info()
        size, tokens = info._get_vision_info(
            image_width=112, image_height=112, num_frames=1
        )
        assert (size.width, size.height) == (112, 112)
        # 112 / 14 patches per side, 1 temporal patch, merge 2x2.
        assert tokens == (1 * 8 * 8) // 4
        assert info.get_num_image_tokens(image_width=112, image_height=112) == 16
        assert (
            info.get_num_video_tokens(image_width=112, image_height=112, num_frames=2)
            == 16
        )

    @pytest.mark.parametrize(("max_tokens", "expected"), [(10, 0), (20, 2)])
    def test_video_frame_search_respects_the_token_ceiling(self, max_tokens, expected):
        """Verifies video frame search respects the token ceiling."""
        info, _ = _processing_info()
        assert info._get_max_video_frames(max_tokens) == expected

    def test_max_tokens_per_item_uses_pixels_frames_and_timestamps(self):
        """Verifies max tokens per item uses pixels frames and timestamps."""
        info, processor = _processing_info()
        processor.serving_budgets["video"].max_pixels = 2 * 14 * 14 * 4 * 10
        result = info.get_mm_max_tokens_per_item(128, {"image": 1, "video": 1})
        # 10 vision tokens + 2 temporal groups * (2 + 2 timestamp tokens) + 2.
        assert result == {"image": 512, "video": 20}

    def test_video_placeholder_interleaves_timestamps_and_image_tokens(self):
        """Verifies video placeholder interleaves timestamps and image tokens."""
        info, processor = _processing_info()
        info._get_video_second_idx_glm46v = lambda *args: [0.0, 1.0]
        placeholder = info._construct_glm5_video_placeholder(
            torch.zeros(2, 4, 4, 3), {}, (2, 4, 4), {}
        )
        assert placeholder == [
            1,
            3,
            *([99] * 4),
            4,
            7,
            8,
            3,
            *([99] * 4),
            4,
            7,
            8,
            2,
        ]

    def test_video_placeholder_rejects_a_sampling_grid_mismatch(self):
        """Verifies video placeholder rejects a sampling grid mismatch."""
        info, _ = _processing_info()
        info._get_video_second_idx_glm46v = lambda *args: [0.0]
        with pytest.raises(ValueError, match="sampling and encoder grid"):
            info._construct_glm5_video_placeholder(
                torch.zeros(2, 4, 4, 3), {}, (2, 4, 4), {}
            )

    def test_video_second_indices_follow_the_sampler(self, monkeypatch):
        """Verifies video second indices follow the sampler."""
        import vllm_fl.models.glm5.processor as processor_module

        calls = []
        monkeypatch.setattr(
            processor_module,
            "glm_select_decoded_frames",
            lambda video_processor, metadata, total, **kwargs: (
                calls.append(("select", total, kwargs)) or ([0], [0])
            ),
        )
        monkeypatch.setattr(
            processor_module,
            "glm_video_timestamp_seconds",
            lambda video_processor, metadata, **kwargs: (
                calls.append(("timestamps", metadata.frames_indices)) or [0, 1]
            ),
        )
        info, _ = _processing_info()
        result = info._get_video_second_idx_glm46v(
            {"do_sample_frames": False, "total_num_frames": 5}, total_frames=5
        )
        assert result == [0, 1]
        assert calls[0] == ("select", 5, {"fps_interval": 0.1})
        assert calls[1][1] == [0]


@dataclass
class _DummyInputs:
    hf_processor_mm_kwargs: dict
    mm_data_items: dict
    seq_len: int = 16


class TestDummyInputs:
    def test_processor_inputs_force_the_maximal_frame_set(self, monkeypatch):
        """Verifies processor inputs force the maximal frame set."""
        monkeypatch.setattr(
            mm.Glm4vDummyInputsBuilder,
            "get_dummy_processor_inputs",
            lambda self, *args: _DummyInputs(
                hf_processor_mm_kwargs={"videos_kwargs": {"max_frames": 8}},
                mm_data_items={"video": []},
            ),
        )
        info, _ = _processing_info()
        builder = object.__new__(mm.Glm5NextDummyInputsBuilder)
        builder.info = info
        inputs = builder.get_dummy_processor_inputs(16, {"video": 1}, {})
        assert inputs.hf_processor_mm_kwargs == {
            "videos_kwargs": {"max_frames": 8, "do_sample_frames": False}
        }

    def test_dummy_media_uses_the_effective_canvases(self):
        """Verifies dummy media uses the effective canvases."""
        info, _ = _processing_info()
        info._vision_budget = lambda modality: SimpleNamespace(
            largest_canvas=lambda frames=1: (8, 8)
        )
        builder = object.__new__(mm.Glm5NextDummyInputsBuilder)
        builder.info = info
        image_calls = []
        video_calls = []
        builder._get_dummy_images = lambda **kwargs: (
            image_calls.append(kwargs) or ["image"]
        )
        builder._get_dummy_videos = lambda **kwargs: (
            video_calls.append(kwargs) or ["video"]
        )
        data = builder.get_dummy_mm_data(32, {"image": 1, "video": 2}, {})
        assert data == {"image": ["image"], "video": ["video"]}
        assert image_calls[0]["width"] == 8
        assert image_calls[0]["num_images"] == 1
        assert video_calls[0]["num_frames"] == 4
        assert video_calls[0]["num_videos"] == 2


class TestProcessorHooks:
    def test_prompt_updates_are_owned_by_vllm(self):
        """Verifies prompt updates are owned by vllm."""
        processor = object.__new__(mm.Glm5NextMultiModalProcessor)
        assert processor._hf_processor_applies_updates("prompt", {}, {}, {}) is False

    def test_empty_mm_data_delegates_to_the_base_processor(self, monkeypatch):
        """Verifies empty mm data delegates to the base processor."""
        processor = object.__new__(mm.Glm5NextMultiModalProcessor)
        calls = []
        monkeypatch.setattr(
            mm.Glm4vMultiModalProcessor,
            "_call_hf_processor",
            lambda self, prompt, mm_data, mm_kwargs, tok_kwargs: (
                calls.append((self, prompt, mm_data, mm_kwargs, tok_kwargs)) or "base"
            ),
        )
        assert processor._call_hf_processor("prompt", {}, {}, {}) == "base"
        assert calls[0][1] == "prompt"
        assert calls[0][2] == {}

    def test_hf_processor_is_built_once_and_serving_configured(self, monkeypatch):
        """Verifies hf processor is built once and serving configured."""
        import vllm_fl.models.glm5.processor as processor_module

        created = {}

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                created["model"] = model
                created["kwargs"] = kwargs
                instance = object.__new__(cls)
                instance.serving_budgets = {}
                return instance

            def configure_serving(self, kwargs):
                created["serving"] = kwargs

        monkeypatch.setattr(processor_module, "Glm5NextProcessor", FakeProcessor)
        info = object.__new__(mm.Glm5NextProcessingInfo)
        info.ctx = SimpleNamespace(
            model_config=SimpleNamespace(
                revision="rev",
                tokenizer_revision="tok-rev",
                hf_token="token",
                trust_remote_code=True,
                model="checkpoint",
            ),
            get_merged_mm_kwargs=lambda kwargs: {"merged": kwargs},
        )
        first = info.get_hf_processor(temperature=1)
        assert first is info._glm5_hf_processor
        assert created["model"] == "checkpoint"
        assert created["kwargs"] == {
            "revision": "rev",
            "tokenizer_revision": "tok-rev",
            "token": "token",
            "trust_remote_code": True,
            "temperature": 1,
        }
        assert created["serving"] == {"merged": {}}
        assert info.get_hf_processor() is first


class TestConditionalGeneration:
    def test_encoder_cudagraph_config_drops_position_embeddings(self, monkeypatch):
        """Verifies encoder cudagraph config drops position embeddings."""
        config = SimpleNamespace(buffer_keys=["pos_embeds", "rotary"])
        monkeypatch.setattr(
            mm.Glm4vForConditionalGeneration,
            "get_encoder_cudagraph_config",
            lambda self: config,
        )
        model = object.__new__(mm.Glm5NextForConditionalGeneration)
        assert model.get_encoder_cudagraph_config().buffer_keys == ["rotary"]

    def test_mamba_state_hooks_delegate_to_the_text_model(self, monkeypatch):
        """Verifies mamba state hooks delegate to the text model."""
        import vllm_fl.models.glm5.model as text_model

        calls = []
        monkeypatch.setattr(
            text_model.Glm5NextForCausalLM,
            "get_mamba_state_dtype_from_config",
            classmethod(lambda cls, cfg: calls.append("dtype") or "dtype"),
        )
        monkeypatch.setattr(
            text_model.Glm5NextForCausalLM,
            "get_mamba_state_shape_from_config",
            classmethod(lambda cls, cfg: calls.append("shape") or "shape"),
        )
        monkeypatch.setattr(
            text_model.Glm5NextForCausalLM,
            "get_mamba_state_copy_func",
            classmethod(lambda cls: calls.append("copy") or "copy"),
        )
        cls = mm.Glm5NextForConditionalGeneration
        assert cls.get_mamba_state_dtype_from_config(None) == "dtype"
        assert cls.get_mamba_state_shape_from_config(None) == "shape"
        assert cls.get_mamba_state_copy_func() == "copy"
        assert calls == ["dtype", "shape", "copy"]

    def test_construction_wires_tower_and_language_model(self, monkeypatch):
        """Verifies construction wires tower and language model."""
        from contextlib import nullcontext

        import vllm_fl.models.glm5.runtime as runtime

        created = SimpleNamespace(vision=None, language=None)

        class FakeVision(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.args = args
                self.kwargs = kwargs
                created.vision = self

        def fake_language(**kwargs):
            created.language = kwargs
            return "language-model"

        monkeypatch.setattr(runtime, "validate_glm5_config", lambda cfg: None)
        monkeypatch.setattr(mm, "Glm5NextVisionTransformer", FakeVision)
        monkeypatch.setattr(mm, "init_vllm_registered_model", fake_language)
        monkeypatch.setattr(
            mm.Glm5NextForConditionalGeneration,
            "_mark_tower_model",
            lambda self, *args, **kwargs: nullcontext(),
        )
        monkeypatch.setattr(
            mm.Glm5NextForConditionalGeneration,
            "_mark_language_model",
            lambda self, *args, **kwargs: nullcontext(),
        )
        config = SimpleNamespace(
            text_config="text-config",
            vision_config="vision-config",
        )
        multimodal_config = SimpleNamespace(
            mm_encoder_tp_mode="data",
            is_multimodal_pruning_enabled=lambda: True,
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=config, multimodal_config=multimodal_config
            )
        )
        model = mm.Glm5NextForConditionalGeneration(
            vllm_config=vllm_config, prefix="root"
        )
        assert model.config is config
        assert model.use_data_parallel is True
        assert model.is_multimodal_pruning_enabled is True
        assert created.vision.kwargs["prefix"] == "root.visual"
        assert created.vision.kwargs["quant_config"] is None
        assert created.language["hf_config"] == "text-config"
        assert created.language["architectures"] == ["Glm5NextForCausalLM"]
        assert model.language_model == "language-model"

    def test_load_weights_filters_and_audits(self, monkeypatch):
        """Verifies load weights filters and audits."""
        from contextlib import nullcontext

        import vllm_fl.models.glm5.loader as loader

        calls = []
        monkeypatch.setattr(
            loader, "audit_text_weights", lambda language: nullcontext()
        )
        monkeypatch.setattr(
            loader, "unquantized_weights", lambda weights: ("filtered", weights)
        )
        monkeypatch.setattr(
            mm.Glm4vForConditionalGeneration,
            "load_weights",
            lambda self, weights: calls.append(weights) or "loaded",
        )
        model = object.__new__(mm.Glm5NextForConditionalGeneration)
        nn.Module.__init__(model)
        model.language_model = "language"
        assert model.load_weights("weights") == "loaded"
        assert calls == [("filtered", "weights")]
