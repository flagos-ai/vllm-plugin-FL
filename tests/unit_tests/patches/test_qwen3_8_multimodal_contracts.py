# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU routing/input contracts from production definitions; no device numerics."""

from __future__ import annotations

import ast
import enum
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def definition(path, name, namespace, *, methods=()):
    tree = ast.parse((ROOT / path).read_text())
    node = next(item for item in tree.body if getattr(item, "name", None) == name)
    if methods:
        node.bases = []
        node.body = [
            item for item in node.body if getattr(item, "name", None) in methods
        ]
        assert {item.name for item in node.body} == set(methods)
    future = ast.ImportFrom("__future__", [ast.alias("annotations")], 0)
    code = ast.fix_missing_locations(ast.Module([future, node], type_ignores=[]))
    exec(compile(code, path, "exec"), namespace)
    return namespace[name]


def install_module(monkeypatch, name, **attrs):
    result = ModuleType(name)
    result.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, result)
    return result


@pytest.fixture(scope="module")
def contracts():
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next/common/multimodal.py"
    spec = importlib.util.spec_from_file_location("qwen38_mm_contracts", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("fps", [1, 2.0, 23.976, 1e-9])
def test_valid_fps(contracts, fps):
    contracts.validate_video_fps(fps, name="source fps")


@pytest.mark.parametrize(
    "fps",
    [
        None,
        0,
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        False,
        "2",
        [],
        10**400,
    ],
)
def test_invalid_fps(contracts, fps):
    with pytest.raises(ValueError, match="finite positive"):
        contracts.validate_video_fps(fps, name="source fps")


@pytest.mark.parametrize("rate", [None, 0, 0.0])
def test_valid_pruning(contracts, rate):
    for config in [None, SimpleNamespace(), SimpleNamespace(video_pruning_rate=rate)]:
        contracts.validate_video_pruning(config)


@pytest.mark.parametrize("rate", [0.5, 1, -1, float("nan"), float("inf"), True, "0"])
def test_invalid_pruning(contracts, rate):
    with pytest.raises(ValueError, match="does not support video pruning"):
        contracts.validate_video_pruning(SimpleNamespace(video_pruning_rate=rate))


class Backend(enum.Enum):
    TORCH_SDPA = "sdpa"
    FLASH_ATTN = "fa"


def test_vision_selection_rejects_before_patching(monkeypatch):
    installs = []
    install_module(
        monkeypatch,
        "vllm_fl.attention.utils",
        patch_mm_encoder_attention=lambda: installs.append(1),
    )
    platform = definition(
        "vllm_fl/platform.py",
        "PlatformFL",
        {
            "AttentionBackendEnum": Backend,
            "logger": SimpleNamespace(info_once=lambda *a: None),
        },
        methods=("get_supported_vit_attn_backends", "get_vit_attn_backend"),
    )
    platform.vendor_name = "thead"
    platform.get_device_capability = lambda: pytest.fail("unexpected device probe")
    assert platform.get_supported_vit_attn_backends() == [Backend.TORCH_SDPA]
    for backend in [None, Backend.TORCH_SDPA]:
        assert platform.get_vit_attn_backend(72, object, backend) is Backend.TORCH_SDPA
    before = len(installs)
    with pytest.raises(ValueError, match="T-Head vision"):
        platform.get_vit_attn_backend(72, object, Backend.FLASH_ATTN)
    assert len(installs) == before
    platform.vendor_name = "nvidia"
    assert platform.get_supported_vit_attn_backends() == [
        Backend.TORCH_SDPA,
        Backend.FLASH_ATTN,
    ]
    assert (
        platform.get_vit_attn_backend(72, object, Backend.FLASH_ATTN)
        is Backend.FLASH_ATTN
    )


@pytest.mark.parametrize(
    "vendor,cuda,expected",
    [
        ("thead", False, "native"),
        ("thead", True, "native"),
        ("nvidia", True, "cuda"),
        ("metax", False, "old"),
    ],
)
def test_oot_binding_is_idempotent_and_scoped(monkeypatch, vendor, cuda, expected):
    class Attention:
        def forward_native(self, *args):
            return "native", args

        def forward_cuda(self, *args):
            return "cuda", args

        def forward_oot(self, *args):
            return "old", args

    for name in [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
    ]:
        install_module(monkeypatch, name)
    install_module(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        MMEncoderAttention=Attention,
    )
    install_module(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(vendor_name=vendor, is_cuda=lambda: cuda),
    )
    install_module(
        monkeypatch, "vllm.v1.attention.backends.registry", AttentionBackendEnum=Backend
    )
    hook = definition(
        "vllm_fl/attention/utils.py",
        "patch_mm_encoder_attention",
        {"logger": SimpleNamespace(info_once=lambda *a: None)},
    )
    hook()
    first = Attention.forward_oot
    hook()
    assert Attention.forward_oot is first
    assert Attention().forward_oot("q", "k", "v") == (expected, ("q", "k", "v"))


@pytest.fixture
def video(monkeypatch):
    import importlib

    import numpy as np

    from vllm.multimodal.media.connector import MediaConnector
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, Qwen3VLVideoBackend

    module = importlib.import_module("vllm_fl.patches.qwen3_8_flash_next")
    monkeypatch.setattr(
        VIDEO_LOADER_REGISTRY, "name2class", dict(VIDEO_LOADER_REGISTRY.name2class)
    )
    monkeypatch.setattr(
        VIDEO_LOADER_REGISTRY,
        "processor2backend",
        dict(VIDEO_LOADER_REGISTRY.processor2backend),
    )
    source = SimpleNamespace(total_frames_num=48, original_fps=24, duration=2)
    frames = np.arange(48 * 2 * 2 * 3, dtype=np.uint8).reshape(48, 2, 2, 3)
    reads = []
    monkeypatch.setattr(Qwen3VLVideoBackend, "open_video_capture", lambda _: "cap")
    monkeypatch.setattr(Qwen3VLVideoBackend, "get_video_metadata", lambda _: source)

    def read(cap, indices, **kwargs):
        reads.append((cap, indices, kwargs))
        return frames[indices], indices

    monkeypatch.setattr(Qwen3VLVideoBackend, "read_frames", read)
    config = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            language_model_only=False,
            video_pruning_rate=0,
            media_io_kwargs={"video": {"fps": 2, "min_frames": 4}},
        )
    )
    return SimpleNamespace(
        module=module,
        registry=VIDEO_LOADER_REGISTRY,
        native=Qwen3VLVideoBackend,
        config=config,
        source=source,
        target=SimpleNamespace(fps=2),
        frames=frames,
        reads=reads,
        connector=MediaConnector,
    )


def test_loader_registration_does_not_patch_other_models(video):
    original = video.native.compute_frames_index_to_sample.__func__
    classes = dict(video.registry.name2class)
    processors = dict(video.registry.processor2backend)
    # Preserve the actual native invalid-input behavior, not just valid output.
    bad_source = SimpleNamespace(total_frames_num=48, original_fps=0, duration=2)
    with pytest.raises(ZeroDivisionError):
        video.native.compute_frames_index_to_sample(bad_source, video.target)
    video.module._configure_qwen3_8_video_loader(video.config)
    configured = video.config.multimodal_config
    video.module._configure_qwen3_8_video_loader(video.config)
    assert video.config.multimodal_config is configured
    assert video.native.compute_frames_index_to_sample.__func__ is original
    assert video.registry.processor2backend == processors
    assert {k: video.registry.name2class[k] for k in classes} == classes
    assert (
        video.registry.get_backend_for_video_processor("Qwen3VLVideoProcessor")
        == "qwen3_vl"
    )
    assert video.registry.name2class["qwen3_vl"] is video.native
    with pytest.raises(ZeroDivisionError):
        video.native.compute_frames_index_to_sample(bad_source, video.target)
    with pytest.raises(ValueError, match="finite positive"):
        video.registry.load("qwen3_8_flash_next").compute_frames_index_to_sample(
            bad_source, video.target
        )


def test_loader_config_is_copy_on_write(video):
    shared = video.config.multimodal_config
    original = shared.media_io_kwargs
    image = {"image_mode": "RGB"}
    original["image"] = image
    other_model = SimpleNamespace(multimodal_config=shared)
    video.module._configure_qwen3_8_video_loader(video.config)
    result = video.config.multimodal_config
    assert result is not shared and result.media_io_kwargs is not original
    assert result.media_io_kwargs["video"] == {
        "fps": 2,
        "min_frames": 4,
        "video_backend": "qwen3_8_flash_next",
    }
    assert result.media_io_kwargs["image"] is image
    assert other_model.multimodal_config is shared
    assert original["video"] == {"fps": 2, "min_frames": 4}


@pytest.mark.parametrize("backend", ["opencv", "qwen2_vl", "custom", None, ""])
def test_explicit_alternate_backend_is_preserved(video, backend):
    shared = video.config.multimodal_config
    shared.media_io_kwargs["video"]["video_backend"] = backend
    before = dict(video.registry.name2class)
    video.module._configure_qwen3_8_video_loader(video.config)
    assert video.config.multimodal_config is shared
    assert shared.media_io_kwargs["video"]["video_backend"] == backend
    assert video.registry.name2class == before


@pytest.mark.parametrize("backend", ["qwen3_vl", "qwen3_8_flash_next"])
def test_explicit_qwen_loader_and_deserialized_config_register_locally(video, backend):
    video.config.multimodal_config.media_io_kwargs["video"]["video_backend"] = backend
    video.module._configure_qwen3_8_video_loader(video.config)
    assert (
        video.config.multimodal_config.media_io_kwargs["video"]["video_backend"]
        == "qwen3_8_flash_next"
    )
    assert (
        video.registry.name2class["qwen3_8_flash_next"]
        is video.module.Qwen3_8FlashNextVideoBackend
    )


@pytest.mark.parametrize("required_kwarg", [False, True])
def test_loader_bad_abi_preserves_registry_and_config(
    video, monkeypatch, required_kwarg
):
    if required_kwarg:

        def original(cls, source, target, *, required):
            return []
    else:

        def original(cls, source, target):
            return []

    monkeypatch.setattr(
        video.native, "compute_frames_index_to_sample", classmethod(original)
    )
    shared = video.config.multimodal_config
    before = dict(video.registry.name2class)
    with pytest.raises(TypeError):
        video.module._configure_qwen3_8_video_loader(video.config)
    assert video.config.multimodal_config is shared
    assert video.registry.name2class == before
    assert video.native.compute_frames_index_to_sample.__func__ is original


@pytest.mark.parametrize("failure", ["collision", "register", "config"])
def test_loader_setup_failure_leaves_existing_objects_unchanged(
    video, monkeypatch, failure
):
    shared = video.config.multimodal_config
    if failure == "collision":
        video.registry.name2class["qwen3_8_flash_next"] = object()
    elif failure == "register":

        def register(name):
            def install(cls):
                video.registry.name2class[name] = cls
                raise RuntimeError("registration failed")

            return install

        monkeypatch.setattr(video.registry, "register", register)
    else:

        class FrozenConfig:
            @property
            def multimodal_config(self):
                return shared

            @multimodal_config.setter
            def multimodal_config(self, value):
                raise RuntimeError("config assignment failed")

        video.config = FrozenConfig()
    before = dict(video.registry.name2class)
    with pytest.raises((ValueError, RuntimeError)):
        video.module._configure_qwen3_8_video_loader(video.config)
    assert video.registry.name2class == before
    assert video.config.multimodal_config is shared
    assert shared.media_io_kwargs["video"] == {"fps": 2, "min_frames": 4}


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["source", "target"])
def test_loader_rejects_before_frame_read(video, fps, field):
    video.module._configure_qwen3_8_video_loader(video.config)
    if field == "source":
        video.source.original_fps = fps
    else:
        video.config.multimodal_config.media_io_kwargs["video"]["fps"] = fps
    loader = video.registry.load("qwen3_8_flash_next")
    with pytest.raises(ValueError, match="finite positive"):
        loader.load_bytes(
            b"test", **video.config.multimodal_config.media_io_kwargs["video"]
        )
    assert video.reads == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("request_override", [False, True])
@pytest.mark.parametrize("target_model", [False, True])
def test_real_connector_routes_model_and_preserves_override(
    video, monkeypatch, asynchronous, request_override, target_model
):
    import asyncio

    import numpy as np

    from vllm.multimodal.media.connector import merge_media_io_kwargs

    original_defaults = video.config.multimodal_config.media_io_kwargs
    video.module._configure_qwen3_8_video_loader(video.config)
    defaults = (
        video.config.multimodal_config.media_io_kwargs
        if target_model
        else original_defaults
    )
    before = {k: dict(v) for k, v in defaults.items()}
    override = (
        {"video": {"video_backend": "qwen3_vl", "fps": 4}} if request_override else None
    )
    merged = merge_media_io_kwargs(defaults, override)
    connector = video.connector(media_io_kwargs=merged)
    selected = []

    def load(url, io, **kwargs):
        selected.append(type(io.video_loader))
        return io.load_bytes(b"test")

    async def load_async(*args, **kwargs):
        return load(*args, **kwargs)

    monkeypatch.setattr(connector, "load_from_url", load)
    monkeypatch.setattr(connector, "load_from_url_async", load_async)
    if asynchronous:
        result, metadata = asyncio.run(
            connector.fetch_video_async("test", video_processor="Qwen3VLVideoProcessor")
        )
    else:
        result, metadata = connector.fetch_video(
            "test", video_processor="Qwen3VLVideoProcessor"
        )
    expected_cls = (
        video.native
        if request_override or not target_model
        else video.module.Qwen3_8FlashNextVideoBackend
    )
    assert selected == [expected_cls]
    # Independent expected indices: 48 source frames at 24 FPS.
    indices = [0, 7, 13, 20, 27, 34, 40, 47] if request_override else [0, 16, 31, 47]
    assert video.reads == [
        ("cap", indices, {"total_frames_num": 48, "frame_recovery": False})
    ]
    np.testing.assert_array_equal(result, video.frames[indices])
    assert metadata == {
        "total_num_frames": 48,
        "fps": 24,
        "duration": 2,
        "video_backend": "opencv",
        "frames_indices": indices,
        "do_sample_frames": False,
    }
    assert defaults == before


def test_loader_execution_error_is_not_retried(video, monkeypatch):
    calls = []

    def original(cls, source, target, **kwargs):
        calls.append((cls, source, target, kwargs))
        raise RuntimeError("native video failed")

    monkeypatch.setattr(
        video.native, "compute_frames_index_to_sample", classmethod(original)
    )
    video.module._configure_qwen3_8_video_loader(video.config)
    with pytest.raises(RuntimeError, match="native video failed"):
        video.registry.load("qwen3_8_flash_next").load_bytes(
            b"test", fps=2, min_frames=8
        )
    assert len(calls) == 1
    assert calls[0][0] is video.module.Qwen3_8FlashNextVideoBackend
    assert calls[0][3] == {"min_frames": 8}
    assert video.reads == []


@pytest.fixture
def processing_info(contracts):
    calls = []

    class Base:
        def _get_video_second_idx(self, **kwargs):
            calls.append(kwargs)
            return [0.5]

    cls = definition(
        "vllm_fl/models/qwen3_8_flash_next/gpu/model.py",
        "Qwen3_8FlashNextProcessingInfo",
        {
            "Qwen3VLProcessingInfo": Base,
            "Qwen3_8FlashNextConfig": object,
            "validate_video_fps": contracts.validate_video_fps,
            "validate_video_pruning": contracts.validate_video_pruning,
        },
    )
    info = cls()
    mm = SimpleNamespace(video_pruning_rate=0)
    info.ctx = SimpleNamespace(
        get_mm_config=lambda: mm, get_hf_config=lambda _: "config"
    )
    return info, mm, calls


def test_processor_preserves_metadata_and_rejects_pruning(processing_info):
    info, mm, calls = processing_info
    assert info.get_hf_config() == "config"
    mm.video_pruning_rate = 0.5
    with pytest.raises(ValueError):
        info.get_hf_config()
    metadata = {"fps": 24, "frames_indices": [0, 24]}
    assert info._get_video_second_idx(metadata, sampled_num_frames=2) == [0.5]
    assert calls[0]["metadata"] is metadata
    assert calls[0]["sampled_fps"] is None
    assert calls[0]["sampled_num_frames"] == 2


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf")])
def test_processor_rejects_source_fps(processing_info, fps):
    info, _, calls = processing_info
    with pytest.raises(ValueError):
        info._get_video_second_idx({"fps": fps})
    assert calls == []


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_processor_rejects_sampled_fps(processing_info, fps):
    info, _, calls = processing_info
    with pytest.raises(ValueError):
        info._get_video_second_idx({"fps": 24}, sampled_fps=fps)
    assert calls == []


def test_config_only_installs_loader_for_multimodal(video, monkeypatch):
    hooks = []
    monkeypatch.setattr(video.module, "_strip_mrope", lambda _: hooks.append("strip"))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hc_count=4,
                ple_layer_ids=[],
                indexer_n_heads=None,
                mamba_ssm_dtype="float32",
            ),
            multimodal_config=None,
        ),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            enable_dbo=False, ubatch_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
    )
    verify = video.module.Qwen3_8FlashNextForConditionalGenerationConfig.verify_and_update_config
    before = dict(video.registry.name2class)
    verify(config)
    assert hooks == [] and video.registry.name2class == before
    mm = video.config.multimodal_config
    mm.language_model_only = True
    config.model_config.multimodal_config = mm
    verify(config)
    assert hooks == ["strip"] and video.registry.name2class == before
    mm.language_model_only = False
    config.speculative_config = object()
    with pytest.raises(NotImplementedError, match="speculative"):
        verify(config)
    assert config.model_config.multimodal_config is mm
    assert video.registry.name2class == before
    config.speculative_config = None
    verify(config)
    updated = config.model_config.multimodal_config
    assert updated is not mm
    assert updated.media_io_kwargs["video"]["video_backend"] == "qwen3_8_flash_next"
    assert "video_backend" not in mm.media_io_kwargs["video"]
    assert (
        video.registry.name2class["qwen3_8_flash_next"]
        is video.module.Qwen3_8FlashNextVideoBackend
    )
    updated.video_pruning_rate = 0.5
    with pytest.raises(ValueError, match="does not support video pruning"):
        verify(config)
    assert hooks == ["strip"]
