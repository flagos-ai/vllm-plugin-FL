# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the GLM MoE DSA Ascend wrapper model.

``vllm_fl/models/glm_moe_dsa.py`` subclasses vLLM's
``GlmMoeDsaForCausalLM`` to install the Ascend dense-attention fallback and
to filter the ModelSlim ``rot.weight`` top-level checkpoint entry out of the
upstream weight stream. The module imports vLLM at import time; this test
loads it under stubbed ``vllm.*`` modules (so it runs without a vLLM install),
then exercises:

* the class wiring (inheritance + module-level registration strings),
* the ``__init__`` prepare/super() contract with a stubbed upstream base,
* the ``load_weights`` filter against a recording upstream loader.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_SOURCE = Path(__file__).resolve().parents[4] / "vllm_fl/models/glm_moe_dsa.py"
_MODULE_NAME = "supp_glm_moe_dsa_under_test"


def model_config_restored(hf_config, vllm_config):
    """Assert the context manager restored the original config objects."""
    assert vllm_config.model_config.hf_config is hf_config
    assert vllm_config.model_config.hf_text_config is hf_config
    return True


class RecordingBase:
    """Recording stand-in for vLLM's GlmMoeDsaForCausalLM."""

    def __init__(self, *, vllm_config, prefix=""):
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.constructed = True
        self.loaded = None

    def load_weights(self, weights):
        self.loaded = list(weights)
        return {"model.embed_tokens.weight"}


@pytest.fixture
def glm_module(monkeypatch):
    """Load vllm_fl/models/glm_moe_dsa.py with the vllm import surface stubbed.

    Only the two vllm imports the module performs at import time are faked:
    ``vllm.config.VllmConfig`` (type annotation) and the upstream base class
    ``vllm.model_executor.models.deepseek_v2.GlmMoeDsaForCausalLM``. The FL
    module itself is executed from source so its real logic is under test.
    """
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    vllm_config = ModuleType("vllm.config")

    class VllmConfig:
        pass

    vllm_config.VllmConfig = VllmConfig
    model_executor = ModuleType("vllm.model_executor")
    model_executor.__path__ = []
    models_pkg = ModuleType("vllm.model_executor.models")
    models_pkg.__path__ = []
    deepseek_v2 = ModuleType("vllm.model_executor.models.deepseek_v2")
    deepseek_v2.GlmMoeDsaForCausalLM = RecordingBase

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models_pkg)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.deepseek_v2", deepseek_v2
    )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)
    module.RecordingBase = RecordingBase
    return module


@pytest.mark.supplementary
def test_wrapper_subclasses_upstream_glm(glm_module):
    """The FL wrapper is a direct subclass of the upstream model class."""
    assert issubclass(glm_module.GlmMoeDsaForCausalLM, glm_module.RecordingBase)
    assert glm_module.__all__ == ["GlmMoeDsaForCausalLM"]


@pytest.mark.supplementary
def test_registration_entry_points_at_wrapper_module():
    """The ModelRegistry entry in vllm_fl.register() names this class."""
    registration = "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM"
    vllm_fl_init = (_SOURCE.parents[1] / "__init__.py").read_text(encoding="utf-8")
    assert f'"{registration}"' in vllm_fl_init
    assert _SOURCE.as_posix().endswith("vllm_fl/models/glm_moe_dsa.py")


@pytest.mark.supplementary
def test_init_runs_prepare_then_upstream_super(monkeypatch, glm_module):
    """__init__ applies the dense fallback before the upstream constructor."""
    from vllm_fl.dispatch.backends.vendor.ascend.patches import (
        ascend_glm_dsa,
    )

    events = []
    monkeypatch.setattr(
        ascend_glm_dsa,
        "prepare_glm_dsa_dense_fallback",
        lambda config: events.append(("prepare", config)) or True,
    )

    def fake_super_init(self, *, vllm_config, prefix):
        events.append(("super", vllm_config, prefix))
        self.constructed = True

    monkeypatch.setattr(glm_module.RecordingBase, "__init__", fake_super_init)

    class HfConfig:
        index_topk = 1024

    hf_config = HfConfig()
    hf_config.index_topk = 256
    hf_config.model_type = "glm_moe_dsa"
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)
    )
    model = glm_module.GlmMoeDsaForCausalLM(vllm_config=vllm_config, prefix="model.0")

    assert events == [
        ("prepare", vllm_config.model_config),
        ("super", vllm_config, "model.0"),
    ]
    assert model.constructed is True
    assert model_config_restored(hf_config, vllm_config)


@pytest.mark.supplementary
def test_init_wraps_super_with_dense_hf_config(monkeypatch, glm_module):
    """The upstream super() runs inside use_glm_dsa_dense_hf_config."""
    from vllm_fl.dispatch.backends.vendor.ascend.patches import (
        ascend_glm_dsa,
    )

    seen = {}
    monkeypatch.setattr(
        ascend_glm_dsa, "prepare_glm_dsa_dense_fallback", lambda config: True
    )

    def fake_super_init(self, *, vllm_config, prefix):
        seen["hf_config"] = vllm_config.model_config.hf_config
        seen["hf_text_config"] = vllm_config.model_config.hf_text_config

    monkeypatch.setattr(glm_module.RecordingBase, "__init__", fake_super_init)

    class GlmConfig:
        index_topk = 1024

    hf_config = GlmConfig()
    hf_config.index_topk = 256
    model_config = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)
    vllm_config = SimpleNamespace(model_config=model_config)

    glm_module.GlmMoeDsaForCausalLM(vllm_config=vllm_config)

    # Inside super() the dense copy hides index_topk; afterwards restored.
    assert seen["hf_config"] is not hf_config
    assert not hasattr(seen["hf_config"], "index_topk")
    assert model_config.hf_config is hf_config
    assert model_config.hf_text_config is hf_config
    assert hf_config.index_topk == 256
    assert GlmConfig.index_topk == 1024


@pytest.mark.supplementary
def test_load_weights_filters_only_top_level_rot_weight(monkeypatch, glm_module):
    """The ModelSlim rot.weight is skipped; every other entry is forwarded."""
    rotation = object()
    embedding = object()
    nested_rotation = object()
    mtp_weight = object()
    similar = object()
    rot_bias = object()

    def recording_load(self, weights):
        self.loaded = list(weights)
        return {"model.embed_tokens.weight"}

    monkeypatch.setattr(glm_module.RecordingBase, "load_weights", recording_load)
    model = glm_module.GlmMoeDsaForCausalLM.__new__(glm_module.GlmMoeDsaForCausalLM)

    loaded = model.load_weights(
        iter(
            [
                ("rot.weight", rotation),
                ("model.embed_tokens.weight", embedding),
                ("model.layers.3.rot.weight", nested_rotation),
                ("mtp.rot.weight", mtp_weight),
                ("my_rot.weight", similar),
                ("rot.bias", rot_bias),
            ]
        )
    )

    assert loaded == {"model.embed_tokens.weight"}
    assert model.loaded == [
        ("model.embed_tokens.weight", embedding),
        ("model.layers.3.rot.weight", nested_rotation),
        ("mtp.rot.weight", mtp_weight),
        ("my_rot.weight", similar),
        ("rot.bias", rot_bias),
    ]


@pytest.mark.supplementary
def test_load_weights_lazy_stream_is_consumed_once(monkeypatch, glm_module):
    """load_weights works with generators (the Iterable contract)."""

    def counting_load(self, weights):
        assert list(weights), "the generator must yield items"
        assert not list(weights), "the generator must be exhausted"
        return {"ok"}

    monkeypatch.setattr(glm_module.RecordingBase, "load_weights", counting_load)
    model = glm_module.GlmMoeDsaForCausalLM.__new__(glm_module.GlmMoeDsaForCausalLM)

    weights = iter([("a.b", 1), ("rot.weight", 2), ("c.d", 3)])
    assert model.load_weights(weights) == {"ok"}
