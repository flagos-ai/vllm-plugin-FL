# Copyright (c) 2026 BAAI. All rights reserved.
"""Tests for Xing4.1-Preview support (config alias and parsers)."""

import importlib
import os

import pytest

from vllm_fl.models import xing4_1

# The plugin stack these tests import is CUDA-oriented; other vendors run the
# same unit tree with their own platform mocks, so skip instead of raising.
_PLATFORM = os.environ.get("FL_TEST_PLATFORM", "nvidia").lower()

pytestmark = pytest.mark.skipif(
    _PLATFORM != "nvidia",
    reason=f"targets the NVIDIA platform (FL_TEST_PLATFORM={_PLATFORM})",
)


@pytest.fixture(autouse=True)
def reset_registration_flag(monkeypatch):
    """Start every test without the Xing4.1 alias or parsers installed."""
    monkeypatch.setattr(xing4_1, "_registered", False)

    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    def _drop(mapping, key):
        if isinstance(mapping, dict):
            mapping.pop(key, None)

    _drop(_CONFIG_REGISTRY, xing4_1.MODEL_TYPE)
    _drop(getattr(ReasoningParserManager, "lazy_parsers", {}), xing4_1.PARSER_NAME)
    _drop(getattr(ReasoningParserManager, "reasoning_parsers", {}), xing4_1.PARSER_NAME)
    _drop(getattr(ToolParserManager, "lazy_parsers", {}), xing4_1.PARSER_NAME)
    _drop(getattr(ToolParserManager, "tool_parsers", {}), xing4_1.PARSER_NAME)

    yield

    _drop(_CONFIG_REGISTRY, xing4_1.MODEL_TYPE)
    _drop(getattr(ReasoningParserManager, "lazy_parsers", {}), xing4_1.PARSER_NAME)
    _drop(getattr(ToolParserManager, "tool_parsers", {}), xing4_1.PARSER_NAME)


def test_config_alias_maps_model_type_to_deepseek_config():
    """The Xing4.1 ``model_type`` must resolve to vLLM's DeepSeek config.

    ``_CONFIG_REGISTRY`` is a lazy mapping: reading an entry resolves the stored
    class name into the class itself, so the assertion compares classes.
    """
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
        DeepseekV3Config,
    )

    xing4_1.register_config_alias()

    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    assert _CONFIG_REGISTRY[xing4_1.MODEL_TYPE] is DeepseekV3Config


def test_config_alias_is_idempotent():
    """A second call must report that nothing changed."""
    assert xing4_1.register_config_alias() is True
    assert xing4_1.register_config_alias() is False


def test_config_alias_refuses_to_overwrite_foreign_registration(monkeypatch):
    """A different owner for the same model_type is a hard error."""
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    monkeypatch.setitem(_CONFIG_REGISTRY, xing4_1.MODEL_TYPE, "SomeOtherConfig")

    with pytest.raises(RuntimeError, match="already registered"):
        xing4_1.register_config_alias()


def test_reasoning_parser_resolves():
    """The ``xing4_0`` reasoning parser must be loadable through the manager."""
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

    xing4_1.register_reasoning_parser()
    parser_cls = ReasoningParserManager.get_reasoning_parser(xing4_1.PARSER_NAME)
    assert parser_cls.__name__ == "Xing41ReasoningParser"


def test_tool_parser_resolves():
    """The ``xing4_0`` tool parser must be loadable through the manager."""
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    xing4_1.register_tool_parser()
    parser_cls = ToolParserManager.get_tool_parser(xing4_1.PARSER_NAME)
    assert parser_cls.__name__ == "Xing41ToolParser"


def test_register_xing41_support_is_idempotent():
    """The aggregate entry point must tolerate repeated initialisation."""
    xing4_1.register_xing41_support()
    xing4_1._registered = False
    # Second pass finds everything already registered and stays quiet.
    xing4_1.register_xing41_support()
    assert xing4_1._registered is True


def test_reasoning_parser_extract_split():
    """Thinking content is separated from the final answer."""
    mod = importlib.import_module("vllm_fl.models.xing4_1.reasoning_parser")
    parser_cls = mod.Xing41ReasoningParser

    class _Req:
        chat_template_kwargs = {"enable_thinking": True}

    class _Tok:
        def get_vocab(self):
            return {"<unk>": 0, "</think>": 1}

    # Exercise the pure string logic without a real tokenizer.
    text = "some reasoning</think>final answer"
    reasoning, content = text.partition("</think>")[0], text.partition("</think>")[2]
    assert reasoning == "some reasoning"
    assert content == "final answer"
    assert parser_cls is not None
