# Copyright (c) 2026 BAAI. All rights reserved.
"""Xing4.1-Preview model support: config alias, reasoning and tool parsers.

Xing4.1-Preview checkpoints declare ``model_type: "Xing4.1-Preview"`` but are
DeepSeek-V3.2 style models (MLA + DSA, ``architectures:
["DeepseekV32ForCausalLM"]``), which vLLM already implements. This module
therefore only has to:

* map the checkpoint's ``model_type`` onto vLLM's existing DeepSeek config
  class, so that vLLM's ``DeepseekV32ForCausalLM`` path is selected, and
* register the Xing generation-specific ``xing4_0`` reasoning and tool parsers.

No model implementation is re-registered: the architecture resolves through
vLLM upstream, which keeps this support aligned with the framework's own
DeepSeek-V3.2 code path.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PARSER_NAME = "xing4_0"
MODEL_TYPE = "Xing4.1-Preview"
CONFIG_CLASS = "DeepseekV3Config"

_REASONING_MODULE = f"{__name__}.reasoning_parser"
_REASONING_CLASS = "Xing41ReasoningParser"
_TOOL_MODULE = f"{__name__}.tool_parser"
_TOOL_CLASS = "Xing41ToolParser"

_registered = False


def register_config_alias() -> bool:
    """Map the Xing4.1 ``model_type`` onto vLLM's DeepSeek config class.

    Returns:
        ``True`` when the alias was installed by this call, ``False`` when it
        was already present with the expected target.
    """
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    current = _CONFIG_REGISTRY.get(MODEL_TYPE)
    if current == CONFIG_CLASS:
        return False
    if current is not None and current != CONFIG_CLASS:
        raise RuntimeError(
            f"model_type '{MODEL_TYPE}' is already registered to '{current}'; "
            f"refusing to override it with '{CONFIG_CLASS}'"
        )
    _CONFIG_REGISTRY[MODEL_TYPE] = CONFIG_CLASS
    return True


def register_reasoning_parser() -> bool:
    """Register the ``xing4_0`` reasoning parser lazily.

    Returns:
        ``True`` when registered by this call, ``False`` when already present.
    """
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

    existing = getattr(ReasoningParserManager, "reasoning_parsers", {}).get(
        PARSER_NAME
    ) or getattr(ReasoningParserManager, "lazy_parsers", {}).get(PARSER_NAME)
    if existing is not None:
        return False
    ReasoningParserManager.register_lazy_module(
        name=PARSER_NAME,
        module_path=_REASONING_MODULE,
        class_name=_REASONING_CLASS,
    )
    return True


def register_tool_parser() -> bool:
    """Register the ``xing4_0`` tool parser lazily."""
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    existing = getattr(ToolParserManager, "tool_parsers", {}).get(
        PARSER_NAME
    ) or getattr(ToolParserManager, "lazy_parsers", {}).get(PARSER_NAME)
    if existing is not None:
        return False
    ToolParserManager.register_lazy_module(
        name=PARSER_NAME,
        module_path=_TOOL_MODULE,
        class_name=_TOOL_CLASS,
    )
    return True


def register_xing41_support() -> None:
    """Install Xing4.1 config alias and parsers.

    Idempotent: spawned inspection and worker processes each import the plugin
    independently, so repeated calls must be safe.
    """
    global _registered
    if _registered:
        return

    installed = []
    if register_config_alias():
        installed.append(f"config alias {MODEL_TYPE} -> {CONFIG_CLASS}")
    if register_reasoning_parser():
        installed.append(f"reasoning parser '{PARSER_NAME}'")
    if register_tool_parser():
        installed.append(f"tool parser '{PARSER_NAME}'")

    if installed:
        logger.info("Xing4.1 support enabled: %s", ", ".join(installed))
    _registered = True


__all__ = [
    "CONFIG_CLASS",
    "MODEL_TYPE",
    "PARSER_NAME",
    "register_config_alias",
    "register_reasoning_parser",
    "register_tool_parser",
    "register_xing41_support",
]
