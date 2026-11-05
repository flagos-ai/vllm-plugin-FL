# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 tool-call envelope grammar using xgrammar's public APIs.

Tool names and argument tag order are constrained. Argument value bodies use
free text, as in the checkpoint's original structural format.
"""

from __future__ import annotations

from typing import Any, Literal

from xgrammar import StructuralTag, normalize_tool_choice
from xgrammar.openai_tool_call_schema import BuiltinToolParam, FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    OrFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

SimplifiedToolChoice = Literal["auto", "required", "forced"]


def _dump_function_tool(tool) -> dict[str, Any]:
    value = (
        tool.model_dump(mode="json", exclude_none=True)
        if hasattr(tool, "model_dump")
        else dict(tool)
    )
    if value.get("type") != "function":
        raise ValueError("Hy4 structural tags support function tools only")
    if "function" in value:
        return value
    return {
        "type": "function",
        "function": {
            key: value[key]
            for key in ("name", "description", "parameters", "strict")
            if key in value
        },
    }


def _dump_tool_choice(choice):
    if isinstance(choice, str) or choice is None:
        return choice
    value = (
        choice.model_dump(mode="json", exclude_none=True)
        if hasattr(choice, "model_dump")
        else dict(choice)
    )
    if value.get("type") == "function" and "name" in value:
        return {"type": "function", "function": {"name": value["name"]}}
    if value.get("type") == "allowed_tools" and "allowed_tools" not in value:
        return {
            "type": "allowed_tools",
            "allowed_tools": {"mode": value["mode"], "tools": value["tools"]},
        }
    return value


def build_request_structural_tag(
    tools, tool_choice, *, reasoning=False, token_suffix=""
):
    if not tools or tool_choice == "none":
        raise ValueError("Hy4 required/named tool choice needs function tools")
    functions, builtin_tools, choice = normalize_tool_choice(
        [_dump_function_tool(tool) for tool in tools], _dump_tool_choice(tool_choice)
    )
    if builtin_tools or not functions:
        raise ValueError("Hy4 structural tags require at least one function tool")
    return get_hy_v4_structural_tag(functions, [], choice, reasoning, token_suffix)


def _hy_v4_argument_keys(parameters: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Split a tool's argument keys into ``(required, optional)``.

    Both lists keep declaration order. Required keys missing from
    ``properties`` are still treated as required.
    """
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    required_set = set(required)
    required_keys = [k for k in properties if k in required_set]
    required_keys += [k for k in required if k not in properties]
    optional_keys = [k for k in properties if k not in required_set]
    return required_keys, optional_keys


def get_hy_v4_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    """Build HYV4 <tool_calls>/<tool_call>/<arg_key>/<arg_value> structural tags.

    Args:
        tools: Normalized function tools the model may call.
        builtin_tools: Unused; HYV4 has no builtin tools.
        tool_choice: Simplified tool choice.
        reasoning: Whether the grammar also covers the reasoning phase.
        token_suffix: Per-checkpoint structural-token suffix including the
            leading colon (e.g. ``":6124c78e"``), or ``""`` when the checkpoint
            uses unsuffixed tokens. The HYV4 tool parser reads it off the
            tokenizer vocab and passes it to this module's request builder.

    """
    del builtin_tools

    think_end = f"</think{token_suffix}>"
    tool_calls_begin = f"<tool_calls{token_suffix}>"
    tool_calls_end = f"</tool_calls{token_suffix}>"
    tool_call_begin = f"<tool_call{token_suffix}>"
    tool_call_end = f"</tool_call{token_suffix}>"
    arg_key_begin = f"<arg_key{token_suffix}>"
    arg_key_end = f"</arg_key{token_suffix}>"
    arg_value_begin = f"<arg_value{token_suffix}>"
    arg_value_end = f"</arg_value{token_suffix}>"

    def arg_pair(key: str) -> SequenceFormat:
        return SequenceFormat(
            elements=[
                ConstStringFormat(value=arg_key_begin),
                ConstStringFormat(value=key),
                ConstStringFormat(value=arg_key_end),
                ConstStringFormat(value=arg_value_begin),
                AnyTextFormat(),
                ConstStringFormat(value=arg_value_end),
            ]
        )

    def single_tool_call(tool: FunctionToolParam) -> TagFormat:
        function = tool.function
        parameters = (
            function.parameters if isinstance(function.parameters, dict) else {}
        )
        required_keys, optional_keys = _hy_v4_argument_keys(parameters)
        elements: list[Any] = [arg_pair(k) for k in required_keys]
        if optional_keys:
            # Optional keys may appear in any order, or not at all.
            optional_pairs = [arg_pair(k) for k in optional_keys]
            elements.append(
                StarFormat(
                    content=optional_pairs[0]
                    if len(optional_pairs) == 1
                    else OrFormat(elements=optional_pairs)
                )
            )
        content: Any = (
            SequenceFormat(elements=elements) if elements else AnyTextFormat()
        )
        return TagFormat(
            begin=tool_call_begin + function.name,
            content=content,
            end=tool_call_end,
        )

    tags = [single_tool_call(tool) for tool in tools]

    if tool_choice == "auto":
        block = TagFormat(
            begin=tool_calls_begin,
            content=TagsWithSeparatorFormat(
                tags=tags,
                separator="",
                at_least_one=True,
            ),
            end=tool_calls_end,
        )
        suffix_tag: Any = TriggeredTagsFormat(triggers=[tool_calls_begin], tags=[block])
    else:
        # HYV4 places tool_call blocks back-to-back with no separator. "forced"
        # has already been filtered down to the single named tool and must emit
        # exactly one block.
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=tool_calls_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                    stop_after_first=tool_choice == "forced",
                ),
                ConstStringFormat(value=tool_calls_end),
            ]
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    # The reasoning phase is constrained too, so the tag must explicitly skip
    # the ``<think>...</think:SUF>`` prefix.
    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=think_end)
    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))
