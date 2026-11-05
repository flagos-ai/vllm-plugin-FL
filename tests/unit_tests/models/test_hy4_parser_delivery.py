# SPDX-License-Identifier: Apache-2.0
"""CPU parser components, public xgrammar formats and lazy registration.

The extractors operate on plain data. Their production definitions are loaded
without importing the native serving adapters, whose target is vLLM 0.28.
"""

import ast
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

MODEL = Path(__file__).resolve().parents[3] / "vllm_fl/models/hy_v4"


def component_module(filename, adapter_name):
    path = MODEL / filename
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(not item.name.startswith("vllm") for item in node.names):
                body.append(node)
        elif isinstance(node, ast.ImportFrom):
            if not (node.module or "").startswith("vllm"):
                body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name != adapter_name:
                body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                isinstance(target, ast.Name) and target.id == "logger"
                for target in targets
            ):
                body.append(node)
    namespace = {
        "__name__": "hy4_plain_parser_contract",
        "logger": logging.getLogger(__name__),
    }

    def partial_tag_overlap(text, tag):
        return max(
            (
                size
                for size in range(min(len(text), len(tag)) + 1)
                if text.endswith(tag[:size])
            ),
            default=0,
        )

    namespace["partial_tag_overlap"] = partial_tag_overlap
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def structural_vocab(suffix=""):
    tokens = ["think", "tool_calls", "tool_call", "arg_key", "arg_value"]
    return {
        token: index
        for index, token in enumerate(
            marker
            for name in tokens
            for marker in (f"<{name}{suffix}>", f"</{name}{suffix}>")
        )
    }


@pytest.mark.parametrize("suffix", ["", ":test"])
def test_reasoning_extraction_and_batched_boundary_delta(suffix):
    module = component_module("hy_v4_reasoning_parser.py", "HYV4ReasoningParser")
    vocab = structural_vocab(suffix)
    extractor = module.HYV4ReasoningExtractor(vocab, suffix)
    end = f"</think{suffix}>"
    assert extractor.extract_reasoning(f"analysis{end}answer") == ("analysis", "answer")
    assert extractor.extract_reasoning("unfinished") == ("unfinished", None)
    end_id = vocab[end]
    delta = extractor.extract_reasoning_streaming(
        "", f"a{end}b", f"a{end}b", [], [90, end_id, 91], [90, end_id, 91]
    )
    assert delta == {"reasoning": "a", "content": "b"}
    assert extractor.extract_content_ids([90, end_id, 91]) == [91]
    assert extractor.count_reasoning_tokens([90, end_id, 91]) == 1
    assert extractor.has_reasoning_ended([90, end_id])


def test_no_think_and_tokenizer_suffix_contract():
    module = component_module("hy_v4_reasoning_parser.py", "HYV4ReasoningParser")
    tokenizer = SimpleNamespace(
        init_kwargs={}, get_vocab=lambda: structural_vocab(":test")
    )
    assert module.detect_token_suffix(tokenizer) == ":test"
    extractor = module.HYV4ReasoningExtractor({}, "", thinking=False)
    assert extractor.extract_reasoning("answer") == (None, "answer")
    assert extractor.count_reasoning_tokens([1, 2]) == 0
    assert extractor.extract_content_ids([1, 2]) == [1, 2]
    tokenizer.init_kwargs = {
        "model_specific_special_tokens": {"think_begin_token": "<think>"}
    }
    with pytest.raises(RuntimeError, match="tokenizer_config"):
        module.detect_token_suffix(tokenizer)


def tool_packet(name="weather", city="Boston", days="3", suffix=""):
    return (
        f"<tool_calls{suffix}><tool_call{suffix}>{name}"
        f"<arg_key{suffix}>city</arg_key{suffix}>"
        f"<arg_value{suffix}>{city}</arg_value{suffix}>"
        f"<arg_key{suffix}>days</arg_key{suffix}>"
        f"<arg_value{suffix}>{days}</arg_value{suffix}>"
        f"</tool_call{suffix}></tool_calls{suffix}>"
    )


def tools():
    return [
        {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        }
    ]


@pytest.mark.parametrize("suffix", ["", ":test"])
def test_tool_extractor_preserves_schema_types_and_malformed_content(suffix):
    module = component_module("hy_v4_tool_parser.py", "HYV4ToolParser")
    extractor = module.HYV4ToolExtractor(structural_vocab(suffix), suffix, strict=True)
    packet = tool_packet(suffix=suffix)
    result = extractor.extract_tool_calls(packet, tools())
    assert result["tools_called"] and result["content"] is None
    assert result["tool_calls"][0]["name"] == "weather"
    assert json.loads(result["tool_calls"][0]["arguments"]) == {
        "city": "Boston",
        "days": 3,
    }
    malformed = packet.replace(f"</arg_value{suffix}>", "", 1)
    assert extractor.extract_tool_calls(malformed, tools()) == {
        "tools_called": False,
        "content": malformed,
        "tool_calls": [],
    }


def test_batched_tool_stream_drains_every_call_and_keeps_shared_state():
    module = component_module("hy_v4_tool_parser.py", "HYV4ToolParser")
    vocab = structural_vocab()
    extractor = module.HYV4ToolExtractor(vocab, "", strict=True)
    one = tool_packet()
    two = tool_packet(city="Paris", days="2")
    packet = one.removesuffix("</tool_calls>") + two.removeprefix("<tool_calls>")
    delta = extractor.extract_tool_calls_streaming(
        "",
        packet,
        packet,
        [],
        [vocab["<tool_calls>"]],
        [vocab["<tool_calls>"]],
        tools(),
    )
    assert [entry["index"] for entry in delta["tool_calls"]] == [0, 1]
    assert [json.loads(entry["arguments"]) for entry in delta["tool_calls"]] == [
        {"city": "Boston", "days": 3},
        {"city": "Paris", "days": 2},
    ]
    assert (
        len(extractor.prev_tool_call_arr) == len(extractor.streamed_args_for_tool) == 2
    )


def test_lazy_parser_registration_uses_plugin_modules_and_is_repeatable(monkeypatch):
    reason, tool = {}, {}
    for name, target, attribute in [
        ("vllm.reasoning", reason, "ReasoningParserManager"),
        ("vllm.tool_parsers.abstract_tool_parser", tool, "ToolParserManager"),
    ]:
        module = ModuleType(name)

        def register(name, module_path, class_name, target=target):
            target[name] = (module_path, class_name)

        setattr(module, attribute, SimpleNamespace(register_lazy_module=register))
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "hy4_registration_contract", MODEL / "register.py"
    )
    registration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registration)
    registration.register_hy_v4_parsers()
    registration.register_hy_v4_parsers()
    assert reason == {
        "hy_v4": ("vllm_fl.models.hy_v4.hy_v4_reasoning_parser", "HYV4ReasoningParser")
    }
    assert tool == {
        "hy_v4": ("vllm_fl.models.hy_v4.hy_v4_tool_parser", "HYV4ToolParser")
    }


def test_required_and_named_grammars_use_checkpoint_suffix_and_tool_name():
    spec = importlib.util.spec_from_file_location(
        "hy4_grammar_contract", MODEL / "tool_structural_tag.py"
    )
    grammar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grammar)
    public_tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    for choice in ["required", {"type": "function", "function": {"name": "weather"}}]:
        tag = grammar.build_request_structural_tag(
            public_tools, choice, reasoning=True, token_suffix=":test"
        )
        value = tag.model_dump_json()
        assert "</think:test>" in value and "<tool_call:test>weather" in value
        assert "<arg_key:test>" in value
    with pytest.raises(ValueError, match="function tools"):
        grammar.build_request_structural_tag([{"type": "web_search"}], "required")
