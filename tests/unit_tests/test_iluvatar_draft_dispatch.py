"""Exercise draft dispatch without importing device-specific dependencies."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest


@pytest.fixture
def propose_draft_token_ids():
    source = Path(__file__).parents[2] / "vllm_fl/worker/iluvatar_model_runner.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "propose_draft_token_ids"
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"Any": Any, "cast": cast}
    exec(
        compile(
            module, str(source), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    return namespace["propose_draft_token_ids"]


def make_runner(method):
    config = SimpleNamespace(
        method=method,
        use_ngram_gpu=lambda: False,
        uses_extract_hidden_states=lambda: False,
        use_eagle=lambda: False,
        use_dflash=lambda: False,
        uses_draft_model=lambda: False,
    )
    return SimpleNamespace(speculative_config=config)


@pytest.mark.parametrize("method", ["unsupported_method", "future_draft_method"])
def test_unsupported_method_raises(propose_draft_token_ids, method):
    runner = make_runner(method)
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=1, num_spec_tokens_to_schedule=1
    )
    with pytest.raises(
        ValueError, match=f"Unsupported speculative decoding method: {method}"
    ):
        propose_draft_token_ids(
            runner, scheduler, [], None, None, None, None, None, None, None
        )


def test_custom_class_returns_proposal(propose_draft_token_ids):
    runner = make_runner("custom_class")
    expected = [[7, 8]]
    runner.drafter = SimpleNamespace(propose=Mock(return_value=expected))
    runner.input_batch = SimpleNamespace(
        num_tokens_no_spec=object(), token_ids_cpu=object()
    )
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=1, num_spec_tokens_to_schedule=2
    )
    sampled = [[3]]
    result = propose_draft_token_ids(
        runner, scheduler, sampled, None, None, None, None, None, None, None
    )
    assert result is expected
    runner.drafter.propose.assert_called_once_with(
        sampled,
        runner.input_batch.num_tokens_no_spec,
        runner.input_batch.token_ids_cpu,
        slot_mappings=None,
    )
