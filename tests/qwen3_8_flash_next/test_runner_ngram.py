"""Exercise runner-owned CPU n-gram preparation without allocating a model."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _method(name="_prepare_ngram_context"):
    # The runner imports distributed worker runtimes. Compile only this pure
    # input-preparation method so this test also runs on CPU-only build hosts.
    path = Path(__file__).parents[2] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerFL"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": torch, "CUDAGraphMode": object}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method.name]


def _function(name):
    path = Path(__file__).parents[2] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"Any": object}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def test_ngram_runner_path_requires_explicit_model_capability():
    uses_ngram = _function("_uses_qwen38_ngram_embedding")
    assert not uses_ngram(SimpleNamespace(ple_layer_ids=[1]))
    assert not uses_ngram(
        SimpleNamespace(supports_qwen38_ngram_context=True, ple_layer_ids=[])
    )
    assert uses_ngram(
        SimpleNamespace(supports_qwen38_ngram_context=True, ple_layer_ids=[1])
    )


@pytest.mark.parametrize("prompt_embeds", [False, True])
def test_ngram_context_chunk_boundary_eos_and_graph_padding(prompt_embeds):
    history = np.array([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=np.int32)
    mask = np.ones_like(history, dtype=bool)
    mask[0, 2] = False
    cpu = np.full((4, 3), -9, dtype=np.int32)
    gpu = torch.full((4, 3), -9, dtype=torch.int32)
    buffer = SimpleNamespace(np=cpu, gpu=gpu)
    buffer.copy_to_gpu = lambda rows: gpu[:rows].copy_(torch.from_numpy(cpu[:rows]))
    runner = SimpleNamespace(
        uses_ngram_embedding=True,
        ngram_context_len=3,
        ngram_eos_token_id=99,
        ngram_context=buffer,
        enable_prompt_embeds=prompt_embeds,
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.array([4, 1]),
            token_ids_cpu=history,
            is_token_ids=mask,
        ),
    )
    result = _method()(runner, 2, 4)
    expected = [
        [11, 99 if prompt_embeds else 12, 13],
        [99, 99, 20],
        [99, 99, 99],
        [99, 99, 99],
    ]
    assert result.tolist() == expected
    # Request reuse must reset the entire left context, not leak prior history.
    runner.input_batch.num_computed_tokens_cpu[:] = 0
    assert _method()(runner, 2, 4).tolist() == [[99] * 3] * 4


def test_non_ple_runner_never_enters_new_metadata_path():
    runner = SimpleNamespace(common_attention_metadata_graph=None)
    # No input batch, device, graph runtime or dispatch is needed by this path.
    assert _method("_run_common_attention_metadata")(runner, 4, object()) is False
