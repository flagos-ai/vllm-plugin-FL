# Copyright (c) 2026 BAAI. All rights reserved.
"""Seeded sampling must survive discarded chunked-prefill samples on TXDA."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


class _AfterGeneratorRollback(Exception):
    pass


class _StopBeforeOutputCopy:
    @property
    def sampled_token_ids(self):
        raise _AfterGeneratorRollback


def _runner_methods():
    # Execute the actual runner methods without importing its accelerator-only
    # dependencies, so the RNG regression also runs on CPU-only CI.
    path = Path(__file__).parents[3] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_sample", "_bookkeeping_sync"}
    ]
    namespace = {
        "torch": torch,
        "np": np,
        "envs": SimpleNamespace(VLLM_COMPUTE_NANS_IN_LOGITS=False),
    }
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["_sample"], namespace["_bookkeeping_sync"]


@pytest.mark.parametrize("speculative", [False, True])
def test_discarded_cpu_sample_restores_state_but_keeps_accepted_sample(speculative):
    sample, bookkeeping = _runner_methods()
    generators = {i: torch.Generator().manual_seed(120 + i) for i in range(2)}
    initial_states = {i: gen.get_state().clone() for i, gen in generators.items()}
    samples = {}

    def draw(*args, **kwargs):
        samples.update({i: torch.rand(17, generator=g) for i, g in generators.items()})
        return _StopBeforeOutputCopy()

    batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(generators=generators),
        update_async_output_token_ids=Mock(),
        num_reqs=2,
        generators=generators,
        req_ids=["discarded", "accepted"],
        req_id_to_index={"discarded": 0, "accepted": 1},
    )
    runner = SimpleNamespace(
        input_batch=batch,
        sampler=draw,
        rejection_sampler=draw,
        use_async_scheduling=False,
        _get_spec_decode_draft_probs=lambda metadata: None,
        discard_request_mask=SimpleNamespace(np=np.array([True, False])),
    )
    result = sample(runner, None, object() if speculative else None)
    accepted_state = generators[1].get_state().clone()
    with pytest.raises(_AfterGeneratorRollback):
        bookkeeping(runner, None, result, None, None, 2)

    assert torch.equal(generators[0].get_state(), initial_states[0])
    assert torch.equal(generators[1].get_state(), accepted_state)
    assert torch.equal(torch.rand(17, generator=generators[0]), samples[0])

    # A subsequent step takes a fresh snapshot, not the previous step's state.
    runner.discard_request_mask.np[:] = [False, True]
    before_second = generators[1].get_state().clone()
    result = sample(runner, None, None)
    with pytest.raises(_AfterGeneratorRollback):
        bookkeeping(runner, None, result, None, None, 2)
    assert torch.equal(generators[1].get_state(), before_second)


def test_accelerator_generator_keeps_philox_offset_rollback():
    sample, bookkeeping = _runner_methods()
    generator = Mock(device=SimpleNamespace(type="cuda"))
    generator.get_offset.return_value = 16
    batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(generators={0: generator}),
        update_async_output_token_ids=Mock(),
        num_reqs=1,
        generators={0: generator},
        req_ids=["discarded"],
        req_id_to_index={"discarded": 0},
    )
    runner = SimpleNamespace(
        input_batch=batch,
        sampler=lambda **kwargs: _StopBeforeOutputCopy(),
        use_async_scheduling=False,
        discard_request_mask=SimpleNamespace(np=np.array([True])),
    )
    result = sample(runner, None, None)
    with pytest.raises(_AfterGeneratorRollback):
        bookkeeping(runner, None, result, None, None, 1)
    generator.get_state.assert_not_called()
    generator.set_state.assert_not_called()
    generator.set_offset.assert_called_once_with(12)
