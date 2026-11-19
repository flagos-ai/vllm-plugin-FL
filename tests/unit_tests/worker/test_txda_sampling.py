# Copyright (c) 2026 BAAI. All rights reserved.
"""V1 seeded sampling preserves discarded CPU states and accelerator offsets."""

import __future__

import ast
import importlib.util
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
    helper_path = path.with_name("sampling_rng.py")
    spec = importlib.util.spec_from_file_location("sampling_rng", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    namespace = {
        "preserve_discarded_generator_offsets": helper.preserve_discarded_generator_offsets,
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


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("consumed", [0, 4, 256, 31040, 124160])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_accelerator_generator_restores_entry_offset(
    speculative, consumed, fail, asynchronous
):
    sample, bookkeeping = _runner_methods()
    generators = {i: Mock(device=SimpleNamespace(type="cuda")) for i in [0, 1]}
    offsets = {0: 16, 1: 32}
    for i, gen in generators.items():
        gen.get_offset.side_effect = lambda i=i: offsets[i]
        gen.set_offset.side_effect = lambda value, i=i: offsets.__setitem__(i, value)

    def draw(*args, **kwargs):
        for i in offsets:
            offsets[i] += consumed
        if fail:
            raise RuntimeError("sampler failed")
        return _StopBeforeOutputCopy()

    batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(generators=generators),
        update_async_output_token_ids=Mock(),
        num_reqs=3,
        generators=generators,
        req_ids=["discarded", "accepted", "unseeded"],
        req_id_to_index={"discarded": 0, "accepted": 1, "unseeded": 2},
    )
    runner = SimpleNamespace(
        input_batch=batch,
        sampler=Mock(side_effect=draw),
        rejection_sampler=Mock(side_effect=draw),
        _draft_token_req_ids=None,
        use_async_scheduling=asynchronous,
        _get_spec_decode_draft_probs=lambda metadata: None,
        discard_request_mask=SimpleNamespace(np=np.array([True, False, True])),
    )
    if fail:
        with pytest.raises(RuntimeError, match="sampler failed"):
            sample(runner, None, object() if speculative else None)
    else:
        result = sample(runner, None, object() if speculative else None)
        with pytest.raises(_AfterGeneratorRollback):
            bookkeeping(runner, None, result, None, None, 3)
    assert offsets == {0: 16, 1: 32 + consumed}
    generators[0].set_offset.assert_called_once_with(16)
    generators[1].set_offset.assert_not_called()
    for gen in generators.values():
        gen.get_state.assert_not_called()
        gen.set_state.assert_not_called()
    assert runner.sampler.call_count + runner.rejection_sampler.call_count == 1


@pytest.mark.parametrize("fail", [False, True])
def test_discarded_cpu_generator_restores_on_exception_and_mapping_replacement(fail):
    path = Path(__file__).parents[3] / "vllm_fl/worker/sampling_rng.py"
    spec = importlib.util.spec_from_file_location("sampling_rng", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original = torch.Generator().manual_seed(17)
    accepted = torch.Generator().manual_seed(29)
    replacement = torch.Generator().manual_seed(31)
    initial = original.get_state().clone()
    replacement_state = replacement.get_state().clone()
    generators = {0: original, 1: accepted}

    def draw():
        with helper.preserve_discarded_generator_offsets(generators, [0, 2]):
            torch.rand(513, generator=original)
            torch.rand(1027, generator=accepted)
            generators[0] = replacement
            if fail:
                raise RuntimeError("sampler failed")

    if fail:
        with pytest.raises(RuntimeError, match="sampler failed"):
            draw()
    else:
        draw()
    assert torch.equal(original.get_state(), initial)
    assert torch.equal(replacement.get_state(), replacement_state)
    reference = torch.Generator().manual_seed(29)
    torch.rand(1027, generator=reference)
    assert torch.equal(accepted.get_state(), reference.get_state())
