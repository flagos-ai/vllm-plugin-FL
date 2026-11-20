# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""P800 TP4 regression for V2 Mamba state migration and prefix reuse."""

from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

MAMBA_CACHE_WORKER_EXTENSION = (
    "tests.e2e_tests.inference.test_kunlunxin_mamba_align.MambaCacheInspection"
)
_GENERATED_TOKENS = 16


def _sampling_params():
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0, max_tokens=_GENERATED_TOKENS, ignore_eos=True
    )


@contextmanager
def _engine(model: Path, *, enforce_eager: bool, prefix_caching: bool):
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = None
    try:
        llm = LLM(
            model=str(model),
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=16,
            block_size=128,
            gpu_memory_utilization=0.7,
            enforce_eager=enforce_eager,
            enable_prefix_caching=prefix_caching,
            worker_extension_cls=MAMBA_CACHE_WORKER_EXTENSION,
        )
        yield llm
    finally:
        if llm is not None:
            with suppress(Exception):
                llm.llm_engine.engine_core.shutdown()
            del llm
        cleanup_dist_env_and_memory()


class MambaCacheInspection:
    def get_align_mamba_block_size(self) -> int:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
        from vllm.v1.worker.mamba_utils import get_mamba_groups

        runner = self.model_runner
        assert isinstance(runner.model_state, MambaHybridModelState)
        assert runner.cache_config.enable_prefix_caching
        assert runner.cache_config.mamba_cache_mode == "align"
        (block_size,) = {
            spec.block_size for spec in get_mamba_groups(runner.kv_cache_config)
        }
        assert block_size == runner.cache_config.mamba_block_size
        return block_size


def assert_mamba_align_prefix_reuse(llm):
    """Check actual Mamba cache reuse across a decode block boundary."""
    config = llm.llm_engine.vllm_config
    assert config.use_v2_model_runner
    assert config.compilation_config.cudagraph_mode.has_full_cudagraphs() == (
        not config.model_config.enforce_eager
    )
    (block_size,) = set(llm.collective_rpc("get_align_mamba_block_size"))

    # Keep one reusable checkpoint in the prompt and force decode across the
    # next boundary. The new block then needs the previous block's state.
    boundary = 2 * block_size
    prompt_length = boundary - 4
    tokenizer = llm.get_tokenizer()
    lead = tokenizer.encode(
        "This is a plain test sentence. " * prompt_length,
        add_special_tokens=False,
    )
    tail = tokenizer.encode("\nThe capital of France is", add_special_tokens=False)
    prompt = {"prompt_token_ids": lead[: prompt_length - len(tail)] + tail}
    assert len(prompt["prompt_token_ids"]) == prompt_length

    params = _sampling_params()
    assert llm.reset_prefix_cache()
    (cold,) = llm.generate([prompt], params, use_tqdm=False)
    (warm,) = llm.generate([prompt], params, use_tqdm=False)

    print(
        f"Mamba block={block_size}, prompt={prompt_length}, boundary={boundary}, "
        f"cold cached={cold.num_cached_tokens}, warm cached={warm.num_cached_tokens}"
    )
    assert cold.num_cached_tokens == 0
    assert warm.num_cached_tokens is not None and warm.num_cached_tokens >= block_size
    assert warm.num_cached_tokens < prompt_length
    cold_tokens = cold.outputs[0].token_ids
    warm_tokens = warm.outputs[0].token_ids
    assert len(cold_tokens) == len(warm_tokens) == _GENERATED_TOKENS
    # The final emitted token is never fed back into the recurrent state.
    assert prompt_length + len(cold_tokens) - 1 > boundary
    assert warm_tokens == cold_tokens, "Prefix hit changed greedy output"
    return prompt, cold_tokens


@pytest.mark.e2e
@pytest.mark.serial
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
def test_mamba_align_prefix_cache(
    backend, device_count, model_base_path, enforce_eager
):
    if backend != "kunlunxin" or device_count < 4:
        pytest.skip("Requires four Kunlunxin P800 accelerators")

    from vllm.platforms import current_platform

    assert current_platform.vendor_name == "kunlunxin", "Load the FL platform plugin"
    model = Path(model_base_path) / "Qwen" / "Qwen3.6-27B"
    assert model.is_dir(), f"Model checkpoint not found: {model}"

    with _engine(model, enforce_eager=enforce_eager, prefix_caching=True) as llm:
        prompt, cold_tokens = assert_mamba_align_prefix_reuse(llm)

    # Cold and warm align runs both migrate state at the decode boundary. Use
    # the same execution mode with prefix caching off as an independent oracle.
    with _engine(model, enforce_eager=enforce_eager, prefix_caching=False) as llm:
        config = llm.llm_engine.vllm_config
        assert config.use_v2_model_runner
        assert config.compilation_config.cudagraph_mode.has_full_cudagraphs() == (
            not enforce_eager
        )
        (reference,) = llm.generate([prompt], _sampling_params(), use_tqdm=False)

    assert reference.num_cached_tokens == 0
    assert cold_tokens == reference.outputs[0].token_ids, (
        "Mamba align changed greedy output compared with prefix caching disabled"
    )
