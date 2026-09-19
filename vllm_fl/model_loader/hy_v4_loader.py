# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-sliced safetensors loader for the packed HY4 checkpoint."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any

import torch
from safetensors.torch import safe_open
from tqdm.auto import tqdm

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.model_loader.weight_utils import enable_tqdm

logger = init_logger(__name__)

_PACKED_EXPERT_NAMES = (
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
)


def _contiguous_runs(expert_ids: list[int]) -> list[tuple[int, int]]:
    """Return half-open contiguous runs for sorted expert IDs."""
    if not expert_ids:
        return []
    runs = []
    start = previous = expert_ids[0]
    for expert_id in expert_ids[1:]:
        if expert_id != previous + 1:
            runs.append((start, previous + 1))
            start = expert_id
        previous = expert_id
    runs.append((start, previous + 1))
    return runs


def _read_local_experts(tensor_slice: Any, expert_ids: list[int]) -> torch.Tensor:
    """Materialize only the selected leading expert rows from safetensors."""
    pieces = [tensor_slice[start:end] for start, end in _contiguous_runs(expert_ids)]
    if not pieces:
        raise ValueError("HY4 expert-sliced loading requires local experts")
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)


class HYV4SafetensorsLoader(DefaultModelLoader):
    """Avoid reading every packed expert on every HY4 expert-parallel rank.

    HY4 stores one layer's 256 routed experts in four fused tensors totaling
    roughly 9.3 GiB. The default safetensors iterator materializes each full
    tensor before the MoE parameter loader selects the rank-local experts.
    This loader performs that selection through ``safe_open.get_slice`` first.
    """

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Reuse vLLM's safetensors discovery for the plugin load format.

        vLLM 0.24 supports out-of-tree loader registration, but
        ``DefaultModelLoader._prepare_weights`` only recognizes its built-in
        format names. Temporarily present this plugin format as explicit
        safetensors while delegating file discovery, then restore the original
        value before any tensors are read.
        """
        original_format = self.load_config.load_format
        self.load_config.load_format = "safetensors"
        try:
            return super()._prepare_weights(
                model_name_or_path,
                subfolder,
                revision,
                fall_back_to_pt,
                allow_patterns_overrides,
            )
        finally:
            self.load_config.load_format = original_format

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        self._init_ep_weight_filter(model_config)
        local_expert_ids = self.local_expert_ids
        if local_expert_ids is None:
            raise ValueError(
                "load_format=hy4_safetensors requires expert-parallel weight "
                "filtering; start vLLM with --enable-expert-parallel and "
                "--enable-ep-weight-filter"
            )
        if model.__class__.__name__ != "HYV4ForCausalLM":
            raise ValueError(
                "load_format=hy4_safetensors is specific to HYV4ForCausalLM"
            )

        expert_ids = sorted(local_expert_ids)
        model._hy4_local_expert_ids = expert_ids
        try:
            super().load_weights(model, model_config)
        finally:
            del model._hy4_local_expert_ids

    def _get_weights_iterator(
        self,
        source: DefaultModelLoader.Source,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        _, weight_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if not use_safetensors:
            raise ValueError("HY4 expert-sliced loading requires safetensors weights")
        if self.local_expert_ids is None:
            raise ValueError("HY4 expert-sliced loading requires local expert IDs")

        expert_ids = sorted(self.local_expert_ids)
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        files = tqdm(
            weight_files,
            desc="Loading HY4 expert-sliced safetensors",
            disable=not enable_tqdm(self.load_config.use_tqdm_on_load),
        )
        for weight_file in files:
            with safe_open(weight_file, framework="pt") as handle:
                for name in handle.keys():  # noqa: SIM118
                    if name.startswith("model.mtp_layers."):
                        continue
                    if "rotary_emb.inv_freq" in name:
                        continue
                    if should_skip_weight(name, self.local_expert_ids):
                        continue
                    if any(pattern in name for pattern in _PACKED_EXPERT_NAMES):
                        loaded_weight = _read_local_experts(
                            handle.get_slice(name),
                            expert_ids,
                        )
                    else:
                        loaded_weight = handle.get_tensor(name)
                    yield source.prefix + name, loaded_weight


__all__ = ["HYV4SafetensorsLoader"]
