# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in checkpoint-name matching for upstream compressed-tensors configs."""

from __future__ import annotations

from copy import copy
from functools import update_wrapper
from typing import TYPE_CHECKING

import regex as re

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

_CONTRACT_ATTRIBUTE = "_fl_checkpoint_contract"
_WRAPPER_MARKER = "_fl_checkpoint_contract_wrapper"


class _CheckpointContract:
    """Serializable metadata state, without bound callbacks or device policy."""

    def __init__(self, mapper):
        self.mapper = mapper
        self.mapper_applied = False
        self.checkpoint_names_enabled = False
        self.ignore_cache = None

    def checkpoint_names(self, name):
        mapper = self.mapper
        if not self.checkpoint_names_enabled or mapper is None:
            return [name]
        canonical = name + ".weight"
        candidates = {canonical}
        # Multiple checkpoint roots may map to the same runtime root. Verify
        # each inverse through the model's unchanged forward weight mapper.
        for mappings, kind in (
            (mapper.orig_to_new_suffix, "suffix"),
            (mapper.orig_to_new_prefix, "prefix"),
            (mapper.orig_to_new_substr, "substr"),
        ):
            if kind == "substr":
                for source, target in reversed(tuple(mappings.items())):
                    if target:
                        candidates.update(
                            key.replace(target, source, 1)
                            for key in tuple(candidates)
                            if target in key
                        )
            else:
                originals = tuple(candidates)
                for source, target in mappings.items():
                    if not target:
                        continue
                    for key in originals:
                        if kind == "prefix" and key.startswith(target):
                            candidates.add(source + key[len(target) :])
                        elif kind == "suffix" and key.endswith(target):
                            candidates.add(key[: -len(target)] + source)
        return [name] + sorted(
            key.removesuffix(".weight")
            for key in candidates - {canonical}
            if mapper._map_name(key) == canonical
        )

    def should_ignore(self, config, name):
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )

        if name is None:
            return False
        signature = tuple(config.ignore)
        cached = self.ignore_cache
        if cached is None or cached[0] != signature:
            literals = frozenset(
                rule for rule in signature if not rule.startswith("re:")
            )
            patterns = tuple(
                re.compile(rule[3:]) for rule in signature if rule.startswith("re:")
            )
            cached = (signature, literals, patterns)
            self.ignore_cache = cached
        _, literals, patterns = cached
        proj_name = name.split(".")[-1]
        if proj_name in config.packed_modules_mapping and name not in literals:
            names = [
                name.replace(proj_name, shard)
                for shard in config.packed_modules_mapping[proj_name]
            ]
        else:
            names = [name]
        # A regex match on a packed root must not become a literal exemption
        # from upstream's shard-consistency checks.
        matched = tuple(
            candidate
            for candidate in names
            if candidate in literals
            or any(pattern.match(candidate) for pattern in patterns)
        )
        return should_ignore_layer(name, matched, config.packed_modules_mapping)


def _resolve_scheme(config, contract, layer, name, original_lookup):
    names = contract.checkpoint_names(name)
    if any(contract.should_ignore(config, alias) for alias in names):
        return None
    # Keep upstream target/class/fused matching and format defaults. A fresh
    # view sees metadata mutations without repeating the resolved ignore scan.
    view = copy(config)
    view.ignore = ()
    schemes = [original_lookup(view, layer, alias) for alias in names]
    matched = [scheme for scheme in schemes if scheme is not None]
    if matched and any(scheme != matched[0] for scheme in matched[1:]):
        raise ValueError(f"Conflicting quantization aliases for {name}")
    return matched[0] if matched else None


def _uniform_scheme(schemes, prefix):
    if all(scheme is None for scheme in schemes):
        return None
    if any(scheme is None or scheme != schemes[0] for scheme in schemes):
        raise ValueError(f"Mixed quantization schemes in packed layer {prefix}")
    return schemes[0]


def _install_checkpoint_contract_wrappers():
    """Extend only explicitly bound configs; preserve other callers exactly."""
    config_cls = CompressedTensorsConfig
    current_mapper = config_cls.apply_vllm_mapper
    if getattr(current_mapper, _WRAPPER_MARKER, False):
        return
    current_lookup = config_cls.get_scheme_dict
    current_method = config_cls.get_quant_method

    def apply_vllm_mapper(self, mapper):
        contract = getattr(self, _CONTRACT_ATTRIBUTE, None)
        if contract is None:
            return current_mapper(self, mapper)
        if contract.mapper != mapper:
            raise ValueError("Quantization mapper already applied")
        if contract.mapper_applied:
            return None
        # Arbitrary regex/renaming rewrites cannot be inverted. Retain the
        # captured upstream mapper behavior for these existing mapper forms.
        if mapper.orig_to_new_regex or mapper.orig_to_new_renamings:
            result = current_mapper(self, mapper)
        else:
            contract.checkpoint_names_enabled = True
            result = None
        contract.mapper_applied = True
        return result

    def get_scheme_dict(self, layer, layer_name=None):
        contract = getattr(self, _CONTRACT_ATTRIBUTE, None)
        if contract is None or layer_name is None:
            return current_lookup(self, layer, layer_name)
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            name = layer_name.rsplit(".", 1)[-1]
            components = self.packed_modules_mapping.get(name)
            if components is not None:
                if any(
                    contract.should_ignore(self, alias)
                    for alias in contract.checkpoint_names(layer_name)
                ):
                    return None
                base = layer_name[: -len(name)]
                return _uniform_scheme(
                    [
                        _resolve_scheme(
                            self, contract, layer, base + component, current_lookup
                        )
                        for component in components
                    ],
                    layer_name,
                )
        return _resolve_scheme(self, contract, layer, layer_name, current_lookup)

    def get_quant_method(self, layer, prefix):
        contract = getattr(self, _CONTRACT_ATTRIBUTE, None)
        if contract is None:
            return current_method(self, layer, prefix)
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, RoutedExperts):
            self._add_fused_moe_to_target_scheme_map()
            _uniform_scheme(
                [
                    self.get_scheme_dict(layer, f"{prefix}.{expert}.{projection}")
                    for expert in range(layer.moe_config.num_experts)
                    for projection in ("gate_proj", "up_proj", "down_proj")
                ],
                prefix,
            )
        elif isinstance(layer, (LinearBase, ParallelLMHead)):
            # Upstream LMHead compatibility fallback catches ValueError. Check
            # metadata consistency before entering that fallback.
            self.get_scheme_dict(layer, prefix)
        return current_method(self, layer, prefix)

    for wrapper, original in (
        (apply_vllm_mapper, current_mapper),
        (get_scheme_dict, current_lookup),
        (get_quant_method, current_method),
    ):
        update_wrapper(wrapper, original)
        setattr(wrapper, _WRAPPER_MARKER, True)
    config_cls.apply_vllm_mapper = apply_vllm_mapper
    config_cls.get_scheme_dict = get_scheme_dict
    config_cls.get_quant_method = get_quant_method


def bind_compressed_tensors_model(
    config,
    mapper: WeightsMapper | None,
    packed_mapping: dict[str, list[str]] | None,
) -> bool:
    """Opt an existing upstream config into a model's checkpoint-name contract.

    Call from the model's SupportsQuant mapping hook before upstream rewrites
    targets. Non-compressed-tensors configs are untouched and return False.
    Rebinding an equivalent mapper is idempotent, including after serialization;
    a different mapping is rejected.
    """
    if not isinstance(config, CompressedTensorsConfig):
        return False
    _install_checkpoint_contract_wrappers()
    contract = getattr(config, _CONTRACT_ATTRIBUTE, None)
    if contract is None:
        contract = _CheckpointContract(mapper)
        setattr(config, _CONTRACT_ATTRIBUTE, contract)
    elif contract.mapper != mapper:
        raise ValueError("Quantization mapper already applied")
    if packed_mapping is not None:
        config.packed_modules_mapping = packed_mapping
    if mapper is not None:
        config.apply_vllm_mapper(mapper)
    return True


__all__ = ["bind_compressed_tensors_model"]
