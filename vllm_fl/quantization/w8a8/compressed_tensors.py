# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend dynamic W8A8 execution for compressed-tensors checkpoints."""

from copy import copy
from functools import cache

import regex as re
import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_fl.dispatch import CachedOp
from vllm_fl.quantization.w8a8.moe import AscendW8A8DynamicMoEMethod


class AscendCompressedTensorsConfig(CompressedTensorsConfig):
    """Keep the upstream parser and checkpoint matching; replace supported math.

    Metadata uses checkpoint names, including regexes. Matching before model
    name translation preserves the exporter's ignore rules and packed shards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_mapper = None
        self._applied_mapper = None
        self.weight_nz_mode = 1
        self._ignore_cache = None
        self._add_fused_moe_to_target_scheme_map()

    def apply_vllm_mapper(self, mapper):
        if self._applied_mapper is mapper:
            return
        if self._applied_mapper is not None:
            raise ValueError("Quantization mapper already applied")
        self._applied_mapper = mapper
        # Regex substitution cannot in general be reversed. Such models retain
        # the upstream canonical-name contract instead of guessing an inverse.
        if mapper.orig_to_new_regex or mapper.orig_to_new_renamings:
            super().apply_vllm_mapper(mapper)
        else:
            self._weight_mapper = mapper

    def _checkpoint_names(self, name):
        """Resolve literal aliases without translating exporter regexes."""
        mapper = self._weight_mapper
        if mapper is None:
            return [name]
        canonical = name + ".weight"
        candidates = {canonical}
        # Several checkpoint roots can map to one runtime root. Keep every
        # valid inverse, then verify it through the model's forward mapper.
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

    def _should_ignore_cached(self, name):
        """Retain compiled rules while preserving upstream fused consistency."""
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )

        if name is None:
            return False
        signature = tuple(self.ignore)
        cached = self._ignore_cache
        if cached is None or cached[0] != signature:
            literals = frozenset(
                rule for rule in signature if not rule.startswith("re:")
            )
            patterns = tuple(
                re.compile(rule[3:]) for rule in signature if rule.startswith("re:")
            )
            cached = (signature, literals, patterns)
            self._ignore_cache = cached
        _, literals, patterns = cached
        proj_name = name.split(".")[-1]
        if proj_name in self.packed_modules_mapping and name not in literals:
            names = [
                name.replace(proj_name, shard)
                for shard in self.packed_modules_mapping[proj_name]
            ]
        else:
            names = [name]
        # Pass only resolved exact names back through the upstream predicate.
        # A regex matching a packed root must not bypass its shard checks.
        matched = tuple(
            candidate
            for candidate in names
            if candidate in literals
            or any(pattern.match(candidate) for pattern in patterns)
        )
        return should_ignore_layer(name, matched, self.packed_modules_mapping)

    def _scheme(self, layer, prefix):
        names = self._checkpoint_names(prefix)
        if any(self._should_ignore_cached(name) for name in names):
            return None
        # Ignore rules have already been resolved across every alias. Keep
        # upstream target/class/fused matching without repeating the regex scan.
        # A fresh shallow view observes current mapper/packed metadata.
        view = copy(self)
        view.ignore = ()
        schemes = [
            CompressedTensorsConfig.get_scheme_dict(view, layer, name) for name in names
        ]
        matched = [scheme for scheme in schemes if scheme is not None]
        if matched and any(scheme != matched[0] for scheme in matched[1:]):
            raise ValueError(f"Conflicting quantization aliases for {prefix}")
        return matched[0] if matched else None

    def _validated_scheme(self, layer, prefix, names):
        schemes = [self._scheme(layer, name) for name in names]
        if all(scheme is None for scheme in schemes):
            return None
        if any(scheme is None for scheme in schemes):
            raise ValueError(f"Mixed W8A8 schemes in packed layer {prefix}")
        if not all(self._dynamic_w8a8(scheme) for scheme in schemes):
            raise NotImplementedError(
                f"Unsupported Ascend compressed-tensors scheme at {prefix}"
            )
        if any(scheme != schemes[0] for scheme in schemes[1:]):
            raise ValueError(f"Mixed W8A8 schemes in packed layer {prefix}")
        if self.kv_cache_scheme or self.transform_config:
            raise NotImplementedError(
                "Ascend W8A8 does not implement KV quantization or transforms"
            )
        return schemes[0]

    def get_scheme_dict(self, layer, layer_name=None):
        if layer_name is None:
            return super().get_scheme_dict(layer, layer_name)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            name = layer_name.rsplit(".", 1)[-1]
            base = layer_name[: -len(name)]
            names = [base + n for n in self.packed_modules_mapping.get(name, [name])]
            return self._validated_scheme(layer, layer_name, names)
        return self._scheme(layer, layer_name)

    def get_scheme(self, layer, layer_name=None):
        scheme = super().get_scheme(layer, layer_name)
        if type(scheme) is CompressedTensorsW8A8Int8:
            return AscendW8A8Int8(
                scheme.strategy, scheme.is_static_input_scheme, scheme.input_symmetric
            )
        return scheme

    @staticmethod
    def _dynamic_w8a8(scheme):
        if scheme is None:
            return False
        w, a = scheme.get("weights"), scheme.get("input_activations")
        return (
            scheme.get("format") == "int-quantized"
            and w is not None
            and a is not None
            and w.num_bits == a.num_bits == 8
            and w.type == a.type == "int"
            and w.strategy == "channel"
            and a.strategy == "token"
            and w.symmetric
            and a.symmetric
            and not w.dynamic
            and a.dynamic
            and w.group_size is None
        )

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        if isinstance(layer, (RoutedExperts, MoERunner)):
            names = [
                f"{prefix}.{i}.{proj}"
                for i in range(layer.moe_config.num_experts)
                for proj in ("gate_proj", "up_proj", "down_proj")
            ]
            scheme = self._validated_scheme(layer, prefix, names)
            if scheme is None:
                return None
            return AscendW8A8DynamicMoEMethod(
                scheme["weights"],
                scheme["input_activations"],
                layer.moe_config,
                layer_name=prefix,
            )
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            # Validate before upstream's LMHead compatibility fallback, which
            # intentionally catches ValueError for unsupported embedding schemes.
            self.get_scheme_dict(layer, prefix)
            layer._fl_ascend_weight_nz_mode = self.weight_nz_mode
        return super().get_quant_method(layer, prefix)


@cache
def register_ascend_compressed_tensors():
    """Override compressed-tensors only in Ascend integration processes."""
    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config("compressed-tensors")(AscendCompressedTensorsConfig)


class AscendW8A8Int8(CompressedTensorsW8A8Int8):
    """Keep the NPU activation contract around upstream weight allocation."""

    def create_weights(
        self,
        layer,
        output_partition_sizes,
        input_size_per_partition,
        params_dtype,
        weight_loader,
        **kwargs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("Dynamic W8A8 Linear requires BF16/FP16")
        return super().create_weights(
            layer,
            output_partition_sizes,
            input_size_per_partition,
            params_dtype,
            weight_loader,
            **kwargs,
        )


_prepare_weight = CachedOp("w8a8_prepare_weight")


_linear = CachedOp("w8a8_dynamic_linear")


class AscendGroupedW8A8LinearMethod(CompressedTensorsLinearMethod):
    """Group projection adapter; ordinary Linear allocation stays upstream."""

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype != torch.int8 or layer.weight.ndim != 2:
            raise ValueError("Expected W8A8 [N,K] int8 weight")
        if layer.weight_scale.shape != (layer.weight.shape[0], 1):
            raise ValueError("Expected one W8A8 scale per output channel")
        if layer.weight.shape[0] >= 65536:
            raise NotImplementedError(
                "Large-output DSA-CP Linear splitting is not enabled"
            )
        if getattr(self, "output_groups", None) is None:
            raise ValueError(
                "Grouped W8A8 requires configure_grouped before preparation"
            )
        if self.quantization_config.weight_nz_mode not in (0, 1, 2):
            raise ValueError("weight_nz_mode must be 0, 1 or 2")
        groups = self.output_groups
        if groups < 1 or layer.weight.shape[0] % groups:
            raise ValueError("Grouped W8A8 outputs must divide the weight rows")
        self.group_input_size = layer.weight.shape[1]
        for group, (weight, scale) in enumerate(
            zip(
                layer.weight.data.chunk(groups, dim=0),
                layer.weight_scale.data.chunk(groups, dim=0),
            )
        ):
            weight, scale = _prepare_weight(
                weight.transpose(0, 1).contiguous(),
                scale.flatten(),
                self.quantization_config.weight_nz_mode,
            )
            layer.register_buffer(
                f"_grouped_w8a8_weight_{group}", weight, persistent=False
            )
            layer.register_buffer(
                f"_grouped_w8a8_scale_{group}", scale, persistent=False
            )
        # Keep parameter ownership while releasing the unused logical bank.
        # Other groups remain module-owned buffers, moved with the layer.
        layer.weight.data = layer._grouped_w8a8_weight_0

    def configure_grouped(self, groups):
        """Set the independent output groups before post-load preparation."""
        if groups < 1:
            raise ValueError("Grouped W8A8 requires positive groups")
        self.output_groups = groups

    def apply_grouped(self, layer, x):
        groups = self.output_groups
        tokens = x.shape[0]
        grouped = x.reshape(tokens, groups, self.group_input_size)
        outputs = [
            _linear(
                grouped[:, group].contiguous(),
                getattr(layer, f"_grouped_w8a8_weight_{group}"),
                getattr(layer, f"_grouped_w8a8_scale_{group}"),
            )
            for group in range(groups)
        ]
        return torch.cat(outputs, dim=-1)
