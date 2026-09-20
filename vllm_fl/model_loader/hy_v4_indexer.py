# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single pairing entry point for HY4 indexer ``wk`` checkpoints.

Both plain FP8 and MXFP8 checkpoints store the indexer ``wk`` projection as
``torch.float8_e4m3fn``, so the weight dtype cannot choose the dequantization
scheme; the scale tensor can.  This loader pairs each layer's ``wk.weight``
with its scale, identifies the format from the scale name/dtype, validates the
pair against the checkpoint quantization metadata, and writes the upcast BF16
weight into the fused ``wk_weights_proj`` parameter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)

FP8_FORMAT = "fp8"
MXFP8_FORMAT = "mxfp8"

_MXFP8_GROUP_SIZE = 32

_WEIGHT_FIELD = "weight"
_FP8_SCALE_FIELD = "fp8_scale"
_MXFP8_SCALE_FIELD = "mxfp8_scale"
_SCALE_FORMAT = {
    _FP8_SCALE_FIELD: FP8_FORMAT,
    _MXFP8_SCALE_FIELD: MXFP8_FORMAT,
}

_NON_FP8_WEIGHT_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
)


def normalize_quant_format(quant_metadata: object) -> str | None:
    """Map a checkpoint quantization label to ``fp8``/``mxfp8`` (or ``None``)."""
    if quant_metadata is None:
        return None
    lowered = str(quant_metadata).lower()
    if "mxfp8" in lowered:
        return MXFP8_FORMAT
    if "fp8" in lowered:
        return FP8_FORMAT
    return None


def quant_metadata_from_quant_config(quant_config: object) -> str | None:
    """Derive the checkpoint format label from a vLLM quantization config."""
    if quant_config is None:
        return None
    get_name = getattr(quant_config, "get_name", None)
    if not callable(get_name):
        return None
    return normalize_quant_format(get_name())


def _validate_matrix_shapes(weight: torch.Tensor, scale: torch.Tensor) -> None:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "HY4 indexer WK expects 2D weight and scale tensors, got "
            f"{tuple(weight.shape)} vs {tuple(scale.shape)}"
        )
    if not weight.numel() or not scale.numel():
        raise ValueError(
            "HY4 indexer WK requires non-empty weight and scale tensors, got "
            f"{tuple(weight.shape)} vs {tuple(scale.shape)}"
        )


def dequantize_mxfp8_wk(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Upcast an MXFP8 indexer ``wk`` weight using this checkpoint's e8m0 scales."""
    _validate_matrix_shapes(weight, scale)
    if weight.shape[:-1] != scale.shape[:-1]:
        raise ValueError(
            "HY4 indexer MXFP8 shape mismatch: "
            f"{tuple(weight.shape)} vs {tuple(scale.shape)}"
        )
    if weight.shape[-1] != scale.shape[-1] * _MXFP8_GROUP_SIZE:
        raise ValueError(
            "HY4 indexer MXFP8 requires last-dim group size "
            f"{_MXFP8_GROUP_SIZE}: "
            f"{tuple(weight.shape)} vs {tuple(scale.shape)}"
        )
    scales = torch.exp2(scale.to(torch.int16).float() - 127.0)
    scales = scales.repeat_interleave(_MXFP8_GROUP_SIZE, dim=-1)
    return (weight.float() * scales).to(torch.bfloat16)


def dequantize_fp8_wk(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Upcast a plain FP8 indexer ``wk`` weight with vLLM block-scale semantics."""
    _validate_matrix_shapes(weight, scale)
    if scale.shape[-1] == 0 or weight.shape[-1] % scale.shape[-1] != 0:
        raise ValueError(
            "HY4 indexer FP8 scale columns do not divide the weight: "
            f"{tuple(weight.shape)} vs {tuple(scale.shape)}"
        )
    block_size = weight.shape[-1] // scale.shape[-1]
    if scale.shape[-2] == 0 or weight.shape[-2] != scale.shape[-2] * block_size:
        raise ValueError(
            "HY4 indexer FP8 scale rows do not match the block grouping: "
            f"{tuple(weight.shape)} vs {tuple(scale.shape)}"
        )
    return scaled_dequantize(
        weight,
        scale,
        group_shape=GroupShape(block_size, block_size),
        out_dtype=torch.bfloat16,
    )


class IndexerWKLoader:
    """Pair, validate and dequantize the HY4 indexer ``wk`` scale tensors."""

    def __init__(
        self,
        params_dict: Mapping[str, nn.Parameter],
        pp_missing_layer_names: Iterable[str] = (),
        quant_metadata: str | Mapping[str, str] | None = None,
    ) -> None:
        self._params_dict = params_dict
        self._pp_missing = tuple(self._normalize_missing(pp_missing_layer_names))
        self._quant_metadata = quant_metadata
        self._pending: dict[str, dict[str, object]] = {}
        self._loaded: dict[str, str] = {}
        self._loaded_params: set[str] = set()

    @staticmethod
    def _normalize_missing(names: Iterable[str]) -> list[str]:
        normalized = []
        for name in names:
            text = str(name)
            normalized.append(text if text.endswith(".") else f"{text}.")
        return normalized

    @staticmethod
    def _new_entry() -> dict[str, object]:
        return {
            _WEIGHT_FIELD: None,
            _FP8_SCALE_FIELD: None,
            _MXFP8_SCALE_FIELD: None,
            "duplicates": [],
        }

    def _is_pp_missing(self, name: str) -> bool:
        return any(
            name == missing[:-1] or name.startswith(missing)
            for missing in self._pp_missing
        )

    @staticmethod
    def _classify(name: str, tensor: torch.Tensor) -> str | None:
        if ".indexer.wk." not in name or "wk_weights" in name:
            return None
        if name.endswith(".weight"):
            if tensor.dtype == torch.float8_e4m3fn:
                return _WEIGHT_FIELD
            if tensor.dtype in _NON_FP8_WEIGHT_DTYPES:
                return None
            raise ValueError(
                f"HY4 indexer WK weight has unsupported dtype {tensor.dtype}: {name}"
            )
        if name.endswith(".weight_scale_inv"):
            if not tensor.is_floating_point():
                raise ValueError(
                    "HY4 indexer FP8 scale must be a floating dtype, got "
                    f"{tensor.dtype}: {name}"
                )
            return _FP8_SCALE_FIELD
        if name.endswith(".weight_scale"):
            if tensor.dtype != torch.uint8:
                raise ValueError(
                    "HY4 indexer MXFP8 scale must be torch.uint8, got "
                    f"{tensor.dtype}: {name}"
                )
            return _MXFP8_SCALE_FIELD
        return None

    def consume(self, name: str, tensor: torch.Tensor) -> bool:
        """Take ownership of ``tensor`` when it is a WK weight or scale."""
        if ".indexer.wk." not in name or name.rsplit(".wk.", 1)[-1] not in (
            "weight",
            "weight_scale",
            "weight_scale_inv",
        ):
            return False
        # A PP rank must not validate or retain tensors belonging to another
        # rank, including checkpoint formats it does not support locally.
        if self._is_pp_missing(name):
            return True
        field = self._classify(name, tensor)
        layer_prefix = name.rsplit(".wk.", 1)[0]
        if layer_prefix in self._loaded:
            raise ValueError(
                f"HY4 indexer WK received unexpected input for already loaded "
                f"layer {layer_prefix}: {name}"
            )
        if field is None:
            if layer_prefix in self._pending:
                raise ValueError(
                    "HY4 indexer WK received a plain weight mixed with "
                    f"quantized inputs for layer {layer_prefix}: {name}"
                )
            # The model's normal stacked loader owns this tensor, but remember
            # it here so a later FP8 pair cannot silently overwrite the shard.
            self._loaded[layer_prefix] = "unquantized"
            return False

        entry = self._pending.setdefault(layer_prefix, self._new_entry())
        if entry[field] is not None:
            duplicates = entry["duplicates"]
            assert isinstance(duplicates, list)
            duplicates.append(name)
            return True
        entry[field] = tensor

        scale_fields = [
            field
            for field in (_FP8_SCALE_FIELD, _MXFP8_SCALE_FIELD)
            if entry[field] is not None
        ]
        if entry[_WEIGHT_FIELD] is None or not scale_fields:
            return True
        duplicates = entry["duplicates"]
        assert isinstance(duplicates, list)
        if duplicates or len(scale_fields) != 1:
            return True
        self._finalize(layer_prefix, entry, scale_fields[0])
        return True

    def _expected_format(self, layer_prefix: str) -> str | None:
        metadata = self._quant_metadata
        if isinstance(metadata, Mapping):
            if layer_prefix not in metadata:
                return None
            return normalize_quant_format(metadata[layer_prefix])
        return normalize_quant_format(metadata)

    def _finalize(
        self,
        layer_prefix: str,
        entry: dict[str, object],
        scale_field: str,
    ) -> None:
        fused_name = f"{layer_prefix}.wk_weights_proj.weight"
        if fused_name not in self._params_dict:
            raise ValueError(
                "HY4 indexer WK target parameter is missing for local layer "
                f"{layer_prefix}: {fused_name}"
            )
        fmt = _SCALE_FORMAT[scale_field]
        expected = self._expected_format(layer_prefix)
        if expected is not None and expected != fmt:
            raise ValueError(
                f"HY4 indexer WK format conflict for {layer_prefix}: "
                f"checkpoint metadata says {expected}, paired scale is {fmt}"
            )

        weight = entry[_WEIGHT_FIELD]
        scale = entry[scale_field]
        assert isinstance(weight, torch.Tensor)
        assert isinstance(scale, torch.Tensor)
        try:
            if fmt == MXFP8_FORMAT:
                weight_bf16 = dequantize_mxfp8_wk(weight, scale)
            else:
                weight_bf16 = dequantize_fp8_wk(weight, scale)
        except ValueError as exc:
            raise ValueError(f"HY4 indexer WK layer {layer_prefix}: {exc}") from exc

        param = self._params_dict[fused_name]
        param.weight_loader(param, weight_bf16, 0)
        del self._pending[layer_prefix]
        self._loaded[layer_prefix] = fmt
        self._loaded_params.add(fused_name)

    def finish(self) -> set[str]:
        """Validate every pending layer and return the loaded fused names."""
        problems = []
        for layer_prefix in sorted(self._pending):
            entry = self._pending[layer_prefix]
            missing = []
            if entry[_WEIGHT_FIELD] is None:
                missing.append("weight")
            scale_fields = [
                field
                for field in (_FP8_SCALE_FIELD, _MXFP8_SCALE_FIELD)
                if entry[field] is not None
            ]
            if not scale_fields:
                missing.append("scale")
            duplicates = entry["duplicates"]
            assert isinstance(duplicates, list)
            if duplicates:
                problems.append(f"{layer_prefix}: duplicate input {sorted(duplicates)}")
            if len(scale_fields) > 1:
                problems.append(
                    f"{layer_prefix}: conflicting scales "
                    f"{sorted(_SCALE_FORMAT[field] for field in scale_fields)}"
                )
            if missing:
                problems.append(f"{layer_prefix}: missing {'/'.join(missing)}")
        if problems:
            raise ValueError("HY4 indexer WK load failed: " + "; ".join(problems))
        self._pending.clear()
        return set(self._loaded_params)
