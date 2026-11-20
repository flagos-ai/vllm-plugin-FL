# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Read ModelSlim checkpoint metadata without importing an Ascend runtime.

Initial scope: local DeepSeek V4.1 exports, symmetric per-channel INT4/INT8,
dynamic per-token A8 and explicitly FLOAT modules. Backend selection remains
in the existing MetaX/FlagOS methods; the on-disk method name stays 'ascend'.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class ModelSlimConfig(QuantizationConfig):
    def __init__(self):
        super().__init__()
        self.packed_modules_mapping = {}
        self.description = {}
        self.expert_schemes = {}
        self.rotation_block = None
        self.embedding_storage = {}
        self.root = None
        self.weight_map = {}
        self.duplicate_keys = set()
        self.num_layers = 0
        self.draft_layers = 0
        self._metadata_hash = "uninitialized-modelslim"

    def compute_hash(self):
        return self._metadata_hash

    @classmethod
    def get_name(cls):
        return "ascend"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        # NVIDIA SM numbers do not describe MetaX capabilities. Actual methods
        # use the platform/vendor dispatch and native shape preflight.
        return 0

    @classmethod
    def get_config_filenames(cls):
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config):
        if config.get("quant_method") != "ascend":
            raise ValueError("expected the original ModelSlim ascend metadata")
        if set(config) - {"quant_method", "model_quant_type"}:
            raise ValueError("unsupported embedded ModelSlim quantization options")
        return cls()

    def get_scaled_act_names(self):
        return []

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        from safetensors import safe_open

        root = Path(model_name)
        if not root.is_dir():
            raise ValueError("ModelSlim support requires a downloaded local checkpoint")
        config = json.loads((root / "config.json").read_text())
        if config.get("model_type") != "deepseek_v41":
            raise NotImplementedError(
                "ModelSlim name mapping currently covers DeepSeek V4.1"
            )
        description = json.loads((root / "quant_model_description.json").read_text())
        index_bytes = (root / "quant_model_weights.safetensors.index.json").read_bytes()
        weight_map = json.loads(index_bytes)["weight_map"]
        if any(
            Path(name).is_absolute() or len(Path(name).parts) != 1
            for name in weight_map.values()
        ):
            raise NotImplementedError(
                "ModelSlim support requires root-level safetensors shards"
            )
        self._metadata_hash = hashlib.sha256(
            (root / "config.json").read_bytes()
            + (root / "quant_model_description.json").read_bytes()
            + index_bytes
        ).hexdigest()
        counts = Counter()
        authoritative = set()
        for path in sorted(root.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118 - safetensors exposes keys(), not iteration.
                    if key not in weight_map:
                        raise ValueError(f"tensor absent from ModelSlim index: {key}")
                    counts[key] += 1
                    if weight_map[key] == path.name:
                        authoritative.add(key)
        if authoritative != set(weight_map):
            raise ValueError("ModelSlim index references missing tensors or shards")
        if description.get("version") != "1.0.0" or description.get("group_size") != 0:
            raise NotImplementedError(
                "expected ModelSlim v1 symmetric per-channel storage"
            )
        allowed = {"FLOAT", "W4A8_DYNAMIC", "W8A8_DYNAMIC", "INT8_SYM"}
        rules = {
            k: v
            for k, v in description.items()
            if k not in {"version", "group_size", "metadata", "optional"}
        }
        if any(v not in allowed for v in rules.values()):
            raise NotImplementedError("unsupported per-parameter ModelSlim rule")
        expert_schemes = {}
        for name, kind in rules.items():
            if ".experts." in name and name.endswith(".weight"):
                group = name.split(".experts.", 1)[0] + ".experts"
                expert_schemes.setdefault(group, set()).add(kind)
        if any(len(kinds) != 1 for kinds in expert_schemes.values()):
            raise ValueError("mixed expert schemes within one routed layer")
        optional = description.get("optional", {})
        storage = optional.get("embedding_storage", {})
        if storage.get("version") != 2:
            raise NotImplementedError(
                "expected explicit ModelSlim embedding storage v2"
            )
        for spec in storage.get("modules", {}).values():
            if any(
                spec.get(k) != v
                for k, v in {
                    "format": "int8_sym",
                    "weight_dtype": "I8",
                    "layout": "row_major",
                    "group_size": 32,
                    "axis": 1,
                    "scale_dtype": "F8_E8M0",
                }.items()
            ):
                raise NotImplementedError("unsupported Engram checkpoint storage")
        rotation = config.get("engram_rotation_config", {})
        if rotation != {
            "value_projection_rotated": True,
            "value_basis": "quarot_global",
            "key_and_gate_basis": "original",
            "runtime_delta_rotation": False,
        }:
            raise NotImplementedError("unsupported Engram rotation contract")
        path = optional.get("quarot", {}).get("rotation_map", {}).get("global_rotation")
        if path != "optional/quarot.safetensors":
            raise ValueError("missing standard QuaRot attachment")
        text = config.get("text_config", config)
        hidden = text["hidden_size"]
        with safe_open(root / path, framework="pt", device="cpu") as f:
            matrix = f.get_tensor("global_rotation")
        if (
            matrix.dtype != torch.float32
            or matrix.shape != (hidden, hidden)
            or hidden % 32
        ):
            raise ValueError("invalid QuaRot matrix shape/dtype")
        block = matrix[:32, :32].contiguous()
        if not torch.isfinite(matrix).all() or not torch.equal(
            matrix, torch.block_diag(*([block] * (hidden // 32)))
        ):
            raise NotImplementedError("only repeated block32 QuaRot is supported")
        self.root = root
        self.weight_map = weight_map
        self.duplicate_keys = {key for key, count in counts.items() if count > 1}
        self.description = rules
        self.expert_schemes = {k: next(iter(v)) for k, v in expert_schemes.items()}
        self.embedding_storage = storage["modules"]
        self.rotation_block = block
        self.num_layers = text["num_hidden_layers"]
        self.draft_layers = text["num_nextn_predict_layers"]

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        # Keep the original namespace: the text mapper deliberately drops MTP
        # keys, while the draft has its own layer numbering and endpoint names.
        # canonical_name applies that same model mapping in the reverse direction.
        pass

    def canonical_name(self, name, *, draft=False):
        name = name.removeprefix("language_model.").removeprefix("model.")
        match = re.match(r"layers\.(\d+)\.(.*)", name)
        if match:
            index, suffix = int(match[1]), match[2]
            if index >= self.num_layers:
                stage = index - self.num_layers
                if stage >= self.draft_layers:
                    raise ValueError("draft layer prefix exceeds checkpoint stages")
                name = f"mtp.{stage}.{suffix}"
            elif draft:
                if index >= self.draft_layers:
                    raise ValueError("invalid local draft layer index")
                name = f"mtp.{index}.{suffix}"
        name = name.replace(".engram.embed_tokens.", ".engram.embed.")
        name = name.replace(".shared_experts.down_proj.", ".shared_experts.w2.")
        if name.endswith(".shared_experts.down_proj"):
            name = name.removesuffix("down_proj") + "w2"
        if name.startswith("main_proj.") or name == "main_proj":
            name = "mtp.0." + name
        if name.startswith("markov_head."):
            name = f"mtp.{self.draft_layers - 1}." + name.replace(
                "markov_w1", "embed"
            ).replace("markov_w2", "head")
        if name.startswith("confidence_head."):
            name = f"mtp.{self.draft_layers - 1}." + name
        if name.startswith("embed_tokens"):
            name = name.replace("embed_tokens", "embed", 1)
        if name.startswith("lm_head"):
            name = name.replace("lm_head", "head", 1)
        return name

    def authoritative_weights(self, weights):
        """Honor the original index while retaining vLLM's default iterator.

        Some exports retain pre-rotation and final Engram tensors in separate
        shards. The default iterator exposes no source filename, so only those
        duplicated keys are reread from their indexed owner, never first/last-wins.
        """
        from safetensors import safe_open

        seen = set()
        for name, tensor in weights:
            if name not in self.weight_map:
                raise ValueError(f"unindexed ModelSlim tensor: {name}")
            if name in self.duplicate_keys:
                if name in seen:
                    continue
                with safe_open(
                    self.root / self.weight_map[name], framework="pt", device="cpu"
                ) as f:
                    tensor = f.get_tensor(name)
                seen.add(name)
            yield name, tensor

    def layer_kind(self, prefix):
        if not self.description:
            raise RuntimeError(
                "ModelSlim metadata must be loaded before layer creation"
            )
        name = self.canonical_name(prefix)
        if name in self.expert_schemes:
            return self.expert_schemes[name]
        fused = {
            "gate_up_proj": ("w1", "w3"),
            "fused_wqa_wkv": ("wq_a", "wkv"),
            "fused_wkv_wgate": ("wkv", "wgate"),
        }
        stem, _, suffix = name.rpartition(".")
        members = (
            [f"{stem}.{part}.weight" for part in fused[suffix]]
            if suffix in fused
            else [name + ".weight"]
        )
        missing = [key for key in members if key not in self.description]
        if missing:
            raise ValueError(f"missing ModelSlim layer rules: {missing}")
        kinds = {self.description[key] for key in members}
        if len(kinds) != 1:
            raise NotImplementedError(f"conflicting fused layer schemes: {members}")
        return next(iter(kinds))

    def get_quant_method(self, layer, prefix):
        from compressed_tensors.quantization import (
            QuantizationArgs,
            QuantizationStrategy,
        )

        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsLinearMethod,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
            CompressedTensorsW8A8Int8,
        )
        from vllm.platforms import current_platform

        if not isinstance(layer, (LinearBase, RoutedExperts)):
            return None
        if getattr(current_platform, "vendor_name", None) != "metax":
            raise NotImplementedError("this ModelSlim execution path requires MetaX")
        kind = self.layer_kind(prefix)
        if isinstance(layer, RoutedExperts):
            if kind == "FLOAT":
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if kind != "W4A8_DYNAMIC":
                raise NotImplementedError(
                    f"unsupported routed ModelSlim scheme: {kind}"
                )
            from .metax_w4a8 import MetaXCompressedTensorsW4A8Method

            return MetaXCompressedTensorsW4A8Method(
                QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="channel",
                    dynamic=False,
                ),
                QuantizationArgs(
                    num_bits=8,
                    type="int",
                    symmetric=True,
                    strategy="token",
                    dynamic=True,
                ),
                layer.moe_config,
            )
        if kind == "FLOAT":
            return UnquantizedLinearMethod()
        if kind != "W8A8_DYNAMIC":
            raise NotImplementedError(f"unsupported ModelSlim linear scheme: {kind}")
        layer.scheme = CompressedTensorsW8A8Int8(
            strategy=QuantizationStrategy.CHANNEL,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        return CompressedTensorsLinearMethod(self)

    def auxiliary(self, name, tensor, *, draft=False):
        """Validate format-only parameters before the model loader consumes them."""
        canonical = self.canonical_name(name, draft=draft)
        if not canonical.endswith((".weight_offset", ".scale_bias")):
            return False
        kind = self.description.get(canonical)
        if kind not in {"W4A8_DYNAMIC", "W8A8_DYNAMIC"}:
            raise ValueError(f"unexpected ModelSlim auxiliary parameter: {canonical}")
        if (
            tensor.dtype != torch.float32
            or tensor.ndim != 2
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid ModelSlim auxiliary tensor: {canonical}")
        if canonical.endswith(".weight_offset"):
            if tensor.shape[1] != 1 or torch.count_nonzero(tensor):
                raise NotImplementedError(
                    "MetaX ModelSlim path requires symmetric zero offsets"
                )
        elif kind != "W4A8_DYNAMIC" or tensor.shape[1] not in (1, 16):
            raise ValueError("invalid per-channel W4A8 scale-bias layout")
        return True

    def is_packed_expert(self, name, *, draft=False):
        return (
            ".experts." in name
            and name.endswith(".weight")
            and self.description.get(self.canonical_name(name, draft=draft))
            == "W4A8_DYNAMIC"
        )


def register_modelslim_format():
    from vllm.model_executor.layers.quantization import (
        get_quantization_config,
        register_quantization_config,
    )

    try:
        existing = get_quantization_config("ascend")
    except ValueError:
        register_quantization_config("ascend")(ModelSlimConfig)
        return
    if existing is not ModelSlimConfig:
        raise RuntimeError("another ascend format implementation is already registered")
