# SPDX-License-Identifier: Apache-2.0
"""gpt-oss packed-expert checkpoints on vLLM 0.24's routed-experts layout.

vLLM 0.24 moved MoE expert parameters under ``...experts.routed_experts.*``,
but ``GptOssModel._load_weights_other`` in some builds still looks up
``...experts.w2_weight`` and raises ``KeyError: 'layers.0.mlp.experts.w2_weight'``
for standard HuggingFace gpt-oss checkpoints (``experts.gate_up_proj`` /
``experts.down_proj``). The plugin adapts it here instead of patching vLLM:
checkpoint names are rewritten to the parameters that actually exist, and names
that already match (older layouts) are left untouched.
"""

import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)


def remap_expert_param_name(name, param_names):
    """Return the routed-experts parameter name when it exists, else ``name``."""
    if ".experts." not in name:
        return name
    prefix, _, rest = name.partition(".experts.")
    candidate = prefix + ".experts.routed_experts." + rest
    return candidate if candidate in param_names else name


def _remap_iterator(weights, param_names):
    for name, tensor in weights:
        yield remap_expert_param_name(name, param_names), tensor


def _patch_model_class(module):
    model_cls = getattr(module, "GptOssModel", None)
    if model_cls is None or getattr(model_cls, "_fl_moe_expert_remap", False):
        return False

    try:
        from vllm.model_executor.model_loader.weight_utils import (
            remap_moe_expert_weights as _vllm_remap,
        )
    except Exception:  # pragma: no cover - older builds
        _vllm_remap = None

    # `load_weights` maps checkpoint names through `hf_to_vllm_mapper` *before*
    # `_load_weights_other` looks them up in `params_dict`, so the remap has to
    # happen on the mapped names that method receives.
    original = model_cls._load_weights_other

    def _load_weights_other(self, ep_rank_end, ep_rank_start, heads_per_rank,
                            head_start, weights, stacked_params_mapping):
        param_names = dict(self.named_parameters())
        remapped = (
            _vllm_remap(weights, param_names)
            if _vllm_remap is not None
            else _remap_iterator(weights, param_names)
        )
        return original(self, ep_rank_end, ep_rank_start, heads_per_rank,
                        head_start, remapped, stacked_params_mapping)

    model_cls._load_weights_other = _load_weights_other
    model_cls._fl_moe_expert_remap = True
    logger.info("gpt-oss MoE expert weight names remapped for vLLM routed experts")
    return True


class _PostImportPatch:
    """Apply the patch right after vLLM's gpt_oss module is imported.

    vLLM loads platform plugins before its model package is importable, so
    importing `vllm.model_executor.models.gpt_oss` from `register()` fails and
    the patch would be skipped. This finder defers it to the real import.
    """

    def __init__(self, target, callback):
        self.target = target
        self.callback = callback

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            try:
                self.callback(module)
            except Exception as exc:  # pragma: no cover - never break imports
                logger.warning("gpt-oss weight patch failed: %s", exc)

        spec.loader.exec_module = exec_module
        return spec


def apply_model_patches():
    """Arrange for the gpt-oss loader patch to run when the model module loads."""
    try:
        import importlib.util
        import sys as _sys
        from vllm.model_executor.models import gpt_oss

        return _patch_model_class(gpt_oss)
    except Exception:
        # Plugin registration happens before vLLM's model package is importable.
        _sys.meta_path.insert(0, _PostImportPatch(
            "vllm.model_executor.models.gpt_oss", _patch_model_class))
        return True
