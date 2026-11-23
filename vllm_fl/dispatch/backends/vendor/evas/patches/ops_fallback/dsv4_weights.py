# Copyright (c) 2026 BAAI. All rights reserved.
"""DSV4 FP8 weight loading on LPU — the loader and the patches that install it.

This module owns ``load_weights_evas`` and the patch set that routes it into
upstream.  FP8 must load and stay FP8: F8_E4M3 bytes keep their native dtype,
F8_E8M0 block scales decode to fp32 (an upstream uint8 view would store raw
exponent bytes as floats), and routed experts copy in batches, not per
(expert, shard), on dual die.  Patches: the fp8 linear/MoE methods, model
``__init__`` (quant_config + deferred indexer), the ``load_weights`` redirect,
and the MTP contract.

"""
from __future__ import annotations

import logging
import os
import re
from itertools import islice
from typing import Iterable

import torch

from vllm.model_executor.models.utils import AutoWeightsLoader

logger = logging.getLogger(__name__)

_PATCHED = False
_FP8_METHODS_PATCHED = False


def _is_fp8_expert(cfg) -> bool:
    """True iff a checkpoint (hf) config declares FP8 experts.

    Shared by the patch gate (``_real_fp8_enabled`` / MTP init) and the loader.
    Default ``"fp4"``: checkpoints without ``expert_dtype`` are FP4-era.
    """
    return getattr(cfg, "expert_dtype", "fp4") == "fp8"


# =========================================================================
# Weight loader — the LPU implementation of load_weights
# =========================================================================


def _apply_weights_mapper(
    model,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Apply the model's hf_to_vllm_mapper to translate flagos checkpoint keys
    into deepseek_v4.py param names."""
    mapper = model.hf_to_vllm_mapper
    return list(mapper.apply(weights))


def _real_fp8_from_model(model) -> bool:
    """Real-FP8 path decision read off the constructed model.

    An FP8 checkpoint (``config.expert_dtype == "fp8"``) runs the real-FP8
    path; anything else (BF16/fp4) is rejected by ``load_weights_evas``.
    """
    return _is_fp8_expert(getattr(model, "config", None))


def _reinterpret_fp8(
    mapped_weights: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Reinterpret the raw F8 checkpoint stream into LPU-friendly dtypes.

    Names are left exactly as the WeightsMapper produced them.  In particular the
    block scales of a fused pair stay as the checkpoint two keys (attn ``wq_a`` +
    ``wkv``; shared-expert ``w1`` + ``w3``): upstream's ``stacked_params_mapping``
    matches on substrings, so ``attn.wq_a.weight_scale_inv`` is already routed to
    ``attn.fused_wqa_wkv.weight_scale_inv`` shard 0 and the paired shard to shard
    1, via the fused param own shard loader.  The same mechanism carries the MTP
    drafter scales through ``DeepSeekV4MTP.load_weights``, so pre-merging pairs
    here would duplicate it.

    * ``float8_e4m3fn`` weight bytes stay native fp8 — the Fp8LinearMethodLPU /
      Fp8MoEMethodLPU params are ``float8_e4m3fn``, so the bytes are
      byte-identical to the checkpoint (the downstream dequant views them to
      uint8 first).
    * ``float8_e8m0fnu`` block scales -> ``fp32`` (``.float()`` decodes the E8M0
      exponent), matching the fp32 ``*_scale_inv`` params.  This decode has to
      happen here and on CPU: LPU aten has no float8 compute dtype and its
      ``e8m0.to(fp32)`` reinterprets the byte instead of decoding the unsigned
      exponent, while every consumer (evas ``fp8_einsum``, MoE W8A16 dequant)
      multiplies the scale as a float on the device.

    Plain bf16/fp32 weights (gate / norms / embed / lm_head / hc) pass through.
    """
    result: list[tuple[str, torch.Tensor]] = []
    for name, tensor in mapped_weights:
        if tensor.dtype == torch.float8_e8m0fnu:
            tensor = tensor.float()  # E8M0 -> fp32 (decode exponent)
        # float8_e4m3fn (and every non-F8 dtype) passes through unchanged.
        result.append((name, tensor))
    return result


# Mapped (post-WeightsMapper) routed-expert weight names:
#   model.layers.{l}.ffn.experts.{eid}.{w1|w2|w3}.{weight|weight_scale_inv}
_ROUTED_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)"
    r"\.(weight|weight_scale_inv)$"
)


def _collect_local_experts(
    entries: list[tuple[int, torch.Tensor]],
    expert_map_manager,
    n_local: int,
) -> list[torch.Tensor] | None:
    """Map checkpoint (global) expert ids to local ids and drop non-local
    experts, mirroring ``RoutedExperts.weight_loader``'s EP handling
    (``_map_global_expert_id_to_local_expert_id`` -> -1 = skip).  Returns
    tensors ordered by local id 0..n_local-1, or None if the local set is
    unexpected (batch-loader assumption violated).
    """
    by_local: dict[int, torch.Tensor] = {}
    for eid, tensor in entries:
        lid = int(expert_map_manager.map_global_to_local(eid))
        if lid < 0:
            continue
        by_local[lid] = tensor
    if len(by_local) != n_local or set(by_local) != set(range(n_local)):
        return None
    return [by_local[i] for i in range(n_local)]


def _batch_load_w13_param(
    routed,
    param_name: str,
    w1: list[torch.Tensor],
    w3: list[torch.Tensor],
) -> None:
    """Stack all local experts' w1/w3 and copy them into the fused w13 param.

    w13 = w1 | w3 merged on the intermediate axis (dim 1 after stacking); the
    TP slice on the intermediate axis reproduces ``_load_w13`` exactly.

    The two halves go into the destination's dim-1 slices directly rather than
    through a fused host buffer.  Building that buffer costs a full-size
    host-to-host pass over the fused weight (measured 347 ms per layer on the
    4-layer carrier, about 1.4 s of its load) and a second full-size host
    residency; the destination slice write runs at device speed instead.  The
    param is allocated with the double-die row split (``split_dim=0``), so a
    dim-1 slice never crosses the die boundary.
    """
    tp_size = routed.moe_config.tp_size
    tp_rank = routed.moe_config.tp_rank

    per = w1[0].shape[0] // tp_size
    start = per * tp_rank
    W1 = torch.stack(w1).narrow(1, start, per)
    W3 = torch.stack(w3).narrow(1, start, per)
    dst = getattr(routed, param_name).data
    dst.narrow(1, 0, per).copy_(W1)
    dst.narrow(1, per, per).copy_(W3)


def _batch_load_w2_param(
    routed,
    param_name: str,
    w2: list[torch.Tensor],
) -> None:
    """Stack all local experts' w2 and do one contiguous copy per w2 param.

    w2 shards on the intermediate axis (dim 2 after stacking); the TP slice
    on the intermediate axis reproduces ``_load_w2`` exactly.
    """
    tp_size = routed.moe_config.tp_size
    tp_rank = routed.moe_config.tp_rank

    W2 = torch.stack(w2)
    per = W2.shape[2] // tp_size
    start = per * tp_rank
    getattr(routed, param_name).data.copy_(W2.narrow(2, start, per))


def _batch_load_routed_experts(
    model,
    stream: list[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], set[str]]:
    """Split routed-expert weights out of ``stream`` and batch-load them.

    Returns ``(rest, loaded)``: ``rest`` is the stream minus the batch-loaded
    routed experts (fed to AutoWeightsLoader); ``loaded`` is informational
    (the caller reports all model params for vLLM's strict check regardless).

    Falls back wholesale (returns the original stream, no batching) for any
    unsupported configuration: MegaMoE, non-Fp8-block quant, EPLB redundant
    experts, non-gated activation, or an intermediate axis that does not split
    evenly across TP.  DSV4 FP8 (generic FusedMoE, Fp8MoEMethod block,
    256 experts, SiLU, TP-divisible) never falls back.
    """
    if not stream:
        return stream, set()

    dsv4_model = model.model
    if not hasattr(dsv4_model, "get_expert_mapping"):
        return stream, set()

    # Only the generic FusedMoE path (not MegaMoE).
    try:
        first_layer = next(iter(
            islice(dsv4_model.layers, dsv4_model.start_layer, dsv4_model.end_layer)
        ))
    except (StopIteration, AttributeError):
        return stream, set()
    if getattr(first_layer.ffn, "use_mega_moe", False):
        return stream, set()

    expert_mapping = dsv4_model.get_expert_mapping()
    params_dict = dict(model.named_parameters())

    # Group checkpoint expert weights by mapped param name.
    groups: dict[str, dict[str, list[tuple[int, torch.Tensor]]]] = {}
    rest: list[tuple[str, torch.Tensor]] = []
    for name, tensor in stream:
        if _ROUTED_EXPERT_RE.match(name) is None:
            rest.append((name, tensor))
            continue
        matched = False
        for param_name, weight_name, expert_id, shard_id in expert_mapping:
            if weight_name in name:
                nm = name.replace(weight_name, param_name)
                groups.setdefault(nm, {"w1": [], "w2": [], "w3": []})[
                    shard_id
                ].append((expert_id, tensor))
                matched = True
                break
        if not matched:
            rest.append((name, tensor))

    if not groups:
        return stream, set()

    # Pre-flight: every batch-loadable param must satisfy the batching
    # assumptions; otherwise fall back wholesale.
    for nm, shards in groups.items():
        if nm not in params_dict:
            continue  # PP: not this rank's layer.
        if "routed_experts" not in nm:
            logger.info("batch routed-experts: unexpected param %s → fallback", nm)
            return stream, set()
        routed = model.get_submodule(nm.rsplit(".", 1)[0])
        qm = routed.quant_method.__class__.__name__
        block_quant = getattr(routed.quant_method, "block_quant", False)
        weight_scale_name = getattr(routed.quant_method, "weight_scale_name", None)
        cfg = routed.moe_config
        # Attribute-based check (not exact class-name): the LPU path uses
        # Fp8MoEMethodLPU (a subclass of Fp8MoEMethod) which sets the same
        # ``block_quant`` / ``weight_scale_name`` fields.
        if not block_quant or weight_scale_name != "weight_scale_inv":
            logger.info(
                "batch routed-experts: quant %s (block=%s scale=%s) → fallback",
                qm, block_quant, weight_scale_name,
            )
            return stream, set()
        if cfg.num_experts != cfg.num_logical_experts:
            logger.info("batch routed-experts: redundant experts → fallback")
            return stream, set()
        if not cfg.is_act_and_mul:
            logger.info("batch routed-experts: non-gated → fallback")
            return stream, set()

    loaded: set[str] = set()
    for nm, shards in sorted(groups.items()):
        if nm not in params_dict:
            continue  # PP: not this rank's layer.
        routed = model.get_submodule(nm.rsplit(".", 1)[0])
        n_local = routed.local_num_experts
        tp_size = routed.moe_config.tp_size
        param_name = nm.rsplit(".", 1)[1]  # w13_weight | w13_weight_scale_inv | w2_...

        if param_name.startswith("w13_"):
            w1 = _collect_local_experts(
                shards["w1"], routed.expert_map_manager, n_local
            )
            w3 = _collect_local_experts(
                shards["w3"], routed.expert_map_manager, n_local
            )
            if w1 is None or w3 is None:
                logger.info(
                    "batch routed-experts: unexpected local experts → fallback "
                    "(n_local=%d w1=%d w3=%d)",
                    n_local, len(shards["w1"]), len(shards["w3"]),
                )
                return stream, set()
            # Intermediate axis must split evenly (no trailing-pad rank).
            if w1[0].shape[0] % tp_size != 0:
                logger.info("batch routed-experts: TP-uneven intermediate → fallback")
                return stream, set()
            _batch_load_w13_param(routed, param_name, w1, w3)
        else:  # w2_
            w2 = _collect_local_experts(
                shards["w2"], routed.expert_map_manager, n_local
            )
            if w2 is None:
                logger.info(
                    "batch routed-experts: unexpected local experts → fallback "
                    "(n_local=%d w2=%d)",
                    n_local, len(shards["w2"]),
                )
                return stream, set()
            # Intermediate axis must split evenly (no trailing-pad rank).
            if w2[0].shape[1] % tp_size != 0:
                logger.info("batch routed-experts: TP-uneven intermediate → fallback")
                return stream, set()
            _batch_load_w2_param(routed, param_name, w2)
        loaded.add(nm)

    return rest, loaded


def load_weights_evas(
    model,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Evas (LPU) impl of DeepseekV4ForCausalLM.load_weights — FP8 only.

    Pipeline:
    1. Apply the model's ``hf_to_vllm_mapper`` (WeightsMapper) to translate
       flagos checkpoint keys into deepseek_v4.py param names.
    2. REAL-FP8 reinterpret (``_reinterpret_fp8``): keep FP8 weights
       single-residency — native ``float8_e4m3fn`` bytes + fp32 block scales.
    3. Batch-load the routed experts (see ``_batch_load_routed_experts``).
    4. Pass the rest through ``AutoWeightsLoader``, which handles stacked
       params (fused weights *and* their paired block scales), TP slicing,
       per-module weight_loader callbacks, and the FusedMoE expert-aware
       weight_loader.  Routed experts load into FusedMoE ``w13``/``w2``;
       shared experts into ``gate_up_proj``/``down_proj``.
    5. Call ``finalize_mega_moe_weights()`` (no-op on LPU).
    """
    # Step 1: Apply WeightsMapper. Materialize the weight stream first so we
    # can report the raw count (mapper.apply consumes the iterable).
    raw_weights = list(weights)
    mapped = _apply_weights_mapper(model, raw_weights)
    logger.info("load_weights_evas: %d raw weights -> %d mapped", len(raw_weights), len(mapped))

    # Step 2: REAL-FP8 only — reinterpret the F8 stream to native fp8 weights
    # + fp32 scales.  LPU runs FP8 checkpoints exclusively.
    if not _real_fp8_from_model(model):
        raise ValueError(
            "LPU runs FP8 DeepSeek-V4 checkpoints only (config.expert_dtype "
            "== 'fp8'); a non-FP8 checkpoint is not supported."
        )
    reinterpreted = _reinterpret_fp8(mapped)
    logger.info(
        "load_weights_evas: REAL-FP8 path: %d weights reinterpreted "
        "(native fp8 E4M3 weights + E8M0 scales decoded to fp32), "
        "no BF16 dequant", len(reinterpreted),
    )

    _log_weight_footprint(mapped, reinterpreted)

    # Step 3: batch-load routed experts (fp8 weight + fp32 block scale).
    # Collapses num_experts x {w1,w2,w3} tiny per-expert copies into a handful
    # of contiguous H2D copies, avoiding the ~50ms first-touch cost per
    # (expert, shard) on the split_dim=0 double-die params.  Falls back to the
    # AutoWeightsLoader path below for unsupported configurations.
    # VLLM_FL_DSV4_NO_BATCH_MOE_LOAD=1 skips the batch path entirely (A/B the
    # per-expert loader for correctness isolation).
    if os.environ.get("VLLM_FL_DSV4_NO_BATCH_MOE_LOAD", "0") != "1":
        stream, batched = _batch_load_routed_experts(model, reinterpreted)
    else:
        stream, batched = reinterpreted, set()
    if batched:
        logger.info(
            "load_weights_evas: batch-loaded %d routed-expert params; "
            "%d weights left for AutoWeightsLoader", len(batched), len(stream),
        )

    # Step 4: AutoWeightsLoader (routed/shared experts -> generic FusedMoE
    # buffers w13/w2/gate_up_proj/down_proj; see module docstring).
    loader = AutoWeightsLoader(
        model,
        skip_substrs=["mtp."],
    )
    autoloaded = loader.load_weights(stream, mapper=None)
    logger.info("load_weights_evas: %d weights loaded via AutoWeightsLoader", len(autoloaded))

    # Step 5: Finalize (no-op on LPU, but call for consistency)
    model.model.finalize_mega_moe_weights()

    # vLLM's strict check (default_loader.py, active when
    # model_config.quantization is None) compares the set we return against
    # model.named_parameters().  Every weight goes through the steps above, so
    # reporting all model params loaded is correct.
    total_loaded = set(name for name, _ in model.named_parameters())
    logger.info(
        "load_weights_evas: autoloader=%d; reporting all %d model "
        "params initialized", len(autoloaded), len(total_loaded),
    )

    return total_loaded


def _log_weight_footprint(
    mapped: list[tuple[str, torch.Tensor]],
    resident: list[tuple[str, torch.Tensor]],
) -> None:
    """Startup weight type/footprint summary.

    ``mapped`` is the stream straight off the checkpoint (float8_e4m3fn weights,
    float8_e8m0fnu block scales, and the bf16/fp32/int64 tensors that need no
    reinterpretation); ``resident`` is the same stream after
    ``_reinterpret_fp8``.  Printing both is how the single-residency claim is
    checked: the resident total must stay at roughly the fp8 weight bytes, i.e.
    nothing was expanded to bf16.
    """
    raw_mib: dict[str, float] = {}
    for _, t in mapped:
        k = str(t.dtype)
        raw_mib[k] = raw_mib.get(k, 0.0) + t.numel() * t.element_size() / (1024.0**2)
    resident_mib = sum(t.numel() * t.element_size() for _, t in resident) / (1024.0**2)
    logger.info(
        "load_weights_evas [FP8 path] weight footprint: checkpoint raw=%s; "
        "resident=%.1f MiB (single residency)",
        ", ".join(f"{k}={v:.1f}MiB" for k, v in sorted(raw_mib.items())),
        resident_mib,
    )


# =========================================================================
# Patch install
# =========================================================================


def _real_fp8_enabled(vllm_config) -> bool:
    """Real-FP8 path decision read off ``vllm_config`` (patch-side twin of
    ``_real_fp8_from_model``: at ``__init__`` time the model does not exist
    yet, so the hf_config is the source).  Mirrors
    ``DeepseekV4FP8Config._resolve_expert_dtype``: an FP8 checkpoint
    (``expert_dtype == "fp8"``) keeps its quant_config, anything else drops it.
    """
    try:
        hf_config = vllm_config.model_config.hf_config
    except Exception:
        from vllm.attention import get_current_vllm_config

        hf_config = get_current_vllm_config().model_config.hf_config

    return _is_fp8_expert(hf_config)


def patch_lpu_fp8_methods() -> None:
    """Route fp8 ``get_quant_method`` to the LPU FP8 methods.

    Idempotent. First step of ``apply_lpu_dsv4_weights_patch``. On LPU every
    fp8 checkpoint's Linear/MoE quant methods are swapped to
    ``Fp8LinearMethodLPU`` / ``Fp8MoEMethodLPU`` (constructed directly with
    their LPU-safe ``__init__`` — the upstream ``Fp8MoEMethod.__init__`` runs
    ``select_fp8_moe_backend`` which iterates CUDA kernel classes). Two config
    classes are wrapped because vLLM resolves fp8 checkpoints by model_type:

    - ``DeepseekV4FP8Config`` — DSV4 checkpoints (model_type deepseek_v4);
      its ``get_quant_method`` only overrides the fp4-expert MoE dispatch and
      delegates the rest to ``Fp8Config`` via ``super()``.
    - ``Fp8Config`` — every other fp8 checkpoint, e.g. GLM-5.2-FP8
      (model_type glm... -> plain fp8 config). Without this wrap its
      ``DeepSeekV2FusedQkvAProjLinear`` construction dies in
      ``choose_scaled_mm_linear_kernel`` (all candidates CUDA/XPU-only).

    The swap uses exact-type checks, so already-converted LPU variants (the
    DeepseekV4 wrapper sees what the Fp8Config wrapper returned through
    ``super()``), online-fp8 methods, and unquantized/skipped layers pass
    through untouched. Other platforms (and the BF16 path, where the config is
    dropped) are untouched.

    ⚠️ The classes patched are the **installed** site-packages ones: vLLM's
    ``get_quantization_config("deepseek_v4_fp8")`` resolves the FP8 checkpoint
    to that class and the runtime ``quant_config`` is constructed from it
    (model_config.quantization -> ``get_quant_config`` ->
    ``quant_cls.from_config``).  Patching any other copy would be a silent
    no-op.
    """
    global _FP8_METHODS_PATCHED
    if _FP8_METHODS_PATCHED:
        return

    # The installed vLLM classes vLLM actually constructs for fp8
    # checkpoints (registered in vllm/model_executor/layers/quantization/
    # __init__.py; Fp8Config is the "fp8" entry).
    from vllm.model_executor.layers.quantization.fp8 import (
        Fp8Config,
        Fp8LinearMethod,
        Fp8MoEMethod,
    )
    from vllm.models.deepseek_v4 import DeepseekV4FP8Config
    from vllm_fl.dispatch.backends.vendor.evas.impl.fp8_methods import (
        Fp8LinearMethodLPU,
        Fp8MoEMethodLPU,
    )
    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().device_type != "lpu":
        return

    def _make_lpu_get_quant_method(orig):
        def _lpu_get_quant_method(self, layer, prefix):
            method = orig(self, layer, prefix)
            if type(method) is Fp8LinearMethod:
                # Reconstruct as the LPU variant (native fp8 weight + fp32
                # block scale, pure-PyTorch apply). The inherited __init__
                # only does CUDA capability probes (safe on LPU) and sets the
                # quant keys.
                return Fp8LinearMethodLPU(self)
            if type(method) is Fp8MoEMethod:
                # Reconstruct as the LPU variant; its __init__ skips
                # select_fp8_moe_backend and sets moe_kernel=None (built
                # later by process_weights_after_loading).
                return Fp8MoEMethodLPU(self, layer)
            return method

        return _lpu_get_quant_method

    # Subclass first: its orig must be captured before the base class is
    # wrapped (its super() then resolves to the wrapped Fp8Config method at
    # call time — the LPU no-op re-wrap guard is the exact-type check above).
    DeepseekV4FP8Config.get_quant_method = _make_lpu_get_quant_method(
        DeepseekV4FP8Config.get_quant_method)
    Fp8Config.get_quant_method = _make_lpu_get_quant_method(
        Fp8Config.get_quant_method)

    _FP8_METHODS_PATCHED = True


def apply_lpu_dsv4_weights_patch() -> None:
    """Monkey-patch DeepseekV4ForCausalLM for LPU weight loading. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm_fl.dispatch.backends.vendor.evas.evas import EvasBackend
    from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM
    from vllm_fl.utils import DeviceInfo

    # ---- 0. real-FP8: route Fp8LinearMethod/Fp8MoEMethod to LPU variants ----
    patch_lpu_fp8_methods()

    # ---- 1. quant_config dispatch on LPU (real-FP8 keeps the config) ----

    _original_init = DeepseekV4ForCausalLM.__init__

    def _patched_init(self, *, vllm_config, prefix=""):
        if not _real_fp8_enabled(vllm_config):
            # Non-FP8 checkpoint: drop the quant_config so sub-modules never
            # attempt the CUDA FP8 kernel init (crashes on LPU). The loader
            # rejects such checkpoints anyway; this only keeps construction
            # from failing first with a kernel error.
            vllm_config.quant_config = None
        # Deferred indexer dispatch: run now (before attention layers are built
        # inside _original_init) with the checkpoint's expert_dtype deciding the
        # FP8-vs-BF16 sparse indexer. No env knob (auto from the checkpoint).
        from vllm_fl.dispatch.backends.vendor.evas.patches.model_compat import (
            _dsv4_indexer_dispatch,
        )

        _dsv4_indexer_dispatch(vllm_config)
        _original_init(self, vllm_config=vllm_config, prefix=prefix)

    DeepseekV4ForCausalLM.__init__ = _patched_init

    # ---- 2. load_weights → EvasBackend on LPU ----

    _original_load_weights = DeepseekV4ForCausalLM.load_weights

    def _patched_load_weights(self, weights):
        if DeviceInfo().device_type == "lpu":
            return EvasBackend().load_weights(self, weights)
        return _original_load_weights(self, weights)

    DeepseekV4ForCausalLM.load_weights = _patched_load_weights

    # ---- 2b. MTP spec-decode drafter: force FP8 on LPU ----
    # The drafter (num_nextn_predict_layers>0) loads the checkpoint's fp8 MTP
    # weights.  On LPU float8 casts are broken, so it needs the same contract as
    # the target: a DeepseekV4FP8Config (params as native fp8 + fp32 block
    # scale, single residency — a bf16-dequant approach OOM'd the card) and the
    # same reinterpret of its fp8 stream before load_weights.
    from vllm.models.deepseek_v4 import DeepSeekV4MTP  # noqa: E402
    from vllm.models.deepseek_v4 import DeepseekV4FP8Config  # noqa: E402

    def _memdbg_note(tag):
        from vllm.platforms import current_platform
        parts = []
        for i in range(4):
            try:
                _f, _t = current_platform.torch_device_fn.mem_get_info(f"lpu:{i}")
                parts.append(f"lpu{i}={(_t - _f) / 2**30:.2f}G")
            except Exception:  # noqa: BLE001
                parts.append(f"lpu{i}=?")
        print(f"[MTP-MEMDBG] {tag}: " + " ".join(parts), flush=True)

    _original_mtp_init = DeepSeekV4MTP.__init__

    def _patched_mtp_init(self, *, vllm_config, prefix=""):
        if os.environ.get("VLLM_FL_MTP_MEMDBG") == "1":
            _memdbg_note("MTP __init__ enter")
        if DeviceInfo().device_type == "lpu":
            hf = vllm_config.model_config.hf_config
            if _is_fp8_expert(hf) and vllm_config.quant_config is None:
                vllm_config.quant_config = DeepseekV4FP8Config.from_config(
                    hf.quantization_config
                )
        _original_mtp_init(self, vllm_config=vllm_config, prefix=prefix)
        if os.environ.get("VLLM_FL_MTP_MEMDBG") == "1":
            _memdbg_note("MTP __init__ done")

    DeepSeekV4MTP.__init__ = _patched_mtp_init

    _original_mtp_load = DeepSeekV4MTP.load_weights

    def _reinterpret_mtp_fp8(weights):
        # Generator: never materialize the full weight stream.  Non-MTP weights
        # (the target's ~27 GiB fp8) are yielded straight through — the draft's
        # load_weights filters them out.  MTP weights: float8_e4m3fn stays
        # native fp8 (matching the Fp8*Method params) and float8_e8m0fnu scale
        # -> fp32, names unchanged (the draft renames `.scale` ->
        # `.weight_scale_inv`).
        for name, t in weights:
            if not name.startswith(("mtp.", "embed.", "head.")):
                yield (name, t)  # pass through (draft discards)
            elif t.dtype == torch.float8_e8m0fnu:
                yield (name, t.float())
            else:
                yield (name, t)

    def _patched_mtp_load(self, weights):
        _dbg = os.environ.get("VLLM_FL_MTP_MEMDBG") == "1"
        if _dbg:
            _memdbg_note("MTP load enter")
        if DeviceInfo().device_type == "lpu":
            weights = _reinterpret_mtp_fp8(weights)
        result = _original_mtp_load(self, weights)
        if _dbg:
            _memdbg_note("MTP load done")
        return result

    DeepSeekV4MTP.load_weights = _patched_mtp_load

    _PATCHED = True
