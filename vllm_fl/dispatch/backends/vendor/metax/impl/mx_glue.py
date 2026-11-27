# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MetaX: fused glue of Llama-style decoder layers, routed by step type (R6/G1 + R7/N6).

Between the projection GEMMs of a layer the compiled graph runs Inductor's add+RMSNorm
reductions, SiLU-and-mul and RoPE kernels, and the attention op runs the vendor KV-cache
write.  On MiniCPM5-2B (MetaX C500) these cost 7.2 ms of an 84 ms chunked-prefill (mixed)
step and 1.18 ms of a 17.9 ms decode step, in 6 of the 13 kernel launches per layer.

Two fusions, each with two implementations and a step-type policy:

* RoPE + KV write (VLLM_FL_MX_ROPE_KV): LlamaAttention hands q / k to the attention op
  UNROTATED; the MetaX FlashAttention impl rotates q and k and writes k / v into the
  paged cache in ONE kernel (positions come from the attention metadata).  Replaces two
  Inductor RoPE kernels and the vendor reshape_and_cache_flash.  Bitwise the same q /
  cache contents.
* add + RMSNorm (VLLM_FL_MX_NORM): a vLLM IR provider ("fl_metax") for
  ``fused_add_rms_norm``, a one-pass kernel with the row in registers, selected through
  the platform's IR op priority (serve arguments unchanged; like the CUDA/ROCm platforms'
  oink/aiter providers).  Registered inplace, so the hot path allocates nothing.

WHICH IMPLEMENTATION, AND WHY IT IS GATED (R7/N6).  R6/G1 had only the FlagGems/Triton
kernels and measured, on the official benchmark, 4k +1.10% / 16k +0.26% over the round-6
base -- exactly the saving of the graph-captured decode steps (0.32 ms/step) and nothing
from the mixed steps, plus 4k mean TTFT 1904 -> 2055 ms.  Reason: mixed steps are not
graph-captured (capture sizes end at ``max_cudagraph_capture_size``), and at M=2048 a
FlagGems/Triton launch of these kernels costs 61-116 us of HOST time against 22-30 us of
GPU time, so the host became the critical path of exactly the steps that set TTFT.
So the choice is keyed on the step's token count:

  tokens <= CAPTURE_MAX   graph-captured decode step: the host cost is paid once, at
                          capture, so the fastest GPU kernel wins (table below);
  tokens >  CAPTURE_MAX   eager (mixed) step: only the native library (impl/csrc,
                          ~4 us of host time per launch, one launch per fusion) may run;
                          without it the whole glue stays on the deployed path.

The tables are pinned from the GPU bench (work/R7/N6-mx-glue-gated/bench_n6.py --emit)
and key on runtime features only (token count, hidden size, head counts, dtype, strides),
never on the benchmark case.

VLLM_FL_MX_GLUE=0 turns everything off; VLLM_FL_MX_NORM=0 and VLLM_FL_MX_ROPE_KV=0 turn
off one fusion each (so a bench session can isolate either leg without a rebuild);
VLLM_FL_MX_GLUE_NATIVE=0 drops the native library (and with it the mixed-step fusions,
which is also what a "d" policy entry does); VLLM_FL_MX_GLUE_FORCE=flaggems|native
overrides the policy for both fusions (A/B of the pipeline, not for serving).
VLLM_FL_MX_GLU (gate_up + SiLU-and-mul as one op) is OFF by default: R6/G1 measured no
GLU-epilogue GEMM that beats the deployed mm + SiLU at any M, and the op pulls the MLP
out of the compiled graph, which costs host time on the mixed steps.
"""

import importlib
import importlib.util
import logging
import os

import torch
from torch import Tensor

from . import mx_glue_native as _nat

logger = logging.getLogger(__name__)


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "off", "no")


ENABLED = _flag("VLLM_FL_MX_GLUE")
ROPE_KV = ENABLED and _flag("VLLM_FL_MX_ROPE_KV")
NORM = ENABLED and _flag("VLLM_FL_MX_NORM")
GLU = ENABLED and _flag("VLLM_FL_MX_GLU", "0")
FORCE = os.environ.get("VLLM_FL_MX_GLUE_FORCE", "").strip().lower()

PROVIDER = "fl_metax"
_KERNELS = "flag_gems.runtime.backend._metax.ops.mx_glue"
_kmod = None

# --------------------------------------------------------------------------- #
# step-type policy
# --------------------------------------------------------------------------- #
# Largest token count that runs inside a captured CUDA graph; install() replaces it with
# compilation_config.max_cudagraph_capture_size of the served configuration.
CAPTURE_MAX = 512
# Implementation per token count: "g" = FlagGems Triton (R6/G1), "n" = native (R7/N6),
# "d" = leave it on the deployed path (only possible before the hook is installed).
# From the C500 GPU run of 2026-10-24 (work/R7/N6-mx-glue-gated/logs/gpu,
# n6_full_20261024_225056.log), per 42-layer step of the served model:
#   captured decode step (M=64)  glue 1.117 ms deployed, 1.090 native, 0.813 FlagGems
#                                -> "g": 0.304 ms/step, bitwise the deployed residual /
#                                   rotated q / cache contents;
#   eager mixed step (M=2048)    add+RMSNorm max(GPU, host) 33.7 + 49.0 us/layer
#                                deployed, 2 x 32.0 native, 2 x max(31.0, 61.2) FlagGems
#                                -> "n": 18.7 us/layer = 0.79 ms/step, and never the
#                                   Triton path, whose host time (292 us/layer) is what
#                                   cost R6/G1 its mixed steps and 8% of the 4k TTFT.
# RoPE+KV on the eager steps: native is 16.9 us/layer cheaper on the GPU (30.2 vs 47.1)
# and about even on the host (51.6 vs 53.8), so "n" -- and a third value "d" (leave the
# fusion to the deployed path) is only expressible for a WHOLE run: once LlamaAttention
# hands q / k over unrotated there is no deployed kernel left to re-enter per step, and
# not installing the hook at all would give up 7.2 of the 8.2 us/layer that the measured
# decode-step saving is made of.  A "d" entry therefore switches its fusion off
# completely (same as VLLM_FL_MX_ROPE_KV=0 / VLLM_FL_MX_NORM=0), which is how the next
# round can A/B "deployed everywhere" without touching the kernels.
# --- pinned glue policy: begin ---
NORM_POLICY = [("capture", "g"), ("rest", "n")]
ROPE_POLICY = [("capture", "g"), ("rest", "n")]
# --- pinned glue policy: end ---

_NORM_FG_MAX = 0   # tokens <= this go to FlagGems, above to native (0: native only)
_ROPE_FG_MAX = 0
_NORM_OFF = False  # a "d" entry: the fusion stays on the deployed path for the whole run
_ROPE_OFF = False


def _resolve_policy() -> None:
    """Turn the pinned tables into the two integers the hot path compares against."""
    global _NORM_FG_MAX, _ROPE_FG_MAX, _NORM_OFF, _ROPE_OFF
    for table, name in ((NORM_POLICY, "norm"), (ROPE_POLICY, "rope")):
        limit, off = 0, False
        for where, impl in table:
            n = CAPTURE_MAX if where == "capture" else (1 << 30 if where == "rest" else int(where))
            if impl == "g":
                limit = max(limit, n)
            elif impl == "d":
                off = True
            elif impl != "n":
                raise ValueError(f"MetaX glue: {name} policy {table}: no implementation {impl!r}")
        if FORCE == "flaggems":
            limit, off = 1 << 30, False
        elif FORCE == "native":
            limit, off = 0, False
        if name == "norm":
            _NORM_FG_MAX, _NORM_OFF = limit, off
        else:
            _ROPE_FG_MAX, _ROPE_OFF = limit, off


_resolve_policy()


def is_metax() -> bool:
    from vllm.platforms import current_platform

    return getattr(current_platform, "vendor_name", None) == "metax"


def kernels_available() -> bool:
    """Whether the installed FlagGems carries the MetaX glue kernels (checked on the
    file system: importing flag_gems initialises its runtime)."""
    try:
        spec = importlib.util.find_spec("flag_gems")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    root = list(spec.submodule_search_locations)[0]
    return os.path.isfile(os.path.join(root, "runtime", "backend", "_metax", "ops", "mx_glue.py"))


def kernels():
    global _kmod
    if _kmod is None:
        _kmod = importlib.import_module(_KERNELS)
    return _kmod


def native_ready(which: str) -> bool:
    """Native library built, loaded and self-tested for this fusion.  The mixed steps
    have no other implementation, so the hooks are installed only when this holds.

    Before the worker's install() has run (the IR provider is registered while the
    platform builds its op priorities, possibly in another process) only the mxcc build
    and the ctypes load can be checked -- mxcc needs no GPU context; the device self-test
    then runs in install(), and a failure there sends the op to its last-resort path."""
    if not _nat.REQUESTED:
        return False
    if not _nat._SETUP_DONE:
        _nat.setup(build_only=True)
        return bool(_nat._LIB)
    return _nat.NORM_OK if which == "norm" else _nat.ROPE_OK


# --------------------------------------------------------------------------- #
# torch custom ops (opaque to Inductor)
# --------------------------------------------------------------------------- #
_NORM_NATIVE_FN = None   # vLLM's own eager implementation (last-resort fallback)


def _mx_fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """In place, like torch.ops._C.fused_add_rms_norm (the vllm_c provider's kernel):
    residual <- x + residual, x <- RMSNorm(x + residual) * weight.  Both kernels read a
    row into registers before they store it, so writing into x / residual is safe.
    One int compare picks the implementation for this step type (see the policy above)."""
    if x.shape[0] <= _NORM_FG_MAX:
        kernels().add_rms_norm(x, residual, weight, epsilon, out=x, residual_out=residual)
        return
    if _nat.NORM_OK and _nat.add_rms_norm(x, residual, weight, epsilon, x, residual):
        return
    # last resort (a self-test failure after the provider was registered): vLLM's own
    # eager implementation, then copy back.  Never taken in a served step.
    if _NORM_NATIVE_FN is not None:
        y, r = _NORM_NATIVE_FN(x, residual, weight, epsilon, None)
    else:
        y, r = kernels().add_rms_norm(x, residual, weight, epsilon)
    x.copy_(y)
    residual.copy_(r)


def _mx_fused_add_rms_norm_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    return None


def _mx_gate_up_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    lead = x.shape[:-1]
    out = kernels().gate_up_silu(x.reshape(-1, x.shape[-1]), weight)
    return out.view(*lead, out.shape[-1])


def _mx_gate_up_silu_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0] // 2), dtype=x.dtype, device=x.device)


_ops_registered = False


def register_ops() -> None:
    global _ops_registered
    if _ops_registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "mx_fused_add_rms_norm"):
        direct_register_custom_op(
            op_name="mx_fused_add_rms_norm",
            op_func=_mx_fused_add_rms_norm,
            mutates_args=["x", "residual"],
            fake_impl=_mx_fused_add_rms_norm_fake,
        )
    if not hasattr(torch.ops.vllm, "mx_gate_up_silu"):
        direct_register_custom_op(
            op_name="mx_gate_up_silu",
            op_func=_mx_gate_up_silu,
            mutates_args=[],
            fake_impl=_mx_gate_up_silu_fake,
        )
    _ops_registered = True


# --------------------------------------------------------------------------- #
# vLLM IR provider for fused_add_rms_norm
# --------------------------------------------------------------------------- #
def _norm_supports(x, x_residual, weight, epsilon, variance_size=None):
    # static properties only (called on fake tensors during IR lowering)
    return (
        variance_size is None
        and weight is not None
        and x.dtype in (torch.bfloat16, torch.float16)
        and x_residual.dtype == x.dtype
        and weight.dtype == x.dtype
        and x.dim() == 2
        and x.shape[-1] == x_residual.shape[-1] == weight.shape[0]
        and x.shape[-1] in _nat.NORM_CFG          # a native config covers this hidden size
        and x.stride(-1) == 1
        and x_residual.stride(-1) == 1
    )


def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """fl_metax provider: one-pass add + RMSNorm, routed by step type.  Registered
    inplace (like the vllm_c provider): the custom op mutates x and x_residual."""
    torch.ops.vllm.mx_fused_add_rms_norm(x, x_residual, weight, epsilon)
    return x, x_residual


_ir_registered = False


def register_ir_providers() -> bool:
    """Register the fl_metax provider of vllm.ir.ops.fused_add_rms_norm (idempotent).
    Returns whether it is registered and supported.  The provider is supported only when
    BOTH implementations exist: the FlagGems kernels for the captured decode steps and
    the native library for the eager mixed steps (which have no other implementation)."""
    global _ir_registered, _NORM_NATIVE_FN
    from vllm import ir

    op = ir.ops.fused_add_rms_norm
    if _NORM_NATIVE_FN is None:
        _NORM_NATIVE_FN = op.impls["native"].impl_fn
    if PROVIDER not in op.impls:
        register_ops()
        supported = (NORM and not _NORM_OFF and is_metax() and kernels_available()
                     and native_ready("norm"))
        op.register_impl(PROVIDER, supports_args=_norm_supports, supported=supported,
                         inplace=True)(fused_add_rms_norm)
    _ir_registered = True
    return op.impls[PROVIDER].supported


def ir_priority_overrides() -> dict:
    """IR op priorities the MetaX platform puts in front of its defaults."""
    if not (NORM and not _NORM_OFF and is_metax() and kernels_available()):
        return {}
    try:
        if not register_ir_providers():
            return {}
    except Exception as e:  # never break config creation
        logger.warning("MetaX glue: IR provider registration failed: %s", e)
        return {}
    return {"fused_add_rms_norm": [PROVIDER, "native"]}


# --------------------------------------------------------------------------- #
# model patches (LlamaAttention: RoPE in the attention op; LlamaMLP: GLU op)
# --------------------------------------------------------------------------- #
_patched = False
_ROPE_REJECTED = False
_counts = {"rope_kv": 0, "rope_kv_skipped": 0, "glu": 0, "glu_skipped": 0}


def _rope_fusable(attn_layer, llama) -> bool:
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    if type(attn_layer) is not llama.LlamaAttention:
        return False
    rope = getattr(attn_layer, "rotary_emb", None)
    if not isinstance(rope, RotaryEmbedding):
        return False
    cls = type(rope)
    if cls.forward_native is not RotaryEmbedding.forward_native:
        return False
    if cls.forward_static is not RotaryEmbedding.forward_static:
        return False
    head = attn_layer.head_dim
    if not rope.is_neox_style or rope.rotary_dim != head or rope.head_size != head:
        return False
    half = head // 2
    if head % 2 or half & (half - 1):
        return False
    cache = getattr(rope, "cos_sin_cache", None)
    if cache is None or cache.dim() != 2 or cache.shape[-1] != head:
        return False
    attn = attn_layer.attn
    impl = getattr(attn, "impl", None)
    if not getattr(impl, "mx_rope_supported", False):
        return False
    backend = getattr(attn, "attn_backend", None)
    if backend is None or not getattr(backend, "forward_includes_kv_cache_update", True):
        return False
    if getattr(attn, "kv_sharing_target_layer_name", None) is not None:
        return False
    if getattr(attn, "query_quant", None) is not None or getattr(attn, "calculate_kv_scales", False):
        return False
    if getattr(attn, "kv_cache_dtype", "auto") not in ("auto", "bfloat16", "float16"):
        return False
    return True


def _glu_fusable(mlp, llama) -> bool:
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        UnquantizedLinearMethod,
    )

    if type(mlp) is not llama.LlamaMLP:
        return False
    gu = mlp.gate_up_proj
    if type(gu) is not MergedColumnParallelLinear:
        return False
    if type(getattr(gu, "quant_method", None)) is not UnquantizedLinearMethod:
        return False
    if getattr(gu, "bias", None) is not None or getattr(gu, "gather_output", False):
        return False
    sizes = getattr(gu, "output_sizes", None)
    if not sizes or len(sizes) != 2 or sizes[0] != sizes[1]:
        return False
    w = getattr(gu, "weight", None)
    if w is None or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
        return False
    act = mlp.act_fn
    if not isinstance(act, SiluAndMul):
        return False
    if type(act).forward_native is not SiluAndMul.forward_native:
        return False
    return True


def install_model_patches() -> None:
    """Patch vLLM's Llama layers (idempotent; call before the model is built)."""
    global _patched
    # the mixed steps have no other implementation, so the hook needs the native kernel
    rope_kv = ROPE_KV and not _ROPE_OFF and native_ready("rope")
    if _patched or not (rope_kv or GLU):
        if ROPE_KV and not rope_kv:
            logger.info("MetaX glue: no native RoPE+KV kernel (%s), RoPE stays in the graph",
                        _nat.STATUS)
        return
    if not (is_metax() and kernels_available()):
        logger.info("MetaX glue: FlagGems mx_glue kernels not installed, Llama glue fusion off")
        return
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.models import llama

    register_ops()
    attn_init = llama.LlamaAttention.__init__
    attn_forward = llama.LlamaAttention.forward
    mlp_init = llama.LlamaMLP.__init__
    mlp_forward = llama.LlamaMLP.forward

    def patched_attn_init(self, *args, **kwargs):
        attn_init(self, *args, **kwargs)
        self._mx_rope_in_attn = bool(rope_kv and _rope_fusable(self, llama))
        if self._mx_rope_in_attn:
            # the attention impl rotates q / k and writes the cache in one kernel
            self.attn.impl.mx_rope = self.rotary_emb
            _counts["rope_kv"] += 1
        else:
            _counts["rope_kv_skipped"] += 1

    def patched_attn_forward(self, positions, hidden_states):
        if not getattr(self, "_mx_rope_in_attn", False):
            return attn_forward(self, positions, hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q / k stay unrotated: RoPE runs inside the attention op, fused with
        # the KV-cache write (MetaX FlashAttentionImpl.forward, mx_rope).
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def patched_mlp_init(self, *args, **kwargs):
        mlp_init(self, *args, **kwargs)
        self._mx_glu = bool(GLU and _glu_fusable(self, llama))
        _counts["glu" if self._mx_glu else "glu_skipped"] += 1

    def patched_mlp_forward(self, x):
        if not getattr(self, "_mx_glu", False) or type(self.gate_up_proj) is not MergedColumnParallelLinear:
            return mlp_forward(self, x)
        x = torch.ops.vllm.mx_gate_up_silu(x, self.gate_up_proj.weight)
        x, _ = self.down_proj(x)
        return x

    patched_attn_forward.__wrapped__ = attn_forward
    patched_mlp_forward.__wrapped__ = mlp_forward
    llama.LlamaAttention.__init__ = patched_attn_init
    llama.LlamaAttention.forward = patched_attn_forward
    llama.LlamaMLP.__init__ = patched_mlp_init
    llama.LlamaMLP.forward = patched_mlp_forward
    _patched = True
    logger.info(
        "MetaX glue: Llama patches installed (rope+kv=%s, glu=%s, norm provider=%s); "
        "policy: FlagGems <= %d tokens (norm) / %d (rope), native above; capture max %d; "
        "native: %s",
        rope_kv, GLU, NORM, _NORM_FG_MAX, _ROPE_FG_MAX, CAPTURE_MAX, _nat.STATUS,
    )


def install(vllm_config=None) -> None:
    """Worker-side entry: native library + self-test, step-type policy, model patches."""
    global CAPTURE_MAX
    if not ENABLED or not is_metax():
        return
    try:
        cc = getattr(vllm_config, "compilation_config", None)
        cap = getattr(cc, "max_cudagraph_capture_size", None)
        if isinstance(cap, int) and cap > 0:
            CAPTURE_MAX = cap
            _resolve_policy()
        _nat.setup()
        install_model_patches()
    except Exception as e:
        logger.warning("MetaX glue: model patches not installed: %s", e)


def summary() -> dict:
    return dict(_counts, capture_max=CAPTURE_MAX, rope_capture_safe=rope_capture_safe(),
                rope_only_ok=_nat.ROPE_ONLY_OK, norm_fg_max=_NORM_FG_MAX,
                rope_fg_max=_ROPE_FG_MAX, norm_off=_NORM_OFF, rope_off=_ROPE_OFF,
                native=_nat.summary())


# --------------------------------------------------------------------------- #
# attention side
# --------------------------------------------------------------------------- #
def positions_from_metadata(attn_metadata, num_tokens):
    """Token positions from query_start_loc / seq_lens (device ops only, no host sync):
    token t of request r sits at seq_lens[r] - (qsl[r+1] - t).  Fallback for metadata
    built without the runner's positions buffer."""
    qsl = attn_metadata.query_start_loc
    seq_lens = attn_metadata.seq_lens
    t = torch.arange(num_tokens, device=qsl.device, dtype=qsl.dtype)
    req = torch.searchsorted(qsl[1:].contiguous(), t, right=True)
    req = req.clamp_(max=seq_lens.shape[0] - 1)
    pos = seq_lens[req].to(torch.int64) - (qsl[req + 1].to(torch.int64) - t.to(torch.int64))
    return pos.clamp_(min=0)


# R10/F: a CUDA-graph capture is in progress on this thread's current stream.  Resolved
# once, because the attribute is what decides whether the guard below can work at all.
_CAPTURE_PROBE = getattr(torch.cuda, "is_current_stream_capturing", None)


def capture_active() -> bool:
    if _CAPTURE_PROBE is None:
        return False
    try:
        return bool(_CAPTURE_PROBE())
    except Exception:  # noqa: BLE001
        return False


def rope_fusion_installed() -> bool:
    """The Llama patch handed at least one layer's RoPE to the attention impl."""
    return bool(_patched and _counts.get("rope_kv", 0))


def rope_capture_safe() -> bool:
    """True when every RoPE this process can run inside a captured region is one of our
    own kernels.  With the fusion installed the attention impl owns the rotation, and the
    INT8 KV path needs the rotation WITHOUT the cache write (_nat.rope_only); the torch
    fallback for that half is capture-illegal -- it reaches FlagGems' generic broadcast
    multiply, which copies a [shape | strides] meta tensor from the host at every call
    (flag_gems/ops/mul.py _launch_generic), and a host->device copy while a capture is
    active aborts the capture with "operation would make the legacy stream depend on a
    capturing blocking stream".  Features that put the rotation on that path inside a
    FULL decode graph must refuse while this is False.
    """
    return (not rope_fusion_installed()) or bool(_nat.ROPE_ONLY_OK)


_ROPE_INPLACE_FALLBACK = False
_ROPE_ONLY_LOGGED = False


def rope_inplace(rope, query, key, attn_metadata):
    """Rotate q / k with ``rope`` at the metadata positions, and nothing else.

    R9/F: the INT8 KV cache writes the pages with its own quantising kernel, so the
    fused RoPE+KV kernel -- which writes bf16 KV -- cannot serve it until the QUANT
    variant of that kernel exists.  This is the half of it the int8 path still needs,
    and having it here is what lets the fusion be INSTALLED whenever it is possible:
    FlashAttentionImpl.mx_rope_supported no longer has to refuse at model-build time
    merely because the INT8 cache was requested, which is a decision the metadata
    builder's setup() latches much later.  A refusing int8 self-test can therefore no
    longer cost the RoPE fusion (<= 0.74 s per 16k run) for the rest of the process.

    Same rotation the unpatched Llama layer does (RotaryEmbedding.__call__, in place on
    q / k), with the torch-native implementation as the fallback, and the ROTATED
    tensors are returned rather than assumed to be the inputs -- so the result does not
    depend on which implementation the CustomOp dispatch picked.
    """
    global _ROPE_INPLACE_FALLBACK, _ROPE_ONLY_LOGGED
    t = query.shape[0]
    positions = getattr(attn_metadata, "positions", None)
    if positions is None:
        positions = positions_from_metadata(attn_metadata, t)
    positions = positions[:t]
    # R10/F: our own kernel first -- one launch, no allocation, no host read, so it is
    # the only implementation of this rotation that a CUDA-graph capture tolerates (see
    # rope_capture_safe).  It rotates q and k IN PLACE, which is what the vendor's
    # rotary_embedding op does too; every thread reads the two halves of the 8 dims it
    # owns before it writes them back, so the aliasing is safe by construction.
    if _nat.ROPE_ONLY_OK and key.shape[0] == t:
        cache = rope.cos_sin_cache
        if cache.dtype != query.dtype or cache.device != query.device:
            cache = cache.to(query.device, dtype=query.dtype)
            rope.cos_sin_cache = cache
        d = cache.shape[-1]
        if query.shape[-1] == d or query.dim() == 2:
            q3 = query if query.dim() == 3 else query.view(t, -1, d)
            k3 = key if key.dim() == 3 else key.view(t, -1, d)
            if (_nat.rope_only_supported(q3, k3, cache)
                    and _nat.rope_only(q3, k3, positions, cache, q3, k3)):
                return q3.view(query.shape), k3.view(key.shape)
        if not _ROPE_ONLY_LOGGED:
            _ROPE_ONLY_LOGGED = True
            logger.warning("MetaX glue: the native RoPE-only kernel rejected %s / %s; "
                           "the rotation falls back to torch, which cannot run inside a "
                           "CUDA-graph capture: %s",
                           tuple(query.shape), tuple(key.shape), _nat.summary())
    # No capture-safe rotation left.  PREVENT the illegal operation instead of catching
    # it: the torch path below ends in a host->device copy, and a CUDA error raised
    # during capture is sticky, so the try/except this replaces could only log it (the
    # fallback raised the same error one line later) before the engine died.
    if capture_active():
        raise RuntimeError(
            "MetaX glue: no capture-safe RoPE for the INT8 KV path (native rope_only: "
            f"{_nat.STATUS}). The torch rotation reaches FlagGems' generic broadcast "
            "multiply, whose host->device meta copy aborts a CUDA-graph capture. Serve "
            "with VLLM_FL_MX_INT8_KV=0, or with VLLM_FL_MX_ROPE_KV=0 to keep RoPE in "
            "the compiled graph.")
    # reshape, not view: a non-viewable layout gives a copy, which the rotation then
    # lands in -- correct either way, because the caller uses what comes back.
    q2 = query.reshape(t, -1)
    k2 = key.reshape(key.shape[0], -1)
    if not _ROPE_INPLACE_FALLBACK:
        try:
            qr, kr = rope(positions, q2, k2)
            return qr.view(query.shape), kr.view(key.shape)
        except Exception as e:  # noqa: BLE001
            _ROPE_INPLACE_FALLBACK = True
            logger.warning(
                "MetaX glue: the in-place RoPE call failed (%s); the INT8 KV path "
                "uses the torch implementation from here on", e)
    qr, kr = rope.forward_native(positions, q2, k2)
    return qr.view(query.shape), kr.view(key.shape)


def attn_rope_kv(rope, query, key, value, key_cache, value_cache, attn_metadata):
    """Rotate q / k with ``rope``'s cos/sin cache at the metadata positions, write rotated
    k and v into the paged cache, return the rotated q.  Rows past the slot mapping (CUDA
    graph padding) rotate at position 0 and write nothing, so one captured graph is valid
    for every step; the implementation is chosen from the step's token count."""
    if key is None or value is None:
        raise RuntimeError("MetaX glue: fused RoPE needs the layer's key and value")
    positions = getattr(attn_metadata, "positions", None)
    if positions is None:
        positions = positions_from_metadata(attn_metadata, query.shape[0])
    cache = rope.cos_sin_cache
    if cache.dtype != query.dtype or cache.device != query.device:
        cache = cache.to(query.device, dtype=query.dtype)
        rope.cos_sin_cache = cache
    t = query.shape[0]
    if t > _ROPE_FG_MAX and _nat.ROPE_OK:
        d = cache.shape[-1]
        q3 = query if query.dim() == 3 else query.view(t, -1, d)
        k3 = key if key.dim() == 3 else key.view(key.shape[0], -1, d)
        v3 = value if value.dim() == 3 else value.view(value.shape[0], -1, d)
        q_out = torch.empty((t, q3.shape[1], d), dtype=query.dtype, device=query.device)
        if _nat.rope_kv(q3, k3, v3, positions, cache, q_out, key_cache, value_cache,
                        attn_metadata.slot_mapping):
            return q_out
        global _ROPE_REJECTED
        if not _ROPE_REJECTED:
            _ROPE_REJECTED = True
            logger.warning("MetaX glue: the native RoPE+KV kernel rejected an eager step "
                           "(%s); these steps fall back to the FlagGems kernel, which "
                           "costs host time: %s", tuple(q3.shape), _nat.summary())
    return kernels().rope_kv_cache(
        query,
        key,
        value,
        positions,
        cache,
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
    )
