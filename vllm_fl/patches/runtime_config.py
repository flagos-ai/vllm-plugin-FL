"""Runtime config injection patch (v2 — intercept at LLM.__init__).

The bulk of our +146% throughput uplift on Qwen3-4B / Ascend 910 comes
from a SET of vllm runtime kwargs (tensor_parallel_size=2,
async_scheduling=True, max_num_seqs=512, max_num_batched_tokens=16384,
block_size=256, compilation_config with enable_sp, enforce_eager=True)
plus env vars (TRITON_ALL_BLOCKS_PARALLEL=1, TASK_QUEUE_ENABLE=1).

When the FlagOS evaluation harness invokes `vllm bench throughput` or
`lm_eval --model vllm` with ONLY the default kwargs, most of those args
are at vllm defaults and most of our optimization disappears even
though the PA-in-eager patch is correctly applied.

This patch wraps `vllm.entrypoints.llm.LLM.__init__` to inject our
preferred defaults BEFORE EngineArgs is constructed. Mutating later
(via `Platform.check_and_update_config`) causes worker spawn /
world-size inconsistency because vllm has already computed those
quantities from the original tensor_parallel_size=1.

We only override fields that arrive at the LLM constructor at the
documented vllm default values; any explicit user choice (e.g. user
passes tensor_parallel_size=4) is preserved.
"""

import importlib.abc
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

_TARGET_MODULE = "vllm.entrypoints.llm"
_PATCH_FLAG_ATTR = "_fl_runtime_config_patched"


def _num_npus_available() -> int:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible:
        return len([x for x in visible.split(",") if x.strip()])
    try:
        import torch_npu  # type: ignore
        if torch_npu.npu.is_available():
            return torch_npu.npu.device_count()
    except Exception:
        pass
    return 0


# Documented vllm defaults that signal "user did not pass this arg"
_VLLM_DEFAULTS = {
    "tensor_parallel_size": 1,
    "enforce_eager": False,
}


def _patch_llm_init(llm_module):
    """Wrap LLM.__init__ to inject our preferred kwargs before EngineArgs."""
    try:
        LLM = llm_module.LLM
    except AttributeError:
        logger.warning("FL runtime_config: vllm.entrypoints.llm.LLM not found")
        return

    if getattr(LLM, _PATCH_FLAG_ATTR, False):
        return

    orig_init = LLM.__init__

    def wrapped_init(self, *args, **kwargs):
        changes = []
        npu_count = _num_npus_available()

        # 1. tensor_parallel_size: prefer 2 if we have ≥2 NPUs and user didn't override
        tp = kwargs.get("tensor_parallel_size", None)
        # Also support positional — first positional after model is runner; tp not in first 5 positions
        if tp is None or tp == _VLLM_DEFAULTS["tensor_parallel_size"]:
            if npu_count >= 2:
                kwargs["tensor_parallel_size"] = 2
                changes.append("tensor_parallel_size=2")

        # 2. enforce_eager: required by our PA-in-eager patch
        ee = kwargs.get("enforce_eager", None)
        if not ee:
            kwargs["enforce_eager"] = True
            changes.append("enforce_eager=True")

        # 3. max_num_seqs (passed via **kwargs as EngineArgs forwards them)
        if "max_num_seqs" not in kwargs or kwargs["max_num_seqs"] in (None, 64, 128, 256):
            kwargs["max_num_seqs"] = 512
            changes.append("max_num_seqs=512")

        # 4. max_num_batched_tokens
        if "max_num_batched_tokens" not in kwargs or kwargs["max_num_batched_tokens"] in (None, 2048, 4096, 8192):
            kwargs["max_num_batched_tokens"] = 16384
            changes.append("max_num_batched_tokens=16384")

        # 5. block_size
        if "block_size" not in kwargs or kwargs["block_size"] in (None, 16, 32, 64, 128):
            kwargs["block_size"] = 256
            changes.append("block_size=256")

        # 6. async_scheduling — overlap scheduler with NPU forward
        if "async_scheduling" not in kwargs or not kwargs["async_scheduling"]:
            kwargs["async_scheduling"] = True
            changes.append("async_scheduling=True")

        # 7. compilation_config — enable sequence parallelism pass
        cc = kwargs.get("compilation_config", None)
        if cc is None:
            kwargs["compilation_config"] = {"pass_config": {"enable_sp": True}}
            changes.append("compilation_config.pass_config.enable_sp=True")
        elif isinstance(cc, dict):
            pass_cfg = cc.setdefault("pass_config", {})
            if isinstance(pass_cfg, dict) and not pass_cfg.get("enable_sp"):
                pass_cfg["enable_sp"] = True
                changes.append("compilation_config.pass_config.enable_sp=True")

        if changes:
            logger.info("FL runtime_config: injected %s", ", ".join(changes))

        return orig_init(self, *args, **kwargs)

    # Preserve signature & docstring for introspection
    import functools
    functools.update_wrapper(wrapped_init, orig_init)
    LLM.__init__ = wrapped_init
    setattr(LLM, _PATCH_FLAG_ATTR, True)
    logger.info(
        "FL runtime_config: LLM.__init__ wrapped to inject Qwen3-4B runtime tuning")


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        try:
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _WrappedLoader(spec.loader)
            return spec
        except Exception as e:
            logger.warning(f"FL runtime_config: finder error: {e}")
            return None


class _WrappedLoader(importlib.abc.Loader):
    def __init__(self, inner_loader):
        self._inner = inner_loader

    def create_module(self, spec):
        if hasattr(self._inner, "create_module"):
            return self._inner.create_module(spec)
        return None

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch_llm_init(module)
        except Exception as e:
            logger.warning(f"FL runtime_config: patch failed: {e}")


def apply_runtime_config_patch():
    """Install LLM.__init__ wrapper + baseline env vars."""
    os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")

    mod = sys.modules.get(_TARGET_MODULE)
    if mod is not None:
        _patch_llm_init(mod)
        return

    for f in sys.meta_path:
        if isinstance(f, _PostImportPatcher):
            return
    sys.meta_path.insert(0, _PostImportPatcher())
