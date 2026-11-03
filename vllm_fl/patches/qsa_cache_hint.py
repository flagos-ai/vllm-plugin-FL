# Copyright (c) 2026 BAAI. All rights reserved.

"""Work around QSA cache-hint miscompilation on Hopper with FlagTree 3.6.

The pinned toolchain emits an uninitialized cache-policy descriptor for the
K-cache loads in upstream QSA prefill and decode. Removing that optional hint
preserves the native kernels and their launch parameters. Requalify and remove
this workaround when the FlagTree toolchain is upgraded.
"""

import ast
from importlib.metadata import PackageNotFoundError, version

_KERNEL_NAMES = (
    "_qsa_mqa_paged_prefill_kernel",
    "_qsa_mqa_paged_uniform_kernel",
)


def _without_k_cache_eviction_hint(kernel) -> str | None:
    """Validate the upstream load before changing its one optional keyword."""
    tree = ast.parse(kernel.src)
    loads = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "keys"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "tl"
        and node.value.func.attr == "load"
        and node.value.args
        and any(
            isinstance(arg, ast.Name) and arg.id == "k_cache_ptr"
            for arg in ast.walk(node.value.args[0])
        )
    ]
    if len(loads) != 1:
        raise RuntimeError(
            "QSA K-cache load changed; revalidate the FlagTree workaround"
        )
    load = loads[0]
    hints = [kw for kw in load.keywords if kw.arg == "eviction_policy"]
    if not hints:
        return None  # Already patched, or fixed by upstream.
    if len(hints) != 1 or not (
        isinstance(hints[0].value, ast.Constant)
        and hints[0].value.value == "evict_first"
    ):
        raise RuntimeError("QSA eviction policy changed; revalidate the workaround")
    load.keywords.remove(hints[0])
    return ast.unparse(tree) + "\n"


def patch_qsa_cache_hint(model_config, device) -> bool:
    """Install before model loading, after the worker selects its actual GPU.

    Both native model runners share this worker hook. Do not apply to other
    architectures, GPU generations, or standard/newer Triton installations.
    """
    if model_config.architecture not in (
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpForCausalLM",
    ):
        return False

    import triton

    if triton.__version__.split(".")[:2] != ["3", "6"]:
        return False
    try:
        version("flagtree")
    except PackageNotFoundError:
        return False

    import torch

    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(device) != (9, 0)
    ):
        return False

    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    # Validate both kernels before mutating either one. A compiled JITFunction
    # also caches device launchers; changing only its source would leave those
    # launchers stale. The worker hook runs before the first model compilation.
    updates = []
    for name in _KERNEL_NAMES:
        kernel = getattr(qsa_indexer, name)
        source = _without_k_cache_eviction_hint(kernel)
        if source is None:
            continue
        if kernel.device_caches:
            raise RuntimeError(
                f"{name} was already compiled; install the QSA workaround "
                "before model loading"
            )
        updates.append((kernel, source))
    for kernel, source in updates:
        kernel._fl_qsa_original_src = kernel.src
        kernel._unsafe_update_src(source)
    return bool(updates)
