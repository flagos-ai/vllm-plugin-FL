"""Pin verified RMS launch parameters by mathematical body, not generated suffix.

AST is read solely to fingerprint a compiler-generated function. No rewriting,
exec, source replacement or arithmetic change is performed.
"""

import ast
import functools
import hashlib
import inspect
import json
import os
import textwrap
from pathlib import Path

_seen = set()


def _fingerprint(fn):
    src = getattr(fn, "src", None)
    if not src:
        src = inspect.getsource(fn.fn)
    tree = ast.parse(textwrap.dedent(src))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    node.name = "same_rms"
    node.decorator_list = []
    # Python 3.12 adds an empty type_params field; normalize to the 3.10 audit schema.
    for item in ast.walk(node):
        if hasattr(item, "type_params"):
            assert not item.type_params, "Generic RMS functions are not supported"
            delattr(item, "type_params")
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def install(policy, audit_dir):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from triton import Config

    if getattr(CachingAutotuner, "_dsv4_semantic_rms_policy", False):
        return
    by_hash = policy["body_fingerprints"]
    expected_warps = policy["rank_warps"]
    folder = Path(audit_dir)
    original_precompile = CachingAutotuner.precompile
    original_recheck = CachingAutotuner.recheck_autotune_cache

    def candidate(obj):
        name = getattr(getattr(obj, "fn", None), "__name__", "")
        return (
            name.startswith("triton_red_fused_add_mean")
            and "rsqrt" in name
            and (getattr(obj, "size_hints", None) or {}).get("r0_") == 4096
        )

    def resolve(obj, reload_kernel=None):
        if not candidate(obj):
            return None
        if getattr(obj.fn, "fn", None) is None:
            reload_kernel = reload_kernel or getattr(obj, "_reload_kernel", None)
            if reload_kernel is None:
                raise RuntimeError("Verified RMS needs source reload")
            obj.fn = reload_kernel().fn
        fingerprint = _fingerprint(obj.fn)
        if fingerprint not in by_hash:
            raise RuntimeError(
                "Unknown RMS mathematical body: "
                + fingerprint
                + " / "
                + obj.fn.__name__
            )
        return fingerprint, by_hash[fingerprint]

    def matches(c, w):
        return (
            c.num_warps == w
            and c.num_stages == policy["stages"]
            and c.kwargs == {"XBLOCK": policy["xblock"], "R0_BLOCK": policy["rblock"]}
        )

    @functools.wraps(original_recheck)
    def recheck(self, reload_kernel_from_src):
        if candidate(self):
            return None
        return original_recheck(self, reload_kernel_from_src)

    @functools.wraps(original_precompile)
    def precompile(
        self, warm_cache_only=False, reload_kernel=None, static_triton_bundle_key=None
    ):
        selected = resolve(self, reload_kernel)
        if selected:
            fingerprint, role = selected
            rank = int(self.device_props.index)
            warps = expected_warps[role][rank]
            self.configs = [
                Config(
                    {"XBLOCK": policy["xblock"], "R0_BLOCK": policy["rblock"]},
                    num_warps=warps,
                    num_stages=policy["stages"],
                )
            ]
            self.compile_results = [
                x for x in self.compile_results if matches(x.config, warps)
            ]
            self.launchers = []
            self.inductor_meta["dynamic_scale_rblock"] = False
            self.inductor_meta["coordinate_descent_tuning"] = False
            self.autotune_cache_info = dict(
                autotune_cache_state="DSV4_RMS4096_PINNED",
                num_warps=warps,
                XBLOCK=policy["xblock"],
                R0_BLOCK=policy["rblock"],
            )
        result = original_precompile(
            self,
            warm_cache_only=warm_cache_only,
            reload_kernel=reload_kernel,
            static_triton_bundle_key=static_triton_bundle_key,
        )
        if selected:
            assert len(self.compile_results) == 1 and all(
                matches(x.config, warps) for x in self.compile_results
            )
            if not warm_cache_only:
                assert len(self.launchers) == 1 and all(
                    matches(x.config, warps) for x in self.launchers
                )
            key = (os.getpid(), rank, role, warps, bool(warm_cache_only))
            if key not in _seen:
                _seen.add(key)
                folder.mkdir(parents=True, exist_ok=True)
                record = {
                    "pid": os.getpid(),
                    "rank": rank,
                    "role": role,
                    "body_sha256": fingerprint,
                    "kernel": self.fn.__name__,
                    "num_warps": warps,
                    "warm_cache_only": bool(warm_cache_only),
                    "XBLOCK": policy["xblock"],
                    "R0_BLOCK": policy["rblock"],
                    "num_stages": policy["stages"],
                }
                with (folder / f"rms_{os.getpid()}.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
        return result

    CachingAutotuner.precompile = precompile
    CachingAutotuner.recheck_autotune_cache = recheck
    CachingAutotuner._dsv4_semantic_rms_policy = True
