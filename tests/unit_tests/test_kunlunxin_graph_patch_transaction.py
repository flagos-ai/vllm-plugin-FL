# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

import pytest

from vllm.platforms import current_platform

import vllm_fl.dispatch.backends.vendor.kunlunxin.patch as patches

pytestmark = pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "kunlunxin",
    reason="Kunlunxin graph patch installation",
)


def _targets():
    import vllm.compilation.breakable_cudagraph as breakable
    import vllm.v1.worker.gpu_model_runner as gpu
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    import vllm_fl.worker.model_runner as fl
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinAttentionBackendImpl,
    )
    from vllm_fl.distributed.communicator import CommunicatorFL

    return [
        (CudagraphDispatcher, "dispatch"),
        (breakable, "is_breakable_cudagraph_enabled"),
        (gpu, "is_breakable_cudagraph_enabled"),
        (fl, "is_breakable_cudagraph_enabled"),
        (breakable.BreakableCUDAGraphWrapper, "__init__"),
        (breakable.BreakableCUDAGraphWrapper, "__call__"),
        (CommunicatorFL, "all_reduce"),
        (CommunicatorFL, "all_gather"),
        (KunlunxinAttentionBackendImpl, "forward"),
        (QwenGatedDeltaNetAttention, "_forward_core"),
    ]


@pytest.mark.parametrize("failure", ["missing_api", "late_prepare", "late_publish"])
@pytest.mark.parametrize("inherited", [False, True])
def test_dependent_graph_patches_restore_objects_and_retry(
    monkeypatch, failure, inherited
):
    import builtins
    import inspect
    from functools import wraps

    import vllm.compilation.breakable_cudagraph as breakable

    targets = _targets()
    # Register restoration before the real patch installer mutates attributes.
    for owner, name in targets:
        original = inspect.unwrap(getattr(owner, name))

        def wrap_original(function):
            @wraps(function)
            def other_owner(*args, **kwargs):
                return function(*args, **kwargs)

            return other_owner

        monkeypatch.setattr(owner, name, wrap_original(original))
    if inherited:
        for owner, name in targets:
            if name == "all_gather":
                monkeypatch.delattr(owner, name)
    originals = [
        (owner, name, getattr(owner, name), name in vars(owner))
        for owner, name in targets
    ]
    monkeypatch.setattr(patches, "_patches_applied", False)
    monkeypatch.setattr(patches, "_patches_initializing", False)
    monkeypatch.setattr(
        patches, "_apply_kunlunxin_patches", patches.patch_graph_adaptations
    )

    with monkeypatch.context() as injection:
        if failure == "missing_api":
            injection.delattr(breakable, "BreakableCUDAGraphCapture")
        elif failure == "late_prepare":

            def fail_prepare(replacements):
                assert replacements
                raise RuntimeError("injected late preparation failure")

            injection.setattr(patches, "patch_eager_all_gather", fail_prepare)
        else:
            count = 0

            def fail_publish(owner, name, replacement):
                nonlocal count
                count += 1
                if count == len(targets):
                    raise RuntimeError("injected publication failure")
                builtins.setattr(owner, name, replacement)

            injection.setattr(patches, "setattr", fail_publish, raising=False)

        with pytest.raises(RuntimeError, match="Kunlunxin graph adaptation"):
            patches.apply_kunlunxin_patches()
        assert not patches._patches_applied
        assert not patches._patches_initializing
        for owner, name, original, owned in originals:
            assert getattr(owner, name) is original
            assert (name in vars(owner)) == owned

    patches.apply_kunlunxin_patches()
    assert patches._patches_applied
    assert not patches._patches_initializing


def test_reentry_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(patches, "_patches_applied", False)
    monkeypatch.setattr(patches, "_patches_initializing", False)
    calls = []

    def initialize():
        calls.append(True)
        patches.apply_kunlunxin_patches()
        assert not patches._patches_applied

    monkeypatch.setattr(patches, "_apply_kunlunxin_patches", initialize)
    patches.apply_kunlunxin_patches()
    assert calls == [True]
    assert patches._patches_applied


def test_graph_retry_does_not_repeat_legacy_source_replacements(monkeypatch):
    monkeypatch.setattr(patches, "_patches_applied", False)
    monkeypatch.setattr(patches, "_patches_initializing", False)
    monkeypatch.setattr(patches, "_legacy_patches_applied", False)
    legacy_calls, graph_calls = [], []
    monkeypatch.setattr(
        patches, "_apply_legacy_kunlunxin_patches", lambda: legacy_calls.append(True)
    )

    def graph():
        graph_calls.append(True)
        if len(graph_calls) == 1:
            raise RuntimeError("preflight failed")

    monkeypatch.setattr(patches, "patch_graph_adaptations", graph)
    with pytest.raises(RuntimeError, match="preflight failed"):
        patches.apply_kunlunxin_patches()
    assert patches._legacy_patches_applied and not patches._patches_applied
    patches.apply_kunlunxin_patches()
    assert patches._patches_applied
    assert len(legacy_calls) == 1 and len(graph_calls) == 2
