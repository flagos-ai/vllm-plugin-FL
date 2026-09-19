# SPDX-License-Identifier: Apache-2.0
"""Fork/spawn scope tests through the production activation entry point.

An activation plan is process-local.  A spawned worker must re-run the plugin
registration and then activate the *real* GLM plan (not a no-op dummy), and a
forked worker that inherits an already-active plan must be idempotent for the
same model and refuse a different one.
"""

import multiprocessing as mp
import os
from types import SimpleNamespace


def _glm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm5_next_text", architectures=["Glm5NextForCausalLM"]
            ),
            hf_text_config=SimpleNamespace(model_type="glm5_next_text"),
            architectures=["Glm5NextForCausalLM"],
        )
    )


def _other_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_8_flash_next", architectures=["Qwen3Model"]
            ),
            hf_text_config=SimpleNamespace(model_type="qwen3_8_flash_next"),
            architectures=["Qwen3Model"],
        )
    )


def _probe_state():
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    from vllm_fl.activation import get_active_plan
    from vllm_fl.dispatch.policy import PolicyManager
    from vllm_fl.models.glm5_next_kpool import Glm5NextIndexerAttentionBackend
    from vllm_fl.patches import glm5_next_kpool_v024 as kpool

    active = get_active_plan()
    kpool_bound = all(
        patch.current() is not patch.pristine
        for patch in kpool.glm5_next_kpool_runtime_patches("probe@1")
    )
    return {
        "active": active.name if active is not None else None,
        "private_indexer": Glm5NextIndexerAttentionBackend.indexes_kv_by_block_stride(),
        "indexer": bool(
            DeepseekV32IndexerBackend.indexes_kv_by_block_stride.__func__(
                DeepseekV32IndexerBackend
            )
        ),
        "kpool_bound": kpool_bound,
        "order": PolicyManager.get_instance()
        .get_policy()
        .get_per_op_order("moe_align_block_size"),
    }


def _child_spawn_activate(conn):
    try:
        # Force the portable GLM provider so the plan contributes the
        # capability-required MoE defaults this test asserts on.
        os.environ["VLLM_FL_GLM5_PROVIDER"] = "flaggems"
        import vllm_fl

        # Production bootstrap: the general-plugin loader calls this entry
        # point in every spawned worker before the model is constructed.
        vllm_fl.register_model()
        from vllm_fl.activation import activate_for_model

        plan = activate_for_model(_glm_config())
        state = _probe_state()
        state["activated"] = plan is not None
        conn.send(state)
    except Exception as exc:  # noqa: BLE001 - surface child failure
        conn.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()


def _child_fork_rebind(conn):
    try:
        from vllm_fl.activation import ActivationConflict, activate_for_model

        inherited = _probe_state()
        # Same model: idempotent, no re-patch.
        same = activate_for_model(_glm_config())
        # Different model: must be rejected.
        conflict = None
        try:
            activate_for_model(_other_config())
        except ActivationConflict:
            conflict = True
        after = _probe_state()
        conn.send(
            {
                "inherited": inherited,
                "same_is_none": same is None,
                "conflict": conflict,
                "after": after,
            }
        )
    except Exception as exc:  # noqa: BLE001 - surface child failure
        conn.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()


def _run(ctx, target, timeout=180):
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=target, args=(child,))
    process.start()
    child.close()
    assert parent.poll(timeout), "child process timed out"
    result = parent.recv()
    parent.close()
    process.join(timeout)
    assert process.exitcode == 0, f"child exited with {process.exitcode}"
    return result


def test_spawn_worker_activates_real_plan_and_policy():
    result = _run(mp.get_context("spawn"), _child_spawn_activate)
    assert "error" not in result, result
    assert result["activated"] is True
    assert result["active"] == "glm5_next_v024"
    assert result["indexer"] is False
    assert result["private_indexer"] is True
    assert result["kpool_bound"] is True
    assert result["order"] == ["flagos", "reference"]


def _protect_real_activation_state(monkeypatch):
    """Snapshot every class/module attribute the real GLM activation patches."""
    from vllm import _custom_ops
    from vllm.model_executor.layers.mhc import (
        MHCFusedPostPreOp,
        MHCPostOp,
        MHCPreOp,
    )
    from vllm.v1.attention.backends.mla import indexer as indexer_backend
    from vllm.v1.worker import utils as worker_utils

    for mhc_cls in (MHCPreOp, MHCPostOp, MHCFusedPostPreOp):
        monkeypatch.setattr(mhc_cls, "forward_oot", mhc_cls.forward_oot)
    try:
        from vllm.model_executor.layers.activation import SiluAndMulWithClamp

        monkeypatch.setattr(
            SiluAndMulWithClamp, "forward_oot", SiluAndMulWithClamp.forward_oot
        )
    except Exception:
        pass
    monkeypatch.setattr(
        indexer_backend.DeepseekV32IndexerBackend,
        "indexes_kv_by_block_stride",
        indexer_backend.DeepseekV32IndexerBackend.indexes_kv_by_block_stride,
    )
    monkeypatch.setattr(
        worker_utils.AttentionGroup,
        "create_metadata_builders",
        worker_utils.AttentionGroup.create_metadata_builders,
    )
    monkeypatch.setattr(
        indexer_backend.DeepseekV32IndexerMetadataBuilder,
        "build",
        indexer_backend.DeepseekV32IndexerMetadataBuilder.build,
    )
    monkeypatch.setattr(
        worker_utils.KVBlockZeroer, "__init__", worker_utils.KVBlockZeroer.__init__
    )
    for name in ("concat_mla_q", "concat_and_cache_mla"):
        if hasattr(_custom_ops, name):
            monkeypatch.setattr(_custom_ops, name, getattr(_custom_ops, name))


def test_fork_after_activation_is_idempotent_and_model_scoped(monkeypatch):
    _protect_real_activation_state(monkeypatch)
    monkeypatch.setenv("VLLM_FL_GLM5_PROVIDER", "flaggems")
    from vllm_fl.kernels.glm5_next import provider

    provider.get_glm5_provider.cache_clear()
    import vllm_fl

    vllm_fl.register_model()
    from vllm_fl.activation import (
        activate_for_model,
        get_active_plan,
        reset_activation_for_tests,
    )

    # reset_activation_for_tests clears the provider registry, so re-run the
    # production registration before activating (mirrors a fresh process).
    reset_activation_for_tests()
    vllm_fl.register_model()
    try:
        assert activate_for_model(_glm_config()) is not None
        assert get_active_plan().name == "glm5_next_v024"

        result = _run(mp.get_context("fork"), _child_fork_rebind)
        assert "error" not in result, result
        assert result["inherited"]["active"] == "glm5_next_v024"
        assert result["inherited"]["order"] == ["flagos", "reference"]
        # Same plan in the child: idempotent, and state stays bound.
        assert result["same_is_none"] is False
        assert result["after"]["kpool_bound"] is True
        assert result["after"]["indexer"] is False
        assert result["after"]["private_indexer"] is True
        # A different model in the child is rejected.
        assert result["conflict"] is True
        # Parent process state is untouched by the child.
        assert get_active_plan().name == "glm5_next_v024"
        assert _probe_state()["order"] == ["flagos", "reference"]
    finally:
        reset_activation_for_tests()
        # Leave the session with the production providers registered.
        vllm_fl.register_model()
