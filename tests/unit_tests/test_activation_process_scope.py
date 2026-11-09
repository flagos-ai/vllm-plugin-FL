# SPDX-License-Identifier: Apache-2.0
"""Fork/spawn scope tests through the production activation entry point.

An activation plan is process-local.  A spawned worker must re-run the plugin
registration and then activate the *real* GLM plan (not a no-op dummy), and a
forked worker that inherits an already-active plan must be idempotent for the
same model and refuse a different one.
"""

import multiprocessing as mp
from types import SimpleNamespace

import pytest


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
    from vllm_fl.models.glm5 import cache as kpool
    from vllm_fl.models.glm5.kpool import Glm5NextIndexerAttentionBackend

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
        # The actual capability check contributes portable MoE defaults.
        from vllm_fl.models.glm5 import provider

        provider._has_vllm_native_extension = lambda: False
        provider._has_nvidia_reference_kernels.cache_clear()
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
    try:
        assert parent.poll(timeout), "child process timed out"
        try:
            result = parent.recv()
        except EOFError as error:
            process.join(timeout)
            raise AssertionError(
                f"child exited without a result (exitcode={process.exitcode})"
            ) from error
        process.join(timeout)
        assert process.exitcode == 0, f"child exited with {process.exitcode}"
        return result
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout)


@pytest.mark.gpu
def test_spawn_worker_activates_real_plan_and_policy():
    result = _run(mp.get_context("spawn"), _child_spawn_activate)
    assert "error" not in result, result
    assert result["activated"] is True
    assert result["active"] == "glm5_next"
    assert result["indexer"] is False
    assert result["private_indexer"] is True
    assert result["kpool_bound"] is True
    assert result["order"] == ["flagos", "reference"]


def _child_fork_after_activation(conn):
    try:
        from vllm_fl.models.glm5 import provider

        provider._has_vllm_native_extension = lambda: False
        provider._has_nvidia_reference_kernels.cache_clear()
        import vllm_fl

        vllm_fl.register_model()
        from vllm_fl.activation import activate_for_model, get_active_plan

        assert activate_for_model(_glm_config()) is not None
        assert get_active_plan().name == "glm5_next"
        result = _run(mp.get_context("fork"), _child_fork_rebind)
        result["parent_active"] = get_active_plan().name
        result["parent_order"] = _probe_state()["order"]
        conn.send(result)
    except Exception as error:
        conn.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        conn.close()


@pytest.mark.gpu
def test_fork_after_activation_is_idempotent_and_model_scoped():
    # Fork the real activated plan in a fresh parent, without inheriting LLVM
    # compiler thread pools initialized by unrelated earlier GPU tests.
    result = _run(mp.get_context("spawn"), _child_fork_after_activation)
    assert "error" not in result, result
    assert result["inherited"]["active"] == "glm5_next"
    assert result["inherited"]["order"] == ["flagos", "reference"]
    assert result["same_is_none"] is False
    assert result["after"]["kpool_bound"] is True
    assert result["after"]["indexer"] is False
    assert result["after"]["private_indexer"] is True
    assert result["conflict"] is True
    assert result["parent_active"] == "glm5_next"
    assert result["parent_order"] == ["flagos", "reference"]
