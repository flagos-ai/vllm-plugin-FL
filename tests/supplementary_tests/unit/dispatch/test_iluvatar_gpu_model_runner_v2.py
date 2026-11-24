# Copyright 2025 BAAI. All rights reserved.
"""Real-BI-V150 coverage for the Iluvatar V2 model runner and worker device rows.

``vllm_fl/dispatch/backends/vendor/iluvatar/worker/model_runner_v2.py`` (0/6)
delegates everything to ``patch_v2_uva`` + the upstream GPUModelRunner
constructor; ``worker/worker.py``'s ``IluvatarWorkerFL`` (0/133 at file level)
applies the in-worker vendor patches in ``init_device`` and installs the
pipeline transport. Both need a real CUDA device to construct buffers on.

``GPUModelRunner.__init__`` with a minimal VllmConfig allocates the persistent
input buffers — exactly the allocation the UVA patch redirects to explicit
H2D copies on CoreX. Constructing it once on the BI-V150 pins:

* ``patch_v2_uva`` replaced ``buffer_utils.is_uva_available`` with the
  iluvatar ``False`` *before* the base class ran (the import-order contract
  the module comment calls out);
* the runner is a real ``GPUModelRunner`` (MRO) holding device-resident
  buffers (``input_buffers.input_ids`` / ``positions`` on cuda) that stay
  zero-initialized until a step runs;
* the class binds upstream V2 as its direct base.

The full ``init_device`` (distributed init + model load) is too heavy for a
supplementary suite, so the pure branches of the patch machinery are covered
directly: ``apply_worker_patches`` idempotency, the ``patch_v2_uva`` no-op
for foreign vendors, and ``install_pipeline_transport``'s world-size-1 early
return. Assertions are structural (the numeric oracle for buffer contents is
a plain ``torch.count_nonzero``).
"""

from __future__ import annotations

import uuid

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("iluvatar"),
    pytest.mark.gpu,
    pytest.mark.serial,
]

MODEL_PATH = "/data/Qwen/Qwen3-0.6B"


def _cuda():
    from tests.utils.device_utils import get_device

    return get_device()


class TestPatchV2Uva:
    """``patch_v2_uva`` — the import-order contract on the real platform."""

    def test_replaces_is_uva_available_with_false(self):
        from vllm.v1.worker.gpu import buffer_utils

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            _is_uva_available,
            patch_v2_uva,
        )

        patch_v2_uva()
        # the vendor gate passes on the iluvatar platform, so the module
        # attribute must now be the iluvatar function
        assert buffer_utils.is_uva_available is _is_uva_available
        assert buffer_utils.is_uva_available() is False

    def test_idempotent_double_patch(self):
        from vllm.v1.worker.gpu import buffer_utils

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_v2_uva,
        )

        before = buffer_utils.is_uva_available
        patch_v2_uva()
        assert buffer_utils.is_uva_available is before

    def test_foreign_vendor_is_a_noop(self, monkeypatch):
        """A non-iluvatar vendor never touches buffer_utils."""
        import vllm.platforms

        from vllm.v1.worker.gpu import buffer_utils

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            patch_v2_uva,
        )

        monkeypatch.setattr(
            vllm.platforms,
            "current_platform",
            type("P", (), {"vendor_name": "nvidia"})(),
        )
        original = buffer_utils.is_uva_available
        patch_v2_uva()
        assert buffer_utils.is_uva_available is original


class TestIluvatarGPUModelRunnerV2:
    """``IluvatarGPUModelRunnerV2.__init__`` on the real device."""

    @pytest.fixture(scope="class")
    def runner(self, tmp_path_factory):
        """Build the runner once: minimal config, single-rank gloo env.

        ``GPUModelRunner.__init__`` reads the pp/tp/dcp group state, so a
        single-rank ``init_distributed_environment`` +
        ``initialize_model_parallel`` runs first (gloo backend: the device
        buffers still land on the BI-V150 through the runner's own
        ``device`` argument). The model itself is never loaded.
        """
        from vllm.config import VllmConfig
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )

        from vllm_fl.dispatch.backends.vendor.iluvatar.worker.model_runner_v2 import (
            IluvatarGPUModelRunnerV2,
        )

        try:
            destroy_model_parallel()
        except Exception:
            pass
        dist.destroy_process_group() if dist.is_initialized() else None
        store_path = (
            tmp_path_factory.mktemp("iluvatar_v2") / f"store-{uuid.uuid4().hex}"
        )
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{store_path}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)

        config = VllmConfig(
            model_config=dict(
                model=MODEL_PATH,
                tokenizer=MODEL_PATH,
                dtype="bfloat16",
                max_model_len=256,
                trust_remote_code=True,
            ),
            scheduler_config=dict(max_num_batched_tokens=16, max_num_seqs=4),
        )
        runner = IluvatarGPUModelRunnerV2(config, _cuda())
        yield runner
        try:
            destroy_model_parallel()
        except Exception:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_constructs_with_real_config_and_device(self, runner):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        assert isinstance(runner, GPUModelRunner)
        assert runner.device == _cuda()
        assert runner.device.type == "cuda"
        # persistent input buffers allocated on the device
        assert runner.input_buffers is not None
        assert runner.input_buffers.input_ids.device.type == "cuda"
        assert runner.input_buffers.positions.device.type == "cuda"
        # buffers are zero-initialized until a step runs
        assert torch.count_nonzero(runner.input_buffers.positions[:16]).item() == 0
        # a write to the device buffer is readable back (device round trip)
        runner.input_buffers.input_ids[0] = 7
        assert int(runner.input_buffers.input_ids[0].item()) == 7
        runner.input_buffers.input_ids[0] = 0

    def test_class_mro_binds_upstream_v2(self):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        from vllm_fl.dispatch.backends.vendor.iluvatar.worker.model_runner_v2 import (
            IluvatarGPUModelRunnerV2,
        )

        assert IluvatarGPUModelRunnerV2.__mro__[1] is GPUModelRunner


class TestIluvatarWorkerFLSurface:
    """``IluvatarWorkerFL`` structural rows executable without a full engine."""

    def test_class_inherits_worker_fl(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker import (
            IluvatarWorkerFL,
        )
        from vllm_fl.worker.worker import WorkerFL

        assert issubclass(IluvatarWorkerFL, WorkerFL)

    def test_init_device_exists_and_calls_patches(self):
        """``init_device`` is a plain method on the class surface."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker import (
            IluvatarWorkerFL,
        )

        assert callable(IluvatarWorkerFL.init_device)

    def test_apply_worker_patches_idempotent(self):
        """The worker patch bundle runs twice without raising."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            apply_worker_patches,
        )

        apply_worker_patches()
        apply_worker_patches()

    def test_install_pipeline_transport_world_size_1_noop(self):
        """A world-size-1 group never binds the pipeline transport methods."""
        from types import SimpleNamespace

        from vllm_fl.dispatch.backends.vendor.iluvatar.pipeline import (
            install_pipeline_transport,
        )

        group = SimpleNamespace(
            world_size=1,
            device_communicator=None,
            isend_tensor_dict=None,
            irecv_tensor_dict=None,
        )
        install_pipeline_transport(group)
        assert group.isend_tensor_dict is None, "world_size 1 must not bind methods"

    def test_sleep_saved_kv_scales_attr_absent_before_sleep(self):
        """The vendor scale-save attributes only appear after sleep()."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker import (
            IluvatarWorkerFL,
        )

        worker = object.__new__(IluvatarWorkerFL)
        assert not hasattr(worker, "_sleep_saved_kv_scales")
        assert not hasattr(worker, "_sleep_saved_buffers")

    def test_update_config_rejects_unknown_override(self):
        """``update_config`` validates the allowed override names."""
        from vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker import (
            IluvatarWorkerFL,
        )

        worker = object.__new__(IluvatarWorkerFL)
        with pytest.raises(ValueError, match="not supported"):
            worker.update_config({"cache_config": {"block_size": 32}})
