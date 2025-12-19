import pytest
import torch
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def fake_vllm_config():
    cfg = MagicMock()

    cfg.model_config.trust_remote_code = False
    cfg.model_config.dtype = torch.float16
    cfg.model_config.seed = 0
    cfg.model_config.enforce_eager = True

    cfg.cache_config.gpu_memory_utilization = 0.5
    cfg.cache_config.kv_cache_memory_bytes = None

    cfg.parallel_config.world_size = 1
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.parallel_config.distributed_executor_backend = "ray"

    cfg.scheduler_config.max_num_seqs = 8
    cfg.scheduler_config.max_num_batched_tokens = 8

    cfg.compilation_config.compile_sizes = []

    return cfg


@pytest.fixture
def mock_model_runner():
    runner = MagicMock()
    runner.model_memory_usage = 1024 * 1024 * 100  # 100MB
    runner.model = MagicMock()
    runner.get_model.return_value = runner.model
    runner.get_supported_tasks.return_value = ()
    runner._dummy_run.return_value = None
    runner.capture_model.return_value = 0
    runner.is_pooling_model = False
    runner.lora_config = None
    runner.maybe_remove_all_loras.return_value = None
    runner.ensure_kv_transfer_shutdown.return_value = None
    return runner


@pytest.fixture
def worker_fl(cuda_available, fake_vllm_config, mock_model_runner):
    with patch(
        "vllm_fl.worker.worker.init_worker_distributed_environment"
    ), patch(
        "vllm_fl.worker.worker.ModelRunnerFL",
        return_value=mock_model_runner,
    ), patch(
        "vllm_fl.worker.worker.register_oot_ops"
    ), patch(
        "vllm_fl.worker.worker.flag_gems.enable"
    ), patch(
        "vllm_fl.worker.worker.report_usage_stats"
    ):
        from vllm_fl.worker.worker import WorkerFL

        worker = WorkerFL(
            vllm_config=fake_vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
            is_driver_worker=True,
        )

        worker.init_device()

        return worker


def test_init_device_cuda(worker_fl):
    assert worker_fl.device.type == "cuda"
    assert worker_fl.device.index == 0
    assert worker_fl.model_runner is not None


def test_initialize_cache(worker_fl):
    worker_fl.initialize_cache(num_gpu_blocks=4, num_cpu_blocks=2)
    assert worker_fl.cache_config.num_gpu_blocks == 4
    assert worker_fl.cache_config.num_cpu_blocks == 2


def test_get_model(worker_fl):
    model = worker_fl.get_model()
    assert model is worker_fl.model_runner.model


def test_execute_dummy_batch(worker_fl):
    worker_fl.execute_dummy_batch()
    worker_fl.model_runner._dummy_run.assert_called_once()


def test_supported_tasks(worker_fl):
    tasks = worker_fl.get_supported_tasks()
    assert tasks == ()


def test_shutdown(worker_fl):
    worker_fl.shutdown()
    worker_fl.model_runner.ensure_kv_transfer_shutdown.assert_called_once()

