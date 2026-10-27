# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for TXDA configuration, independent of vendor libraries."""

import __future__

import ast
import os
import runpy
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_function(path, name, namespace, *, class_name=None):
    source = _ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = tree.body
    if class_name is not None:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    result = namespace[name]
    return result.__func__ if isinstance(result, classmethod) else result


@pytest.mark.parametrize("requested", [None, "0", "1"])
def test_txda_registration_defaults_to_fl_runner_and_preserves_explicit_choice(
    monkeypatch, requested
):
    fake_utils = ModuleType("vllm_fl.utils")
    fake_utils.DeviceInfo = lambda: SimpleNamespace(vendor_name="tsingmicro")
    monkeypatch.setitem(sys.modules, "vllm_fl.utils", fake_utils)
    if requested is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", requested)
    # register() also selects a multiprocessing default. Keep it test-local.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    namespace = {"os": os, "logger": Mock()}
    for name in (
        "_init_vendor_device",
        "_arm_cpu_platform",
        "_patch_custom_ops",
        "_patch_flash_attn_import",
        "_get_op_config",
    ):
        namespace[name] = lambda: None
    register = _load_function("vllm_fl/__init__.py", "register", namespace)

    assert register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == (
        "0" if requested is None else requested
    )


class _CompilationMode(IntEnum):
    NONE = 0
    STOCK_TORCH_COMPILE = 1
    DYNAMO_AS_IS = 2
    VLLM_COMPILE = 3


class _GraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3
    FULL_DECODE_ONLY = 4


@pytest.fixture
def configure_platform(monkeypatch):
    for name in (
        "TXDA_LAUNCH_VIA_GGL",
        "TXDA_LAUNCH_KERNEL_SYNC",
        "VLLM_USE_BREAKABLE_CUDAGRAPH",
    ):
        monkeypatch.setenv(name, os.environ.get(name, "unchanged"))
    txda_partition = ModuleType("vllm_fl.compilation.txda_partition")
    txda_partition.install_txda_graph_partition = Mock()
    monkeypatch.setitem(sys.modules, txda_partition.__name__, txda_partition)
    config_module = ModuleType("vllm.config")
    config_module.CUDAGraphMode = _GraphMode
    compilation_module = ModuleType("vllm.config.compilation")
    compilation_module.CompilationMode = _CompilationMode
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", compilation_module)
    return _load_function(
        "vllm_fl/platform.py",
        "check_and_update_config",
        {"logger": Mock(), "os": os},
        class_name="PlatformFL",
    )


def _config(
    mode=_CompilationMode.NONE, graph=_GraphMode.NONE, *, tensor_parallel_size=1
):
    return SimpleNamespace(
        use_v2_model_runner=False,
        model_config=None,
        parallel_config=SimpleNamespace(
            worker_cls=None,
            all2all_backend=None,
            data_parallel_size=1,
            tensor_parallel_size=tensor_parallel_size,
        ),
        scheduler_config=SimpleNamespace(),
        cache_config=SimpleNamespace(block_size=None),
        compilation_config=SimpleNamespace(
            mode=mode,
            cudagraph_mode=graph,
            compile_sizes=None,
            backend="eager",
            splitting_ops=None,
            _attention_ops=[
                "vllm::unified_attention_with_output",
                "vllm::qwen_gdn_attention_core",
            ],
            pass_config=SimpleNamespace(
                fuse_norm_quant=True,
                fuse_act_quant=True,
                fuse_attn_quant=True,
                enable_sp=True,
                fuse_gemm_comms=True,
                eliminate_noops=True,
            ),
        ),
    )


def test_txda_rejects_explicit_upstream_v2_runner(configure_platform):
    platform = SimpleNamespace(device_type="txda", vendor_name="tsingmicro")
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        configure_platform(platform, SimpleNamespace(use_v2_model_runner=True))


@pytest.mark.parametrize(
    "graph", [mode for mode in _GraphMode if mode != _GraphMode.NONE]
)
@pytest.mark.parametrize("mode", [_CompilationMode.NONE, _CompilationMode.VLLM_COMPILE])
@pytest.mark.parametrize("tensor_parallel_size", [1, 8])
@pytest.mark.parametrize("data_parallel_size", [1, 8])
def test_txda_rejects_all_graph_modes_without_launch_side_effects(
    configure_platform,
    monkeypatch,
    graph,
    mode,
    tensor_parallel_size,
    data_parallel_size,
):
    config = _config(mode, graph, tensor_parallel_size=tensor_parallel_size)
    config.parallel_config.data_parallel_size = data_parallel_size
    config.parallel_config.world_size_across_dp = (
        tensor_parallel_size * data_parallel_size
    )
    platform = SimpleNamespace(
        device_type="txda", vendor_name="tsingmicro", dist_backend="flagcx"
    )
    monkeypatch.setenv("TXDA_LAUNCH_VIA_GGL", "existing-ggl")
    monkeypatch.setenv("TXDA_LAUNCH_KERNEL_SYNC", "existing-sync")
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    before = dict(os.environ)
    partition = sys.modules["vllm_fl.compilation.txda_partition"]

    with pytest.raises(ValueError, match="graph mode is not supported") as exc:
        configure_platform(platform, config)

    assert "--enforce-eager" in str(exc.value)
    assert config.compilation_config.cudagraph_mode == graph
    assert dict(os.environ) == before
    partition.install_txda_graph_partition.assert_not_called()


@pytest.mark.parametrize(
    "mode", [mode for mode in _CompilationMode if mode != _CompilationMode.NONE]
)
@pytest.mark.parametrize("backend", ["", "eager", "inductor"])
def test_txda_requires_eager_even_when_graphs_are_disabled(
    configure_platform, mode, backend
):
    config = _config(mode)
    config.compilation_config.backend = backend
    before = dict(os.environ)
    with pytest.raises(ValueError, match="compilation must be disabled") as exc:
        configure_platform(
            SimpleNamespace(device_type="txda", vendor_name="tsingmicro"), config
        )
    assert "--enforce-eager" in str(exc.value)
    assert config.compilation_config.mode == mode
    assert config.compilation_config.backend == backend
    assert dict(os.environ) == before
    partition = sys.modules["vllm_fl.compilation.txda_partition"]
    partition.install_txda_graph_partition.assert_not_called()


@pytest.mark.parametrize("tensor_parallel_size", [1, 8])
def test_txda_eager_preserves_launch_environment(
    configure_platform, tensor_parallel_size
):
    before = dict(os.environ)
    config = _config(tensor_parallel_size=tensor_parallel_size)
    configure_platform(
        SimpleNamespace(
            device_type="txda", vendor_name="tsingmicro", dist_backend="flagcx"
        ),
        config,
    )
    assert config.compilation_config.cudagraph_mode == _GraphMode.NONE
    assert config.compilation_config.mode == _CompilationMode.NONE
    assert dict(os.environ) == before
    partition = sys.modules["vllm_fl.compilation.txda_partition"]
    partition.install_txda_graph_partition.assert_not_called()


@pytest.mark.parametrize(
    "device,vendor,expected",
    [("txda", "tsingmicro", False), ("cuda", "nvidia", True), ("npu", "ascend", True)],
)
def test_txda_reports_no_static_graph_support(device, vendor, expected):
    method = _load_function(
        "vllm_fl/platform.py", "support_static_graph_mode", {}, class_name="PlatformFL"
    )
    assert method(SimpleNamespace(device_type=device, vendor_name=vendor)) is expected


def test_txda_can_import_shared_graph_module_without_vendor_graph_api(monkeypatch):
    modules = {
        "torch": {"Tensor": object},
        "vllm.envs": {},
        "vllm.compilation.counter": {"compilation_counter": SimpleNamespace()},
        "vllm.compilation.monitor": {"validate_cudagraph_capturing_enabled": Mock()},
        "vllm.config": {"CUDAGraphMode": _GraphMode, "VllmConfig": object},
        "vllm.distributed.device_communicators.pynccl_allocator": {
            "set_graph_pool_id": Mock()
        },
        "vllm.forward_context": {
            "BatchDescriptor": object,
            "get_forward_context": Mock(),
            "is_forward_context_available": Mock(),
        },
        "vllm.logger": {"init_logger": Mock(return_value=Mock())},
        "vllm.platforms": {"current_platform": SimpleNamespace(device_type="txda")},
        "vllm.utils.torch_utils": {"current_stream": Mock()},
        "vllm.compilation.cuda_graph": {"CUDAGraphStat": object},
    }
    installed = {}
    for name, attributes in modules.items():
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            key = ".".join(parts[:index])
            if key not in installed:
                module = ModuleType(key)
                module.__path__ = []
                installed[key] = module
                monkeypatch.setitem(sys.modules, key, module)
                if index > 1:
                    setattr(
                        installed[".".join(parts[: index - 1])],
                        parts[index - 1],
                        module,
                    )
        installed[name].__dict__.update(attributes)
    # Eager ModelRunnerFL imports this shared module even when TXDA has no
    # native graph API. Only the shared import/type dispatch is exercised.
    module = runpy.run_path(str(_ROOT / "vllm_fl/compilation/graph.py"))
    assert module["Graph"].graph is None
    assert "GraphWrapper" in module


def test_txda_accepts_eager_configuration(configure_platform):
    config = _config()
    configure_platform(
        SimpleNamespace(device_type="txda", vendor_name="tsingmicro"), config
    )
    assert config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"
    assert config.cache_config.block_size == 16


@pytest.mark.parametrize("speculative_config", [SimpleNamespace(), {}])
def test_txda_rejects_speculative_decoding(configure_platform, speculative_config):
    config = _config()
    config.speculative_config = speculative_config
    with pytest.raises(ValueError, match="Remove --speculative-config"):
        configure_platform(
            SimpleNamespace(device_type="txda", vendor_name="tsingmicro"), config
        )


def test_txda_accepts_explicitly_disabled_speculative_decoding(configure_platform):
    config = _config()
    config.speculative_config = None
    configure_platform(
        SimpleNamespace(device_type="txda", vendor_name="tsingmicro"), config
    )
    assert config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"


def test_speculative_guard_does_not_disable_other_backends(configure_platform):
    config = _config()
    config.speculative_config = SimpleNamespace()
    configure_platform(
        SimpleNamespace(device_type="cuda", vendor_name="nvidia"), config
    )
    assert config.speculative_config is not None


def test_txda_guard_does_not_disable_other_backends(configure_platform):
    config = _config(_CompilationMode.VLLM_COMPILE, _GraphMode.PIECEWISE)
    configure_platform(
        SimpleNamespace(device_type="cuda", vendor_name="nvidia"), config
    )
    assert config.compilation_config.mode == _CompilationMode.VLLM_COMPILE
    assert config.compilation_config.cudagraph_mode == _GraphMode.PIECEWISE


@pytest.fixture
def txda_memory_budget(monkeypatch):
    monkeypatch.delenv("VLLM_TXDA_KV_CACHE_SIZE", raising=False)
    namespace = {
        "current_platform": SimpleNamespace(device_type="txda"),
        "managed_inference_mode": lambda: lambda fn: fn,
        "logger": Mock(),
        "os": os,
        "GiB_bytes": 1024**3,
        "maybe_apply_startup_plan": Mock(
            side_effect=AssertionError("TXDA must not profile")
        ),
    }
    determine = _load_function(
        "vllm_fl/worker/worker.py",
        "determine_available_memory",
        namespace,
        class_name="WorkerFL",
    )

    def budget(explicit=None):
        worker = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_memory_bytes=explicit)
        )
        return determine(worker)

    return budget


@pytest.mark.parametrize("legacy", ["1024", "invalid", "0", "-10"])
def test_txda_explicit_budget_wins_over_legacy_override(
    txda_memory_budget, monkeypatch, legacy
):
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", legacy)
    assert txda_memory_budget(4096) == 4096


def test_txda_legacy_budget_and_default_remain_available(
    txda_memory_budget, monkeypatch
):
    assert txda_memory_budget() == 20 * 1024**3
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", "2097152")
    assert txda_memory_budget() == 2097152


@pytest.mark.parametrize("legacy", ["0", "-1", "1.5", "abc", ""])
def test_txda_rejects_invalid_legacy_budgets(txda_memory_budget, monkeypatch, legacy):
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", legacy)
    with pytest.raises(ValueError, match="positive"):
        txda_memory_budget()


@pytest.mark.parametrize("explicit", [0, -1])
def test_txda_rejects_nonpositive_explicit_budgets(txda_memory_budget, explicit):
    with pytest.raises(ValueError, match="positive"):
        txda_memory_budget(explicit)


def test_txda_visibility_and_runtime_device_mapping(monkeypatch):
    utils = runpy.run_path(str(_ROOT / "vllm_fl/utils.py"))
    monkeypatch.delenv("FLAGOS_DEVICE_CONTROL_ENV_VAR", raising=False)
    assert utils["get_device_type"]("tsingmicro") == "txda"
    assert utils["get_device_name"]("tsingmicro") == "txda"
    assert utils["get_device_control_env_var"]("tsingmicro") == "TXDA_VISIBLE_DEVICES"
    monkeypatch.setenv("FLAGOS_DEVICE_CONTROL_ENV_VAR", "CUSTOM_VISIBLE_DEVICES")
    assert utils["get_device_control_env_var"]("tsingmicro") == "CUSTOM_VISIBLE_DEVICES"


def test_txda_uses_tccl_without_affecting_cuda():
    source = _ROOT / "vllm_fl/platform.py"
    node = next(
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "dist_backend_dict"
            for target in node.targets
        )
    )
    backends = ast.literal_eval(node.value)
    assert backends["txda"] == "tccl"
    assert backends["cuda"] == "nccl"


@pytest.fixture
def txda_test_device_utils(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False, device_count=lambda: 0
    )
    fake_torch.txda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 8)
    fake_torch.device = Mock(side_effect=lambda value: value)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_txda", ModuleType("torch_txda"))
    # An installed GCU SDK must not be reused with this fixture's fake torch.
    monkeypatch.setitem(sys.modules, "torch_gcu", None)
    monkeypatch.setitem(sys.modules, "torch_npu", None)

    def load(backend=""):
        monkeypatch.setenv("FL_BACKEND", backend)
        return runpy.run_path(str(_ROOT / "tests/utils/device_utils.py"))

    return load, fake_torch


@pytest.mark.parametrize("backend", ["", "tsingmicro", "txda"])
def test_test_helpers_select_txda_device_and_count(txda_test_device_utils, backend):
    load, fake_torch = txda_test_device_utils
    helpers = load(backend)
    assert helpers["get_backend"]() == "tsingmicro"
    assert helpers["is_accelerator_available"]() is True
    assert helpers["get_device_count"]() == 8
    assert helpers["get_device"](3) == "txda:3"
    fake_torch.device.assert_called_once_with("txda:3")


def test_test_helpers_prefer_txda_over_cuda_compatibility_alias(txda_test_device_utils):
    load, fake_torch = txda_test_device_utils
    fake_torch.cuda.is_available = lambda: True
    assert load()["get_backend"]() == "tsingmicro"


def test_test_helpers_preserve_explicit_other_backend(txda_test_device_utils):
    load, fake_torch = txda_test_device_utils
    helpers = load("nvidia")
    assert helpers["get_backend"]() == "nvidia"
    assert helpers["get_device"](1) == "cuda:1"
    fake_torch.device.assert_called_once_with("cuda:1")


def test_test_helpers_report_missing_txda_runtime(txda_test_device_utils, monkeypatch):
    load, _ = txda_test_device_utils
    monkeypatch.setitem(sys.modules, "torch_txda", None)
    helpers = load("tsingmicro")
    assert helpers["is_accelerator_available"]() is False
    assert helpers["get_device_count"]() == 0
    assert load()["get_backend"]() == "cpu"


def test_test_helpers_report_unavailable_txda_hardware(txda_test_device_utils):
    load, fake_torch = txda_test_device_utils
    fake_torch.txda.is_available = lambda: False
    fake_torch.txda.device_count = lambda: 0
    helpers = load("tsingmicro")
    assert helpers["is_accelerator_available"]() is False
    assert helpers["get_device_count"]() == 0
    assert load()["get_backend"]() == "cpu"


@pytest.mark.parametrize(
    "case,model",
    [
        ("27b_tp8_eager", "Qwen3.6-27B"),
        ("35b_a3b_tp8_eager", "Qwen3.6-35B-A3B"),
    ],
)
def test_txda_smoke_config_preserves_engine_and_serving_limits(case, model):
    from tests.utils.model_config import ModelConfig
    from tests.utils.platform_config import PlatformConfig

    platform = PlatformConfig.load("tsingmicro", "tx8110")
    cases = platform.get_e2e_tests().get_cases()
    assert {item["task"] for item in cases if item["case"] == case} == {
        "inference",
        "serving",
    }
    assert platform.env_defaults["FL_BACKEND"] == "tsingmicro"
    assert platform.env_defaults["TXDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    config = ModelConfig.load("qwen3_6", case, platform="tsingmicro", device="tx8110")
    engine = config.engine_kwargs()
    assert engine["model"] == f"/models/Qwen/{model}"
    assert engine["tensor_parallel_size"] == 8
    assert engine["max_model_len"] == 1024
    assert engine["max_num_seqs"] == 2
    assert engine["kv_cache_memory_bytes"] == 134217728
    assert engine["enforce_eager"] is True
    assert engine["trust_remote_code"] is True
    args = config.serve_args()
    assert args[args.index("--kv-cache-memory-bytes") + 1] == "134217728"
    assert "--enforce-eager" in args
    assert "--trust-remote-code" in args


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "world_size_across_dp",
    ],
)
def test_txda_multi_device_requires_flagcx(configure_platform, field):
    config = _config()
    setattr(config.parallel_config, field, 8)
    platform = SimpleNamespace(
        device_type="txda", vendor_name="tsingmicro", dist_backend="tccl"
    )
    with pytest.raises(ValueError, match="requires FlagCX"):
        configure_platform(platform, config)


def test_txda_tp8_accepts_flagcx_and_disables_cuda_all_reduce(configure_platform):
    config = _config()
    config.parallel_config.tensor_parallel_size = 8
    config.parallel_config.world_size_across_dp = 8
    config.parallel_config.disable_custom_all_reduce = False
    platform = SimpleNamespace(
        device_type="txda", vendor_name="tsingmicro", dist_backend="flagcx"
    )
    configure_platform(platform, config)
    assert config.parallel_config.disable_custom_all_reduce is True


@pytest.mark.parametrize(
    "field", ["prefill_context_parallel_size", "decode_context_parallel_size"]
)
def test_txda_rejects_context_parallelism_even_with_flagcx(configure_platform, field):
    config = _config()
    setattr(config.parallel_config, field, 2)
    platform = SimpleNamespace(
        device_type="txda", vendor_name="tsingmicro", dist_backend="flagcx"
    )
    with pytest.raises(ValueError, match="does not support context parallelism"):
        configure_platform(platform, config)


@pytest.mark.parametrize(
    "device, mode, expected",
    [
        ("txda", _CompilationMode.VLLM_COMPILE, 0),
        ("txda", _CompilationMode.NONE, 0),
        ("npu", _CompilationMode.VLLM_COMPILE, 1),
        ("npu", _CompilationMode.NONE, 0),
        ("cuda", _CompilationMode.VLLM_COMPILE, 0),
    ],
)
@pytest.mark.parametrize("graph_mode", [_GraphMode.NONE, _GraphMode.PIECEWISE])
def test_runner_resolves_cached_ops_before_compilation(
    monkeypatch, device, mode, expected, graph_mode
):
    source = _ROOT / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerFL"
    )
    load = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    warmup = next(
        n
        for n in load.body
        if isinstance(n, ast.If)
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "vllm_fl.dispatch"
            and any(alias.name == "warmup_cached_ops" for alias in child.names)
            for child in ast.walk(n)
        )
    )
    first_compile = next(
        child
        for child in ast.walk(load)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "compile"
    )
    assert warmup.end_lineno < first_compile.lineno
    dispatch = ModuleType("vllm_fl.dispatch")
    dispatch.warmup_cached_ops = Mock()
    monkeypatch.setitem(sys.modules, dispatch.__name__, dispatch)
    exec(
        compile(ast.Module(body=[warmup], type_ignores=[]), str(source), "exec"),
        {
            "current_platform": SimpleNamespace(device_type=device),
            "self": SimpleNamespace(
                compilation_config=SimpleNamespace(mode=mode, cudagraph_mode=graph_mode)
            ),
            "CompilationMode": _CompilationMode,
            "CUDAGraphMode": _GraphMode,
        },
    )
    assert dispatch.warmup_cached_ops.call_count == expected


@pytest.mark.parametrize(
    "device_type,vendor_name,expected",
    [
        ("txda", "tsingmicro", True),
        ("cuda", "nvidia", True),
        ("cuda", "thead", True),
        ("cuda", "iluvatar", True),
        ("cuda", "metax", False),
        ("npu", "ascend", False),
        ("musa", "mthreads", False),
    ],
)
def test_txda_collectives_use_opaque_ops_without_changing_other_platforms(
    device_type, vendor_name, expected
):
    use_custom_ops = _load_function(
        "vllm_fl/platform.py",
        "use_custom_op_collectives",
        {},
        class_name="PlatformFL",
    )
    platform = SimpleNamespace(device_type=device_type, vendor_name=vendor_name)
    assert use_custom_ops(platform) is expected
