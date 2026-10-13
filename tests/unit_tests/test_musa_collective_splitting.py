"""CPU contracts for production platform/configuration and FX split bodies.

The plugin and upstream modules are not imported: their actual selected AST
bodies execute with explicit configuration/compilation doubles and real CPU
torch/FX/dispatcher. These checks do not prove MUSA ABI or native graph safety.
"""

import ast
import builtins
import dataclasses
import enum
import importlib.util
import operator
import os
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import fx
from torch.fx._lazy_graph_module import _use_lazy_graph_module

SOURCE = Path(__file__).resolve().parents[2]
UPSTREAM = os.environ.get("VLLM_UPSTREAM_SOURCE_ROOT")
if UPSTREAM:
    UPSTREAM = Path(UPSTREAM) / "vllm"
else:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("upstream vLLM Python source is required", allow_module_level=True)
    UPSTREAM = Path(next(iter(spec.submodule_search_locations)))


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def node(path, name):
    return next(item for item in tree(path).body if getattr(item, "name", None) == name)


def execute(nodes, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<actual source bodies>", "exec"),
        namespace,
    )


LOGGER = SimpleNamespace(info=lambda *args: None, warning_once=lambda *args: None)
CONFIG_NS = {"enum": enum, "logger": LOGGER}
config_path = UPSTREAM / "config/compilation.py"
execute(
    [node(config_path, "CompilationMode"), node(config_path, "CUDAGraphMode")],
    CONFIG_NS,
)
Mode = CONFIG_NS["CompilationMode"]
GraphMode = CONFIG_NS["CUDAGraphMode"]
config_class = node(config_path, "CompilationConfig")
config_methods = [
    item
    for item in config_class.body
    if getattr(item, "name", None)
    in {
        "__post_init__",
        "set_splitting_ops_for_v1",
        "set_splitting_ops_for_attn_fusion",
        "splitting_ops_contain_attention",
        "is_attention_compiled_piecewise",
    }
    or (
        isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_attention_ops"
            for target in item.targets
        )
    )
    or (
        isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and item.target.id == "_attention_ops"
    )
]
execute(
    [
        ast.ClassDef(
            name="ConfigContract",
            bases=[],
            keywords=[],
            body=config_methods,
            decorator_list=[],
        )
    ],
    CONFIG_NS,
)
ConfigContract = CONFIG_NS["ConfigContract"]


def platform_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ("vllm.config", "vllm.config.compilation") and level == 0:
        return SimpleNamespace(CUDAGraphMode=GraphMode, CompilationMode=Mode)
    return builtins.__import__(name, globals, locals, fromlist, level)


class BasePlatformContract:
    backend_calls = []

    @classmethod
    def get_compile_backend(cls):
        BasePlatformContract.backend_calls.append(cls.device_type)
        return "inductor"


PLATFORM_NS = {
    "BasePlatformContract": BasePlatformContract,
    "logger": LOGGER,
    "__builtins__": {**vars(builtins), "__import__": platform_import},
}
platform_class = node(SOURCE / "vllm_fl/platform.py", "PlatformFL")
execute(
    [
        ast.ClassDef(
            name="PlatformContract",
            bases=[ast.Name(id="BasePlatformContract", ctx=ast.Load())],
            keywords=[],
            decorator_list=[],
            body=[
                item
                for item in platform_class.body
                if getattr(item, "name", None)
                in {
                    "check_and_update_config",
                    "use_custom_op_collectives",
                    "is_cuda_alike",
                    "get_compile_backend",
                }
            ],
        )
    ],
    PLATFORM_NS,
)
PlatformContract = PLATFORM_NS["PlatformContract"]


def fixture(
    *,
    device="musa",
    backend="mccl",
    tp=4,
    split=None,
    graph_mode=None,
    compile_mode=None,
    inductor=False,
    moe=None,
    dp=3,
    **passes,
):
    config = ConfigContract()
    config.mode = Mode.VLLM_COMPILE if compile_mode is None else compile_mode
    config.cudagraph_mode = GraphMode.PIECEWISE if graph_mode is None else graph_mode
    config.compile_sizes = None
    config.inductor_compile_config = {}
    config.splitting_ops = split
    config.use_inductor_graph_partition = inductor
    config.pass_config = SimpleNamespace(
        **{
            "fuse_attn_quant": False,
            "fuse_rope_kvcache": True,
            "fuse_qk_norm_rope_kvcache": True,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            **passes,
        }
    )
    calls = []
    initializer = config.set_splitting_ops_for_v1

    def observed(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        return initializer(*args, **kwargs)

    config.set_splitting_ops_for_v1 = observed
    model = (
        None
        if moe is None
        else SimpleNamespace(is_moe=moe, is_mm_prefix_lm=False, use_mla=False)
    )
    platform = type(
        "ActualPlatformContract",
        (PlatformContract,),
        {
            "device_type": device,
            "vendor_name": "mthreads" if device == "musa" else "nvidia",
            "dist_backend": backend,
        },
    )
    vconfig = SimpleNamespace(
        use_v2_model_runner=False,
        compilation_config=config,
        model_config=model,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            all2all_backend="allgather_reducescatter",
            worker_cls="auto",
        ),
        scheduler_config=SimpleNamespace(),
        cache_config=SimpleNamespace(block_size=64),
        attention_config=None,
    )
    return platform, vconfig, calls


@pytest.mark.parametrize("device", ["musa", "cuda"])
@pytest.mark.parametrize(
    "initial_options",
    [
        {},
        {"combo_kernels": True, "benchmark_combo_kernel": True},
        {"combo_kernels": False, "benchmark_combo_kernel": False},
        {"combo_kernels": False},
        {"benchmark_combo_kernel": False},
        {"combo_kernels": True},
        {"benchmark_combo_kernel": True},
    ],
)
def test_combo_autotuning_is_disabled_only_on_musa(
    monkeypatch, device, initial_options
):
    platform, vconfig, _ = fixture(
        device=device, backend="mccl" if device == "musa" else "nccl"
    )
    config = vconfig.compilation_config
    config.inductor_compile_config.update(max_autotune=False, **initial_options)
    options = config.inductor_compile_config
    config.inductor_passes = {}
    config.custom_ops = []
    config.pass_config.enable_qk_norm_rope_fusion = False
    config.backend = "inductor"
    config.cudagraph_mm_encoder = False
    config.encoder_cudagraph_token_budgets = []
    capture_sizes = [1, 2, 4, 512]
    config.cudagraph_capture_sizes = capture_sizes
    monkeypatch.setitem(
        CONFIG_NS, "is_torch_equal_or_newer", lambda version: version == "2.9.0.dev"
    )
    monkeypatch.setitem(CONFIG_NS, "envs", SimpleNamespace(VLLM_LOGGING_LEVEL="INFO"))
    monkeypatch.setitem(
        CONFIG_NS,
        "current_platform",
        SimpleNamespace(is_cpu=lambda: False, get_compile_backend=lambda: "inductor"),
    )

    # Execute upstream's real defaults before the plugin hook, as config
    # construction does, then repeat them to catch later reinitialization.
    config.__post_init__()
    before_platform = options.copy()
    if not initial_options:
        assert options["combo_kernels"] is True
        assert options["benchmark_combo_kernel"] is True
    for _ in range(2):
        platform.check_and_update_config(vconfig)
        config.__post_init__()
        config.set_splitting_ops_for_v1("allgather_reducescatter", 3)
        expected = before_platform.copy()
        if device == "musa":
            expected.update(combo_kernels=False, benchmark_combo_kernel=False)
        assert options == expected
        assert config.inductor_compile_config is options
        assert config.backend == "inductor"
        assert config.mode is Mode.VLLM_COMPILE
        assert config.cudagraph_mode is GraphMode.PIECEWISE
        assert config.cudagraph_capture_sizes is capture_sizes


@pytest.mark.parametrize(
    "device,expected_backend,delegates",
    [
        ("musa", "eager", False),
        ("npu", "eager", False),
        ("cuda", "inductor", True),
        ("gcu", "inductor", True),
    ],
)
def test_real_compile_backend_default_and_parent_delegation(
    device, expected_backend, delegates
):
    platform, _, _ = fixture(device=device)
    before = len(BasePlatformContract.backend_calls)
    assert platform.get_compile_backend() == expected_backend
    assert BasePlatformContract.backend_calls[before:] == (
        [device] if delegates else []
    )


@pytest.mark.parametrize("device", ["musa", "cuda"])
@pytest.mark.parametrize("requested_backend", ["", "inductor", "eager"])
def test_real_compile_config_default_preserves_graph_contract(
    monkeypatch, device, requested_backend
):
    platform, vconfig, _ = fixture(
        device=device, backend="mccl" if device == "musa" else "nccl"
    )
    config = vconfig.compilation_config
    config.backend = requested_backend
    config.inductor_passes = {}
    config.custom_ops = []
    config.pass_config.enable_qk_norm_rope_fusion = False
    config.cudagraph_mm_encoder = False
    config.encoder_cudagraph_token_budgets = []
    # Preserve the full upstream default sequence observed in the failed CI.
    capture_sizes = [1, 2, 4, 8, *range(16, 257, 8), *range(272, 513, 16)]
    assert len(capture_sizes) == 51
    expected_sizes = tuple(capture_sizes)
    config.cudagraph_capture_sizes = capture_sizes
    monkeypatch.setitem(
        CONFIG_NS, "is_torch_equal_or_newer", lambda version: version == "2.9.0.dev"
    )
    monkeypatch.setitem(CONFIG_NS, "envs", SimpleNamespace(VLLM_LOGGING_LEVEL="INFO"))
    monkeypatch.setitem(
        CONFIG_NS,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: False, get_compile_backend=platform.get_compile_backend
        ),
    )
    # The real config initializer selects a platform default only for an empty
    # backend. The real hook and repeated upstream initializer must preserve it.
    expected_backend = requested_backend or (
        "eager" if device == "musa" else "inductor"
    )
    for _ in range(2):
        config.__post_init__()
        assert config.backend == expected_backend
        platform.check_and_update_config(vconfig)
        config.set_splitting_ops_for_v1("allgather_reducescatter", 3)
        assert config.backend == expected_backend
        assert config.mode is Mode.VLLM_COMPILE
        assert config.cudagraph_mode is GraphMode.PIECEWISE
        assert config.cudagraph_capture_sizes is capture_sizes
        assert tuple(config.cudagraph_capture_sizes) == expected_sizes
        assert config.use_inductor_graph_partition is False
        assert config.is_attention_compiled_piecewise()
    if device == "musa":
        assert config.splitting_ops.count("vllm::all_reduce") == 1
    else:
        assert "vllm::all_reduce" not in config.splitting_ops


def test_actual_eager_fx_compiler_keeps_piecewise_graph_wrapper():
    sample = torch.tensor([-2.0, 1.0])
    graph = fx.symbolic_trace(lambda value: torch.relu(-value))
    counter = SimpleNamespace(num_eager_compiles=0)
    compiler_ns = {
        "CompilerInterface": object,
        "compilation_counter": counter,
    }
    execute(
        [node(UPSTREAM / "compilation/compiler_interface.py", "EagerAdaptor")],
        compiler_ns,
    )
    platform, vconfig, _ = fixture()
    config = vconfig.compilation_config
    config.backend = platform.get_compile_backend()
    assert config.backend == "eager"
    platform.check_and_update_config(vconfig)
    compiled, handle = compiler_ns["EagerAdaptor"]().compile(graph, [sample], {}, None)
    assert compiled is graph
    assert handle is None
    assert counter.num_eager_compiles == 1
    torch.testing.assert_close(compiled(sample), torch.tensor([2.0, 0.0]))
    wrapped = []

    class GraphWrapperDouble:
        def __init__(self, **kwargs):
            wrapped.append(kwargs)

    def compiler_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "cuda_graph":
            return SimpleNamespace(CUDAGraphOptions=lambda **kwargs: kwargs)
        return builtins.__import__(name, globals, locals, fromlist, level)

    wrapper_ns = {
        "__builtins__": {**vars(builtins), "__import__": compiler_import},
        "current_platform": SimpleNamespace(
            get_static_graph_wrapper_cls=lambda: "GraphWrapperDouble"
        ),
        "resolve_obj_by_qualname": lambda name: GraphWrapperDouble,
        "CUDAGraphMode": GraphMode,
    }
    execute(
        [node(UPSTREAM / "compilation/backends.py", "wrap_with_cudagraph_if_needed")],
        wrapper_ns,
    )
    result = wrapper_ns["wrap_with_cudagraph_if_needed"](
        compiled, vconfig, config, True, True
    )
    assert isinstance(result, GraphWrapperDouble)
    assert len(wrapped) == 1
    assert wrapped[0]["runnable"] is graph
    assert wrapped[0]["runtime_mode"] is GraphMode.PIECEWISE
    assert wrapped[0]["vllm_config"] is vconfig


@pytest.mark.parametrize("moe,expected_dp", [(None, 3), (False, 1), (True, 3)])
def test_defaults_and_upstream_second_initialization(moe, expected_dp):
    platform, vconfig, calls = fixture(moe=moe)
    config = vconfig.compilation_config
    attention_before = list(ConfigContract._attention_ops)
    platform.check_and_update_config(vconfig)
    assert config.splitting_ops == attention_before + [
        "vllm::unified_kv_cache_update",
        "vllm::unified_mla_kv_cache_update",
        "vllm::all_reduce",
    ]
    assert calls == [
        (
            (),
            {
                "all2all_backend": "allgather_reducescatter",
                "data_parallel_size": expected_dp,
            },
        )
    ]
    config.set_splitting_ops_for_v1("allgather_reducescatter", expected_dp)
    platform.check_and_update_config(vconfig)
    assert config.splitting_ops.count("vllm::all_reduce") == 1
    assert ConfigContract._attention_ops == attention_before
    assert not config.pass_config.fuse_rope_kvcache
    assert not config.pass_config.fuse_qk_norm_rope_kvcache


def test_explicit_list_keeps_identity_and_existing_order():
    original = ["custom::boundary", "vllm::all_reduce"]
    platform, vconfig, _ = fixture(split=original)
    platform.check_and_update_config(vconfig)
    platform.check_and_update_config(vconfig)
    assert vconfig.compilation_config.splitting_ops is original
    assert original == ["custom::boundary", "vllm::all_reduce"]


@pytest.mark.parametrize(
    "passes,expected",
    [
        ({}, GraphMode.NONE),
        ({"fuse_attn_quant": True}, GraphMode.FULL),
        ({"enable_sp": True}, GraphMode.FULL),
        ({"fuse_gemm_comms": True}, GraphMode.FULL),
    ],
)
def test_initializer_graph_mode_changes_are_preserved(passes, expected):
    platform, vconfig, _ = fixture(split=[] if not passes else None, **passes)
    platform.check_and_update_config(vconfig)
    assert vconfig.compilation_config.cudagraph_mode is expected
    assert vconfig.compilation_config.splitting_ops == []
    vconfig.compilation_config.set_splitting_ops_for_v1("allgather_reducescatter", 3)
    assert vconfig.compilation_config.cudagraph_mode is expected


@pytest.mark.parametrize(
    "options",
    [
        {"device": "cuda", "backend": "nccl"},
        {"backend": "flagcx"},
        {"tp": 1},
        {"graph_mode": GraphMode.NONE},
        {"compile_mode": Mode.NONE},
        {"inductor": True},
    ],
)
def test_unrelated_paths_do_not_initialize_or_add_split(options):
    platform, vconfig, calls = fixture(**options)
    platform.check_and_update_config(vconfig)
    assert calls == []
    assert vconfig.compilation_config.splitting_ops is None


@pytest.mark.parametrize(
    "device,backend,expected",
    [
        ("musa", "mccl", True),
        ("musa", "flagcx", False),
        ("cuda", "nccl", True),
    ],
)
def test_real_custom_op_collective_boolean(device, backend, expected):
    platform, _, _ = fixture(device=device, backend=backend)
    assert platform.use_custom_op_collectives() is expected
    if device == "musa":
        assert platform().is_cuda_alike() is True


def test_actual_fx_all_reduce_partition_excludes_graph_wrapper():
    libraries = []
    try:
        contract_namespace = "musa_collective_contract_" + uuid.uuid4().hex
        contract_library = torch.library.Library(contract_namespace, "FRAGMENT")
        libraries.append(contract_library)
        register_ns = {
            "torch": torch,
            "infer_schema": torch.library.infer_schema,
            "vllm_lib": contract_library,
        }
        execute(
            [node(UPSTREAM / "utils/torch_utils.py", "direct_register_custom_op")],
            register_ns,
        )

        def cpu_collective(input_: torch.Tensor, group_name: str) -> torch.Tensor:
            return input_.clone()

        # Registration capability has its own namespace: never replace a
        # real vendor implementation if another test already imported vLLM.
        register_ns["direct_register_custom_op"](
            "all_reduce",
            cpu_collective,
            dispatch_key="PrivateUse1",
            target_lib=contract_library,
        )
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            contract_namespace + "::all_reduce", "PrivateUse1"
        )
        had_vllm_op = hasattr(torch.ops.vllm, "all_reduce")
        original_vendor_keys = (
            {
                key: torch._C._dispatch_has_kernel_for_dispatch_key(
                    "vllm::all_reduce", key
                )
                for key in ("CUDA", "PrivateUse1")
            }
            if had_vllm_op
            else None
        )
        library = torch.library.Library("vllm", "FRAGMENT")
        libraries.append(library)
        if not had_vllm_op:
            register_ns["direct_register_custom_op"](
                "all_reduce", cpu_collective, dispatch_key="CPU", target_lib=library
            )
        elif not torch._C._dispatch_has_kernel_for_dispatch_key(
            "vllm::all_reduce", "CPU"
        ):
            library.impl("all_reduce", cpu_collective, "CPU")
        if original_vendor_keys is not None:
            assert original_vendor_keys == {
                key: torch._C._dispatch_has_kernel_for_dispatch_key(
                    "vllm::all_reduce", key
                )
                for key in original_vendor_keys
            }
        graph = fx.Graph()
        value = graph.placeholder("value")
        before = graph.call_function(torch.neg, (value,))
        collective = graph.call_function(
            torch.ops.vllm.all_reduce.default, (before, "cpu_contract")
        )
        after = graph.call_function(torch.relu, (collective,))
        graph.output(after)
        sample = torch.tensor([-2.0, 1.0])
        for item in graph.nodes:
            item.meta["example_value"] = sample
        gm = fx.GraphModule(torch.nn.Module(), graph)
        ns = {
            "torch": torch,
            "fx": fx,
            "dataclasses": dataclasses,
            "operator": operator,
            "defaultdict": defaultdict,
            "_use_lazy_graph_module": _use_lazy_graph_module,
            "is_torch_equal_or_newer": lambda version: False,
        }
        execute([node(UPSTREAM / "compilation/partition_rules.py", "should_split")], ns)
        backend_source = UPSTREAM / "compilation/backends.py"
        execute(
            [
                node(backend_source, name)
                for name in [
                    "SplitItem",
                    "_is_empty_allocation_node",
                    "_merge_empty_only_subgraphs",
                    "_decompose_size_nodes",
                    "split_graph",
                ]
            ],
            ns,
        )
        platform, vconfig, _ = fixture()
        platform.check_and_update_config(vconfig)
        split_gm, items = ns["split_graph"](
            gm, vconfig.compilation_config.splitting_ops
        )
        boundary = [item for item in items if item.is_splitting_graph]
        assert len(boundary) == 1
        assert any(
            item.op == "call_function"
            and item.target is torch.ops.vllm.all_reduce.default
            for item in boundary[0].graph.graph.nodes
        )
        compile_names = [
            item.submod_name for item in items if not item.is_splitting_graph
        ]
        assert len(compile_names) == 2
        wrapped = []

        class BackendDouble:
            def __init__(self, submod, config, index, count, *args, **kwargs):
                self.is_first_graph = index == 0
                self.is_last_graph = index == count - 1

        class GraphWrapperDouble:
            def __init__(self, **kwargs):
                wrapped.append(kwargs)

        def compiler_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and name == "piecewise_backend":
                return SimpleNamespace(PiecewiseBackend=BackendDouble)
            if level == 1 and name == "cuda_graph":
                return SimpleNamespace(CUDAGraphOptions=lambda **kwargs: kwargs)
            return builtins.__import__(name, globals, locals, fromlist, level)

        ns.update(
            {
                "__builtins__": {**vars(builtins), "__import__": compiler_import},
                "current_platform": SimpleNamespace(
                    get_static_graph_wrapper_cls=lambda: "GraphWrapperDouble"
                ),
                "resolve_obj_by_qualname": lambda name: GraphWrapperDouble,
                "CUDAGraphMode": GraphMode,
                "compilation_counter": SimpleNamespace(
                    num_piecewise_capturable_graphs_seen=0
                ),
                "instrument": lambda **kwargs: lambda function: function,
            }
        )
        execute(
            [
                node(backend_source, "wrap_with_cudagraph_if_needed"),
                node(backend_source, "PiecewiseCompileInterpreter"),
            ],
            ns,
        )
        interpreter = ns["PiecewiseCompileInterpreter"](
            split_gm, compile_names, vconfig, object()
        )
        boundary_before = getattr(split_gm, boundary[0].submod_name)
        for item in items:
            interpreter.call_module(item.submod_name, (sample,), {})
        assert getattr(split_gm, boundary[0].submod_name) is boundary_before
        assert boundary[0].submod_name not in compile_names
        assert len(wrapped) == 2
        assert all(item["runtime_mode"] is GraphMode.PIECEWISE for item in wrapped)
        assert ns["compilation_counter"].num_piecewise_capturable_graphs_seen == 2
    finally:
        for library in reversed(libraries):
            library._destroy()
