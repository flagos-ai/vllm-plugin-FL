# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual branches for vllm_fl.compilation.graph.

The graph module binds its native graph class from the live platform at import
time, so it is loaded from the real source with a fully stubbed accelerator
surface. Capture/replay are exercised only through mocks.
"""

import importlib.util
import sys
from contextlib import nullcontext
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _repo_root() -> Path:
    import os

    candidates = [Path("/tmp/flsrc")]
    if os.environ.get("VLLM_FL_ROOT"):
        candidates.insert(0, Path(os.environ["VLLM_FL_ROOT"]))
    for base in [*candidates, Path.cwd(), *Path(__file__).resolve().parents]:
        if not base.is_dir():
            continue
        if (base / "vllm_fl/compilation/graph.py").is_file():
            return base
    raise RuntimeError("cannot locate vllm_fl source tree")


ROOT = _repo_root()


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install(monkeypatch, mapping):
    for name, obj in mapping.items():
        monkeypatch.setitem(sys.modules, name, obj)
    for name in mapping:
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            full = ".".join(parts[:index])
            if full not in sys.modules:
                monkeypatch.setitem(sys.modules, full, ModuleType(full))
            if index > 1:
                parent = sys.modules[".".join(parts[:index - 1])]
                setattr(parent, parts[index - 1], sys.modules[full])


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class _T:
    def __init__(self, address=1):
        self._address = address
        self.device = SimpleNamespace(type="musa", index=0)

    def data_ptr(self):
        return self._address


class BatchDescriptor:
    """Hashable stand-in for vllm.forward_context.BatchDescriptor.

    A plain class keeps identity hashing/equality, unlike SimpleNamespace which
    defines __eq__ and is therefore unhashable as a dict key.
    """

    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens


class FakeGraph:
    instances = []

    def __init__(self):
        self.replays = 0
        FakeGraph.instances.append(self)

    def replay(self):
        self.replays += 1


def load_comp(monkeypatch, device="musa", logging_level="INFO", pool="pool"):
    state = SimpleNamespace()
    state.platform = SimpleNamespace(
        device_type=device,
        vendor_name=None,
        get_global_graph_pool=lambda: pool,
        graph_pool_handle=lambda: "handle",
        torch_device_fn=SimpleNamespace(
            graph=MagicMock(return_value=nullcontext()),
            current_stream=MagicMock(return_value="native-stream"),
            synchronize=MagicMock(),
        ),
    )
    state.context = SimpleNamespace(
        batch_descriptor=BatchDescriptor(),
        cudagraph_runtime_mode=Mode.FULL,
    )
    state.available = True
    state.validate = MagicMock()
    state.set_pool = MagicMock()
    state.counter = SimpleNamespace(num_cudagraph_captured=0)
    state.logger = SimpleNamespace(
        debug=MagicMock(), info=MagicMock(), warning=MagicMock()
    )
    state.upstream_weak_ref = MagicMock(side_effect=lambda value: value)
    state.iluvatar = MagicMock(side_effect=lambda value: ("iluvatar", value))

    torch_stub = ModuleType("torch")
    torch_stub.Tensor = _T
    torch_stub._C = SimpleNamespace(
        _construct_storage_from_data_pointer=MagicMock(return_value="storage"),
        _set_conj=MagicMock(),
        _set_neg=MagicMock(),
    )
    torch_stub.accelerator = SimpleNamespace(empty_cache=lambda: None)
    torch_stub.cuda = SimpleNamespace(CUDAGraph=FakeGraph)
    torch_stub.npu = SimpleNamespace(NPUGraph=FakeGraph)
    torch_stub.musa = SimpleNamespace(MUSAGraph=FakeGraph)
    torch_stub.gcu = SimpleNamespace(GCUGraph=FakeGraph)
    torch_stub.ptpu = SimpleNamespace(PTPUGraph=FakeGraph)
    state.torch = torch_stub

    platform_fl = type(
        "PlatformFL", (), {"empty_cache": staticmethod(lambda *a, **k: None)}
    )
    mapping = {
        "torch": torch_stub,
        "vllm": _module("vllm"),
        "vllm.envs": _module("vllm.envs", VLLM_LOGGING_LEVEL=logging_level),
        "vllm.compilation.counter": _module(
            "vllm.compilation.counter", compilation_counter=state.counter
        ),
        "vllm.compilation.monitor": _module(
            "vllm.compilation.monitor",
            validate_cudagraph_capturing_enabled=state.validate,
        ),
        "vllm.config": _module(
            "vllm.config",
            CUDAGraphMode=Mode,
            VllmConfig=type("VllmConfig", (), {}),
        ),
        "vllm.distributed.device_communicators.pynccl_allocator": _module(
            "vllm.distributed.device_communicators.pynccl_allocator",
            set_graph_pool_id=state.set_pool,
        ),
        "vllm.forward_context": _module(
            "vllm.forward_context",
            BatchDescriptor=type("BatchDescriptor", (), {}),
            get_forward_context=lambda: state.context,
            is_forward_context_available=lambda: state.available,
        ),
        "vllm.logger": _module("vllm.logger", init_logger=lambda name: state.logger),
        "vllm.platforms": _module("vllm.platforms", current_platform=state.platform),
        "vllm.utils.torch_utils": _module(
            "vllm.utils.torch_utils",
            current_stream=MagicMock(return_value="vllm-stream"),
            weak_ref_tensors=state.upstream_weak_ref,
        ),
        "vllm.compilation.cuda_graph": _module(
            "vllm.compilation.cuda_graph",
            CUDAGraphStat=type("CUDAGraphStat", (), {}),
        ),
        "vllm_fl": _module("vllm_fl"),
        "vllm_fl.platform": _module("vllm_fl.platform", PlatformFL=platform_fl),
        "vllm_fl.dispatch.backends.vendor.iluvatar.graph": _module(
            "vllm_fl.dispatch.backends.vendor.iluvatar.graph",
            _weak_ref_iluvatar_tensors=state.iluvatar,
        ),
    }
    _install(monkeypatch, mapping)

    source = ROOT / "vllm_fl/compilation/graph.py"
    spec = importlib.util.spec_from_file_location("w23_comp_graph", source)
    module = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses resolve annotations via
    # ``sys.modules[cls.__module__].__dict__``; register before executing.
    sys.modules["w23_comp_graph"] = module
    spec.loader.exec_module(module)
    state.module = module
    return state


@pytest.fixture
def comp(monkeypatch):
    FakeGraph.instances.clear()
    return load_comp(monkeypatch)


def _wrapper(state, runnable=None, mode=Mode.FULL, **options):
    if runnable is None:
        runnable = MagicMock(return_value="output")
    return state.module.GraphWrapper(
        runnable,
        SimpleNamespace(compilation_config=SimpleNamespace()),
        mode,
        state.module.GraphOptions(**options),
    )


# --------------------------------------------------------------------------- #
# GCU weak refs: both optional flags unset
# --------------------------------------------------------------------------- #
class _GcuTensor:
    def __init__(self):
        self.device = SimpleNamespace(type="gcu")
        self.shape = (2, 2)
        self.dtype = "float32"

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: 16, data_ptr=lambda: 100)

    def stride(self):
        return (2, 1)

    def storage_offset(self):
        return 0

    def is_conj(self):
        return False

    def is_neg(self):
        return False


def test_gcu_weak_ref_leaves_plain_tensor_flags_untouched(comp, monkeypatch):
    native = MagicMock(return_value="gcu-tensor")
    monkeypatch.setitem(
        sys.modules,
        "torch_gcu",
        _module(
            "torch_gcu",
            _C=SimpleNamespace(
                _construct_GCU_Tensor_From_Storage_And_Metadata=native
            ),
        ),
    )
    value = _GcuTensor()
    assert comp.module._weak_ref_gcu_tensor(value) == "gcu-tensor"
    # Neither is_conj nor is_neg is set on a plain view.
    comp.torch._C._set_conj.assert_not_called()
    comp.torch._C._set_neg.assert_not_called()
    # The metadata passed through is the borrowed-storage descriptor.
    metadata = native.call_args.args[0]
    assert metadata["nbytes"] == 16
    assert metadata["storage_offset"] == 0


# --------------------------------------------------------------------------- #
# Offloader lifecycle (success path for both suppress blocks)
# --------------------------------------------------------------------------- #
def _install_offloader(monkeypatch, offloader):
    base = _module("vllm.model_executor.offloader.base", get_offloader=lambda: offloader)
    pkg = _module("vllm.model_executor.offloader")
    parent = _module("vllm.model_executor")
    for name, module in (
        ("vllm.model_executor", parent),
        ("vllm.model_executor.offloader", pkg),
        ("vllm.model_executor.offloader.base", base),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["vllm"], "model_executor", parent, raising=False)
    parent.offloader = pkg
    pkg.base = base


def test_capture_and_replay_sync_the_offloader(comp, monkeypatch):
    calls = []

    class _Offloader:
        def sync_prev_onload(self):
            calls.append("sync")

        def join_after_forward(self):
            calls.append("join")

    _install_offloader(monkeypatch, _Offloader())
    wrapper = _wrapper(comp, weak_ref_output=False)
    tensor = _T()

    wrapper(tensor)  # capture: pre-sync then join
    assert calls == ["sync", "join"]

    calls.clear()
    wrapper(tensor)  # replay: pre-sync only
    assert calls == ["sync"]


# --------------------------------------------------------------------------- #
# Debug logging disabled branch during capture
# --------------------------------------------------------------------------- #
def test_capture_with_debug_logging_disabled(comp):
    wrapper = _wrapper(comp, weak_ref_output=False, debug_log_enable=False)
    wrapper(_T())
    comp.logger.debug.assert_not_called()
    assert comp.counter.num_cudagraph_captured == 1
