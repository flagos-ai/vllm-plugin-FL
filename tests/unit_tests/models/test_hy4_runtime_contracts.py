# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for shared Hy4 hosts and NVIDIA runtime integration.

Shared runtime/metadata and NVIDIA metadata are imported from their actual
production files in one isolated package hierarchy. NVIDIA runtime host
definitions are selected with AST to avoid this machine's different native
vLLM imports. This is not full-package or target CUDA capture/replay validation.
No native vLLM modules are fabricated and no GPU work is submitted.
"""

import ast
import copy
import importlib.util
import inspect
import sys
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODEL = Path(__file__).resolve().parents[3] / "vllm_fl/models/hy_v4"
NVIDIA = MODEL / "nvidia"


def load_source(monkeypatch, name, path, *, package=False):
    options = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def selected_native_hosts(path, namespace):
    tree = ast.parse(path.read_text())
    body = ast.parse("from __future__ import annotations").body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node = copy.deepcopy(node)
            node.decorator_list = []
            body.append(node)
        elif (
            isinstance(node, ast.ClassDef)
            and node.name == "Hy4RuntimeHooks"
            or isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_INSTALLED" for t in node.targets
            )
            or isinstance(node, ast.ImportFrom)
            and node.level == 2
            and node.module in ("metadata", "runtime")
        ):
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def model_context(monkeypatch):
    package = "hy4_runtime_contract"
    load_source(monkeypatch, package, MODEL / "__init__.py", package=True)
    metadata = load_source(monkeypatch, package + ".metadata", MODEL / "metadata.py")
    shared = load_source(monkeypatch, package + ".runtime", MODEL / "runtime.py")
    load_source(monkeypatch, package + ".nvidia", NVIDIA / "__init__.py", package=True)
    cuda_metadata = load_source(
        monkeypatch, package + ".nvidia.metadata", NVIDIA / "metadata.py"
    )
    return SimpleNamespace(
        package=package, metadata=metadata, shared=shared, cuda_metadata=cuda_metadata
    )


@pytest.fixture
def graph(model_context):
    return model_context.shared


@pytest.fixture
def staging(model_context):
    return model_context.metadata


@pytest.fixture
def runtime_context(monkeypatch, model_context):
    name = model_context.package + ".nvidia.runtime"
    spec = importlib.util.spec_from_file_location(name, NVIDIA / "runtime.py")
    native = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, native)
    native.__dict__.update(wraps=wraps, torch=torch)
    selected_native_hosts(NVIDIA / "runtime.py", native.__dict__)
    model_context.native = native
    return model_context


def common(q=4, reqs=2):
    return SimpleNamespace(
        max_query_len=q,
        num_reqs=reqs,
        num_actual_tokens=q * reqs,
        query_start_loc_cpu=torch.arange(reqs + 1) * q,
        is_prefilling=torch.zeros(reqs, dtype=torch.bool),
        causal=True,
        mm_req_doc_ranges=None,
    )


def test_draft_plan_accepts_verified_batch(graph):
    assert graph.decode_plan(
        common(), (1, 2, 4, 8, 16, 32, 64, 128, 256), verified_step=True
    ) == (4, 8, 2)


@pytest.mark.parametrize(
    "condition",
    ["unverified", "nonuniform", "prefill", "noncausal", "media", "capacity"],
)
def test_draft_plan_declines_before_running_model(graph, condition):
    c = common()
    verified = True
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    if condition == "unverified":
        verified = False
    elif condition == "nonuniform":
        c.query_start_loc_cpu[1] = 3
    elif condition == "prefill":
        c.is_prefilling[1] = True
    elif condition == "noncausal":
        c.causal = False
    elif condition == "media":
        c.mm_req_doc_ranges = {0: [(0, 2)]}
    elif condition == "capacity":
        sizes = (1, 2, 4)
    assert graph.decode_plan(c, sizes, verified_step=verified) is None


def test_q1_capacity_is_distinct_from_q4_rows(graph):
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert graph.decode_plan(common(1, 64), sizes, verified_step=True) == (1, 64, 64)
    assert graph.decode_plan(common(1, 65), sizes, verified_step=True) is None


def test_input_maps_follow_reordered_request_history(staging):
    runner = SimpleNamespace(
        prev_positions=SimpleNamespace(np=[-1, 1, 0]),
        input_batch=SimpleNamespace(req_ids=["new", "old-a", "old-b"]),
        prev_num_spec_tokens=3,
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"old-a": [123]})
    maps = staging.build_input_maps(runner, scheduled, 3, 8, [2, 6, 8])
    assert maps.sample_dst == [4, 7]
    assert maps.sample_src == [1, 0]
    assert maps.draft_dst == [5]
    assert maps.draft_src == [3]
    assert maps.num_sampled == 2
    assert maps.copy_base is True


def test_unchanged_dense_sample_map_stays_native(staging):
    runner = SimpleNamespace(
        prev_positions=SimpleNamespace(np=[0, 1]),
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
        prev_num_spec_tokens=3,
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={})
    assert staging.build_input_maps(runner, scheduled, 2, 2, [1, 2]) is None


def test_host_packet_layout_and_untouched_tail(staging):
    import numpy as np

    packet = np.full(32, 99, dtype=np.int64)
    used = staging.pack_host_packet(
        packet, [4, 7, 5], [1, 0, 3], [-1, 1, 0], [0, 8, 16], [0, 0, 0, 0, 0]
    )
    assert used == 17
    assert packet[:used].tolist() == [
        4,
        7,
        5,
        1,
        0,
        3,
        -1,
        1,
        0,
        0,
        8,
        16,
        0,
        0,
        0,
        0,
        0,
    ]
    assert packet[used:].tolist() == [99] * 15


def test_packet_proof_copies_only_the_transferred_facts(staging):
    import numpy as np

    packet = np.array(
        [4, 7, 5, 1, 0, 3, -1, 1, 0, 0, 8, 16, 0, 0, 0, 0, 0], dtype=np.int64
    )
    ids = ("new", "old-a", "old-b")
    args = [
        "owner",
        1,
        ids,
        packet,
        3,
        3,
        5,
        [2, 6, 8],
        [20, 40, 60],
        {"old-a": 1, "old-b": 0},
    ]
    result = staging._cpu_sequence_proof(*args)
    assert isinstance(result, staging.CPUSequenceLengthsProof)
    assert result.query_starts == (0, 2, 6, 8)
    assert result.computed == (0, 8, 16)
    assert result.previous_positions == (-1, 1, 0)
    assert result.previous_drafts == (-1, 0, 0)
    packet[:] = 999
    assert result.computed == (0, 8, 16)


def test_patch_plan_restores_objects_and_removes_added_attributes(runtime_context):
    runtime = runtime_context.shared

    class FailOnce(type):
        def __setattr__(cls, name, value):
            super().__setattr__(name, value)
            if name == "second" and cls.fail:
                super().__setattr__("fail", False)
                raise RuntimeError("assignment failed")

    class Owner(metaclass=FailOnce):
        first = object()
        second = object()
        fail = True

    first, second = Owner.first, Owner.second
    plan = [
        (Owner, "first", first, object()),
        (Owner, "added", runtime._ABSENT, object()),
        (Owner, "second", second, object()),
    ]
    with pytest.raises(RuntimeError, match="assignment failed"):
        runtime.apply_patch_plan(plan)
    assert Owner.first is first and Owner.second is second
    assert "added" not in Owner.__dict__


def test_non_target_preserves_arguments_and_result(runtime_context):
    runtime = runtime_context.native
    calls = []
    result = object()

    def original(self, *args, **kwargs):
        calls.append((args, kwargs))
        return result

    def implementation(*args, **kwargs):
        raise AssertionError("unselected adapter")

    fn = runtime._scoped_method(
        original, implementation, None, "runner", "execute_model"
    )
    assert fn(SimpleNamespace(), 1, named=2) is result
    assert calls == [((1,), {"named": 2})]


def test_selected_error_restores_nested_context_without_retry(runtime_context):
    runtime, metadata = runtime_context.native, runtime_context.metadata
    outer_proof, step_proof = object(), object()
    owner = SimpleNamespace(_hy4_runtime_enabled=True)
    outer_runner = runtime_context.shared._CURRENT_RUNNER.set("caller")
    outer_token = metadata.push_current_cpu_seq_lens_proof(outer_proof)
    calls = []

    def original(*args, **kwargs):
        raise AssertionError("execution cannot retry")

    def implementation(self, *args, **kwargs):
        assert runtime_context.shared.current_runner() is self
        assert metadata.get_current_cpu_seq_lens_proof() is None
        metadata.set_current_cpu_seq_lens_proof(step_proof)
        calls.append((args, kwargs))
        raise RuntimeError("selected execution failed")

    try:
        fn = runtime._scoped_method(
            original, implementation, None, "runner", "execute_model"
        )
        with pytest.raises(RuntimeError, match="selected execution failed"):
            fn(owner, 3, option=4)
        assert calls == [((3,), {"option": 4})]
        assert runtime_context.shared.current_runner() == "caller"
        assert metadata.get_current_cpu_seq_lens_proof() is outer_proof
    finally:
        runtime_context.shared._CURRENT_RUNNER.reset(outer_runner)
        metadata.reset_current_cpu_seq_lens_proof(outer_token)


def test_constructor_delegates_original_once(runtime_context, monkeypatch):
    runtime = runtime_context.native
    common, stager = object(), object()
    policy = SimpleNamespace(
        make_common_metadata_graph=lambda: common,
        make_input_staging=lambda runner: stager,
    )
    selections = []

    def select(config):
        selections.append(config)
        return policy

    monkeypatch.setattr(runtime, "create_hy4_runtime", select)
    calls = []

    def native(self, config, device):
        calls.append(("native", config, device))
        self.uses_mrope = False

    @wraps(native)
    def original(self, *args, **kwargs):
        calls.append(("original", args, kwargs))
        return native(self, *args, **kwargs)

    owner = SimpleNamespace()
    fn = runtime._scoped_method(original, None, SimpleNamespace(), "runner", "__init__")
    assert fn(owner, "config", device="cpu") is None
    assert calls == [
        ("original", ("config",), {"device": "cpu"}),
        ("native", "config", "cpu"),
    ]
    assert selections == ["config"]
    assert owner._model_runtime is policy
    assert owner.common_attention_metadata_graph is common
    assert owner._fl_async_input_staging is stager


def test_other_model_returns_before_target_selection(runtime_context, monkeypatch):
    runtime = runtime_context.native

    def forbidden(config):
        raise AssertionError("non-target dependency")

    monkeypatch.setattr(runtime, "create_hy4_runtime", forbidden)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
    )
    assert runtime.prepare_runtime_for_worker(config) is False


def test_explicit_v2_target_rejected_before_install(runtime_context, monkeypatch):
    runtime = runtime_context.native
    monkeypatch.setattr(runtime, "create_hy4_runtime", lambda config: object())
    monkeypatch.setattr(
        runtime,
        "install_runtime_patches",
        lambda: pytest.fail("must reject before installation"),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="hy_v4")),
        use_v2_model_runner=True,
    )
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        runtime.prepare_runtime_for_worker(config)


def fixed_runtime_owners():
    """Host-only dependency doubles for the pinned 0.28 installation boundary."""

    class FailOnce(type):
        def __setattr__(cls, name, value):
            super().__setattr__(name, value)
            if name == cls.fail_on:
                super().__setattr__("fail_on", None)
                raise RuntimeError("install assignment failed")

    class Runner(metaclass=FailOnce):
        fail_on = None

        def __init__(self, vllm_config, device):
            self.native_init = (vllm_config, device)
            self.uses_mrope = False

    class Proposer:
        def __init__(
            self, vllm_config, device, pass_hidden_states_to_model, runner=None
        ):
            self.native_init = (
                vllm_config,
                device,
                pass_hidden_states_to_model,
                runner,
            )

    def native_method(self, *args, **kwargs):
        return args, kwargs

    for name in (
        "_prepare_input_ids",
        "_prepare_inputs",
        "_build_attention_metadata",
        "_bookkeeping_sync",
        "synchronize_input_prep",
        "_get_slot_mappings",
        "execute_model",
        "sample_tokens",
        "load_model",
        "_get_prompt_logprobs_dict",
        "_dummy_run",
        "_cleanup_profiling_kv_cache",
        "profile_cudagraph_memory",
        "capture_model",
        "_warmup_and_capture",
    ):
        setattr(Runner, name, native_method)
    Proposer.initialize_cudagraph_keys = native_method
    Proposer.propose = native_method
    return SimpleNamespace(GPUModelRunner=Runner), SimpleNamespace(
        SpecDecodeBaseProposer=Proposer
    )


def test_fixed_runtime_install_is_idempotent(runtime_context):
    runtime = runtime_context.native
    runner_module, proposer_module = fixed_runtime_owners()
    runner, proposer = (
        runner_module.GPUModelRunner,
        proposer_module.SpecDecodeBaseProposer,
    )
    assert runtime._install_runtime_patches(runner_module, proposer_module) is True
    installed = dict(runner.__dict__)
    installed_proposer = dict(proposer.__dict__)
    assert runtime._install_runtime_patches(runner_module, proposer_module) is False
    assert all(runner.__dict__[name] is value for name, value in installed.items())
    assert all(
        proposer.__dict__[name] is value for name, value in installed_proposer.items()
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
    )
    instance = runner(config, device="cpu")
    assert instance.native_init == (config, "cpu")
    assert instance.execute_model("scheduled", intermediate_tensors="hidden") == (
        ("scheduled",),
        {"intermediate_tensors": "hidden"},
    )
    with pytest.raises(AttributeError, match="_prepare_decode_mrope"):
        instance._prepare_decode_mrope("scheduled")


def test_install_failure_restores_plan_and_remains_uninstalled(runtime_context):
    runtime = runtime_context.native
    runner_module, proposer_module = fixed_runtime_owners()
    runner, proposer = (
        runner_module.GPUModelRunner,
        proposer_module.SpecDecodeBaseProposer,
    )
    originals = dict(runner.__dict__)
    proposer_originals = dict(proposer.__dict__)
    runner.fail_on = "_get_slot_mappings"
    with pytest.raises(RuntimeError, match="install assignment failed"):
        runtime._install_runtime_patches(runner_module, proposer_module)
    assert runtime._INSTALLED is False
    assert all(runner.__dict__[name] is value for name, value in originals.items())
    assert all(
        proposer.__dict__[name] is value for name, value in proposer_originals.items()
    )
    assert "_prepare_decode_mrope" not in runner.__dict__
    assert "_run_common_attention_metadata" not in runner.__dict__


def test_host_adapters_accept_fixed_runner_call_arguments(runtime_context):
    runtime = runtime_context.native
    bookkeeping = inspect.signature(runtime.runner_bookkeeping_sync)
    native_args = [object()] * 6
    bound = bookkeeping.bind(*native_args)
    bound.apply_defaults()
    assert bound.arguments["num_scheduled_tokens"] is native_args[-1]
    assert bound.arguments["prompt_logprobs_hidden_states"] is None
    prompt = inspect.signature(runtime.runner_get_prompt_logprobs_dict)
    bound = prompt.bind(object(), object(), {})
    bound.apply_defaults()
    assert bound.arguments["retained_prompt_hidden_states"] is None
    for fn, parameter in (
        (runtime.runner_get_slot_mappings, "slot_mapping_is_current"),
        (runtime.runner_build_attention_metadata, "block_table_rows_are_current"),
    ):
        assert inspect.signature(fn).parameters[parameter].default is False


@pytest.mark.parametrize("failure", [False, True])
def test_input_event_lease_order_and_source_retention(runtime_context, failure):
    runtime, metadata = runtime_context.shared, runtime_context.cuda_metadata
    events = []
    owner = object()

    class Staging(metadata.AsyncInputStaging):
        def begin_step(self):
            assert events[-1] == "wait"
            super().begin_step()
            events.append("begin")

        def end_step(self):
            super().end_step()
            events.append("end")

    stager = object.__new__(Staging)
    stager.plan, stager.in_step, stager._cpu_proof_generation = owner, False, 0
    event = SimpleNamespace(
        synchronize=lambda: events.append("wait"),
        record=lambda: events.append("record"),
    )
    runner = SimpleNamespace(prepare_inputs_event=event, _fl_async_input_staging=stager)

    def prepare():
        with runtime.runnersynchronize_input_prep(runner):
            assert stager.in_step and stager._cpu_proof_generation == 1
            assert stager.plan is None
            stager.plan = owner
            events.append("body")
            if failure:
                raise RuntimeError("body failed")

    if failure:
        with pytest.raises(RuntimeError, match="body failed"):
            prepare()
    else:
        prepare()
    assert events == ["wait", "begin", "body", "record", "end"]
    assert stager.plan is owner and not stager.in_step
    with runtime.runnersynchronize_input_prep(runner):
        assert stager.plan is None and stager._cpu_proof_generation == 2


@pytest.mark.parametrize(
    "case",
    [
        "no_runner",
        "old_proof_no_runner",
        "unselected",
        "other_owner",
        "missing_proof",
        "current_proof",
        "stale_proof",
    ],
)
def test_common_binding_initializes_current_instance_proof(runtime_context, case):
    runtime, metadata = runtime_context.shared, runtime_context.metadata
    policy, proof = object(), object()
    runner = SimpleNamespace(_model_runtime=policy)
    common = SimpleNamespace(_model_runtime=policy)
    if case == "no_runner":
        runner, common = None, SimpleNamespace()
    elif case == "old_proof_no_runner":
        runner = None
        common.cpu_seq_lens_proof = proof
    elif case == "unselected":
        runner = SimpleNamespace()
        common.cpu_seq_lens_proof = proof
    elif case == "other_owner":
        common._model_runtime = object()
        common.cpu_seq_lens_proof = proof
    elif case == "current_proof":
        common.cpu_seq_lens_proof = proof
    elif case == "stale_proof":
        common.cpu_seq_lens_proof = object()
    runner_token = runtime_context.shared._CURRENT_RUNNER.set(runner)
    proof_token = metadata.push_current_cpu_seq_lens_proof(proof)
    try:
        assert runtime.bind_common_runtime_metadata(common) is common
        assert common.cpu_seq_lens_proof is (proof if case == "current_proof" else None)
        assert metadata.get_current_cpu_seq_lens_proof() is proof
    finally:
        runtime_context.shared._CURRENT_RUNNER.reset(runner_token)
        metadata.reset_current_cpu_seq_lens_proof(proof_token)


@pytest.mark.parametrize(
    "use_graph,ubatching", [(False, False), (True, False), (True, True)]
)
def test_common_metadata_submission_uses_fixed_step_contract(
    runtime_context, use_graph, ubatching
):
    runtime, metadata = runtime_context.native, runtime_context.cuda_metadata
    calls = []
    result = object()
    none_mode, full_mode = object(), object()

    def submit(*args, **kwargs):
        inspect.signature(metadata.CommonAttentionMetadataGraphRunner.run).bind(
            object(), *args, **kwargs
        )
        calls.append((args, kwargs))
        return result

    block_table = object()
    starts = torch.arange(5, dtype=torch.int32)
    positions = torch.arange(8, dtype=torch.int64)
    seq_lens = torch.arange(4, dtype=torch.int32)
    computed = torch.zeros(4, dtype=torch.int32)
    runner = SimpleNamespace(
        _hy4_runtime_module=SimpleNamespace(
            CUDAGraphMode=SimpleNamespace(NONE=none_mode)
        ),
        common_attention_metadata_graph=SimpleNamespace(run=submit),
        max_num_reqs=4,
        parallel_config=SimpleNamespace(use_ubatching=ubatching),
        input_batch=SimpleNamespace(block_table=block_table),
        query_start_loc=SimpleNamespace(gpu=starts),
        positions=positions,
        seq_lens=seq_lens,
        num_computed_tokens=computed,
        _mrope_decode_state=None,
    )
    assert (
        runtime.runner_run_common_attention_metadata(
            runner, 2, full_mode if use_graph else none_mode, capture=True
        )
        is result
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is block_table and args[1] == 4
    assert args[2].data_ptr() == starts.data_ptr() and args[2].numel() == 5
    assert args[3] is positions
    assert args[4].data_ptr() == seq_lens.data_ptr()
    assert args[5].data_ptr() == computed.data_ptr()
    assert kwargs == {
        "use_graph": use_graph and not ubatching,
        "capture": True,
        "extra_tensors": (),
    }


def test_platform_adapters_share_runtime_identity(runtime_context):
    native, shared = runtime_context.native, runtime_context.shared
    assert native._CURRENT_RUNNER is shared._CURRENT_RUNNER
    assert native._ABSENT is shared._ABSENT
    assert native._LOCK is shared._LOCK
    assert native.Hy4ExecuteModelState is shared.Hy4ExecuteModelState
    assert (
        runtime_context.cuda_metadata._cpu_sequence_proof
        is runtime_context.metadata._cpu_sequence_proof
    )
    token = shared._CURRENT_RUNNER.set("owner")
    try:
        assert native._CURRENT_RUNNER.get() == "owner"
    finally:
        shared._CURRENT_RUNNER.reset(token)


def test_full_draft_plan_preserves_q4_q1_capacity_contract(graph):
    assert graph.full_draft_plan(common(4, 2), (4, 8, 16), (1, 2, 4)) == (8, 2)
    assert graph.full_draft_plan(common(4, 2), (4, 8, 16), (1, 4)) is None
    assert graph.full_draft_plan(common(1, 2), (4, 8, 16), (1, 2, 4)) is None
