# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G18 coverage: ModelRunnerFL.sample_tokens / load_model (814b).

``sample_tokens`` (4802-5060) is exercised through its branch matrix:
early PP-only return, grammar bitmask, bookkeeping/logprob passthrough,
async output + broadcast, EAGLE/DFlash GPU tokens, ngram-GPU tokens,
CPU (post-bookkeeping) drafters and the zeroed-draft fallback.

``load_model`` (5527-5747) is exercised through dummy weights, LoRA,
drafter + EPLB, main MoE EPLB, OOM, multimodal pruning, NPU dispatch
warmup, STOCK_TORCH_COMPILE and the three cudagraph wrapping modes.

Only the local module under test is used; heavy collaborators are patched
at the ``vllm_fl.worker.model_runner`` namespace boundary.
"""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm_fl.worker.model_runner as model_runner_module
from vllm_fl.worker.model_runner import ModelRunnerFL


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


@contextlib.contextmanager
def _fake_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class _FakeMRO:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.only = False

    @classmethod
    def with_kv_conn_output_only(cls, kv_connector_output):
        obj = cls(kv_connector_output=kv_connector_output)
        obj.only = True
        return obj


class _FakeAsync:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sampled_token_ids_cpu = kwargs["sampled_token_ids"]
        self.async_copy_ready_event = object()


class _FakeEagle:
    def __init__(self):
        self.prepare_calls = []
        self.dummy_calls = []

    def prepare_next_token_ids_padded(self, *args):
        self.prepare_calls.append(args)
        return torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.int64)

    def dummy_run(self, num_tokens=1):
        self.dummy_calls.append(num_tokens)


class _FakeNgram:
    def __init__(self):
        self.update_calls = []

    def update_token_ids_ngram(self, *args):
        self.update_calls.append(args)
        return (
            torch.zeros(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int64),
            None,
        )


class _InputBatch:
    def __init__(self, req_ids=("a",)):
        self.req_ids = list(req_ids)
        self.sampling_metadata = object()
        self.vocab_size = 32
        self.prev_sampled_token_ids = object()
        self.async_calls = []

    def set_async_sampled_token_ids(self, ids, event):
        self.async_calls.append((ids, event))


def _spec(**flags):
    base = dict(
        use_eagle=lambda: False,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: False,
        use_ngram_gpu=lambda: False,
        use_dflash=lambda: False,
        disable_padded_drafter_batch=False,
    )
    base.update(flags)
    return SimpleNamespace(**base)


@contextlib.contextmanager
def _sample_env(pp=None):
    calls = SimpleNamespace(grammar=[], advance=[])
    pp = pp or SimpleNamespace(is_last_rank=True, world_size=1, rank=0)
    with _stack(
        (model_runner_module, "ModelRunnerOutput", _FakeMRO),
        (model_runner_module, "AsyncGPUModelRunnerOutput", _FakeAsync),
        (model_runner_module, "apply_grammar_bitmask",
         lambda *a: calls.grammar.append(a)),
        (model_runner_module, "record_function_or_nullcontext",
         lambda *a, **k: contextlib.nullcontext()),
        (model_runner_module, "get_pp_group", lambda: pp),
        (model_runner_module, "advance_io_step",
         lambda: calls.advance.append(True)),
        (model_runner_module, "EagleProposer", _FakeEagle),
        (model_runner_module, "NgramProposerGPU", _FakeNgram),
    ):
        yield calls


def _sample_runner(
    *,
    spec=None,
    fits=True,
    async_sched=False,
    event=None,
    supports_mm=False,
    ec=None,
    logprobs=None,
    prompt_lp=None,
    drafter=None,
    dp=1,
):
    ib = _InputBatch()
    sampled = torch.tensor([[3], [7]], dtype=torch.int64)
    sampler = SimpleNamespace(sampled_token_ids=sampled, logprobs_tensors=None)
    scheduler = SimpleNamespace(total_num_scheduled_tokens=2)
    logits = torch.zeros(2, 8)
    state = (
        scheduler,
        logits,
        object(),
        object(),
        torch.zeros(2, 4),
        torch.zeros(2, 4),
        None,
        ec,
        "cgstats",
        {"slot": torch.zeros(2)},
    )
    calls = SimpleNamespace(
        sample=[], update_states=[], receive=[], broadcast=[], propose=[],
        copy_draft=[], bookkeep=[], finalize=[], eplb=[], copy_valid=[],
    )
    runner = SimpleNamespace(
        execute_model_state=state,
        kv_connector_output="kv-in",
        use_async_scheduling=async_sched,
        broadcast_pp_output=False,
        input_batch=ib,
        speculative_config=spec,
        num_spec_tokens=2,
        device=torch.device("cpu"),
        requests={"a": object()},
        discard_request_mask=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        token_ids_gpu_tensor=torch.zeros(2, dtype=torch.int32),
        num_tokens_no_spec_gpu=torch.zeros(2, dtype=torch.int32),
        valid_sampled_token_count_event=event,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        supports_mm_inputs=supports_mm,
        check_ep_fault=False,
        _draft_token_ids=None,
        _draft_probs=object(),
        _draft_prob_req_ids=object(),
        _draft_token_req_ids=object(),
        valid_sampled_token_count_gpu=object(),
        drafter=drafter,
        calls=calls,
    )
    runner._sample = lambda lg, md: calls.sample.append((lg, md)) or sampler
    runner._update_states_after_model_execute = (
        lambda ids, so: calls.update_states.append((ids, so))
    )
    runner._pp_receive_prev_sampled_token_ids_to_input_batch = (
        lambda: calls.receive.append(True)
    )
    runner._pp_broadcast_prev_sampled_token_ids = (
        lambda ids: calls.broadcast.append(ids)
    )
    runner._input_fits_in_drafter = lambda md: fits
    runner.propose_draft_token_ids = (
        lambda *a, **k: calls.propose.append(a) or "draft"
    )
    runner._copy_draft_token_ids_to_cpu = (
        lambda so, zeros_only=False: calls.copy_draft.append(zeros_only)
    )
    runner._copy_valid_sampled_token_count = (
        lambda n, c: calls.copy_valid.append((n, c))
    )
    runner._bookkeeping_sync = lambda *a, **k: calls.bookkeep.append(a) or (
        0, None, logprobs, [4, 5], prompt_lp,
        {"a": 0}, {"a": 0}, [1],
    )
    runner.finalize_kv_connector = lambda: calls.finalize.append(True)
    runner.eplb_step = lambda: calls.eplb.append(True)
    runner._get_or_create_async_output_copy_stream = lambda: "copy-stream"
    runner.sample_tokens = ModelRunnerFL.sample_tokens.__get__(
        runner, ModelRunnerFL
    )
    return runner


def test_sample_tokens_without_state_returns_kv_only():
    runner = _sample_runner()
    runner.execute_model_state = None
    with _sample_env():
        out = runner.sample_tokens(None)
    assert isinstance(out, _FakeMRO)
    assert out.only is True
    assert out.kwargs["kv_connector_output"] == "kv-in"
    assert runner.kv_connector_output is None
    assert runner.calls.receive == []


def test_sample_tokens_without_state_async_receives_prev_ids():
    runner = _sample_runner(async_sched=True)
    runner.execute_model_state = None
    pp = SimpleNamespace(is_last_rank=False, world_size=2, rank=0)
    with _sample_env(pp=pp):
        out = runner.sample_tokens(None)
    assert out.only is True
    assert runner.calls.receive == [True]




def test_sample_tokens_applies_grammar_bitmask():
    runner = _sample_runner()
    sched, logits = runner.execute_model_state[0], runner.execute_model_state[1]
    grammar = object()
    with _sample_env() as env:
        runner.sample_tokens(grammar)
    assert len(env.grammar) == 1
    assert env.grammar[0] == (sched, grammar, runner.input_batch, logits)


def test_sample_tokens_logprobs_passthrough():
    runner = _sample_runner(logprobs="LP", prompt_lp={"p": 1})
    with _sample_env():
        out = runner.sample_tokens(None)
    assert out.kwargs["logprobs"] == "LP"
    assert out.kwargs["prompt_logprobs_dict"] == {"p": 1}
    assert out.kwargs["ec_connector_output"] is None


def test_sample_tokens_async_output_sets_tokens_no_broadcast():
    runner = _sample_runner(async_sched=True)
    with _sample_env():
        out = runner.sample_tokens(None)
    assert isinstance(out, _FakeAsync)
    assert out.kwargs["vocab_size"] == 32
    assert out.kwargs["async_output_copy_stream"] == "copy-stream"
    assert runner.calls.broadcast == []
    assert len(runner.input_batch.async_calls) == 1


def test_sample_tokens_async_broadcasts_when_multi_rank():
    runner = _sample_runner(async_sched=True)
    pp = SimpleNamespace(is_last_rank=True, world_size=2, rank=1)
    with _sample_env(pp=pp):
        runner.sample_tokens(None)
    assert len(runner.calls.broadcast) == 1


def test_sample_tokens_async_skips_broadcast_for_broadcast_pp_output():
    runner = _sample_runner(async_sched=True)
    runner.broadcast_pp_output = True
    pp = SimpleNamespace(is_last_rank=True, world_size=2, rank=1)
    with _sample_env(pp=pp):
        runner.sample_tokens(None)
    assert runner.calls.broadcast == []


def test_sample_tokens_eagle_gpu_tokens_propose_and_finalize():
    drafter = _FakeEagle()
    runner = _sample_runner(
        spec=_spec(use_eagle=lambda: True), drafter=drafter, fits=True
    )
    with _sample_env():
        runner.sample_tokens(None)
    assert len(runner.calls.propose) == 1
    assert runner.calls.finalize == [True]
    assert runner.calls.copy_draft == [False]


def test_sample_tokens_eagle_not_fits_event_prepares_and_dummies():
    drafter = _FakeEagle()
    runner = _sample_runner(
        spec=_spec(use_eagle=lambda: True),
        drafter=drafter,
        fits=False,
        event=object(),
        dp=2,
    )
    with _sample_env():
        runner.sample_tokens(None)
    assert len(drafter.prepare_calls) == 1
    assert drafter.dummy_calls == [1]
    assert len(runner.calls.copy_valid) == 1
    assert runner.calls.copy_draft[-1] is True
    assert runner._draft_token_ids.shape == (1, 2)


def test_sample_tokens_ngram_gpu_proposes_when_fits():
    runner = _sample_runner(
        spec=_spec(use_ngram_gpu=lambda: True), drafter=_FakeNgram(), fits=True
    )
    with _sample_env():
        runner.sample_tokens(None)
    assert len(runner.calls.propose) == 1


def test_sample_tokens_ngram_not_fits_updates_token_ids():
    drafter = _FakeNgram()
    runner = _sample_runner(
        spec=_spec(use_ngram_gpu=lambda: True),
        drafter=drafter,
        fits=False,
        event=object(),
        dp=1,
    )
    with _sample_env():
        runner.sample_tokens(None)
    assert len(drafter.update_calls) == 1
    assert len(runner.calls.copy_valid) == 1


def test_sample_tokens_draft_after_bookkeeping_uses_valid_ids():
    runner = _sample_runner(spec=_spec(), fits=True)
    with _sample_env():
        runner.sample_tokens(None)
    assert len(runner.calls.propose) == 1
    assert runner.calls.propose[0][1] == [4, 5]
    assert runner.calls.finalize == [True]


def test_sample_tokens_draft_after_bookkeeping_not_fits_dummies():
    drafter = _FakeEagle()
    runner = _sample_runner(
        spec=_spec(
            use_eagle=lambda: True, disable_padded_drafter_batch=True
        ),
        drafter=drafter,
        fits=False,
        dp=2,
    )
    with _sample_env():
        runner.sample_tokens(None)
    assert drafter.dummy_calls == [1]
    assert runner.calls.copy_draft[-1] is True


def test_sample_tokens_zeros_draft_when_no_fit_no_event():
    drafter = _FakeEagle()
    runner = _sample_runner(
        spec=_spec(use_eagle=lambda: True), drafter=drafter, fits=False
    )
    with _sample_env():
        runner.sample_tokens(None)
    assert drafter.prepare_calls == []
    assert drafter.dummy_calls == []
    assert runner._draft_token_ids.shape == (1, 2)
    assert runner._draft_probs is None
    assert runner.calls.copy_draft[-1] is True


# ---------------------------------------------------------------- load_model


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


class _CompileModel(nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def compile(self, **kwargs):
        self.calls.append(kwargs)
        return self


class _Loader:
    def __init__(self, calls, model=None, raise_oom=False):
        self.calls = calls
        self.model = model
        self.raise_oom = raise_oom

    def load_model(self, *, vllm_config, model_config):
        self.calls.append(("load_model", vllm_config, model_config))
        if self.raise_oom:
            raise torch.OutOfMemoryError("oom")
        return self.model


class _Profiler:
    def __init__(self, consumed_memory=2048):
        self.consumed_memory = consumed_memory

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Offloader:
    def __init__(self, calls):
        self.calls = calls

    def post_init(self):
        self.calls.append("post_init")


class _Recorder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _EplbState:
    def __init__(self, parallel_config, device):
        self.parallel_config = parallel_config
        self.device = device
        self.added = []
        self.is_async = True
        self.started = False

    def add_model(self, model, config):
        self.added.append((model, config))

    def start_async_loop(self):
        self.started = True


class _Counter:
    def __init__(self):
        self.stock_torch_compile_count = 0


def _load_runner(
    model,
    *,
    spec_config=None,
    drafter=None,
    enable_eplb=False,
    mode=None,
    cudagraph_mode=None,
    use_ubatching=False,
    lora_config=None,
    model_config=None,
    graph_runtime=None,
):
    from vllm.config import CompilationMode, CUDAGraphMode

    if mode is None:
        mode = CompilationMode.NONE
    if cudagraph_mode is None:
        cudagraph_mode = CUDAGraphMode.NONE
    compilation = SimpleNamespace(
        mode=mode,
        cudagraph_mode=cudagraph_mode,
        init_backend=lambda cfg: "backend",
    )
    model_config = model_config or SimpleNamespace(
        model="t/m",
        enforce_eager=True,
        quantization=None,
        multimodal_config=None,
        revision="r",
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation, speculative_config=spec_config
    )
    calls = SimpleNamespace(
        loader=[], comm=[], hooks=[], warmup=[], io=[], offload=[], lora=[]
    )
    runner = SimpleNamespace(
        model_config=model_config,
        load_config=SimpleNamespace(load_format="safetensors"),
        vllm_config=vllm_config,
        parallel_config=SimpleNamespace(
            enable_eplb=enable_eplb,
            enable_elastic_ep=False,
            use_ubatching=use_ubatching,
        ),
        lora_config=lora_config,
        device=torch.device("cpu"),
        model=model,
        _moe_model=None,
        compilation_config=compilation,
        graph_runtime=graph_runtime,
        use_aux_hidden_state_outputs=False,
        speculative_config=spec_config,
        jit_warmup_registry=SimpleNamespace(
            activate=lambda: contextlib.nullcontext()
        ),
        eplb_state=None,
        calls=calls,
    )
    runner._setup_eagle3_aux_hidden_state_outputs = (
        ModelRunnerFL._setup_eagle3_aux_hidden_state_outputs.__get__(
            runner, ModelRunnerFL
        )
    )
    runner.get_model = lambda: runner.model
    runner.load_lora_model = (
        lambda m, cfg, dev: calls.lora.append(m) or model
    )
    runner.load_model = ModelRunnerFL.load_model.__get__(runner, ModelRunnerFL)
    if drafter is not None:
        runner.drafter = drafter
    return runner


@contextlib.contextmanager
def _load_env(
    runner,
    loader,
    *,
    moe_fn=None,
    pruner=False,
    breakable=False,
    wrapper_cls=None,
    ubatch_cls=None,
    platform=None,
):
    calls = runner.calls
    moe_fn = moe_fn or (lambda m: None)
    wrapper_cls = wrapper_cls or _Recorder
    ubatch_cls = ubatch_cls or _Recorder
    pairs = [
        (model_runner_module, "init_io_dump_from_env",
         lambda *a, **k: calls.io.append(a)),
        (model_runner_module, "get_model_loader", lambda cfg: loader),
        (model_runner_module, "DeviceMemoryProfiler", _Profiler()),
        (model_runner_module, "warmup_cached_ops",
         lambda *a, **k: calls.warmup.append(True)),
        (model_runner_module, "_prepare_comm_buffer",
         lambda m: calls.comm.append(m)),
        (model_runner_module, "register_io_module_hooks",
         lambda m: calls.hooks.append(m)),
        (model_runner_module, "supports_multimodal_pruning", lambda m: pruner),
        (model_runner_module, "get_offloader",
         lambda: _Offloader(calls.offload)),
        (model_runner_module, "get_mixture_of_experts_model", moe_fn),
        (model_runner_module, "EplbState", _EplbState),
        (model_runner_module, "format_gib", lambda x: "0.0"),
        (model_runner_module, "is_breakable_cudagraph_enabled",
         lambda: breakable),
        (model_runner_module, "BreakableCUDAGraphWrapper", wrapper_cls),
        (model_runner_module, "UBatchWrapper", ubatch_cls),
        (model_runner_module, "resolve_obj_by_qualname", lambda q: wrapper_cls),
        (model_runner_module, "compilation_counter", _Counter()),
    ]
    if platform is not None:
        pairs.append((model_runner_module, "current_platform", platform))
    with _stack(*pairs):
        yield calls


def test_load_model_happy_path_stores_model_and_flags():
    model = _Model()
    runner = _load_runner(model)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader):
        runner.load_model()
    assert ("load_model", runner.vllm_config, runner.model_config) in (
        runner.calls.loader
    )
    assert runner.model is model
    assert runner.model_memory_usage == 2048
    assert runner.is_multimodal_pruning_enabled is False
    assert runner.requires_sequential_video_encoding is False
    assert runner.calls.comm == [model]
    assert runner.calls.hooks == [model]
    assert len(runner.calls.warmup) == 1
    assert runner.calls.offload == ["post_init"]
    assert runner.calls.io[0] == (True,)
    assert runner._moe_model is None


def test_load_model_dummy_weights_skips_comm_and_io_hooks():
    model = _Model()
    runner = _load_runner(model)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader):
        runner.load_model(load_dummy_weights=True)
    assert runner.load_config.load_format == "dummy"
    assert runner.calls.comm == []
    assert runner.calls.hooks == []
    assert len(runner.calls.warmup) == 1
    assert runner.calls.offload == ["post_init"]


def test_load_model_lora_wraps_model():
    model = _Model()
    runner = _load_runner(model, lora_config=SimpleNamespace())
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader):
        runner.load_model()
    assert runner.calls.lora == [model]
    assert runner.model is model


def test_load_model_drafter_eplb_registers_moe():
    model = _Model()
    drafter_model = _Model()
    drafter = SimpleNamespace(
        model=drafter_model,
        load_model=lambda m: None,
        set_eplb_state=lambda state: setattr(drafter, "state", state),
    )
    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(model="draft/m")
    )

    def moe_fn(m):
        return ("moe", m) if m is drafter_model else None

    runner = _load_runner(
        model, spec_config=spec_config, drafter=drafter, enable_eplb=True
    )
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=moe_fn):
        runner.load_model()
    assert runner.eplb_state is not None
    assert runner.eplb_state.added == [
        (("moe", drafter_model), spec_config.draft_model_config)
    ]
    assert drafter.state is runner.eplb_state
    assert runner._moe_model is None


def test_load_model_main_moe_eplb_starts_async_loop():
    model = _Model()
    runner = _load_runner(model, enable_eplb=True)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, moe_fn=lambda m: ("moe", m)):
        runner.load_model()
    assert runner._moe_model == ("moe", model)
    assert runner.eplb_state.added == [(("moe", model), runner.model_config)]
    assert runner.eplb_state.started is True


def test_load_model_out_of_memory_reraises():
    model = _Model()
    runner = _load_runner(model)
    loader = _Loader(runner.calls.loader, raise_oom=True)
    with _load_env(runner, loader):
        with pytest.raises(torch.OutOfMemoryError):
            runner.load_model()
    assert runner.calls.comm == []


def test_load_model_multimodal_pruning_and_sequential_video():
    model = _Model()
    model.requires_sequential_video_encoding = True
    mm_config = SimpleNamespace(is_multimodal_pruning_enabled=lambda: True)
    model_config = SimpleNamespace(
        model="t/m",
        enforce_eager=True,
        quantization=None,
        multimodal_config=mm_config,
        revision="r",
    )
    runner = _load_runner(model, model_config=model_config)
    loader = _Loader(runner.calls.loader, model)
    with _load_env(runner, loader, pruner=True):
        runner.load_model()
    assert runner.is_multimodal_pruning_enabled is True
    assert runner.requires_sequential_video_encoding is True


def test_load_model_npu_dispatch_warmup():
    from vllm.config import CompilationMode

    model = _Model()
    runner = _load_runner(model, mode=CompilationMode.VLLM_COMPILE)
    loader = _Loader(runner.calls.loader, model)
    dispatch_calls = []
    platform = SimpleNamespace(device_type="npu")
    with _fake_module(
        "vllm_fl.dispatch",
        warmup_cached_ops=lambda: dispatch_calls.append(True),
    ):
        with _load_env(runner, loader, platform=platform):
            runner.load_model()
    assert dispatch_calls == [True]
    assert len(runner.calls.warmup) == 1
    assert runner.calls.offload == ["post_init"]


def test_load_model_stock_torch_compile_compiles_and_returns():
    from vllm.config import CompilationMode

    model = _Model()
    compile_calls = []
    compile_model = _CompileModel(compile_calls)
    runner = _load_runner(compile_model, mode=CompilationMode.STOCK_TORCH_COMPILE)
    loader = _Loader(runner.calls.loader, compile_model)
    env_calls = []
    with _fake_module(
        "vllm.env_override",
        _apply_constrain_to_fx_strides_patch=lambda: env_calls.append(True),
    ):
        with _load_env(runner, loader):
            runner.load_model()
    assert env_calls == [True]
    assert compile_calls == [{"fullgraph": True, "backend": "backend"}]
    assert runner.calls.offload == []


def test_load_model_cudagraph_wrapping_modes():
    from vllm.config import CUDAGraphMode

    def run_case(*, breakable, cudagraph_mode, use_ubatching, drafter=None,
                 platform=None):
        model = _Model()
        runner = _load_runner(
            model,
            cudagraph_mode=cudagraph_mode,
            use_ubatching=use_ubatching,
            drafter=drafter,
        )
        loader = _Loader(runner.calls.loader, model)
        with _load_env(
            runner, loader, breakable=breakable, platform=platform
        ):
            runner.load_model()
        return runner, model

    runner, model = run_case(
        breakable=True,
        cudagraph_mode=CUDAGraphMode.FULL,
        use_ubatching=False,
        drafter=SimpleNamespace(model=_Model(), load_model=lambda m: None),
    )
    assert isinstance(runner.model, _Recorder)
    assert runner.model.args[0] is model
    assert runner.calls.offload == ["post_init"]

    platform = SimpleNamespace(
        device_type="cpu", get_static_graph_wrapper_cls=lambda: "W"
    )
    runner, model = run_case(
        breakable=False,
        cudagraph_mode=CUDAGraphMode.FULL,
        use_ubatching=False,
        platform=platform,
    )
    assert isinstance(runner.model, _Recorder)
    assert runner.model.kwargs["runtime_mode"] == CUDAGraphMode.FULL

    runner, model = run_case(
        breakable=False,
        cudagraph_mode=CUDAGraphMode.FULL,
        use_ubatching=True,
    )
    assert isinstance(runner.model, _Recorder)
    assert runner.model.args[2] == CUDAGraphMode.FULL

    runner, model = run_case(
        breakable=False,
        cudagraph_mode=CUDAGraphMode.NONE,
        use_ubatching=True,
    )
    assert isinstance(runner.model, _Recorder)
    assert runner.model.args[2] == CUDAGraphMode.NONE
