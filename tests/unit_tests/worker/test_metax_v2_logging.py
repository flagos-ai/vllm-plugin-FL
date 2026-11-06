"""Real official logging hierarchy contracts, isolated from pytest's root logger."""

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _official_logger_source():
    source = os.environ.get("VLLM_UPSTREAM_SOURCE_ROOT")
    if source:
        path = Path(source) / "vllm/logger.py"
        assert path.is_file(), path
        return path
    spec = importlib.util.find_spec("vllm")
    if spec is None:
        pytest.skip("Requires official vLLM 0.32 source or installed package")
    return Path(spec.origin).parent / "logger.py"


def _logging_contract(logger_path, worker_path, helper_path, vendor, use_v2):
    # This function runs in a fresh CPU-only Python interpreter. In particular,
    # pytest caplog handlers cannot make the old vllm_fl INFO visible by accident.
    import __future__

    import ast
    import builtins
    import copy
    import functools
    import io
    import logging
    import logging.config
    from collections.abc import Hashable
    from contextlib import redirect_stdout
    from pathlib import Path
    from types import MethodType, SimpleNamespace
    from typing import cast

    source = ast.parse(Path(logger_path).read_text(encoding="utf-8"))
    names = {"_print_info_once", "_VllmLogger", "init_logger"}
    nodes = [
        node
        for node in source.body
        if getattr(node, "name", None) in names
        or isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_METHODS_TO_PATCH"
            for target in node.targets
        )
        or isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "DEFAULT_LOGGING_CONFIG"
    ]
    scope = {
        "logging": logging,
        "Logger": logging.Logger,
        "MethodType": MethodType,
        "lru_cache": functools.lru_cache,
        "Hashable": Hashable,
        "cast": cast,
        "LogScope": str,
        # The test represents local rank0. Do not import distributed/GPU modules.
        "_should_log_with_scope": lambda scope: True,
        "envs": SimpleNamespace(
            VLLM_LOGGING_LEVEL="INFO", VLLM_LOGGING_STREAM="ext://sys.stdout"
        ),
        "_DATE_FORMAT": "%m-%d %H:%M:%S",
        "_FORMAT": "%(name)s:%(message)s",
        "_use_color": lambda: False,
    }
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            logger_path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    configuration = copy.deepcopy(scope["DEFAULT_LOGGING_CONFIG"])
    assert set(configuration["loggers"]) == {"vllm"}
    assert configuration["loggers"]["vllm"]["propagate"] is False
    # Only replace the custom formatter factory to avoid importing vLLM/Torch.
    # The actual official handler, levels and logger hierarchy are unchanged.
    configuration["formatters"] = {"plain": {"format": "%(name)s:%(message)s"}}
    for handler in configuration["handlers"].values():
        handler["formatter"] = "plain"
    logging.root.setLevel(logging.WARNING)
    logging.root.handlers = []
    stream = io.StringIO()
    with redirect_stdout(stream):
        logging.config.dictConfig(configuration)
    init_logger = scope["init_logger"]

    worker_tree = ast.parse(Path(worker_path).read_text(encoding="utf-8"))
    helper_tree = ast.parse(Path(helper_path).read_text(encoding="utf-8"))

    def logger_assignment(tree):
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "logger"
                for target in node.targets
            )
        )

    generic = {"__name__": "vllm_fl.worker.worker", "init_logger": init_logger}
    exec(
        compile(
            ast.Module(body=[logger_assignment(worker_tree)], type_ignores=[]),
            worker_path,
            "exec",
        ),
        generic,
    )
    old_logger = generic["logger"]
    assert old_logger.name == "vllm_fl.worker.worker"
    assert old_logger.getEffectiveLevel() == logging.WARNING
    assert not old_logger.isEnabledFor(logging.INFO)
    # Original9be marker behavior: plugin namespace INFO is silently filtered.
    old_logger.info_once("MetaX V2 model runner: %s.%s", "old_actual", "Object")
    assert stream.getvalue() == ""

    helper_scope = {"__name__": "vllm_fl.worker.metax_v2", "init_logger": init_logger}
    exec(
        compile(
            ast.Module(body=[logger_assignment(helper_tree)], type_ignores=[]),
            helper_path,
            "exec",
        ),
        helper_scope,
    )
    assert helper_scope["logger"].name == "vllm.metax_v2"
    prepare = next(
        node
        for node in helper_tree.body
        if getattr(node, "name", None) == "prepare_metax_v2_runtime"
    )
    fallback_info = next(
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "logger.info_once"
    )
    events = []

    def prepare_cpu(config):
        events.append("prepare")
        exec(
            compile(
                ast.Module(body=[fallback_info], type_ignores=[]), helper_path, "exec"
            ),
            helper_scope,
        )

    class ActualReturnedRunner:
        pass

    def official(config, device):
        events.append("construct")
        return ActualReturnedRunner()

    def selected_import(name, *args, **kwargs):
        if name == "vllm_fl.worker.metax_v2":
            return SimpleNamespace(prepare_metax_v2_runtime=prepare_cpu)
        if name == "vllm.v1.worker.gpu.model_runner":
            return SimpleNamespace(GPUModelRunner=official)
        if name == "vllm_fl.worker.model_runner":
            return SimpleNamespace(ModelRunnerFL=official)
        raise AssertionError(name)

    worker = next(
        node
        for node in worker_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    initialize = next(
        node for node in worker.body if getattr(node, "name", None) == "init_device"
    )
    branch = next(
        node
        for node in ast.walk(initialize)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.use_v2_model_runner"
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "vllm.v1.worker.gpu.model_runner"
            for child in node.body
        )
    )
    state = SimpleNamespace(
        use_v2_model_runner=use_v2, vllm_config=object(), device=object()
    )
    namespace = {
        "self": state,
        "current_platform": SimpleNamespace(vendor_name=vendor),
        "init_logger": init_logger,
        "logger": old_logger,
        "__builtins__": {**vars(builtins), "__import__": selected_import},
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
            worker_path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    actual = type(state.model_runner)
    expected = "MetaX V2 model runner: " + actual.__module__ + "." + actual.__qualname__
    output = stream.getvalue()
    should_log = vendor == "metax" and use_v2
    assert (expected in output) is should_log
    assert ("MetaX V2 uses upstream NonUvaBuffer" in output) is should_log
    assert events == (["prepare", "construct"] if should_log else ["construct"])
    assert old_logger.getEffectiveLevel() == logging.WARNING and not old_logger.handlers
    assert "old_actual.Object" not in output
    assert "GPUModelRunner" not in expected  # A hard-coded type marker would fail.
    assert logging.getLogger("vllm.metax_v2").getEffectiveLevel() == logging.INFO
    return {
        "vendor": vendor,
        "use_v2": use_v2,
        "expected_actual_type_marker": expected,
        "output": output,
        "events": events,
        "old_effective_level": old_logger.getEffectiveLevel(),
        "native_effective_level": logging.getLogger(
            "vllm.metax_v2"
        ).getEffectiveLevel(),
        "old_general_worker_logger_unchanged": True,
    }


@pytest.mark.parametrize(
    "vendor,use_v2",
    [("metax", True), ("metax", False), ("nvidia", True), ("ascend", True)],
)
def test_metax_v2_actual_object_marker_visible_in_official_logging_hierarchy(
    vendor, use_v2
):
    paths = [
        str(_official_logger_source()),
        str(ROOT / "vllm_fl/worker/worker.py"),
        str(ROOT / "vllm_fl/worker/metax_v2.py"),
        vendor,
        use_v2,
    ]
    code = (
        inspect.getsource(_logging_contract)
        + "\nimport json\nprint(json.dumps(_logging_contract(*json.loads("
        + repr(json.dumps(paths))
        + "))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(result.stdout)
    assert proof["old_general_worker_logger_unchanged"] is True
    assert proof["old_effective_level"] == 30 and proof["native_effective_level"] == 20
