"""Exercise real vLLM HIP dispatch with CPU tensors and an isolated provider."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_hip_activations_keep_oot_dispatch_and_native_controls():
    # Import the production activation module without importing FlagGems, whose
    # initialization requires an accelerator. Select the real UnspecifiedPlatform
    # through vLLM's plugin API before its first import: NVML can still detect
    # devices when CUDA_VISIBLE_DEVICES is empty. CachedOp and HIP identity are
    # stand-ins; registration, construction, controls and torch.compile are real.
    probe = textwrap.dedent(
        r"""
        import importlib.util
        from importlib import metadata
        import sys
        import types
        from pathlib import Path
        from unittest.mock import patch

        def contract_platform():
            return "vllm.platforms.interface.UnspecifiedPlatform"

        real_entry_points = metadata.entry_points
        platform_entry = metadata.EntryPoint(
            name="cpu_contract",
            value="__main__:contract_platform",
            group="vllm.platform_plugins",
        )

        def isolated_entry_points(*args, **kwargs):
            if kwargs.get("group") == "vllm.platform_plugins":
                return metadata.EntryPoints((platform_entry,))
            return real_entry_points(*args, **kwargs)

        # A test-local entry point isolates bootstrap without inventing native
        # ABI modules. All other entry-point groups retain their real contents.
        metadata.entry_points = isolated_entry_points
        import torch
        import vllm.platforms as platforms
        from vllm.config import CompilationConfig, CompilationMode
        from vllm.model_executor import custom_op
        from vllm.model_executor.layers import activation as upstream
        from vllm.platforms.interface import UnspecifiedPlatform

        assert type(platforms.current_platform) is UnspecifiedPlatform
        # Also verify selection when the host exposes a native CUDA candidate,
        # as THead's CUDA-compatible NVML does, without loading its NVIDIA ABI.
        with patch.dict(
            platforms.builtin_platform_plugins,
            {"cuda": lambda: "vllm.platforms.cuda.CudaPlatform"},
            clear=True,
        ):
            assert platforms.resolve_current_platform_cls_qualname() == (
                "vllm.platforms.interface.UnspecifiedPlatform"
            )
        assert "vllm.platforms.cuda" not in sys.modules
        assert "vllm._C_stable_libtorch" not in sys.modules

        root = Path(sys.argv[1])
        calls = []

        class CachedOp:
            def __init__(self, name):
                self.name = name

            def __call__(self, obj, x):
                calls.append(self.name)
                gate, up = x.chunk(2, dim=-1)
                if self.name == "silu_and_mul":
                    return torch.nn.functional.silu(gate) * up
                return torch.nn.functional.gelu(
                    gate, approximate=obj.approximate
                ) * up

        package = types.ModuleType("vllm_fl")
        package.__path__ = [str(root / "vllm_fl")]
        dispatch = types.ModuleType("vllm_fl.dispatch")
        dispatch.CachedOp = CachedOp
        sys.modules["vllm_fl"] = package
        sys.modules["vllm_fl.dispatch"] = dispatch
        spec = importlib.util.spec_from_file_location(
            "hip_activation_under_test", root / "vllm_fl/ops/activation.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        hygon = types.SimpleNamespace(
            is_rocm=lambda: True,
            is_cuda_alike=lambda: False,
            is_cpu=lambda: False,
            is_tpu=lambda: False,
            is_xpu=lambda: False,
            is_out_of_tree=lambda: True,
        )
        with patch.object(custom_op, "current_platform", hygon), patch.object(
            upstream, "current_platform", hygon
        ):
            for name, replacement, make in (
                ("SiluAndMul", module.SiluAndMulFL, upstream.SiluAndMul),
                ("GeluAndMul", module.GeluAndMulFL, upstream.GeluAndMul),
            ):
                custom_op.CustomOp.register_oot(replacement, name=name)
                for controls, enabled in (
                    (["all"], True),
                    (["none"], False),
                    (["all", f"-{name}"], False),
                    (["none", f"+{name}"], True),
                ):
                    config = CompilationConfig(
                        mode=CompilationMode.NONE, custom_ops=controls
                    )
                    with patch.object(
                        custom_op, "get_cached_compilation_config", lambda: config
                    ):
                        layer = make()
                        assert isinstance(layer, replacement)
                        assert layer._forward_method.__name__ == (
                            "forward_hip" if enabled else "forward_native"
                        )
                        assert not hasattr(layer, "op")
                        x = torch.linspace(-3, 3, 48).reshape(3, 16)
                        calls.clear()
                        torch.testing.assert_close(layer(x), layer.forward_native(x))
                        assert len(calls) == int(enabled)
                        calls.clear()
                        graphs = []

                        def backend(graph, _example_inputs):
                            graphs.append(graph)
                            return graph.forward

                        compiled = torch.compile(layer, backend=backend, fullgraph=True)
                        compiled(x)
                        calls.clear()
                        updated = x * 0.25 + 0.5
                        actual = compiled(updated)
                        torch.testing.assert_close(actual, layer.forward_native(updated))
                        assert len(calls) == int(enabled)
                        assert len(graphs) == 1
                        if enabled:
                            op_name = (
                                "silu_and_mul" if name == "SiluAndMul" else "gelu_and_mul"
                            )
                            op = getattr(torch.ops.vllm_fl, op_name)
                            assert any(
                                node.target in (op, op.default)
                                for node in graphs[0].graph.nodes
                            ), graphs[0].code
                        torch._dynamo.reset()
        """
    )
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        VLLM_PLUGINS="cpu_contract",
        XMLIR_ENABLE_MOCK_TORCH_COMPILE="false",
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, str(Path(__file__).resolve().parents[3])],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
