"""Check real vLLM HIP dispatch with CPU tensors and an isolated RoPE provider."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_hip_rotary_keeps_oot_dispatch_and_native_contracts():
    # The provider and HIP identity are stand-ins. vLLM registration, the
    # constructor, forward selection and native controls use the real classes.
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
        entry = metadata.EntryPoint(
            name="cpu_contract", value="__main__:contract_platform",
            group="vllm.platform_plugins",
        )

        def entry_points(*args, **kwargs):
            if kwargs.get("group") == "vllm.platform_plugins":
                return metadata.EntryPoints((entry,))
            return real_entry_points(*args, **kwargs)

        metadata.entry_points = entry_points
        import torch
        from vllm.config import CompilationConfig, CompilationMode
        from vllm.model_executor import custom_op
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        from vllm.model_executor.layers.rotary_embedding import common as rotary_common

        torch.set_default_device("cpu")
        root = Path(sys.argv[1])
        calls = []
        failure = [None]

        class CachedOp:
            def __init__(self, name):
                self.name = name

            def __call__(self, obj, query, key, cos, sin, positions,
                         rotary_interleaved, inplace):
                assert self.name == "rotary_embedding"
                assert positions.ndim == 1
                assert rotary_interleaved == (not obj.is_neox_style)
                assert inplace is True
                calls.append((obj, positions.clone()))
                if failure[0] is not None:
                    query.add_(10)
                    raise failure[0]
                c = cos.index_select(0, positions).unsqueeze(1)
                s = sin.index_select(0, positions).unsqueeze(1)

                def rotate(x):
                    if rotary_interleaved:
                        even, odd = x[..., ::2], x[..., 1::2]
                        return torch.stack(
                            (even * c - odd * s, odd * c + even * s), dim=-1
                        ).flatten(-2)
                    first, second = x.chunk(2, dim=-1)
                    return torch.cat(
                        (first * c - second * s, second * c + first * s), dim=-1
                    )

                query.copy_(rotate(query))
                key.copy_(rotate(key))
                return query, key

        package = types.ModuleType("vllm_fl")
        package.__path__ = [str(root / "vllm_fl")]
        dispatch = types.ModuleType("vllm_fl.dispatch")
        dispatch.CachedOp = CachedOp
        sys.modules["vllm_fl"] = package
        sys.modules["vllm_fl.dispatch"] = dispatch
        spec = importlib.util.spec_from_file_location(
            "hip_rotary_under_test", root / "vllm_fl/ops/rotary_embedding.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        hip = types.SimpleNamespace(
            device_type="cuda", is_rocm=lambda: True, is_cpu=lambda: False,
            is_tpu=lambda: False, is_xpu=lambda: False,
            is_out_of_tree=lambda: True,
        )
        # UnspecifiedPlatform is not CPU: the real inner ApplyRotaryEmb would
        # otherwise import a GPU-only FlashAttention provider during construction.
        # Isolate that unused capability while outer CustomOp still sees HIP.
        cpu = types.SimpleNamespace(is_cpu=lambda: True)
        native_cuda_calls = []

        def unavailable_cuda(*args, **kwargs):
            native_cuda_calls.append((args, kwargs))
            raise NotImplementedError("native CUDA rotary backend unavailable")

        custom_op.CustomOp.register_oot(
            module.RotaryEmbeddingFL, name="RotaryEmbedding"
        )
        name = module.RotaryEmbeddingFL.name
        positions = torch.tensor([[5, 0, 3]])
        original_query = torch.linspace(-2, 2, 48).reshape(1, 3, 16)
        original_key = torch.linspace(3, -1, 24).reshape(1, 3, 8)
        with patch.object(custom_op, "current_platform", hip), patch.object(
            module, "current_platform", hip
        ), patch.object(rotary_common, "current_platform", cpu), patch.object(
            rotary_common, "import_module",
            side_effect=AssertionError("CPU RoPE probe requested GPU FlashAttention"),
        ) as optional_attention_import, patch.object(
            RotaryEmbedding, "forward_cuda", unavailable_cuda
        ):
            for controls, enabled in (
                (["all"], True), (["none"], False),
                (["all", f"-{name}"], False),
                (["none", f"+{name}"], True),
            ):
                config = CompilationConfig(
                    mode=CompilationMode.NONE, custom_ops=controls
                )
                with patch.object(
                    custom_op, "get_cached_compilation_config", lambda: config
                ):
                    for rotary_dim in (4, 8):
                        for neox in (False, True):
                            layer = RotaryEmbedding(
                                8, rotary_dim, 8, 10000.0, neox, torch.float32
                            )
                            assert isinstance(layer, module.RotaryEmbeddingFL)
                            assert layer.apply_rotary_emb.apply_rotary_emb_flash_attn is None
                            assert layer._forward_method.__name__ == (
                                "forward_hip" if enabled else "forward_native"
                            )
                            query, key = original_query.clone(), original_key.clone()
                            expected = layer.forward_native(
                                positions, query.clone(), key.clone()
                            )
                            calls.clear()
                            actual = layer(positions, query, key)
                            for value, reference in zip(actual, expected):
                                torch.testing.assert_close(value, reference)
                            assert len(calls) == int(enabled)
                            if enabled:
                                assert calls[0][0] is layer
                                assert torch.equal(calls[0][1], positions.flatten())
                                torch.testing.assert_close(query, expected[0])
                                torch.testing.assert_close(key, expected[1])
                                if rotary_dim == layer.head_size:
                                    assert actual[0].data_ptr() == query.data_ptr()
                                    assert actual[1].data_ptr() == key.data_ptr()
                            else:
                                torch.testing.assert_close(query, original_query)
                                torch.testing.assert_close(key, original_key)
                            torch.testing.assert_close(
                                query.view(3, 2, 8)[..., rotary_dim:],
                                original_query.view(3, 2, 8)[..., rotary_dim:],
                            )
                            if enabled:
                                calls.clear()
                                error = RuntimeError("provider failed after a write")
                                failure[0] = error
                                query, key = original_query.clone(), original_key.clone()
                                try:
                                    layer(positions, query, key)
                                except RuntimeError as caught:
                                    assert caught is error
                                else:
                                    raise AssertionError("provider error was swallowed")
                                assert len(calls) == 1
                                torch.testing.assert_close(
                                    query.view(3, 2, 8)[..., :rotary_dim],
                                    original_query.view(3, 2, 8)[..., :rotary_dim] + 10,
                                )
                                torch.testing.assert_close(key, original_key)
                                failure[0] = None
            assert not native_cuda_calls

            # Query-only RoPE is outside the public Q/K provider's contract.
            # Preserve the pre-existing upstream HIP/AITER branch unchanged.
            native_hip_calls = []
            native_error = [None]

            def native_hip(obj, positions, query, key=None):
                native_hip_calls.append((obj, positions, query, key))
                if native_error[0] is not None:
                    raise native_error[0]
                return query, key

            config = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"])
            with patch.object(
                custom_op, "get_cached_compilation_config", lambda: config
            ), patch.object(RotaryEmbedding, "forward_hip", native_hip):
                layer = RotaryEmbedding(8, 8, 8, 10000.0, True, torch.float32)
                assert layer.apply_rotary_emb.apply_rotary_emb_flash_attn is None
                calls.clear()
                query = original_query.clone()
                out_query, out_key = layer(positions, query)
                assert out_query is query and out_key is None
                assert len(native_hip_calls) == 1 and not calls
                assert native_hip_calls[0][0] is layer
                assert native_hip_calls[0][1] is positions
                assert native_hip_calls[0][2] is query
                assert native_hip_calls[0][3] is None
                torch.testing.assert_close(query, original_query)
                error = RuntimeError("native HIP error")
                native_error[0] = error
                native_hip_calls.clear()
                try:
                    layer(positions, query, key=None)
                except RuntimeError as caught:
                    assert caught is error
                else:
                    raise AssertionError("native HIP error was swallowed")
                assert len(native_hip_calls) == 1 and not calls
            optional_attention_import.assert_not_called()
        """
    )
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", VLLM_PLUGINS="cpu_contract")
    result = subprocess.run(
        [sys.executable, "-c", probe, str(Path(__file__).resolve().parents[3])],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
