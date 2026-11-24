from types import SimpleNamespace
import pytest
import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType
from vllm_fl.quantization.w8a8 import int8_mode, packed

def _args(*, num_bits=8, qtype=QuantizationType.INT, strategy=QuantizationStrategy.CHANNEL, symmetric=True, dynamic=False):
    return SimpleNamespace(num_bits=num_bits, type=qtype, strategy=strategy, symmetric=symmetric, dynamic=dynamic)

def _token_args(**overrides):
    return _args(strategy=QuantizationStrategy.TOKEN, symmetric=True, dynamic=True, **overrides)
PACK = CompressionFormat.pack_quantized.value

def test_is_packed_int8_weight_accepts_supported_metadata():
    assert int8_mode.is_packed_int8_weight(_args(), PACK) is True
    assert int8_mode.is_packed_int8_weight(_args(strategy=QuantizationStrategy.GROUP), PACK) is True

@pytest.mark.parametrize(('weight_quant', 'quant_format'), [(None, PACK), (_args(), 'naive-quantized'), (_args(num_bits=4), PACK), (_args(qtype=object()), PACK), (_args(symmetric=False), PACK), (_args(dynamic=True), PACK), (_args(strategy=QuantizationStrategy.TOKEN), PACK)])
def test_is_packed_int8_weight_rejects_other_metadata(weight_quant, quant_format):
    assert int8_mode.is_packed_int8_weight(weight_quant, quant_format) is False

def test_packed_scheme_rejects_input_not_divisible_by_four():
    scheme = packed.CompressedTensorsPackedW8A8Int8(layer_name='model.linear')
    with pytest.raises(ValueError, match='divisible by 4'):
        scheme.create_weights(torch.nn.Module(), output_partition_sizes=[4], input_size_per_partition=6, params_dtype=torch.bfloat16, weight_loader=lambda *args, **kwargs: None)

def test_packed_scheme_create_and_process_weights(monkeypatch):
    monkeypatch.setattr('vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8.init_int8_linear_kernel', lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr('vllm.model_executor.parameter.get_tensor_model_parallel_rank', lambda : 0)
    monkeypatch.setattr('vllm.model_executor.parameter.get_tensor_model_parallel_world_size', lambda : 1)
    scheme = packed.CompressedTensorsPackedW8A8Int8(layer_name='model.linear')
    layer = torch.nn.Module()
    scheme.create_weights(layer, output_partition_sizes=[4, 4], input_size_per_partition=8, params_dtype=torch.bfloat16, weight_loader=lambda *args, **kwargs: None)
    assert scheme.input_size_per_partition == 8
    assert not hasattr(layer, 'weight')
    assert layer.weight_packed.shape == (8, 2)
    assert layer.weight_shape.shape == (2,)
    unpacked = torch.ones((8, 8), dtype=torch.int8)
    monkeypatch.setattr(packed, 'unpack_uint8b128_int32', lambda *args, **kwargs: unpacked)
    processed = []
    monkeypatch.setattr(packed.CompressedTensorsW8A8Int8, 'process_weights_after_loading', lambda self, layer: processed.append(layer))
    scheme.process_weights_after_loading(layer)
    assert not hasattr(layer, 'weight_packed')
    assert torch.equal(layer.weight, unpacked)
    assert processed == [layer]

def test_install_packed_w8a8_scheme_dispatches_and_is_idempotent(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors as ct_module
    config_cls = ct_module.CompressedTensorsConfig
    original = lambda self, w, i, output_quant=None, format=None, layer_name=None: 'orig'
    monkeypatch.setattr(config_cls, '_get_scheme_from_parts', original)
    marker_scheme = SimpleNamespace(name='packed-scheme')
    monkeypatch.setattr(packed, 'CompressedTensorsPackedW8A8Int8', lambda layer_name=None: (marker_scheme, layer_name))
    assert packed.install_packed_w8a8_scheme() is True
    installed = config_cls._get_scheme_from_parts
    assert getattr(installed, packed._PATCH_MARKER) is True
    assert packed.install_packed_w8a8_scheme() is True
    assert config_cls._get_scheme_from_parts is installed
    config = object.__new__(config_cls)
    config.quant_format = PACK
    monkeypatch.setattr(packed, 'should_use_packed_w8a8', lambda *args: True)
    (returned, layer_name) = installed(config, SimpleNamespace(), SimpleNamespace(), format=None, layer_name='m')
    assert returned is marker_scheme
    assert layer_name == 'm'
    monkeypatch.setattr(packed, 'should_use_packed_w8a8', lambda *args: False)
    assert installed(config, SimpleNamespace(), SimpleNamespace(), format=None) == 'orig'

def test_install_returns_early_when_marker_already_set(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors as ct_module
    config_cls = ct_module.CompressedTensorsConfig

    def already_patched(*args, **kwargs):
        return 'already'
    setattr(already_patched, packed._PATCH_MARKER, True)
    monkeypatch.setattr(config_cls, '_get_scheme_from_parts', already_patched)
    assert packed.install_packed_w8a8_scheme() is True
    assert config_cls._get_scheme_from_parts is already_patched