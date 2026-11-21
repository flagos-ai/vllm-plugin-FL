# Copyright (c) 2026 BAAI. All rights reserved.

"""MACA runtime ABI contracts, without loading a GPU library."""

import ctypes
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class TestMacaRuntime(unittest.TestCase):
    def setUp(self):
        names = (
            "torch",
            "vllm",
            "vllm.envs",
            "vllm.distributed",
            "vllm.distributed.device_communicators",
            "vllm.distributed.device_communicators.cuda_wrapper",
            "vllm.distributed.device_communicators.all_reduce_utils",
            "vllm.v1.kv_offload.cpu.host_register",
            "vllm.unrelated",
        )
        modules = {name: types.ModuleType(name) for name in names}
        for name, module in modules.items():
            if "." in name:
                parent, child = name.rsplit(".", 1)
                if parent in modules:
                    setattr(modules[parent], child, module)
        self.target = modules[names[5]]
        self.all_reduce_utils = modules[names[6]]
        self.host_register = modules[names[7]]
        self.unrelated = modules["vllm.unrelated"]
        self.original_runtime = type("CudaRTLibrary", (), {})
        self.original_handle = type("cudaIpcMemHandle_t", (), {})
        self.target.CudaRTLibrary = self.original_runtime
        self.target.cudaIpcMemHandle_t = self.original_handle
        for consumer in (self.all_reduce_utils, self.host_register, self.unrelated):
            consumer.CudaRTLibrary = self.original_runtime
        self.unrelated.cudaIpcMemHandle_t = self.original_handle
        self.custom_handle = self.host_register.cudaIpcMemHandle_t = object()
        self.modules_patch = patch.dict(sys.modules, modules)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        path = Path(__file__).resolve().parents[3] / (
            "vllm_fl/dispatch/backends/vendor/metax/patches/cuda_wrapper.py"
        )
        spec = importlib.util.spec_from_file_location("_metax_cuda_abi_test", path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(self.module)
        self.runtime = self.module.CudaRTLibrary.__new__(self.module.CudaRTLibrary)

    def test_loaded_consumers_and_ipc_handle_use_maca_abi(self):
        for consumer in (self.target, self.all_reduce_utils, self.host_register):
            self.assertIs(consumer.CudaRTLibrary, self.module.CudaRTLibrary)
        self.assertIs(self.target.cudaIpcMemHandle_t, self.module.cudaIpcMemHandle_t)
        self.assertEqual(ctypes.sizeof(self.target.cudaIpcMemHandle_t), 128)
        self.assertIs(self.unrelated.CudaRTLibrary, self.original_runtime)
        self.assertIs(self.unrelated.cudaIpcMemHandle_t, self.original_handle)
        self.assertIs(self.host_register.cudaIpcMemHandle_t, self.custom_handle)

    def test_host_registration_returns_errors_for_pageable_fallback(self):
        register, unregister, last_error = (
            Mock(return_value=801),
            Mock(return_value=3),
            Mock(return_value=801),
        )
        self.runtime.funcs = {
            "cudaHostRegister": register,
            "cudaHostUnregister": unregister,
            "cudaGetLastError": last_error,
        }
        self.assertEqual(self.runtime.cudaHostRegister(0x1234, 4096, 2), 801)
        ptr, size, flags = register.call_args.args
        self.assertEqual((ptr.value, size, flags), (0x1234, 4096, 2))
        self.assertEqual(self.runtime.cudaHostUnregister(0x1234), 3)
        self.assertEqual(unregister.call_args.args[0].value, 0x1234)
        self.assertEqual(self.runtime.cudaGetLastError(), 801)

    def test_ipc_open_uses_maca_flag_and_close_checks_errors(self):
        open_handle, close_handle = Mock(return_value=0), Mock(return_value=7)
        self.runtime.funcs = {
            "cudaIpcOpenMemHandle": open_handle,
            "cudaIpcCloseMemHandle": close_handle,
            "cudaGetErrorString": Mock(return_value=b"bad handle"),
        }
        self.runtime.cudaIpcOpenMemHandle(self.module.cudaIpcMemHandle_t())
        self.assertEqual(open_handle.call_args.args[2], 0)
        with self.assertRaisesRegex(RuntimeError, "bad handle"):
            self.runtime.cudaIpcCloseMemHandle(ctypes.c_void_p(0x1234))

    def test_memcpy_preserves_explicit_direction(self):
        memcpy = Mock(return_value=0)
        self.runtime.funcs = {"cudaMemcpy": memcpy}
        self.runtime.cudaMemcpy(10, 20, 64, 2)
        memcpy.assert_called_once_with(10, 20, 64, 2)


if __name__ == "__main__":
    unittest.main()
