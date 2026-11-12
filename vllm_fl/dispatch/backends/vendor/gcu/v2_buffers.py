# Copyright (c) 2025 BAAI. All rights reserved.

"""Select upstream explicit-copy V2 buffers on GCU empty builds."""


def install_gcu_v2_buffers():
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu import buffer_utils

    original = buffer_utils.is_uva_available
    if getattr(original, "_fl_gcu_uva_guard", False):
        return

    def is_uva_available():
        # Pinned allocations on GCU are valid, but the empty build has no
        # get_cuda_view_from_cpu_tensor op. Use NonUvaBuffer/StagedWriteTensor's
        # existing device-copy branches without disabling pinned allocations.
        return current_platform.device_type != "gcu" and original()

    is_uva_available._fl_gcu_uva_guard = True
    buffer_utils.is_uva_available = is_uva_available

    from vllm.v1.worker.gpu.block_table import BlockTables

    original_make_ptr_tensor = BlockTables._make_ptr_tensor

    def make_ptr_tensor(self, tensors):
        raw = original_make_ptr_tensor(self, tensors)
        if self.device.type != "gcu":
            return raw
        from vllm_fl.dispatch.backends.vendor.gcu.impl.kv_zeroer import (
            _device_segment_addresses,
        )

        return _device_segment_addresses(raw)

    BlockTables._make_ptr_tensor = make_ptr_tensor
