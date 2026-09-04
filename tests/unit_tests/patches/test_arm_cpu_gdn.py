from types import SimpleNamespace

import torch

from vllm_fl.patches.arm_cpu_gdn import (
    apply_arm_cpu_gdn_state_indices_patch,
)


class _Builder:
    def __init__(self, state_indices):
        self.state_indices = state_indices

    def build(self):
        return SimpleNamespace(
            non_spec_state_indices_tensor=self.state_indices,
        )


def test_patch_materializes_singleton_cpu_state_index_stride():
    state_indices = torch.arange(8, dtype=torch.int32).reshape(1, 8)[:, 0]
    assert state_indices.is_contiguous()
    assert state_indices.stride() == (8,)

    class Builder(_Builder):
        pass

    assert apply_arm_cpu_gdn_state_indices_patch(Builder) is True
    metadata = Builder(state_indices).build()

    assert metadata.non_spec_state_indices_tensor.tolist() == [0]
    assert metadata.non_spec_state_indices_tensor.dtype == torch.int32
    assert metadata.non_spec_state_indices_tensor.stride() == (1,)
    assert metadata.non_spec_state_indices_tensor.data_ptr() != state_indices.data_ptr()


def test_patch_is_idempotent_and_avoids_unnecessary_copy():
    state_indices = torch.tensor([3, 5], dtype=torch.int32)

    class Builder(_Builder):
        pass

    original = Builder.build
    assert apply_arm_cpu_gdn_state_indices_patch(Builder) is True
    patched = Builder.build
    assert apply_arm_cpu_gdn_state_indices_patch(Builder) is False
    assert Builder.build is patched
    assert patched._vllm_fl_original is original

    metadata = Builder(state_indices).build()
    assert metadata.non_spec_state_indices_tensor is state_indices


def test_patch_preserves_missing_state_indices():
    class Builder(_Builder):
        pass

    assert apply_arm_cpu_gdn_state_indices_patch(Builder) is True
    metadata = Builder(None).build()
    assert metadata.non_spec_state_indices_tensor is None
