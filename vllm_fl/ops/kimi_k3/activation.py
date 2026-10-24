from vllm.model_executor.layers.activation import SituAndMul
from vllm_fl.dispatch import CachedOp

situ_out = CachedOp("kimi_k3_situ_out")
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.activations import _can_fuse_situ_input


class K3SituAndMul(SituAndMul):
    """K3's explicit OOT entry; no global platform override or extra registration."""

    def forward_oot(self, x):
        if _can_fuse_situ_input(x, self.beta, self.linear_beta):
            # The inherited CUDA-alike entry calls the one registered situ_out.
            return self.forward_cuda(x)
        # Preserve the actual old OOT fallback, without adding an out-copy.
        return self.forward_native(x)
