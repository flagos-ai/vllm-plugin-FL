from typing import Optional, Union
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from flag_gems.modules.normalization import gems_rms_forward

class RMSNormFL(RMSNorm):
    def __init__(self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return gems_rms_forward(x, residual, self.weight, self.variance_epsilon)
    
__all__ = ["RMSNormFL"]
        
