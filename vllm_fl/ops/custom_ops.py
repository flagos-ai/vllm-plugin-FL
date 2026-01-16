# Copyright (c) 2025 BAAI. All rights reserved.

from vllm.model_executor.custom_op import CustomOp
from .layernorm import *
from .activation import *
from .rotary_embedding import *
from .fused_moe import *

    
def register_oot_ops():
    CustomOp.register_oot(_decorated_op_cls=SiluAndMulFL, name="SiluAndMul")
    CustomOp.register_oot(_decorated_op_cls=RMSNormFL, name="RMSNorm")
    CustomOp.register_oot(_decorated_op_cls=RotaryEmbeddingFL, name="RotaryEmbedding")
    CustomOp.register_oot(_decorated_op_cls=FusedMoEFL, name="FusedMoE")
    CustomOp.register_oot(_decorated_op_cls=UnquantizedFusedMoEMethodFL, name="UnquantizedFusedMoEMethod")

