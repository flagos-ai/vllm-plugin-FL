from vllm.model_executor.custom_op import CustomOp

from .activation import *
from .layernorm import *
from .rotary_embedding import *


def register_oot_ops():
    CustomOp.register_oot(_decorated_op_cls=SiluAndMulFL, name="SiluAndMul")
    CustomOp.register_oot(_decorated_op_cls=RMSNormFL, name="RMSNorm")
    CustomOp.register_oot(_decorated_op_cls=RotaryEmbeddingFL, name="RotaryEmbedding")
