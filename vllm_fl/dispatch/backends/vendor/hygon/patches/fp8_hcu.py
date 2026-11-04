"""Task snapshot: HCU FP8 channel dense/Marlin experts behind FL on vLLM empty.

No vllm_hcu platform/import hook and no old vLLM shared libraries are loaded.
Weight layout and LightOp calls follow the installed HCU 0.21 FP8 reference.
Only TP, dynamic token activations and channel FP8 weights are supported here.
"""
import logging
import torch
from torch.nn import Parameter
from compressed_tensors.quantization import QuantizationStrategy
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import CompressedTensorsW8A8Fp8MoEMethod
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.utils import replace_parameter

logger=logging.getLogger(__name__)
_libs=[]

def pack_fp8(weight):
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 3 or weight.shape[-1]%64:
        raise ValueError('HCU FP8 Marlin requires E,N,K e4m3fn with K divisible by 64')
    e,n,k=weight.shape
    return weight.transpose(1,2).reshape(e,k//64,64,n).transpose(2,3).contiguous().reshape(e,k//64,n*64)

def run_fp8_moe(layer,x,weights,ids,inplace=True):
    from lightop.moe import fused_experts_impl_fp8_marlin
    return fused_experts_impl_fp8_marlin(
        hidden_states=x,w1=layer.w13_weight,w2=layer.w2_weight,
        topk_weights=weights,topk_ids=ids,inplace=inplace,
        activation=layer.activation.value,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        use_fp8_w8a8=True,per_channel_quant=True,
        global_num_experts=layer.global_num_experts,expert_map=layer.expert_map,
        w1_scale=layer.w13_weight_scale,w2_scale=layer.w2_weight_scale,
        a1_scale=layer.w13_input_scale,a2_scale=layer.w2_input_scale,
        use_nn_moe=False,i_q=None,i_s=None,shared_output=None,
        routed_scaling_factor=1.0)

class HcuFp8MarlinMoE(CompressedTensorsW8A8Fp8MoEMethod):
    def __init__(self,weight_quant,input_quant,moe,layer_name=None):
        FusedMoEMethodBase.__init__(self,moe)
        if not (weight_quant.strategy==QuantizationStrategy.CHANNEL and input_quant.strategy==QuantizationStrategy.TOKEN and input_quant.dynamic):
            raise ValueError('HCU reference requires channel FP8 + dynamic token FP8')
        p=moe.moe_parallel_config
        if p.dp_size>1 or p.use_ep or p.use_all2all_kernels:
            raise NotImplementedError('This HCU FP8 adapter is TP only')
        self.weight_quant=weight_quant;self.input_quant=input_quant
        self.static_input_scales=False;self.weight_block_size=None;self.block_quant=False
        from lightop.moe import fused_experts_impl_fp8_marlin

    @property
    def supports_internal_mk(self): return True
    @property
    def is_monolithic(self): return False
    @property
    def supports_eplb(self): return False

    def get_fused_moe_quant_config(self,layer):
        return fp8_w8a8_moe_quant_config(w1_scale=layer.w13_weight_scale,w2_scale=layer.w2_weight_scale,a1_scale=None,a2_scale=None,per_act_token_quant=True,per_out_ch_quant=False,block_shape=None)

    def process_weights_after_loading(self,layer):
        self.moe_quant_config=self.get_fused_moe_quant_config(layer)
        replace_parameter(layer,'w13_weight',pack_fp8(layer.w13_weight))
        replace_parameter(layer,'w2_weight',pack_fp8(layer.w2_weight))
        logger.info('HCU_REFERENCE_FP8_MOE: LightOp Marlin, W1=%s W2=%s',tuple(layer.w13_weight.shape),tuple(layer.w2_weight.shape))

    def apply(self,layer,x,topk_weights,topk_ids,shared_experts=None,shared_experts_input=None):
        return run_fp8_moe(layer,x,topk_weights,topk_ids,inplace=not getattr(self.moe,'disable_inplace',False))

def install():
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import compressed_tensors_moe_w8a8_fp8 as moe_module
    moe_module.CompressedTensorsW8A8Fp8MoEMethod=HcuFp8MarlinMoE

    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import ChannelWiseTorchFP8ScaledMMLinearKernel,TorchFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
    original=CompressedTensorsW8A8Fp8.process_weights_after_loading
    def process(self,layer):
        original(self,layer)
        if self.strategy==QuantizationStrategy.CHANNEL:
            # Upstream stores K,N; HCU channelwise GEMM consumes contiguous N,K.
            layer.weight=Parameter(layer.weight.T.contiguous(),requires_grad=False)
    CompressedTensorsW8A8Fp8.process_weights_after_loading=process
    TorchFP8ScaledMMLinearKernel.get_output_padding=lambda self:None
    def dense(self,*,A,B,As,Bs,out_dtype,bias,output_shape):
        from lightop.gemm_ops import hipblaslt_w8a8_channelwise_gemm
        m,k=A.shape;n=B.shape[0]
        _,out=hipblaslt_w8a8_channelwise_gemm(a=A,b=B,scale_a=As,scale_b=Bs,m=m,n=n,k=k,transpose_flag='NT',out_dtype=out_dtype,bias=bias)
        return out.reshape(-1,n)[:m].view(*output_shape[:-1],n)
    ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm=dense

    # empty supplies schemas through FL; provide a real vendor device kernel.
    name='_C::dynamic_per_token_scaled_fp8_quant'
    if not torch._C._dispatch_has_kernel_for_dispatch_key(name,'CUDA'):
        lib=torch.library.Library('_C','IMPL','CUDA')
        def token_quant(result,x,scale,scale_ub):
            if scale_ub is not None:
                raise NotImplementedError('Vendor token FP8 adapter has no scale_ub support')
            from lightop.quant import per_token_quant_fp8
            per_token_quant_fp8(x,out_q=result,out_scale=scale)
        lib.impl('dynamic_per_token_scaled_fp8_quant',token_quant)
        _libs.append(lib)
    logger.info('HCU_REFERENCE_FP8_ENABLED: dense=LightOp channelwise NT; MoE=LightOp FP8 Marlin; token quant=LightOp; vLLM native extensions absent')

    cache_lib=torch.library.Library('_C_cache_ops','FRAGMENT')
    cache_lib.define('concat_and_cache_mla(Tensor kv_c, Tensor k_pe, Tensor(a!) kv_cache, Tensor slot_mapping, str kv_cache_dtype, Tensor scale) -> ()')
    def cache_write(kv,rope,cache,slots,dtype,scale):
        if dtype!='fp8_ds_mla':
            raise NotImplementedError('This cache bridge only supports fp8_ds_mla')
        from .hcu_fp8_cache import write_hcu_cache
        write_hcu_cache(kv,rope,cache,slots)
    cache_lib.impl('concat_and_cache_mla',cache_write,'CUDA')
    cache_lib.define('cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor(a!) dst, Tensor block_table, Tensor seq_lens, Tensor workspace_starts, int batch_size) -> ()')
    def gather(src,dst,blocks,lens,starts,batch):
        import lightop
        lightop.op.cp_gather_and_upconvert_fp8_kv_cache(src,dst,blocks,lens,starts,batch)
    cache_lib.impl('cp_gather_and_upconvert_fp8_kv_cache',gather,'CUDA')
    cache_lib.define('concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor(a!) q_out) -> ()')
    def concat(nope,rope,out): torch.cat((nope,rope),dim=-1,out=out)
    cache_lib.impl('concat_mla_q',concat,'CUDA')
    _libs.append(cache_lib)
    logger.info('EMPTY_FP8_CACHE_BRIDGE: HCU-scale packed writer; LightOp gather; torch.cat query concat')
