"""Host selection policies; native JIT source and cache identity stay intact."""
import torch
import triton
import triton.language as tl
from flag_gems.utils.libentry import LibTuner
from ._qualified_grouped_mm import mm_kernel as _qualified_grouped_mm_jit

class _QualifiedMMTuner(LibTuner.get('default')):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._qualified_grouped_fn=triton.heuristics(dict(self.fn.values))(_qualified_grouped_mm_jit)

    @staticmethod
    def select(a,b,c,m,n,k):
        if not (a.dtype==b.dtype==c.dtype==torch.bfloat16
                and a.shape==(m,k) and b.shape==(k,n) and c.shape==(m,n)
                and a.stride()==(k,1) and b.stride()==(1,k) and c.stride()==(n,1)):
            return None
        if isinstance(m,int) and 1793<=m<=2048 and (n,k) in [(2560,2048),(2048,2048)]:
            return ((128,128,32,8,2) if n==2560 else (256,256,32,16,2),False,'qkv_o')
        if isinstance(m,int) and (897<=m<=1024 or 1152<=m<=2048) and (n,k)==(2048,6144):
            return ((128,128,32,8,2) if 1152<=m<=1536 else (256,256,32,16,2),False,'down')
        if 1793<=m<=2048 and (n,k)==(12288,2048):
            return ((256,256,32,16,2),True,'grouped')
        original={(2560,2048):(256,128,32,8,2),
                  (2048,2048):(256,128,32,8,2),
                  (12288,2048):(256,256,32,16,2),
                  (2048,6144):(256,128,32,8,2)}
        if 1793<=m<=2048 and (n,k) in original:
            return (original[n,k],False,'near_full')
        return None

    def run(self,*args,**kwargs):
        selected=self.select(*args[:6]) if len(args)>=12 else None
        if selected is None:
            return super().run(*args,**kwargs)
        (bm,bn,bk,warps,stages),grouped,rule=selected
        config=triton.Config({'BLOCK_M':bm,'BLOCK_N':bn,'BLOCK_K':bk,'SPLIT_K':1},
                            num_warps=warps,num_stages=stages,num_ctas=1)
        assert config.all_kwargs() in [c.all_kwargs() for c in self.configs]
        options=dict(kwargs)
        for key,value in config.all_kwargs().items():
            assert key not in options or options[key]==value
            options[key]=value
        if grouped:
            assert kwargs['GROUP_M']==8
        if rule in ['qkv_o','down']:
            assert options['fp8_fast_accum'] is True and options['AB_DTYPE']==tl.bfloat16
        self.best_config=config
        self.nargs=dict(zip(self.arg_names,args))
        self.benchmark_success_count=self.benchmark_cache_hit_count=0
        self.bench_time=0.
        full={**self.nargs,**options}
        if getattr(self,'shared_config_pre_hook',None) is not None:
            self.shared_config_pre_hook(full)
        elif config.pre_hook is not None:
            config.pre_hook(full)
        try:
            function=self._qualified_grouped_fn if grouped else self.fn
            return function.run(*args,**options)
        finally:
            self.nargs=None


class _QualifiedLinearTuner(LibTuner.get("default")):
    # Inherit the EXACT default tuner selected by the original decorator.

    @staticmethod
    def qualified(a, w, bias, out, m, n, k):
        return ((3 <= m <= 8 or m in (17, 64)) and n == 130560 and k == 2048
                and a.dtype == w.dtype == out.dtype == torch.bfloat16
                and a.shape == (m, 2048) and w.shape == (130560, 2048)
                and out.shape == (m, 130560)
                and a.stride() == (2048, 1) and w.stride() == (2048, 1)
                and out.stride() == (130560, 1))

    def run(self, *args, **kwargs):
        bound = {**dict(zip(self.arg_names, args)), **kwargs}
        if (len(args) < 7 or not self.qualified(*args[:7])
                or bound.get('BIAS') is not False or bound.get('SPLIT_K', 1) != 1):
            return super().run(*args, **kwargs)
        m = args[4]
        if m <= 4:
            bm, bn, bk = 4, 128, 32
        elif m <= 8:
            bm, bn, bk = 8, 128, 32
        elif m == 17:
            bm, bn, bk = 32, 128, 64
        else:
            bm, bn, bk = 64, 256, 32
        config = triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
                               num_warps=8, num_stages=1, num_ctas=1)
        options = dict(kwargs)
        for key, value in config.all_kwargs().items():
            assert key not in options or options[key] == value
            options[key] = value
        full = {**dict(zip(self.arg_names, args)), **options}
        self.best_config = config
        self.nargs = dict(zip(self.arg_names, args))
        self.benchmark_success_count = self.benchmark_cache_hit_count = 0
        self.bench_time = 0.
        if getattr(self, 'shared_config_pre_hook', None) is not None:
            self.shared_config_pre_hook(full)
        elif config.pre_hook is not None:
            config.pre_hook(full)
        try:
            return self.fn.run(*args, **options)
        finally:
            self.nargs = None

