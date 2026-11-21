"""Preserve registered native LM-head math; select one qualified pipeline config."""
import hashlib
import inspect
import json
import os
import types
from pathlib import Path


def install():
    import torch
    import triton
    import flag_gems
    from flag_gems.utils.libentry import LibTuner
    registered = flag_gems.linear
    source = Path(inspect.getfile(registered))
    assert registered.__module__ == '_iluvatar.ops.linear'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'd431a4fa144ed67dc3deab46dd5beace3612858075f85c39e602447ba3bf768a'
    target = entry = registered.__globals__['linear_kernel']
    while not isinstance(target, LibTuner):
        target = target.fn
    assert not getattr(target, '_lm_head_r739', False)
    original = target.run
    config = triton.Config({'BLOCK_M':64, 'BLOCK_N':256, 'BLOCK_K':32}, num_warps=8, num_stages=1, num_ctas=1)
    small_config = triton.Config({'BLOCK_M':8, 'BLOCK_N':128, 'BLOCK_K':32}, num_warps=8, num_stages=1, num_ctas=1)
    tail4_config = triton.Config({'BLOCK_M':4, 'BLOCK_N':128, 'BLOCK_K':32}, num_warps=8, num_stages=1, num_ctas=1)
    mid_config = triton.Config({'BLOCK_M':32, 'BLOCK_N':128, 'BLOCK_K':64}, num_warps=8, num_stages=1, num_ctas=1)
    stats = {'selected_tuner_calls':0, 'delegated_tuner_calls':0, 'receipts':[]}

    def eligible(a, w, bias, out, m, n, k):
        return ((3 <= m <= 8 or m in (17,64)) and n == 130560 and k == 2048
                and a.dtype == w.dtype == out.dtype == torch.bfloat16
                and a.shape == (m,2048) and w.shape == (130560,2048) and out.shape == (m,130560)
                and a.stride() == (2048,1) and w.stride() == (2048,1) and out.stride() == (130560,1))

    def selected(self, *args, **kwargs):
        bound = {**dict(zip(self.arg_names, args)), **kwargs}
        if (len(args) < 7 or not eligible(*args[:7])
                or bound.get('BIAS') is not False or bound.get('SPLIT_K',1) != 1):
            stats['delegated_tuner_calls'] += 1
            return original(*args, **kwargs)
        selected_config = (tail4_config if args[4] <= 4 else small_config) if 3 <= args[4] <= 8 else (mid_config if args[4] == 17 else config)
        options = dict(kwargs)
        for key, value in selected_config.all_kwargs().items():
            assert key not in options or options[key] == value
            options[key] = value
        full = {**dict(zip(self.arg_names,args)), **options}
        self.best_config = selected_config
        self.nargs = dict(zip(self.arg_names,args))
        self.benchmark_success_count = self.benchmark_cache_hit_count = 0
        self.bench_time = 0.
        if getattr(self,'shared_config_pre_hook',None) is not None:
            self.shared_config_pre_hook(full)
        elif selected_config.pre_hook is not None:
            selected_config.pre_hook(full)
        try:
            result = self.fn.run(*args, **options)
        finally:
            self.nargs = None
        stats['selected_tuner_calls'] += 1
        receipt = {'shape':[args[4],130560,2048], 'config':selected_config.all_kwargs(),
                   'kernel_hash':result.hash, 'registered_module':registered.__module__,
                   'kernel_body_replaced':False, 'count_scope':'First tuner entry only, not per-call frequency.'}
        if not any(r['shape'][0] == args[4] for r in stats['receipts']):
            stats['receipts'].append(receipt)
            if os.environ.get('TIANSHU_LM_HEAD_RECEIPTS'):
                root = Path(os.environ['TIANSHU_LM_HEAD_RECEIPTS'])
                root.mkdir(parents=True, exist_ok=True)
                (root/f'M{args[4]}_N130560_K2048.json').write_text(json.dumps(receipt,indent=2))
        return result

    target.run = types.MethodType(selected,target)
    target._lm_head_r739 = True
    return {'target':target, 'entry':entry, 'original_run':original, 'selected_run':target.run,
            'eligible':eligible, 'stats':stats}


def install_on_runner(cls):
    import functools
    original = cls.load_model
    @functools.wraps(original)
    def load(self,*args,**kwargs):
        self._lm_head_r739 = install()
        return original(self,*args,**kwargs)
    cls.load_model = load
