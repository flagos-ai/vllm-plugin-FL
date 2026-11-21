"""Experiment-local selection of existing native configs for near-full M tiles.

The registered native kernel and heuristics remain in use. The 1793..2048
domain has eight 256-row tiles; four BF16 linear shapes with exact strides
are qualified separately. No persistent ConfigCache is written by this rule.
"""
import hashlib
import inspect
import json
import os
import types
from pathlib import Path

CONFIG = {(2560, 2048): (256, 128, 32, 8, 2),
          (2048, 2048): (256, 128, 32, 8, 2),
          (12288, 2048): (256, 256, 32, 16, 2),
          (2048, 6144): (256, 128, 32, 8, 2)}


def install_near_full_mm_selection():
    import torch
    import triton
    import flag_gems
    from flag_gems.utils.libentry import LibTuner

    registered = flag_gems.mm
    source = Path(inspect.getfile(registered))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert registered.__module__ == '_iluvatar.ops.mm'
    assert source_hash == '4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    launch = registered.__globals__['_launch_mm']
    assert launch.__module__ == registered.__module__
    target = launch.__globals__['mm_kernel']
    while not isinstance(target, LibTuner):
        target = target.fn
    assert not getattr(target, '_near_full_selection_installed', False)
    native_options = [c.all_kwargs() for c in target.configs]
    configurations = {}
    for shape, (bm, bn, bk, warps, stages) in CONFIG.items():
        config = triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 'SPLIT_K': 1},
                               num_warps=warps, num_stages=stages, num_ctas=1)
        assert config.all_kwargs() in native_options, 'Only existing native configurations are eligible'
        configurations[shape] = config
    original = target.run
    stats = {'eligible_tuner_calls': 0, 'original_tuner_calls': 0, 'selected_shapes': {}}

    def eligible(a, b, c, m, n, k):
        return (1793 <= m <= 2048 and (n, k) in configurations
                and a.dtype == b.dtype == c.dtype == torch.bfloat16
                and a.shape == (m, k) and b.shape == (k, n) and c.shape == (m, n)
                and a.stride() == (k, 1) and b.stride() == (1, k) and c.stride() == (n, 1))

    def selected(self, *args, **kwargs):
        if len(args) < 12 or not eligible(*args[:6]):
            stats['original_tuner_calls'] += 1
            return original(*args, **kwargs)
        a, b, c, m, n, k = args[:6]
        config = configurations[n, k]
        options = dict(kwargs)
        for key, value in config.all_kwargs().items():
            assert key not in options or options[key] == value, 'Conflicting native launch options'
            options[key] = value
        full_args = dict(zip(self.arg_names, args))
        full_args.update(options)
        self.best_config = config
        self.nargs = dict(zip(self.arg_names, args))
        self.benchmark_success_count = self.benchmark_cache_hit_count = 0
        self.bench_time = 0.
        if getattr(self, 'shared_config_pre_hook', None) is not None:
            self.shared_config_pre_hook(full_args)
        elif config.pre_hook is not None:
            config.pre_hook(full_args)
        try:
            result = self.fn.run(*args, **options)
        finally:
            self.nargs = None
        stats['eligible_tuner_calls'] += 1
        shape_key = f'M{m}_N{n}_K{k}'
        if shape_key not in stats['selected_shapes']:
            receipt = {'shape': [m, n, k], 'config': config.all_kwargs(),
                       'strides': [list(a.stride()), list(b.stride()), list(c.stride())],
                       'registered_module': registered.__module__, 'native_source_sha256': source_hash,
                       'kernel_body_replaced': False, 'persistent_ConfigCache_written_by_rule': False,
                       'count_scope': 'Tuner selection calls; LibEntry cache and graph replay bypass this observer.'}
            stats['selected_shapes'][shape_key] = receipt
            directory = os.environ.get('TIANSHU_NEAR_FULL_RECEIPTS')
            if directory:
                root = Path(directory)
                root.mkdir(parents=True, exist_ok=True)
                (root / (shape_key + '.json')).write_text(json.dumps(receipt, indent=2))
        return result

    target.run = types.MethodType(selected, target)
    target._near_full_selection_installed = True
    return {'target': target, 'original_run': original, 'selected_run': target.run,
            'eligible': eligible, 'stats': stats, 'native_source_sha256': source_hash,
            'registered_module': registered.__module__}
