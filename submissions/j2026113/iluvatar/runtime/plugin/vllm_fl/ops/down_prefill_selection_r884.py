"""Select existing native MM configurations for qualified prefill down shapes."""
import functools
import hashlib
import inspect
import json
import os
import types
from pathlib import Path


def install(parent):
    import flag_gems
    import torch
    import triton

    target = parent['target']
    assert target._near_full_selection_installed
    assert target._grouped_gate_up_installed
    assert not getattr(target, '_down_prefill_r884', False)
    source = Path(inspect.getfile(flag_gems.mm))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == '4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    original = target.run
    wide = triton.Config(dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, SPLIT_K=1),
                          num_warps=16, num_stages=2, num_ctas=1)
    middle = triton.Config(dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, SPLIT_K=1),
                            num_warps=8, num_stages=2, num_ctas=1)
    native_configs = [c.all_kwargs() for c in target.configs]
    assert wide.all_kwargs() in native_configs and middle.all_kwargs() in native_configs
    stats = {'selected_shapes': {}}

    def eligible(a, b, c, m, n, k):
        return (isinstance(m, int) and (897 <= m <= 1024 or 1152 <= m <= 2048)
                and (n, k) == (2048, 6144)
                and a.dtype == b.dtype == c.dtype == torch.bfloat16
                and a.shape == (m, k) and b.shape == (k, n) and c.shape == (m, n)
                and a.stride() == (k, 1) and b.stride() == (1, k) and c.stride() == (n, 1))

    def selected(self, *args, **kwargs):
        if len(args) < 12 or not eligible(*args[:6]):
            return original(*args, **kwargs)
        m, n, k = args[3:6]
        config = middle if 1152 <= m <= 1536 else wide
        options = dict(kwargs)
        for key, value in config.all_kwargs().items():
            assert key not in options or options[key] == value
            options[key] = value
        assert options['fp8_fast_accum'] is True and options['AB_DTYPE'] == triton.language.bfloat16
        full_args = {**dict(zip(self.arg_names, args)), **options}
        self.best_config = config
        self.nargs = dict(zip(self.arg_names, args))
        self.benchmark_success_count = self.benchmark_cache_hit_count = 0
        self.bench_time = 0.
        if getattr(self, 'shared_config_pre_hook', None) is not None:
            self.shared_config_pre_hook(full_args)
        elif config.pre_hook is not None:
            config.pre_hook(full_args)
        try:
            # The original registered Heuristics and JIT body remain in place.
            result = self.fn.run(*args, **options)
        finally:
            self.nargs = None
        key = f'M{m}_N{n}_K{k}'
        if key not in stats['selected_shapes']:
            row = {'shape': [m, n, k], 'config': config.all_kwargs(),
                   'registered_module': flag_gems.mm.__module__,
                   'native_source_SHA256': digest, 'kernel_body_replaced': False,
                   'kernel_hash': result.hash, 'shared_bytes': result.metadata.shared,
                   'count_scope': 'Distinct tuner selections, not replay frequency.',
                   'native_K_reduction_block': 32}
            stats['selected_shapes'][key] = row
            directory = os.environ.get('TIANSHU_DOWN_PREFILL_RECEIPTS')
            if directory:
                root = Path(directory)
                root.mkdir(exist_ok=True)
                (root / (key + '.json')).write_text(json.dumps(row, indent=2))
        return result

    target.run = types.MethodType(selected, target)
    target._down_prefill_r884 = True
    return {'eligible': eligible, 'stats': stats, 'original_run': original,
            'native_source_SHA256': digest}


def install_on_runner(cls):
    original = cls.load_model

    @functools.wraps(original)
    def load(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self._down_prefill_r884 = install(self._ts_near_full_selection)
        return result

    cls.load_model = load
