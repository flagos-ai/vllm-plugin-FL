"""Experiment-local G4 gate/up traversal, preserving the registered MM ABI."""
import hashlib
import inspect
import json
import os
import types
from pathlib import Path


def install_grouped_gate_up_selection(parent):
    import torch
    import triton
    from . import prefill_mm_grouped_fixed4_r413 as grouped

    target = parent['target']
    assert target._near_full_selection_installed
    assert not getattr(target, '_grouped_gate_up_installed', False)
    original = target.run
    original_heuristics = target.fn
    assert isinstance(original_heuristics, triton.runtime.Heuristics)
    native_jit = original_heuristics.fn
    assert native_jit.arg_names == grouped.mm_kernel.arg_names
    # Same signature/constexpr positions keep LibEntry cached launches compatible.
    assert [p.is_constexpr for p in native_jit.params] == [p.is_constexpr for p in grouped.mm_kernel.params]
    replacement = triton.heuristics(dict(original_heuristics.values))(grouped.mm_kernel)
    config = triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1},
                          num_warps=16, num_stages=2, num_ctas=1)
    assert config.all_kwargs() in [c.all_kwargs() for c in target.configs]
    kernel_hash = hashlib.sha256(Path(inspect.getfile(grouped)).read_bytes()).hexdigest()
    stats = {'selected_tuner_calls': 0, 'delegated_tuner_calls': 0, 'selected_shapes': {}}

    def eligible(a, b, c, m, n, k):
        return (1793 <= m <= 2048 and (n, k) == (12288, 2048)
                and a.dtype == b.dtype == c.dtype == torch.bfloat16
                and a.shape == (m, k) and b.shape == (k, n) and c.shape == (m, n)
                and a.stride() == (k, 1) and b.stride() == (1, k) and c.stride() == (n, 1))

    def selected(self, *args, **kwargs):
        if len(args) < 12 or not eligible(*args[:6]):
            stats['delegated_tuner_calls'] += 1
            return original(*args, **kwargs)
        assert kwargs['GROUP_M'] == 8, 'Preserve native ABI; fixed G4 is in kernel mapping only'
        options = dict(kwargs)
        for key, value in config.all_kwargs().items():
            assert key not in options or options[key] == value
            options[key] = value
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
            result = replacement.run(*args, **options)
        finally:
            self.nargs = None
        stats['selected_tuner_calls'] += 1
        m, n, k = args[3:6]
        key = f'M{m}_N{n}_K{k}'
        if key not in stats['selected_shapes']:
            receipt = {'shape': [m, n, k], 'config': config.all_kwargs(),
                       'mapping_group_m': 4, 'native_GROUP_M_argument': kwargs['GROUP_M'],
                       'kernel_source_sha256': kernel_hash, 'compiled_kernel_hash': result.hash,
                       'compiled_kernel_name': result.name,
                       'count_scope': 'Distinct tuner selections only; cached calls/replays bypass this observer.'}
            stats['selected_shapes'][key] = receipt
            directory = os.environ.get('TIANSHU_GROUPED_RECEIPTS')
            if directory:
                root = Path(directory)
                root.mkdir(parents=True, exist_ok=True)
                (root / (key + '.json')).write_text(json.dumps(receipt, indent=2))
        return result

    target.run = types.MethodType(selected, target)
    target._grouped_gate_up_installed = True
    return {'target': target, 'original_run': original, 'selected_run': target.run,
            'eligible': eligible, 'stats': stats, 'kernel_source_sha256': kernel_hash}
