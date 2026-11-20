"""Exact logical-byte identities for frozen benchmark inputs; no tensor data cache."""
import hashlib
import json
import weakref

import torch

_hash_cache = {}
_records = []
_CHUNK_BYTES = 64 * 1024 * 1024


def _tensor_identity(tensor):
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return {'python_type': type(tensor).__name__, 'value': repr(tensor)}
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    key = id(tensor)
    cached = _hash_cache.get(key)
    metadata = {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
                'dtype': str(tensor.dtype), 'logical_bytes': tensor.numel() * tensor.element_size()}
    if cached and cached[0]() is tensor and cached[1] == version and cached[2] == metadata:
        digest = cached[3]
    else:
        digest = hashlib.sha256()
        raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), _CHUNK_BYTES):
            chunk = raw[start:start + _CHUNK_BYTES].cpu()
            digest.update(memoryview(chunk.numpy()))
        digest = digest.hexdigest()
        # Hold only a weak tensor reference and the digest, never decoded weights/data.
        _hash_cache[key] = (weakref.ref(tensor), version, metadata, digest)
    return dict(metadata, sha256=digest)


def record(precision, shape, inputs, selected, profile_selected):
    names = ['hidden_states'] + [f'weight_or_scale_{i}' for i in range(1, len(inputs) - 2)] + ['topk_weights', 'topk_ids']
    tensors = {name: _tensor_identity(value) for name, value in zip(names, inputs)}
    bank = {name: value for name, value in tensors.items() if name.startswith('weight_or_scale_')}
    bank_sha = hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    record = {'event': 'input_identity', 'generation_index': len(_records),
              'precision': precision, 'shape': list(shape), 'seed': 20261119 + sum(shape),
              'selected': selected, 'profile_selected': profile_selected,
              'weight_bank_sha256': bank_sha, 'inputs': tensors,
              'hash_scope': 'Full logical tensor bytes, not sampled; 64MiB host chunks; weak-reference digest cache only'}
    _records.append(record)
    print(json.dumps(record), flush=True)


def finish(path):
    if path:
        from pathlib import Path
        Path(path).write_text(json.dumps({'method': 'replay all original _gen calls before filtering',
                                         'records': _records}, indent=2))
