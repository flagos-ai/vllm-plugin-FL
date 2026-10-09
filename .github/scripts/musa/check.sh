#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
set -euo pipefail
: "${MTHREADS_VISIBLE_DEVICES:?Set four allocated physical devices}"
: "${MUSA_VISIBLE_DEVICES:?Set four logical devices}"
: "${MUSA_CI_PCI_BUS_IDS:?Set the corresponding four logical PCI bus IDs}"
if command -v mthreads-gmi >/dev/null 2>&1; then
  mthreads-gmi
fi
python -I -B - <<'PY'
import ctypes, hashlib, json, os, pathlib, re, stat
import torch
import torch_musa
physical = os.environ['MTHREADS_VISIBLE_DEVICES'].split(',')
logical = os.environ['MUSA_VISIBLE_DEVICES'].split(',')
expected = os.environ['MUSA_CI_PCI_BUS_IDS'].split(',')
assert len(physical) == 4 and all(re.fullmatch(r'0|[1-9][0-9]*', x) for x in physical) and len({int(x) for x in physical}) == 4
assert logical == ['0', '1', '2', '3'], logical
assert len(set(expected)) == 4 and all(re.fullmatch(r'0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', p) for p in expected)
assert torch.musa.is_available() and torch.musa.device_count() == 4
nodes = []
for node in sorted(pathlib.Path('/dev').glob('mtgpu.*')):
    info = node.lstat(); assert stat.S_ISCHR(info.st_mode)
    pci = (pathlib.Path('/sys/dev/char') / f'{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}' / 'device').resolve(strict=True).name
    nodes.append({'node': str(node), 'rdev': info.st_rdev, 'pci': pci})
assert len(nodes) == 4 and {x['pci'] for x in nodes} == set(expected), nodes
loaded = set()
for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
    parts = line.split()
    if len(parts) >= 6 and parts[-1].startswith('/') and 'libmusart.so' in pathlib.Path(parts[-1]).name:
        loaded.add(str(pathlib.Path(parts[-1]).resolve(strict=True)))
assert len(loaded) == 1, loaded
library = pathlib.Path(next(iter(loaded))); api = ctypes.CDLL(str(library))
forward = api.musaDeviceGetPCIBusId
forward.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
forward.restype = ctypes.c_int
inverse = api.musaDeviceGetByPCIBusId
inverse.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]
inverse.restype = ctypes.c_int
mapping = []
for ordinal, wanted in enumerate(expected):
    buffer = ctypes.create_string_buffer(64)
    rc = forward(buffer, len(buffer), ordinal); assert rc == 0 and b'\0' in buffer.raw
    actual = buffer.value.decode('ascii').lower()[-12:]
    assert actual == wanted, (ordinal, actual, wanted)
    returned = ctypes.c_int(-1); inverse_rc = inverse(ctypes.byref(returned), buffer.value)
    assert inverse_rc == 0 and returned.value == ordinal
    mapping.append({'ordinal': ordinal, 'pci': actual, 'forward_exit': rc,
                    'inverse_exit': inverse_rc, 'inverse_ordinal': returned.value})
tensor = torch.ones((32, 32), device='musa:0'); torch.musa.synchronize()
print(json.dumps({'physical_mask': physical, 'logical_mask': logical, 'nodes': nodes,
                  'native_ordinal_pci_mapping': mapping, 'runtime': str(library),
                  'runtime_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                  'tensor_device': str(tensor.device)}, indent=2))
PY
