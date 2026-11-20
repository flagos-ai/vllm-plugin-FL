import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys


def observe_runtime(rank, group):
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and Path(parts[5]).name.startswith(('libbkcl.so', 'libflagcx.so')):
            paths.add(parts[5])
    libraries = [{'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in sorted(paths)]
    observation = {'rank': rank, 'libraries': libraries, 'route': 'vendor_kccl_pg',
                   'backend_type': str(type(group._get_backend(__import__('torch').device('cuda', int(os.environ['LOCAL_RANK']))))),
                   'xmlir_package': version('xmlir'), 'plugin_imported': any(name.startswith('vllm_fl') for name in sys.modules),
                   'flagcx_imported': any(name.startswith('flagcx') for name in sys.modules),
                   'selected_env': {name: os.environ.get(name) for name in ('BKCL_INFERENCE','BKCL_ENABLE_IBGDA','BKCL_ENABLE_XDR','BKCL_USE_RDMA','XMLIR_ENABLE_NEW_PG')}}
    os.write(1, ('RUNTIME_OBSERVATION ' + json.dumps(observation, sort_keys=True) + '\n').encode())
    assert not observation['plugin_imported'] and not observation['flagcx_imported']
    assert not any(Path(item['path']).name.startswith('libflagcx.so') for item in libraries)
    assert len(libraries) == 1, 'Need one measured native BKCL library'
