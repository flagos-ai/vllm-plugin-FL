import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

out = Path('/audit')
packages = {}
for d in md.distributions():
    packages[d.metadata['Name']] = {'version': d.version, 'requires': d.requires or [], 'location': str(d.locate_file(''))}
(out / 'packages.json').write_text(json.dumps(packages, indent=2))
facts = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
         'environment': {k: os.environ.get(k) for k in ['PATH','PYTHONPATH','CUDA_HOME','LD_LIBRARY_PATH']},
         'tools': {k: shutil.which(k) for k in ['git','gcc','g++','nvcc','cmake','ninja','ffmpeg','compiler','uv']}}
for label, cmd in {
    'pip-check': [sys.executable, '-m', 'pip', 'check'],
    'torch': [sys.executable, '-c', 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); x=torch.ones(16,device="cuda"); print((x+x).sum().item())'],
    'triton': [sys.executable, '-c', 'import triton; print(triton.__version__,triton.__file__)'],
    'flag_gems': [sys.executable, '-c', 'import flag_gems; print(flag_gems.__version__,flag_gems.__file__)'],
    'torchvision': [sys.executable, '-c', 'import torchvision; print(torchvision.__version__)'],
    'torchaudio': [sys.executable, '-c', 'import torchaudio; print(torchaudio.__version__)'],
    'flagcx': [sys.executable, '-c', 'import flagcx; print(flagcx.__file__)'],
    'compiler': ['bash','-lc','compiler'],
    'gpu': ['nvidia-smi'],
    'nvcc': ['bash','-lc','nvcc --version'],
}.items():
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        facts[label] = {'exit': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except Exception as e:
        facts[label] = {'error': str(e)}
    (out / 'facts.json').write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
