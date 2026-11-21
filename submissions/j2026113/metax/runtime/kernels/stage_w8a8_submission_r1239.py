"""Wire the Triton launch table into an independent copy of the vendor stage."""
import argparse
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('stage',type=Path)
    args=p.parse_args()
    shutil.copytree(args.reference,args.stage,symlinks=True)
    path=args.stage/'plugin/vllm_fl/quantization/w8a8/metax_native.py'
    body=path.read_text()
    cache='''@lru_cache(maxsize=8)
def _gemm_for_device(index):
    from mctlassEx.ScaledGEMM import ScaledGEMM
    return ScaledGEMM()
'''
    call='''    bias_native = None if bias is None else bias.reshape(n, 1).contiguous()
    _gemm_for_device(x.device.index)([m, n, k], a, weight, out,
                                    sa, weight_scale, bias_native)
'''
    assert body.count(cache)==body.count(call)==1
    body=body.replace(cache,'').replace('from functools import lru_cache\n','')
    body=body.replace('import torch\n','import torch\nfrom w8a8_submission_route_r1239 import launch\n',1)
    body=body.replace(call,'''    if bias is not None:
        raise ValueError("The MiniCPM5 C500 W8A8 route requires bias-free linear layers")
    launch(a, weight, sa, weight_scale, out, m, n, k)
''')
    body=body.replace('dynamic-token INT8, native scaled GEMM','dynamic-token INT8, Triton scaled GEMM')
    path.write_text(body)


if __name__=='__main__':
    main()
