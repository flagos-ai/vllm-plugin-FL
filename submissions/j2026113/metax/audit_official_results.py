"""Recompute the official C500 summaries from all eight original CSV rows."""
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import sys

BASELINES = {
    (4096,1024,64,256): (5089.645,3199.435),
    (16384,1024,64,128): (7029.675,27197.135),
}
KEYS = ('Prefill','Decode','Conc','Num Prompts')


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def key(row):
    return tuple(int(row[k]) for k in KEYS)


def audit(raw, summary):
    runs,summaries=read(raw),read(summary)
    assert len(runs)==8 and len(summaries)==2
    assert [key(r) for r in runs]==[k for k in BASELINES for _ in range(4)]
    assert [key(r) for r in summaries]==list(BASELINES)
    verified=[]
    for s in summaries:
        case=key(s)
        inp,out,_,n=case
        group=[r for r in runs if key(r)==case]
        assert all(r['Run Status']=='SUCCESS' and float(r['Successful Requests'])==n for r in group)
        assert all(float(r['Total Input Tokens'])==inp*n and float(r['Total Output Tokens'])==out*n for r in group)
        for k,v in s.items():
            if k not in KEYS:
                # CSV inputs are displayed with limited precision before summary rounding.
                assert abs(float(v)-mean(float(row[k]) for row in group[1:]))<=0.011,(case,k)
        total,ttft=float(s['Total tok/s']),float(s['Mean TTFT (ms)'])
        ref_total,ref_ttft=BASELINES[case]
        verified.append(dict(case=f'{inp//1024}K / 1024 / C64',total_tok_s=total,mean_ttft_ms=ttft,
            baseline_total_tok_s=ref_total,baseline_ttft_ms=ref_ttft,
            total_change_pct=100*(total/ref_total-1),ttft_change_pct=100*(ttft/ref_ttft-1),
            passes=total>=0.99*ref_total and ttft<=1.01*ref_ttft))
    return dict(verified=True,raw_rows=8,summary_rows=2,skip_first=1,rows=verified,
        sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in (raw,summary)})


def check_controller(result, path):
    run=json.loads(path.read_bytes())
    assert run['stage']=='finished' and run['complete'] and run['benchmark_exit']==0
    assert run['terminal']['Status']=='exited' and run['terminal']['ExitCode']==0
    assert run['restarts']==0 and not run['terminal']['OOMKilled']
    assert run['runs_per_case']==4 and run['skip_first']==1
    assert run['initial_study_files']==0 and run['additional_warmup_requests']==0
    assert run['csv']==result['sha256']
    assert run['serve']==['vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
        '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
    assert run['benchmark_sha256']=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    assert json.loads(run['benchmark_cmd'][-1])==[list(case) for case in BASELINES]
    result['controller_sha256']=sha256(path.read_bytes()).hexdigest()
    result['container_id']=run['container_id']
    result['clean_start_verified']=True


if __name__=='__main__':
    result=audit(Path(sys.argv[1]),Path(sys.argv[2]))
    if len(sys.argv)>3:
        check_controller(result,Path(sys.argv[3]))
    print(json.dumps(result,indent=2))
