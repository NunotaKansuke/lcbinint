"""Paired direct/automatic calls on the event-held-out cohort.

Build the Python extension before running; never rebuild it during this job.
"""
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'tests/diagnostics/results/recal2026/predictive-route-20261015'

def read(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))

def main():
    spec=importlib.util.spec_from_file_location('train',Path(__file__).with_name('train_adaptive_route.py'))
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    model=json.loads((OUT/'model.json').read_text())['tree']
    measurements=json.loads((OUT/'measurements.json').read_text())
    refs={r['geom_id']:float(r['reference'])for r in train.read(ROOT/'figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv')}
    for r in train.read(ROOT/'figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv'):
        refs[r['geom_id']]=float(r['adjudicated_reference'])
    chosen={r['id']:r for r in measurements if r['split']=='test'}
    lines={r.split()[0]:r for r in (OUT/'selected-inputs.tsv').read_text().splitlines()}
    env={k:v for k,v in os.environ.items() if not k.startswith('LCBININT_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    os.sched_setaffinity(0,{0})
    auto=OUT/'automatic';records=[]
    direct=ROOT/'tests/diagnostics/results/recal2026/refinement-work-optimization-20261015/selected-runner/runner'
    for index,key in enumerate(sorted(chosen,key=int)):
        inp=auto/f'{key}.inputs';inp.write_text(lines[key]+'\n')
        for label in (['direct','auto'] if index%2==0 else ['auto','direct']):
            path=auto/f'{key}-{label}.tsv'
            with path.with_suffix('.log').open('w') as log:
                subprocess.run([str(direct if label=='direct'else auto/'runner'),str(inp),str(path),'16','2'],env=env,stdout=log,stderr=log,check=True,timeout=60)
        dd=read(auto/f'{key}-direct.tsv');aa=read(auto/f'{key}-auto.tsv')
        for target in [1e-3,1e-4]:
            r=next(r for r in measurements if r['id']==key and r['target']==target)
            method='polar'if train.predict(model,r['x'])else'cartesian'
            d=[a for a in dd if a['method']==method and float(a['target'])==target]
            a=[a for a in aa if float(a['target'])==target]
            assert len(d)==len(a)==3
            for field in ['cold_magnification','cold_indicator','cold_met','cold_maps','cold_subdivisions','stop_reason']:
                assert a[0][field]==d[0][field],(key,target,method,field,a[0][field],d[0][field])
            records.append(dict(id=key,profile=r['profile'],target=target,selected=method,
                auto_ms=statistics.median(float(z['cold_ms'])for z in a),
                selected_direct_ms=statistics.median(float(z['cold_ms'])for z in d),
                cartesian_ms=statistics.median(float(z['cold_ms'])for z in dd if z['method']=='cartesian'and float(z['target'])==target),
                polar_ms=statistics.median(float(z['cold_ms'])for z in dd if z['method']=='polar'and float(z['target'])==target),
                accepted=a[0]['cold_met']=='1',error=abs(float(a[0]['cold_magnification'])/refs[key]-1)))
        if(index+1)%16==0:print(index+1,'/',len(chosen),flush=True)
    groups=[]
    for profile in ['uniform','linear']:
        for target in [1e-3,1e-4]:
            rr=[r for r in records if r['profile']==profile and r['target']==target]
            finite=[r for r in rr if math.isfinite(r['cartesian_ms'])and math.isfinite(r['polar_ms'])]
            groups.append(dict(profile=profile,target=target,n=len(rr),latency_pairs=len(finite),
                polar_selected=sum(r['selected']=='polar'for r in rr),
                not_accepted=sum(not r['accepted']for r in rr),false_accepts=sum(r['accepted']and r['error']>target for r in rr),
                median_direct_over_auto=statistics.median(r['selected_direct_ms']/r['auto_ms']for r in rr),
                total_polar_over_auto=sum(r['polar_ms']for r in finite)/sum(r['auto_ms']for r in finite),
                geometric_oracle_regret=math.exp(statistics.mean(math.log(r['auto_ms']/min(r['cartesian_ms'],r['polar_ms']))for r in finite))))
    result=dict(groups=groups,records=records,parity_conditions=len(records),
        timing='CPU0 single thread, alternate per input direct/auto order; median3 seed misses; auto includes feature calculation and decision; centre roots and caustics prepared outside both')
    (OUT/'automatic-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
