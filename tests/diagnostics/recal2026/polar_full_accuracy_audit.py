"""Resumable accuracy/contract audit of the saved six-panel input corpus."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--initial-bins',type=int,default=2)
    args=parser.parse_args()
    os.sched_setaffinity(0,{0})
    source=ROOT/'figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv'
    native_path=next((ROOT/'build-adaptive-production').glob('_lcbinint*.so'))
    with source.open() as stream:inputs=list(csv.DictReader(stream,delimiter='\t'))
    if args.limit:inputs=inputs[:args.limit]
    identity=dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),limit=args.limit,
        relative_tolerances=[1e-3,1e-4],initial_bins=args.initial_bins,initial_angular_ratio=256,
        reference_scope='saved VBM RelTol=1e-6 agreement; disagreements require independent adjudication')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/'identity.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:raise RuntimeError('audit identity changed')
    else:manifest.write_text(json.dumps(identity,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('_lcbinint',native_path)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    started=time.monotonic();groups={}
    for index,inp in enumerate(inputs):
        shard=args.output/f'{index:05d}.json'
        if shard.exists():rows=json.loads(shard.read_text())
        else:
            rows=[]
            for goal in (1e-3,1e-4):
                row=dict(id=inp['geom_id'],profile=inp['profile'],input=inp,target=goal)
                try:
                    result=native.binary_adaptive_polar_flux(float(inp['time']),float(inp['y']),
                        s=float(inp['s']),q=1/float(inp['q']),rho=float(inp['rho']),
                        limb_darkening=native.LimbDarkening(float(inp['c']),0),relative_tolerance=goal,initial_bins=args.initial_bins)
                    discrepancy=abs(result['magnification']/float(inp['reference'])-1)
                    contract=(not math.isnan(result['error_indicator']) and
                        result['error_indicator']==result['angular_error_indicator']+result['radial_error_indicator'] and
                        result['new_lattice_maps']<=1000000 and
                        (not result['estimated_target_met'] or
                         (result['stop_reason']=='estimated_target_met' and
                          math.isfinite(result['error_indicator']) and
                          result['error_indicator']<=result['requested_error_budget'])))
                    row.update(result=result,reference_discrepancy=discrepancy,contract=contract,
                        accepted_reference_disagreement=result['estimated_target_met'] and discrepancy>goal)
                except Exception as exc:row.update(exception=str(exc),contract=False)
                rows.append(row)
            temporary=shard.with_suffix('.new');temporary.write_text(json.dumps(rows)+'\n');temporary.replace(shard)
        for row in rows:
            key=row['profile']+':'+str(row['target'])
            group=groups.setdefault(key,dict(conditions=0,accepted=0,exceptions=0,contract_failures=0,
                accepted_reference_disagreements=[],rejections=[]))
            group['conditions']+=1
            group['accepted']+=bool(row.get('result',{}).get('estimated_target_met'))
            group['exceptions']+='exception' in row
            group['contract_failures']+=not row.get('contract',False)
            if row.get('accepted_reference_disagreement'):group['accepted_reference_disagreements'].append(row['id'])
            if 'result' in row and not row['result']['estimated_target_met']:group['rejections'].append(row['id'])
        if (index+1)%64==0 or index+1==len(inputs):
            summary=dict(groups=groups,completed_inputs=index+1,total_inputs=len(inputs),
                         seconds=time.monotonic()-started,certified=False,
                         status='generated' if index+1==len(inputs) else 'running')
            temporary=args.output/'summary.new';temporary.write_text(json.dumps(summary,indent=2)+'\n')
            temporary.replace(args.output/'summary.json')
            print('inputs',index+1,'/',len(inputs),'seconds',round(summary['seconds'],2),flush=True)
    return int(any(g['exceptions'] or g['contract_failures'] for g in groups.values()))


if __name__=='__main__':raise SystemExit(main())
