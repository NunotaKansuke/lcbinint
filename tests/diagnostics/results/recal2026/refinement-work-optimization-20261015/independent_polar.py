"""Replay independent references and difficult cases for an optimized kernel."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

ROOT=Path('/rogue1_8/nunota/lcbinint')
BASE=ROOT/'tests/diagnostics/results/recal2026'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--radial-allocation',type=float,default=4.)
    args=parser.parse_args()
    out=Path(__file__).resolve().parent/args.name
    out.mkdir(exist_ok=False)
    for key in list(os.environ):
        if key.startswith('LCBININT_'):os.environ.pop(key)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        LCBININT_DIAGNOSTIC_POLAR_RADIAL_ALLOCATION=str(args.radial_allocation))
    os.sched_setaffinity(0,{0})
    native_path=next(((Path(__file__).resolve().parent/'baseline')if args.baseline else ROOT/'build-adaptive-production').glob('_lcbinint*.so'))
    spec=importlib.util.spec_from_file_location('_lcbinint',native_path)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    files=['polar-quadrature-rework/independent-angular-fresh-final-20261015/validation.json',
        'polar-quadrature-rework/independent-angular-extreme-final-20261015/validation.json',
        'polar-quadrature-rework/fold-footprint-final-regression/validation.json',
        'polar-quadrature-rework/estimator-parent-boole-fresh-small/validation.json']
    datasets=[]
    for file in files:
        payload=json.loads((BASE/file).read_text())
        datasets.append((file,payload))
    summary={};rows=[]
    for name,payload in datasets:
        refs={str(r['id']):r for r in payload['references']}
        prior={(str(r['id']),r['target']):r for r in payload['rows']}
        group=dict(conditions=0,accepted=0,false_accepts=[],new_rejections=[],exceptions=[],contract_failures=[],uncertain=0)
        for inp in payload['inputs']:
            key=str(inp.get('id',inp.get('geom_id')))
            ref=refs[key];value=ref.get('value',ref.get('fine'))
            for target in [1e-3,1e-4,1e-5]:
                row=dict(dataset=name,id=key,target=target,reference_valid=ref['valid'])
                try:
                    result=native.binary_adaptive_polar_flux(float(inp['time']),float(inp['y']),s=float(inp['s']),q=1/float(inp['q']),rho=float(inp['rho']),
                        limb_darkening=native.LimbDarkening(float(inp['c']),0),initial_bins=2,relative_tolerance=target)
                    accepted=result['estimated_target_met']
                    error=abs(result['magnification']/value-1) if value else None
                    contract=(result['error_indicator']==result['angular_error_indicator']+result['radial_error_indicator'] and
                        result['new_lattice_maps']<=1000000 and (not accepted or result['error_indicator']<=result['requested_error_budget']))
                    row.update(result=result,relative_error=error,contract=contract)
                    group['accepted']+=accepted
                    if accepted and ref['valid'] and error>target:group['false_accepts'].append([key,target,error])
                    old=prior.get((key,target),{})
                    if old.get('result',{}).get('estimated_target_met') and not accepted:group['new_rejections'].append([key,target])
                    if not contract:group['contract_failures'].append([key,target])
                except Exception as e:
                    row['exception']=str(e);group['exceptions'].append([key,target,str(e)])
                group['conditions']+=1;group['uncertain']+=not ref['valid'];rows.append(row)
        summary[name]=group
        print(name,group,flush=True)
    report=dict(summary=summary,rows=rows,initial_bins=2,radial_allocation=args.radial_allocation,
        native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(((Path(__file__).resolve().parent/'baseline/source/src/lcbinint/magnification/adaptive_polar_impl.inc')if args.baseline else ROOT/'src/lcbinint/magnification/adaptive_polar_impl.inc').read_bytes()).hexdigest())
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return int(any(g['false_accepts'] or g['new_rejections'] or g['exceptions'] or g['contract_failures'] for g in summary.values()))


if __name__=='__main__':raise SystemExit(main())
