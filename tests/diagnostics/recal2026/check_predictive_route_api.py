"""Opt-in API parity, invalid-input checks and full-wrapper timing."""
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'tests/diagnostics/results/recal2026/predictive-route-20261015'

def main():
    os.sched_setaffinity(0,{0})
    for k in list(os.environ):
        if k.startswith('LCBININT_'):del os.environ[k]
    path=next((ROOT/'build-adaptive-production').glob('_lcbinint*.so'))
    spec=importlib.util.spec_from_file_location('_lcbinint',path);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
    rows=list(csv.DictReader((ROOT/'figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv').open(),delimiter='\t'))
    byid={r['geom_id']:r for r in rows}
    measurements=json.loads((OUT/'measurements.json').read_text())
    ids=[]
    for profile in ['uniform','linear']:
        ids+=sorted({r['id']for r in measurements if r['split']=='test'and r['profile']==profile},key=int)[::3][:24]
    records=[]
    for index,key in enumerate(ids):
        r=byid[key];x=float(r['time']);y=float(r['y'])
        for target in [1e-3,1e-4]:
            kw=dict(s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),limb_darkening=n.LimbDarkening(float(r['c']),0),relative_tolerance=target)
            auto=n.binary_adaptive_auto_flux(x,y,**kw)
            fn=n.binary_adaptive_polar_flux if auto['selected_method']=='polar'else n.binary_adaptive_flux
            direct=fn(x,y,**kw)
            for field in ['magnification','error_indicator','estimated_target_met','stop_reason','lattice_maps']:
                if field in direct:assert auto[field]==direct[field],(key,target,field)
            times={'auto':[],'direct':[]}
            for repeat in range(3):
                order=['auto','direct']if(index+repeat)%2==0 else['direct','auto']
                for label in order:
                    start=time.perf_counter();(n.binary_adaptive_auto_flux if label=='auto'else fn)(x,y,**kw)
                    times[label].append((time.perf_counter()-start)*1000)
            records.append(dict(id=key,target=target,profile=r['profile'],method=auto['selected_method'],
                auto_ms=statistics.median(times['auto']),direct_ms=statistics.median(times['direct']),
                extra_ms=statistics.median(times['auto'])-statistics.median(times['direct'])))
    r=byid[ids[0]];kw=dict(s=float(r['s']),q=1/float(r['q']),rho=float(r['rho']),limb_darkening=n.LimbDarkening(0,0))
    rejected=0
    for bad in [dict(s=-1),dict(q=0),dict(rho=math.nan),dict(relative_tolerance=0),dict(limb_darkening=n.LimbDarkening(.5,.1)),dict(max_lattice_maps=0)]:
        try:n.binary_adaptive_auto_flux(float(r['time']),float(r['y']),**(kw|bad))
        except ValueError:rejected+=1
        else:raise AssertionError(bad)
    fallback=[]
    for bad in [dict(relative_tolerance=1e-5),dict(limb_darkening=n.LimbDarkening(.7,0))]:
        a=n.binary_adaptive_auto_flux(float(r['time']),float(r['y']),**(kw|bad))
        assert a['selected_method']=='polar'and not a['route_within_training_domain']
        fallback.append(a['selected_method'])
    result=dict(parity_conditions=len(records),invalid_inputs_rejected=rejected,out_of_domain_checks=len(fallback),
        native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=records,
        median_direct_over_auto=statistics.median(r['direct_ms']/r['auto_ms']for r in records),
        median_extra_us=statistics.median(r['extra_ms']*1000 for r in records),
        timing='Full Python wrappers including preparation, duplicate centre solve in auto; CPU0, median3, alternating order')
    (OUT/'api-validation.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items()if k!='records'})

if __name__=='__main__':main()
