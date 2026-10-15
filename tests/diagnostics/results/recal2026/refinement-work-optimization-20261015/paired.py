"""Frozen-library paired timings, with every failure kept in the report."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

ROOT=Path('/rogue1_8/nunota/lcbinint')
BASE=ROOT/'tests/diagnostics/results/recal2026/refinement-work-optimization-20261015'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--polar-bins',type=int,default=2)
    parser.add_argument('--cartesian-bins',type=int,default=16)
    parser.add_argument('--cartesian-row-ratio',type=int,choices=[1,2],default=1)
    parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--policy',type=int,default=0)
    parser.add_argument('--blocks',action='store_true')
    parser.add_argument('--batch',type=int,choices=[0,4,8],default=0)
    parser.add_argument('--first-call',action='store_true')
    parser.add_argument('--radial-allocation',type=float,default=4.0)
    args=parser.parse_args()
    out=BASE/args.name;out.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('seed_bench',ROOT/'tests/diagnostics/recal2026/seed_inclusive_cart_polar_benchmark.py')
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    inputs=bench.read_tsv(bench.INPUTS);rng=random.Random(202610151718)
    selected=[]
    for profile in ['uniform','linear']:
        selected+=rng.sample([r for r in inputs if r['profile']==profile],args.count//2)
    selected.sort(key=lambda r:int(r['geom_id']))
    refs={r['geom_id']:float(r['reference']) for r in selected}
    for r in bench.read_tsv(ROOT/'figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv'):
        if r['geom_id'] in refs:refs[r['geom_id']]=float(r['adjudicated_reference'])
    if args.first_call:
        source=bench.RUNNER_SOURCE.read_text()
        source=source.replace('magnifier.ensure_limb_darkening_table();','// Keep the LD table cold for the first timed call.')
        source=source.replace('warmup < 2','warmup < 0')
        bench.RUNNER_SOURCE=out/'first_call_runner.cpp';bench.RUNNER_SOURCE.write_text(source)
    libraries={};commands={}
    for version in ['baseline','candidate']:
        location=out/version;location.mkdir()
        bench.BUILD=BASE/'baseline' if version=='baseline' else ROOT/'build-adaptive-production'
        libraries[version]=bench.sha256(bench.BUILD/'liblcbinint_magnification.a')
        commands[version]=bench.compile_runner(location)
    os.sched_setaffinity(0,{0})
    for k in list(os.environ):
        if k.startswith('LCBININT_'):os.environ.pop(k)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    records=[]
    for index,inp in enumerate(selected):
        key=inp['geom_id'];input_path=out/f'{key}-input.tsv';input_path.write_text(bench.native_line(inp)+'\n')
        order=['baseline','candidate'] if index%2==0 else ['candidate','baseline']
        for version in order:
            os.environ.pop('LCBININT_DIAGNOSTIC_POLAR_RADIAL_ALLOCATION',None)
            os.environ.pop('LCBININT_DIAGNOSTIC_CARTESIAN_INITIAL_ROW_RATIO',None)
            if version=='candidate':os.environ['LCBININT_DIAGNOSTIC_CARTESIAN_INITIAL_ROW_RATIO']=str(args.cartesian_row_ratio)
            if version in ['baseline','candidate']:
                os.environ['LCBININT_DIAGNOSTIC_POLAR_RADIAL_ALLOCATION']=str(args.radial_allocation)
            os.environ['LCBININT_DIAGNOSTIC_WORK_SCHEDULER']=str(args.policy if version=='candidate'else 0)
            os.environ.pop('LCBININT_DIAGNOSTIC_INTERIOR_DISCRETE_BLOCKS',None)
            if args.blocks and version=='candidate':os.environ['LCBININT_DIAGNOSTIC_INTERIOR_DISCRETE_BLOCKS']='1'
            os.environ.pop('LCBININT_DIAGNOSTIC_CARTESIAN_BATCH',None)
            if args.batch and version=='candidate':os.environ['LCBININT_DIAGNOSTIC_CARTESIAN_BATCH']=str(args.batch)
            raw_path=out/version/f'{key}.tsv'
            bins=2 if version=='baseline' else args.polar_bins
            with raw_path.with_suffix('.log').open('w') as log:
                subprocess.run([str(out/version/'runner'),str(input_path),str(raw_path),str(16 if version=='baseline' else args.cartesian_bins),str(bins)],stdout=log,stderr=log,check=True,timeout=60)
            raw=bench.read_tsv(raw_path)
            assert len(raw)==12 and len({(r['method'],r['target'],r['repeat']) for r in raw})==12
            for method in ['cartesian','polar']:
                for target in [1e-3,1e-4]:
                    rr=[r for r in raw if r['method']==method and float(r['target'])==target]
                    r=next(r for r in rr if r['repeat']=='0');mag=float(r['cold_magnification'])
                    error=abs(mag/refs[key]-1) if math.isfinite(mag) else math.inf
                    records.append(dict(id=key,profile=inp['profile'],version=version,method=method,target=target,
                        ms=float(r['cold_ms']) if args.first_call else statistics.median(float(r['cold_ms']) for r in rr),
                        magnification=mag,error=error,accepted=r['cold_met']=='1',maps=int(r['cold_maps']),
                        finite=math.isfinite(mag),stop_reason=r['stop_reason']))
        if (index+1)%16==0:print(index+1,'/',len(selected),flush=True)
    groups=[]
    for profile in ['uniform','linear']:
        for method in ['cartesian','polar']:
            for target in [1e-3,1e-4]:
                subset=[r for r in records if r['profile']==profile and r['method']==method and r['target']==target]
                old={r['id']:r for r in subset if r['version']=='baseline'}
                new={r['id']:r for r in subset if r['version']=='candidate'}
                good=lambda r:r['accepted'] and r['error']<=target
                paired=[i for i in old if good(old[i]) and good(new[i])]
                identical=[i for i in old if old[i]['finite'] and new[i]['finite'] and old[i]['magnification']==new[i]['magnification'] and old[i]['maps']==new[i]['maps']]
                groups.append(dict(profile=profile,method=method,target=target,inputs=len(old),accurate_pairs=len(paired),
                    median_speedup=statistics.median(old[i]['ms']/new[i]['ms'] for i in paired) if paired else None,
                    baseline_median_ms=statistics.median(old[i]['ms'] for i in paired) if paired else None,
                    candidate_median_ms=statistics.median(new[i]['ms'] for i in paired) if paired else None,
                    bit_identical_value_and_maps=len(identical),
                    new_failures=[i for i in old if good(old[i]) and not good(new[i])],
                    baseline_false_accepts=[i for i in old if old[i]['accepted'] and old[i]['error']>target],
                    candidate_false_accepts=[i for i in new if new[i]['accepted'] and new[i]['error']>target],
                    baseline_exceptions=[i for i in old if not old[i]['finite']],candidate_exceptions=[i for i in new if not new[i]['finite']]))
    report=dict(batch=args.batch,blocks=args.blocks,policy=args.policy,groups=groups,records=records,library_sha256=libraries,compile_commands=commands,ids=[r['geom_id'] for r in selected],
        runner_source_sha256=bench.sha256(bench.RUNNER_SOURCE),cpu=0,threads=1,first_call=args.first_call,
        timing='first cold call without LD-table initialization or warmups' if args.first_call else 'median3 seed-inclusive cold calls; roots, caustics and LD table prepared; alternating version order',
        baseline_cartesian_bins=16,baseline_polar_bins=2,candidate_cartesian_bins=args.cartesian_bins,candidate_cartesian_row_ratio=args.cartesian_row_ratio,candidate_polar_bins=args.polar_bins,initial_bins_matched=(args.polar_bins==2 and args.cartesian_bins==16 and args.cartesian_row_ratio==1),radial_allocation=args.radial_allocation)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
