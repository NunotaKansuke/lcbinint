import csv,json,pathlib,hashlib,subprocess
base=pathlib.Path(__file__).resolve().parent;root=base.parents[2]
keys={(r['id'],r['u']) for r in json.load(open(root/'evidence/holonomic/seeded_inverse_comparison_v7/summary.json'))['inverse_normal']['1']['pairs']}
def read(p):return list(csv.DictReader(open(p),delimiter=' '))
def normal(n):
 rows=[r for r in read(base/(n+'.tsv')) if r['tol']=='0.0001' and (r['id'],r['u']) in keys];assert len(rows)==87
 return {k:sum(float(r[k]) for r in rows)/len(rows) for k in ['candidate_integration_ms','candidate_no_caustic_ms']}
s={'normal':{n:normal(n) for n in ['before','after','before_reverse','after_reverse']}}
for suffix in ['', '_reverse']:s['normal']['reduction'+suffix]={k:1-s['normal']['after'+suffix][k]/s['normal']['before'+suffix][k] for k in s['normal']['before'+suffix]}
a=read(base/'before.tsv')+read(root/'evidence/holonomic/node_dispatch_v18/fold.tsv')+read(root/'evidence/holonomic/node_dispatch_v18/deeper.tsv')
b=read(base/'after.tsv')+read(base/'fold.tsv')+read(base/'deeper.tsv')
assert [(r['id'],r['u'],r['tol']) for r in a]==[(r['id'],r['u'],r['tol']) for r in b]
s['validation']={'conditions':len(b),'resolved_reference':sum(r['d14_converged']=='1' for r in b),'unresolved_reference':sum(r['d14_converged']!='1' for r in b),'candidate_failures_on_resolved_reference':sum(r['d14_converged']=='1' and r['candidate_converged']!='1' for r in b),'maximum_relative_change':max(abs(float(x['candidate_mu'])-float(y['candidate_mu']))/max(abs(float(x['candidate_mu'])),1e-300) for x,y in zip(a,b)),'node_count_changes':sum(x['nodes']!=y['nodes'] for x,y in zip(a,b)),'maximum_discrepancy_over_tolerance':max(float(r['relative_difference'])/float(r['tol']) for r in b if r['d14_converged']=='1'),'audited_conditions':sum(len(read(base/(n+'.tsv'))) for n in ['audit_initial','audit_fold','audit_deeper'])}
(base/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
paths=['src/lcbinint/magnification/holonomic/'+n+'.hpp' for n in ['adaptive_epoch','adaptive_radial','radius_terms','root_pair','v2_profile','nested_fejer2']]+['benchmarks/holonomic/bench_seeded_radial_full.cpp']
source=base/'source';source.mkdir(exist_ok=True)
for p in paths:(source/pathlib.Path(p).name).write_bytes((root/p).read_bytes())
(base/'manifest.json').write_text(json.dumps({'checkout':str(root),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'timing':'one thread; all compilation completed before serial timing; fresh source workspaces; normal87 tol1e-4,5 reps, mean of medians; holdout3; audit1','sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
print(json.dumps(s,indent=2))
