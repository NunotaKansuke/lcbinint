import csv,json,pathlib,hashlib,subprocess
base=pathlib.Path(__file__).resolve().parent;root=base.parents[2]
keys={(r['id'],r['u']) for r in json.load(open(root/'evidence/holonomic/seeded_inverse_comparison_v7/summary.json'))['inverse_normal']['1']['pairs']}
def read(p):return list(csv.DictReader(open(p),delimiter=' '))
def normal(n):
 rows=[r for r in read(base/(n+'.tsv')) if r['tol']=='0.0001' and (r['id'],r['u']) in keys];assert len(rows)==87
 return {k:sum(float(r[k]) for r in rows)/len(rows) for k in ['candidate_integration_ms','candidate_no_caustic_ms']}
s={'normal':{n:normal(n) for n in ['before_matched','after_matched','before_reverse','after_reverse']}}
for suffix in ['_matched', '_reverse']:s['normal']['reduction'+suffix]={k:1-s['normal']['after'+suffix][k]/s['normal']['before'+suffix][k] for k in s['normal']['before'+suffix]}
a=read(base/'before_matched.tsv')+read(root/'evidence/holonomic/arc_transport_v22/after_fold.tsv')+read(root/'evidence/holonomic/arc_transport_v22/after_deeper.tsv')
b=read(base/'after_initial.tsv')+read(base/'after_fold.tsv')+read(base/'after_deeper.tsv')
assert [(r['id'],r['u'],r['tol']) for r in a]==[(r['id'],r['u'],r['tol']) for r in b]
s['validation']={'conditions':len(b),'resolved_reference':sum(r['d14_converged']=='1' for r in b),'unresolved_reference':sum(r['d14_converged']!='1' for r in b),'candidate_failures_on_resolved_reference':sum(r['d14_converged']=='1' and (r['candidate_converged']!='1' or float(r['relative_difference'])>float(r['tol'])) for r in b),'maximum_relative_change':max(abs(float(x['candidate_mu'])-float(y['candidate_mu']))/max(abs(float(x['candidate_mu'])),1e-300) for x,y in zip(a,b)),'node_count_changes':sum(x['nodes']!=y['nodes'] for x,y in zip(a,b)),'maximum_discrepancy_over_tolerance':max(float(r['relative_difference'])/float(r['tol']) for r in b if r['d14_converged']=='1'),'audited_conditions':sum(len(read(base/(n+'.tsv'))) for n in ['audit_initial','audit_fold','audit_deeper'])}
s['profile']={}
for name in ['before','after']:
 rows=[fields for l in open(base/('profile_'+name+'_matched.log')) if (fields:=l.split()) and fields[0]=='PROFILE' and fields[3]=='0.0001' and (fields[1],fields[2]) in keys]
 s['profile'][name]={label:sum(float(r[i]) for r in rows)/len(rows) for i,label in [(4,'arc_ms'),(5,'pair_ms'),(48,'assembly_ms'),(49,'untransported_ms'),(50,'pair_cold_ms'),(21,'allocation_ms'),(22,'allocation_count'),(12,'physical_ms'),(19,'integration_ms')]}
(base/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
paths=['src/lcbinint/magnification/holonomic/'+n+'.hpp' for n in ['adaptive_epoch','adaptive_radial','radius_terms','root_pair','v2_profile','nested_fejer2','chebyshev_prediction']]+['benchmarks/holonomic/bench_seeded_radial_full.cpp','benchmarks/holonomic/bench_seeded_integration_profile.cpp','tests/holonomic_cpp/test_quartic_real_transport.cpp']
source=base/'source';source.mkdir(exist_ok=True)
for p in paths:(source/pathlib.Path(p).name).write_bytes((root/p).read_bytes())
(base/'manifest.json').write_text(json.dumps({'checkout':str(root),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'timing':'one thread; all compilation completed before serial timing; fresh source workspaces; normal87 tol1e-4,5 reps, mean of medians; holdout3; audit1','sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
print(json.dumps(s,indent=2))
