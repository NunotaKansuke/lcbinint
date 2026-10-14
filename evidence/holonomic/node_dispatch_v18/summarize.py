import csv,json,pathlib,hashlib,subprocess
base=pathlib.Path(__file__).resolve().parent;root=base.parents[2]
keys={(r['id'],r['u']) for r in json.load(open(root/'evidence/holonomic/seeded_inverse_comparison_v7/summary.json'))['inverse_normal']['1']['pairs']}
def read(path):return list(csv.DictReader(open(path),delimiter=' '))
def normal(name):
 rows=[r for r in read(base/(name+'.tsv')) if r['tol']=='0.0001' and (r['id'],r['u']) in keys]
 assert len(rows)==87
 return {k:sum(float(r[k]) for r in rows)/len(rows) for k in ['candidate_integration_ms','candidate_no_caustic_ms','planning_ms','seed_ms']}
s={'normal':{n:normal(n) for n in ['before','flat','after','before_reverse','after_reverse']}}
for suffix in ['', '_reverse']:
 s['normal']['reduction'+suffix]={k:1-s['normal']['after'+suffix][k]/s['normal']['before'+suffix][k] for k in s['normal']['before'+suffix]}
a=read(base/'before.tsv')+read(root/'evidence/holonomic/bulk_hotpath_v16/fold.tsv')+read(root/'evidence/holonomic/bulk_hotpath_v16/deeper.tsv')
b=read(base/'after.tsv')+read(base/'fold.tsv')+read(base/'deeper.tsv')
assert [(r['id'],r['u'],r['tol']) for r in a]==[(r['id'],r['u'],r['tol']) for r in b]
s['validation']={'conditions':len(b),'resolved_reference':sum(r['d14_converged']=='1' for r in b),'unresolved_reference':sum(r['d14_converged']!='1' for r in b),'candidate_failures_on_resolved_reference':sum(r['d14_converged']=='1' and r['candidate_converged']!='1' for r in b),'maximum_relative_change':max(abs(float(x['candidate_mu'])-float(y['candidate_mu']))/max(abs(float(x['candidate_mu'])),1e-300) for x,y in zip(a,b)),'node_count_changes':sum(x['nodes']!=y['nodes'] for x,y in zip(a,b)),'maximum_discrepancy_over_tolerance':max(float(r['relative_difference'])/float(r['tol']) for r in b if r['d14_converged']=='1')}
s['profile']={}
for name in ['before','after']:
 rows=[fields for l in open(base/('profile_'+name+'.log')) if (fields:=l.split()) and fields[0]=='PROFILE' and fields[3]=='0.0001' and (fields[1],fields[2]) in keys]
 s['profile'][name]={label:sum(float(r[i]) for r in rows)/len(rows) for i,label in [(12,'physical'),(19,'integral'),(26,'header'),(25,'node_loop'),(21,'allocation')]}
(base/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
paths=['src/lcbinint/magnification/holonomic/'+n+'.hpp' for n in ['adaptive_epoch','adaptive_radial','radius_terms','root_pair','v2_profile']]+['benchmarks/holonomic/bench_seeded_radial_full.cpp','benchmarks/holonomic/bench_seeded_integration_profile.cpp']
source=base/'source';source.mkdir(exist_ok=True)
for p in paths:(source/pathlib.Path(p).name).write_bytes((root/p).read_bytes())
(base/'manifest.json').write_text(json.dumps({'checkout':str(root),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'timing':'one thread; compilation completed before serial timing; fresh workspaces; normal 87 conditions tol1e-4 with5 repetitions, per-input median then mean; holdout3; profile3 arithmetic means','master_seed_checkout':'/rogue1_8/nunota/lcbinint','source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
print(json.dumps(s,indent=2))
