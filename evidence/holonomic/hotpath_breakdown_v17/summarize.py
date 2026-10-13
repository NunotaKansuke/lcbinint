import csv,json,pathlib,hashlib,subprocess
base=pathlib.Path(__file__).resolve().parent;root=base.parents[2]
keys={(str(r['id']),str(r['u'])) for r in json.load(open(root/'evidence/holonomic/seeded_inverse_comparison_v7/summary.json'))['inverse_normal']['1']['pairs']}
names={4:'arc',5:'rootpair',6:'endpoint',7:'K',12:'physical',13:'estimator',14:'scheduler',15:'setup',16:'event',17:'panel',19:'integral',20:'count',21:'allocation',23:'arc_body',24:'tail',25:'node_loop',26:'header',27:'coeff',28:'rescue',29:'rescue_calls',30:'k_attempts',31:'k_success',32:'reject_vfloor',33:'reject_tmax',34:'reject_s2',35:'reject_b',36:'reject_disagreement'}
rows=[s for l in open(base/'final.log') if (s:=l.split()) and s[0]=='PROFILE' and s[3]=='0.0001' and (s[1],s[2]) in keys]
summary={}
for label in ['all','uniform','limb_darkened']:
 rs=[r for r in rows if label=='all' or r[2]==('0' if label=='uniform' else '0.5')]
 v={name:sum(float(r[i]) for r in rs)/len(rs) for i,name in names.items()};v['repetitions']=len(rs)
 v['mapped_other']=v['physical']-v['arc']-v['count']-v['arc_body']-v['tail']-v['header']-v['coeff']
 v['node_loop_other']=v['node_loop']-v['physical']
 v['integral_other']=v['integral']-v['node_loop']-v['setup']-v['estimator']-v['scheduler']-v['allocation']
 summary[label]=v
baseline=list(csv.DictReader(open(root/'evidence/holonomic/bulk_hotpath_v16/after.tsv'),delimiter=' '));current=list(csv.DictReader(open(base/'final.tsv'),delimiter=' '))
assert [(r['id'],r['u'],r['tol']) for r in current]==[(r['id'],r['u'],r['tol']) for r in baseline]
summary['profile_validation']={'conditions':len(current),'magnification_changes':sum(r['candidate_mu']!=b['candidate_mu'] for r,b in zip(current,baseline)),'node_count_changes':sum(r['nodes']!=b['nodes'] for r,b in zip(current,baseline))}
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
paths=['src/lcbinint/magnification/holonomic/'+n+'.hpp' for n in ['adaptive_epoch','adaptive_radial','v2_profile']]+['benchmarks/holonomic/bench_seeded_integration_profile.cpp']
source=base/'source';source.mkdir(exist_ok=True)
for p in paths:(source/pathlib.Path(p).name).write_bytes((root/p).read_bytes())
(base/'manifest.json').write_text(json.dumps({'checkout':str(root),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'changes':'profiling only; new scopes enabled by HOLO_FRONTIER_COST_PROFILE','timing':'serial; fresh per-source workspaces; no compilation concurrent; one thread; nested per-repetition arithmetic means; 87 normal conditions tol1e-4, three repetitions','source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
print(json.dumps(summary,indent=2))
