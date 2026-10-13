import csv, hashlib, json, pathlib, platform, subprocess
base=pathlib.Path(__file__).resolve().parent
root=base.parents[2]
def read(name):
    return list(csv.DictReader(open(base/(name+'.tsv')),delimiter=' '))
keys={(r['id'],r['u']) for r in json.load(open(root/'evidence/holonomic/seeded_inverse_comparison_v7/summary.json'))['inverse_normal']['1']['pairs']}
def normal(name):
    rows=[r for r in read(name) if r['tol']=='0.0001' and (r['id'],r['u']) in keys]
    return {k:sum(float(r[k]) for r in rows)/len(rows) for k in ['candidate_integration_ms','candidate_no_caustic_ms','planning_ms','seed_ms']}
summary={'normal_count':len(keys),'normal':{n:normal(n) for n in ['before','sign','after','before_reverse','after_reverse']}}
a=read('before')+read('before_fold')+read('before_deeper')
b=read('after')+read('fold')+read('deeper')
assert [(r['id'],r['u'],r['tol']) for r in a]==[(r['id'],r['u'],r['tol']) for r in b]
summary['validation']={'conditions':len(b),'resolved_d14':sum(r['d14_converged']=='1' for r in b),'unresolved_d14':sum(r['d14_converged']!='1' for r in b),'candidate_failures_on_resolved_d14':sum(r['d14_converged']=='1' and r['candidate_converged']!='1' for r in b),'maximum_relative_change_vs_v15':max(abs(float(x['candidate_mu'])-float(y['candidate_mu']))/max(abs(float(x['candidate_mu'])),1e-300) for x,y in zip(a,b)),'node_count_changes':sum(x['nodes']!=y['nodes'] for x,y in zip(a,b)), 'maximum_relative_difference_on_resolved_d14':max(float(r['relative_difference']) for r in b if r['d14_converged']=='1')}
for suffix in ['', '_reverse']:
    before=summary['normal']['before'+suffix]; after=summary['normal']['after'+suffix]
    summary['normal']['reduction'+suffix]={k:1-after[k]/before[k] for k in before}
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
source=base/'source';source.mkdir(exist_ok=True)
paths=['src/lcbinint/magnification/holonomic/'+n+'.hpp' for n in ['adaptive_epoch','radius_terms','root_pair','seeded_radial_frontier']]
for rel in paths:(source/pathlib.Path(rel).name).write_bytes((root/rel).read_bytes())
manifest={'checkout':str(root),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'host':platform.node(),'master_seed_checkout':'/rogue1_8/nunota/lcbinint','timing':'serial; one thread; no compilation concurrent; per-condition median then mean; normal 5 repetitions, holdout 3; fresh workspaces; caustic construction excluded, master seeds included','sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},'accepted_trials':[1,3,4],'rejected_trials':[2]}
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
