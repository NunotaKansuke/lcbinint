import csv,json,pathlib,hashlib,subprocess
D=pathlib.Path(__file__).resolve().parent;root=D.parents[2]
def read(p):return list(csv.DictReader(open(p),delimiter=' '))
keys={(r['id'],r['u']) for r in json.load(open(root/'evidence/holonomic/seeded_inverse_comparison_v7/summary.json'))['inverse_normal']['1']['pairs']}
s={'normal':{}}
for n in ['before_matched','after_matched','before_reverse','after_reverse']:
 rows=[r for r in read(D/(n+'.tsv')) if r['tol']=='0.0001' and (r['id'],r['u']) in keys];assert len(rows)==87
 s['normal'][n]={k:sum(float(r[k]) for r in rows)/len(rows) for k in ['seed_ms','planning_ms','candidate_integration_ms','candidate_no_caustic_ms']}
for tag in ['matched','reverse']:s['normal']['reduction_'+tag]={k:1-s['normal']['after_'+tag][k]/s['normal']['before_'+tag][k] for k in s['normal']['before_'+tag]}
a=[];b=[];audit=[]
for n in ['initial','fold','deeper']:
 a+=read(root/'evidence/holonomic/frontier_reuse_v28'/('after_'+n+'.tsv'));b+=read(D/('after_'+n+'.tsv'))
assert [(x['id'],x['u'],x['tol']) for x in a]==[(x['id'],x['u'],x['tol']) for x in b]
s['validation']={'conditions':len(b),'reference_resolved_before':sum(r['d14_converged']=='1' for r in a),'reference_resolved_after':sum(r['d14_converged']=='1' for r in b),'unresolved_reference':sum(r['d14_converged']!='1' for r in b),'candidate_failures_on_resolved_reference':sum(r['d14_converged']=='1' and (r['candidate_converged']!='1' or float(r['relative_difference'])>float(r['tol'])) for r in b),'candidate_failures_against_before_reference':sum(x['d14_converged']=='1' and (y['candidate_converged']!='1' or abs(float(x['d14_mu'])-float(y['candidate_mu']))/max(abs(float(x['d14_mu'])),1e-300)>float(x['tol'])) for x,y in zip(a,b)),'magnification_changes':sum(x['candidate_mu']!=y['candidate_mu'] for x,y in zip(a,b)),'maximum_relative_change':max(abs(float(x['candidate_mu'])-float(y['candidate_mu']))/max(abs(float(x['candidate_mu'])),1e-300) for x,y in zip(a,b)),'node_count_changes':sum(x['nodes']!=y['nodes'] for x,y in zip(a,b)),'maximum_discrepancy_over_tolerance':max(float(r['relative_difference'])/float(r['tol']) for r in b if r['d14_converged']=='1')}
s['profile']={}
labels={4:'arc_ms',5:'pair_ms',6:'endpoint_ms',7:'K_ms',12:'physical_ms',13:'estimator_ms',14:'scheduler_ms',15:'setup_ms',16:'setup_event_ms',17:'setup_panel_ms',19:'integration_ms',20:'dynamic_count_ms',21:'allocation_ms',22:'allocation_count',23:'arc_body_ms',24:'tail_ms',25:'node_loop_ms',26:'header_ms',27:'coeff_ms',28:'rescue_ms',37:'prediction_ms',44:'map_ms',58:'frontier_count_ms',59:'frontier_event_ms'}
for name,p in [('before',D/'profile_before.log'),('after',D/'profile_after.log')]:
 rows=[f for l in open(p) if (f:=l.split()) and f[0]=='PROFILE' and f[3]=='0.0001' and (f[1],f[2]) in keys]
 s['profile'][name]={n:sum(float(r[i]) for r in rows)/len(rows) for i,n in labels.items()}
 if name=='after':s['image_starts']={'attempts':sum(int(r[60]) for r in rows),'accepted':sum(int(r[61]) for r in rows),'fraction':sum(int(r[61]) for r in rows)/sum(int(r[60]) for r in rows)}
(D/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
paths=['src/lcbinint/magnification/holonomic/'+n+'.hpp' for n in ['adaptive_epoch','adaptive_radial','radius_terms','quartic_sturm','v2_profile','seeded_radial_frontier','seeded_boundary_events']]+['tests/holonomic_cpp/test_adaptive_radial.cpp','tests/holonomic_cpp/test_seeded_radial_frontier.cpp','tests/holonomic_cpp/test_chebyshev_prediction.cpp','benchmarks/holonomic/build_seeded_radial_master.sh','benchmarks/holonomic/bench_seeded_radial_full.cpp','benchmarks/holonomic/bench_seeded_integration_profile.cpp']
source=D/'source';source.mkdir(exist_ok=True)
for p in paths:(source/pathlib.Path(p).name).write_bytes((root/p).read_bytes())
binaries=['build-seeded-events/bench_push_'+n+'_v29' for n in ['before','after','profile','profile_before']]
(D/'manifest.json').write_text(json.dumps({'checkout':str(root),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'timing':'one thread; no concurrent compilation; same87 normal pairs tol1e-4,5 repetitions, mean of per-condition medians; reverse order repeated; fresh geometry/source objects; seed included, caustic construction excluded','sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},'binaries':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in binaries},'environment':subprocess.check_output(['uname','-a'],text=True).strip()},indent=2)+'\n')
m=json.loads((D/'manifest.json').read_text())
m['inputs']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'evidence/holonomic/caustic_seeded_connection_v4'/(n+'_minimal_seeds.tsv') for n in ['initial','fold','deeper']]}
m['baseline_frontier_sha256']=hashlib.sha256((D/'before_include/lcbinint/magnification/holonomic/seeded_radial_frontier.hpp').read_bytes()).hexdigest()
m['seed_object_sha256']=hashlib.sha256((root/'build-seeded-events/seed_after_v26.o').read_bytes()).hexdigest()
m['reference_exclusions']=142
m['profile_timing']='fresh before and after serial diagnostic runs,3 reps'
(D/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(s,indent=2))
