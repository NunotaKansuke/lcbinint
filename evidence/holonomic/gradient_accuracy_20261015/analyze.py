from pathlib import Path
import csv,json,math,statistics,hashlib,subprocess
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
ref=list(csv.DictReader((out/'reference.csv').open()))
refs={}
for t in sorted(set(float(x['t']) for x in ref)):
 for p in ['t0','u0','rho','q','s']:
  r=[x for x in ref if float(x['t'])==t and x['param']==p]
  vals={k:[float(x[k]) for x in r] for k in ['vbm_fd','native_fd']}
  refs[t,p]={k:statistics.median(v) for k,v in vals.items()}
  refs[t,p].update({k+'_spread':max(v)-min(v) for k,v in vals.items()})
rows=[];parity={};counts={}
for tol in ['1e-4','1e-5','1e-6']:
 base=list(csv.DictReader((out/f'baseline_{tol}.tsv').open(),delimiter='\t'))
 for variant in ['baseline','allmotion_post2','allmotion_posth']:
  data=list(csv.DictReader((out/f'{variant}_{tol}.tsv').open(),delimiter='\t'))
  parity[variant+'_'+tol]=all(a['mu']==b['mu'] for a,b in zip(base,data)) and len(base)==len(data)
  for t in sorted(set(float(x['t']) for x in data)):
   r=[x for x in data if float(x['t'])==t];g=[float(x['g']) for x in r]
   c,s=math.cos(.5),math.sin(.5)
   physical=dict(t0=-(c*g[0]+s*g[1])/10,u0=-s*g[0]+c*g[1],rho=g[2],q=-g[3]*1e8,s=g[4])
   for p,v in physical.items():
    reference=refs[t,p]
    rows.append(dict(t=t,tol=tol,variant=variant,param=p,gradient=v,**reference,vbm_abs_difference=abs(v-reference['vbm_fd']),native_abs_difference=abs(v-reference['native_fd']),mu=float(r[0]['mu']),nodes=int(r[0]['nodes']),kernel_ms=float(r[0]['ms']),value_converged=r[0]['value_converged'],stop=r[0]['stop']))
with (out/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
summary={'API_constraint':'mu absolute/relative tolerances only; no derivative-error-driven adaptation in candidates', 'candidate':'allmotion_posth: material motion on every radial span, full-circle prefactors included, one deterministic h-split of every accepted radial panel', 'value_bitwise_parity':parity,'reference_caveat':'At t=0.1 VBM t0 FD is unstable; VBM/native q and s FD disagree. Keep both references; do not treat all rows as independently validated truth.', 'sample_count':7,'tol_settings':[1e-4,1e-5,1e-6], 'parameter_order':['t0','u0','rho','q','s'],'timing':'one exploratory evaluation per condition, excludes seed construction; not a speed claim','promotion':'research prototype only; loose 1e-4 tolerance still has large gradient errors'}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
paths=list(out.glob('*.py'))+list(out.glob('*.cpp'))+list(out.glob('*.sh'))
paths += list((root/'benchmarks/holonomic/fused_gradient').glob('*.py'))+[root/'benchmarks/holonomic/fused_gradient/fused_k.hpp',root/'benchmarks/holonomic/fused_gradient/fused_motion.hpp',root/'src/lcbinint/magnification/holonomic/fused_gradient.cpp',root/'src/lcbinint/model/lens_model.cpp',root/'src/lcbinint/magnification/finite_source_magnifier.cpp']
paths += list((root/'build-algebraic-boundary/gradient-probe').glob('probe*'))
paths += list((root/'build-algebraic-boundary/python-api').glob('*.a'))
paths += list((root/'build-algebraic-boundary/python-api/lcbinint').glob('*.so'))
paths += [root/'python/lcbinint/holonomic_fused.py']
paths += list(out.glob('*.tsv'))+list(out.glob('*.csv'))+[out/'config.json']
provenance={'checkout':str(root),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0]}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
(out/'native_source.diff').write_text(subprocess.check_output(['git','diff','--','src/lcbinint/model/lens_model.cpp','src/lcbinint/magnification/finite_source_magnifier.cpp'],cwd=root,text=True))
print('Saved comparison.csv, summary.json, provenance.json; exact primal parity:',all(parity.values()))
