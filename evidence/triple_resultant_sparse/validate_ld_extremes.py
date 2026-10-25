import pathlib,json,sys,os
P=pathlib.Path(__file__).resolve().parent;B=P.parents[1]/'build-triple-native';os.sched_setaffinity(0,{0});sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable' not in str(f)];sys.path.insert(0,str(B));import lcbinint
refs=json.load(open(P/'contact-conditions.json'));cs=refs['cases'];rows=[]
# Reuse original tight u=.5 references and verify U and u=1 via the linear
# source-profile identity: pi rho^2 F(u) is affine before normalization.
reference={(r['case'],r['limb']):r['result']['samples'][-1] for r in refs['references']}
for name in ['contact','large','collinear','contact_rotated']+[f'grazing{i}' for i in range(12)]:
 c=cs[name];r0=reference[name,0.];rh=reference[name,.5];mu1=(5*rh['mu']-3*r0['mu'])/2;err1=(5*rh['error_estimate']+3*r0['error_estimate'])/2
 for u in [0.,1.]:
  mu,err=(r0['mu'],r0['error_estimate']) if u==0 else (mu1,err1)
  for tol in [.001,.0001]:
   v=lcbinint.triple_algebraic([complex(*z) for z in c['lenses']],c['masses'],complex(*c['source']),c['rho'],limb=u,atol=0,rtol=tol);dif=abs(v['mu']-mu);row=dict(case=name,limb=u,rtol=tol,value=v,reference_mu=mu,reference_error=err);rows.append(row);assert v['value_converged'] and v['support_complete'] and dif<=tol*abs(mu) and dif<=v['error_estimate']+err,row
(P/'ld-extremes.json').write_text(json.dumps(dict(status='PASS',rows=rows,reference='u=1 inferred from flux-affine exact linear LD identity, tight U and u=.5 references'),indent=2)+'\n');print(len(rows),'PASS')
