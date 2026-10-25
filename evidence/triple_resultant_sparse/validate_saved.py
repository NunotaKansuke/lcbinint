import pathlib,json,sys,os
P=pathlib.Path(__file__).resolve().parent;B=P.parents[1]/'build-triple-native';os.sched_setaffinity(0,{0});sys.meta_path=[f for f in sys.meta_path if '_lcbinint_editable' not in str(f)];sys.path.insert(0,str(B));import lcbinint
assert pathlib.Path(lcbinint._native.__file__).resolve().is_relative_to(B)
rows=[]
for family in ['heldout','contact-heldout']:
 old=json.load(open(P.parent/'triple_uniform_profile'/f'{family}.json'))
 for ref in old['references']:
  a=list(map(float,ref['input'].split()));c=dict(source=a[:2],rho=a[2],lenses=[complex(a[j],a[j+1]) for j in [3,6,9]],masses=[a[j] for j in [5,8,11]])
  for tol in [.001,.0001]:
   v=lcbinint.triple_algebraic(c['lenses'],c['masses'],complex(*c['source']),c['rho'],limb=ref['limb'],atol=0,rtol=tol);r=ref['result'];error=abs(v['mu']-r['mu']);row=dict(family=family,source=c['source'],limb=ref['limb'],rtol=tol,value=v,reference=r,relative_error=error/abs(r['mu']));rows.append(row)
   assert v['value_converged'] and v['support_complete'] and v['parameter_derivative_evaluations']==0 and error<=tol*abs(r['mu']) and error<=v['error_estimate']+r['error_estimate'],row
(P/'saved-heldout.json').write_text(json.dumps(dict(status='PASS',rows=rows,extension=lcbinint._native.__file__),indent=2)+'\n');print(len(rows),'PASS')
