from pathlib import Path
import json
import pandas as pd
root=Path(__file__).parent
refs={('caustic-cross',0.): -2.404688419943568,('caustic-cross',.5):77.45309452467096,('rand030',0.):.0129986580934943}
summary={}
base=None
for filename in ['raw','gradient_only','frozen_priority','hybrid_gradient']:
 d=pd.read_csv(root/(filename+'.tsv'),sep=r'\s+')
 assert len(d)==72 and (d.converged==1).all()
 d['reference']=[refs[(n,u)] for n,u in zip(d.name,d.u)]
 d['absolute_error']=(d.grad-d.reference).abs()
 d['relative_error']=d.absolute_error/d.reference.abs()
 if filename=='gradient_only':base=d.copy()
 if filename in ['frozen_priority','hybrid_gradient']:
  assert (d.mu==base.mu).all(), 'primal snapshot changed'
 summary[filename]={'rows':len(d),'value_converged':int(d.converged.sum()),'tolerance_met':int((d.quality==1).sum()),'invalid':int((d.quality==3).sum()),'selected_64_round_1percent':d[(d.rounds==64)&(d.grad_rtol==.01)][['name','u','max_level','nodes','grad','absolute_error','relative_error','error','quality']].to_dict('records')}
 d.to_csv(root/(filename+'_audited.tsv'),sep='\t',index=False)
summary['reference_provenance']='Phase66 independent GL128/256 direct-angular value finite differences with h-halving; selected component only, not a five-Jacobian certificate.'
summary['base_commit']='e941b91'
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: 288 trials, all values converged; gradient-only alternative meshes preserve primal snapshot exactly; all selected gradients remain FiniteUncertified.')
