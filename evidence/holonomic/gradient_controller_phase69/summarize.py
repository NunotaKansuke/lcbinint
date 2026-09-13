from pathlib import Path
import pandas as pd
import json
r=Path(__file__).parent
p=pd.read_csv(r/'raw.tsv',sep=r'\s+')
refs={('caustic-cross',0.): -2.404688419943568,('caustic-cross',.5):77.45309452467096,('rand030',0.):.0129986580934943}
p['reference']=[refs[(n,u)] for n,u in zip(p.name,p.u)]
p['relative_error']=(p.grad-p.reference).abs()/p.reference.abs()
p.to_csv(r/'audited.tsv',sep='\t',index=False)
c=pd.read_csv(r/'corpus.tsv',sep=r'\s+',header=None)
a=c[c[0]==0].reset_index(drop=True);b=c[c[0]==1].reset_index(drop=True)
assert len(a)==len(b)==220
assert (a[4]==b[4]).all() and (a[5]==b[5]).all()
newinvalid=sum(int(((a[j]!=3)&(b[j]==3)).sum()) for j in (9,11,13,15,17))
assert newinvalid==0
s={'base_commit':'4cfbdc8','difficult_trials':len(p),'corpus_paired_rows':220,'value_converged_each':int(a[5].sum()),'primal_changes':0,'new_invalid_components':newinvalid,'certified_to_uncertified':sum(int(((a[j]==1)&(b[j]==2)).sum()) for j in (9,11,13,15,17)),'selected':p[(p.rounds==64)&(p.grad_rtol==.01)].to_dict('records')}
if (r/'reference.csv').exists():
 ref=pd.read_csv(r/'reference.csv');s['reference']={'rows':len(ref),'usable':int(ref.reference_usable.sum()),'violations':int((ref.violation==1).sum())}
(r/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({k:v for k,v in s.items() if k!='selected'}))
