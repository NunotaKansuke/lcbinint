import json,math
from pathlib import Path
P=Path(__file__).resolve().parent;OLD=P.parent/'cartesian-final-redesign-20261015'
def same(x,y):
 if isinstance(x,float)and isinstance(y,float)and math.isnan(x)and math.isnan(y):return True
 if isinstance(x,dict)and isinstance(y,dict):return x.keys()==y.keys()and all(same(v,y[k])for k,v in x.items())
 if isinstance(x,list)and isinstance(y,list):return len(x)==len(y)and all(same(a,b)for a,b in zip(x,y))
 return x==y
reports={}
for method,baseline in [('cartesian','selected-v7-full16'),('polar','selected-v7-polar-full')]:
 new=P/f'final-{method}-full';old=OLD/baseline;changed=[];count=0
 for f in sorted(new.glob('[0-9]*.json')):
  a=json.loads(f.read_text());b=json.loads((old/f.name).read_text());assert len(a)==len(b)
  for x,y in zip(a,b):
   assert (x['id'],x['target'])==(y['id'],y['target']);count+=1
   if not same(x.get('result'),y.get('result'))or x.get('exception')!=y.get('exception'):changed.append([x['id'],x['target']])
 assert count==14432
 r=dict(conditions=count,changed=changed,new_identity=json.loads((new/'identity.json').read_text()),old_native_sha256=json.loads((old/'identity.json').read_text())['native_sha256']);(P/f'final-{method}-parity.json').write_text(json.dumps(r,indent=2)+'\n');reports[method]=dict(conditions=count,changed=len(changed));assert not changed
print(reports)
