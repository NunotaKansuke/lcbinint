import argparse,json,math,statistics
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();b=args.directory

def read(path):
 lines=path.read_text().splitlines();keys=lines[0].split()
 return [dict(zip(keys,line.split())) for line in lines[1:]]

def metrics(values):
 a=sorted(values)
 return {'count':len(a),'median':statistics.median(a),'p90':a[int(.9*(len(a)-1))],'sum':sum(a)} if a else {'count':0}

result={'CALIBRATED':False,'groups':{},'inverse_normal':{}}
for name in ['initial','fold','deeper']:
 rows=read(b/(name+'_final.tsv'));good=[x for x in rows if x['d14_converged']=='1']
 d={'rows':len(rows),'comparable':len(good),'reference_unresolved':len(rows)-len(good),'failures':sum(x['candidate_converged']!='1' or x['support']!='1' or not math.isfinite(float(x['relative_difference'])) or float(x['relative_difference'])>float(x['tol']) for x in good)}
 for column in ['candidate_no_caustic_ms','candidate_ms','seed_ms','planning_ms','candidate_integration_ms','d14_ms','nodes']:
  d[column]=metrics([float(x[column]) for x in good])
 result['groups'][name]=d
paths=[b/'current_inverse_native.tsv',b/'current_inverse_native_low.tsv']
if all(p.exists() for p in paths):
 inv=sum([read(p) for p in paths],[])
 candidates={(x['id'],x['u']):x for x in read(b/'initial_final.tsv') if float(x['tol'])==1e-4 and x['d14_converged']=='1'}
 for engine in ['1','2']:
  pairs=[];unmatched=[];levels={}
  for key,x in candidates.items():
   accepted=[v for v in inv if (v['id'],v['u'])==key and v['engine']==engine and v['converged']=='1' and math.isfinite(float(v['relative_difference'])) and float(v['relative_difference'])<=1e-4]
   if not accepted:unmatched.append(key);continue
   # Deliberately favour inverse ray: choose the measured fastest acceptable
   # fixed grid per case, with no runtime cost for selecting the resolution.
   y=min(accepted,key=lambda v:float(v['no_caustic_ms']));levels[y['nbin']]=levels.get(y['nbin'],0)+1
   pairs.append({'id':key[0],'u':key[1],'nbin':int(y['nbin']),'candidate':float(x['candidate_no_caustic_ms']),'inverse':float(y['no_caustic_ms']),'candidate_full':float(x['candidate_ms']),'inverse_full':float(y['total_ms']),'relative_difference':float(y['relative_difference'])})
  cs=[v['candidate'] for v in pairs];ir=[v['inverse'] for v in pairs]
  result['inverse_normal'][engine]={'matched':len(pairs),'unmatched':unmatched,'selected_nbin':levels,'candidate_ms':metrics(cs),'inverse_ms':metrics(ir),'candidate_over_inverse_sum':sum(cs)/sum(ir),'median_inverse_over_candidate':statistics.median(v['inverse']/v['candidate'] for v in pairs),'candidate_wins':sum(v['candidate']<v['inverse'] for v in pairs),'candidate_over_inverse_full_sum':sum(v['candidate_full'] for v in pairs)/sum(v['inverse_full'] for v in pairs),'pairs':pairs}
(b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'inverse_normal':{k:{kk:vv for kk,vv in v.items() if kk!='pairs'} for k,v in result['inverse_normal'].items()}},indent=2))
