import pathlib,json,subprocess,sys,statistics,os
P=pathlib.Path(__file__).resolve().parent;os.sched_setaffinity(0,{0});conditions=[json.loads(f.read_text()) for f in (P/'conditions').glob('*-algebraic-*.json')];report=dict(status='RUNNING',rows=[],pairs=[])
for name in ['contact','large']:
 for tol in [.001,.0001]:
  cs={u:next(c for c in conditions if (c['name'],c['limb'],c['rtol'])==(name,u,tol)) for u in [0.,.5]}
  for kind in ['centre','nearby']:
   keys=[('before',0.),('after',0.),('before',.5),('after',.5)]
   workers={k:subprocess.Popen([sys.executable,str(P/'paired_stream.py'),k[0]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1) for k in keys}
   for k,w in workers.items():w.stdin.write(json.dumps(cs[k[1]])+'\n');w.stdin.flush()
   def call(k,s):
    w=workers[k];w.stdin.write(json.dumps(s)+'\n');w.stdin.flush();line=w.stdout.readline();assert line,w.stderr.read();return json.loads(line)
   for _ in range(9):
    for k in keys:call(k,cs[0.]['case']['source'])
   samples=[cs[0.]['case']['source']] if kind=='centre' else cs[0.]['sources'][9:];pairs=[]
   for cycle in range(32 if kind=='centre' else 4):
    for j,s in enumerate(samples):
     r=dict(case=name,rtol=tol,kind=kind,index=j,cycle=cycle,source=s)
     offset=(cycle+j)%4;order=keys[offset:]+keys[:offset]
     if (cycle+j)%2:order=order[::-1]
     for k in order:
      v=call(k,s);ref=json.load(open(P/'references'/f'{name}-u{k[1]}-i{0 if kind=="centre" else j+1}.json'))['result'];error=abs(v['mu']-ref['mu']);assert v['converged'] and error<=tol*abs(ref['mu']) and error<=v['error']+ref['error_estimate'],(r,k,v);r[f'{k[0]}-{k[1]}']=v
     pairs.append(r);report['pairs'].append(r)
   for w in workers.values():w.stdin.close();assert w.wait()==0,w.stderr.read()
   row=dict(case=name,rtol=tol,kind=kind,pairs=len(pairs),before_ld_over_u=statistics.median(r['before-0.5']['seconds']/r['before-0.0']['seconds'] for r in pairs),after_ld_over_u=statistics.median(r['after-0.5']['seconds']/r['after-0.0']['seconds'] for r in pairs),uniform_speedup=statistics.median(r['before-0.0']['seconds']/r['after-0.0']['seconds'] for r in pairs),ld_speedup=statistics.median(r['before-0.5']['seconds']/r['after-0.5']['seconds'] for r in pairs),median_ms={f'{k[0]}-{k[1]}':1000*statistics.median(r[f'{k[0]}-{k[1]}']['seconds'] for r in pairs) for k in keys});report['rows'].append(row);(P/'paired-profiles.json').write_text(json.dumps(report,indent=2)+'\n');print(row,flush=True)
report['status']='PASS';(P/'paired-profiles.json').write_text(json.dumps(report,indent=2)+'\n')
