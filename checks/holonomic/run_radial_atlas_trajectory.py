#!/usr/bin/env python3
"""Bounded, resumable paired trajectory run. Each shard has its own physical CPU.
No statistical row is published until all four epochs and both tolerances finish.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('input');p.add_argument('output');p.add_argument('--cpus',default='4,5,6,7');p.add_argument('--reps',type=int,default=1);a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=True);cpus=list(map(int,a.cpus.split(',')))
rows=[r for r in Path(a.input).read_text().splitlines() if r and not r.startswith('#')];groups={}
for row in rows:
 f=row.split();groups.setdefault(tuple(f[:4]),[]).append(row)
shards=[[] for _ in cpus]
for i,(key,rs) in enumerate(sorted(groups.items(),key=lambda kv:(int(kv[0][0]),int(kv[0][1]),kv[0][2],int(kv[0][3])))):
 assert len(rs)==4
 shards[i%len(shards)].extend(rs)
manifest={'binary_sha256':hashlib.sha256(Path(a.binary).read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'cpus':cpus,'reps':a.reps,'rows':len(rows)*2,'atlas_config':'double first, 2048 boxes, contract, tubes, warm range probe; no old solver fallback','timing':'outer call incl topology and temporary destruction; LensParams outside; first trajectory epoch cold, later three warm; unmeasured trajectory per lane','status':'running'}
mp=root/'manifest.json'
if mp.exists():
 old=json.loads(mp.read_text());assert all(old[k]==manifest[k] for k in ('binary_sha256','input_sha256','cpus','reps')),'refuse mixing different experiments'
 if old['status']=='generation complete' and all((root/f'{m}.tsv').exists() or (root/f'{m}.tsv.gz').exists() for m in ('incumbent','atlas')):
  print('complete artifacts already present');sys.exit(0)
mp.write_text(json.dumps(manifest,indent=2)+'\n')
def work(i):
 inp=root/f'input_{i}.tsv';inp.write_text('\n'.join(shards[i])+'\n')
 # Paired modes on the same pinned core. No compiler/test concurrency.
 for mode in ('incumbent','atlas'):
  dest=root/f'{mode}_{i}.tsv';marker=root/f'{mode}_{i}.done'
  if marker.exists():continue
  env={k:v for k,v in os.environ.items() if not k.startswith('ATLAS_')}
  if mode=='atlas':env['ATLAS_MODE']='1'
  with (root/f'{mode}_{i}.log').open('w') as log:
   subprocess.run(['taskset','-c',str(cpus[i]),a.binary,str(inp),str(dest),str(a.reps)],env=env,stdout=log,stderr=log,check=True)
  count=sum(1 for l in dest.open() if l and l[0]!='#')-1
  assert count==len(shards[i])*2,(mode,i,count)
  marker.write_text(str(count)+'\n')
with cf.ThreadPoolExecutor(max_workers=len(cpus)) as pool:list(pool.map(work,range(len(cpus))))
for mode in ('incumbent','atlas'):
 with (root/f'{mode}.tsv').open('w') as out:
  for i in range(len(cpus)):
   lines=(root/f'{mode}_{i}.tsv').read_text().splitlines()
   out.write('\n'.join(lines if i==0 else lines[2:])+'\n')
manifest['status']='generation complete';mp.write_text(json.dumps(manifest,indent=2)+'\n')
