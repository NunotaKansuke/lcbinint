#!/usr/bin/env python3
import hashlib,json,os,subprocess,sys
from pathlib import Path
import pandas as pd
binary,inp,out=sys.argv[1:];root=Path(out);root.mkdir(parents=True,exist_ok=True)
variants={'default':{},'no_contract':{'ATLAS_NO_CONTRACT':'1'},'no_tubes':{'ATLAS_NO_TUBES':'1'},'no_range_probe':{'ATLAS_NO_RANGE_PROBE':'1'},'initial_dd':{'ATLAS_INITIAL_DD':'1'},'boxes8192':{'ATLAS_MAX_BOXES':'8192'}}
summary={'binary_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(Path(inp).read_bytes()).hexdigest(),'cpu':4,'reps':1,'variants':{}}
for name,changes in variants.items():
 dest=root/f'{name}.tsv';env={k:v for k,v in os.environ.items() if not k.startswith('ATLAS_')};env['ATLAS_MODE']='1';env.update(changes)
 with (root/f'{name}.log').open('w') as log:subprocess.run(['taskset','-c','4',binary,inp,str(dest),'1'],env=env,stdout=log,stderr=log,check=True)
 d=pd.read_csv(dest,sep=r'\s+',comment='#');assert len(d)==128
 result={'env':changes,'rows':len(d),'cold_coverage':int(d.cold_value_converged.sum()),'warm_coverage':int(d.warm_value_converged.sum()),'cold_cover':d.cold_atlas.value_counts().to_dict(),'warm_cover':d.warm_atlas.value_counts().to_dict()}
 for col in ['cold_ms','warm_ms','radial_ms','cold_boxes','warm_boxes','cold_contracted','warm_contracted','cold_tubes','warm_tubes','cold_coefficient_ms','cold_proof_ms','cold_refine_ms','warm_coefficient_ms','warm_proof_ms','warm_refine_ms','warm_range_accepted']:
  result[col]={'p50':float(d[col].median()),'p90':float(d[col].quantile(.9)),'p99':float(d[col].quantile(.99))}
 result['observed_reference_violations']=int(((d.cold_value_converged==1)&(abs(d.cold_mu/d.reference-1)>d.target)).sum())
 summary['variants'][name]=result;(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(name,result['cold_coverage'],result['cold_ms'],flush=True)
