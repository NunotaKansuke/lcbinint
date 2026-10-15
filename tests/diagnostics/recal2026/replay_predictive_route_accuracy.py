"""Replay selection against prior complete per-method numerical audits.

No new integral is computed here. This broadens failure visibility, not the
independence of the learned router's held-out timing assessment.
"""
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'tests/diagnostics/results/recal2026/predictive-route-20261015'
OLD=ROOT/'tests/diagnostics/results/recal2026/refinement-work-optimization-20261015'

def main():
    spec=importlib.util.spec_from_file_location('train',Path(__file__).with_name('train_adaptive_route.py'))
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    model=json.loads((OUT/'model.json').read_text())['tree']
    features={(r['id'],float(r['target'])):[float(r['f'+str(k)])for k in range(15)] for r in train.read(OUT/'features.tsv')}
    inputs=train.read(ROOT/'figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv')
    frozen=OLD/'frozen-scalar-v3/liblcbinint_magnification.a'
    current=ROOT/'build-adaptive-production/liblcbinint_magnification.a'
    def members(path):
        names=subprocess.check_output(['ar','t',str(path)],text=True).splitlines()
        return {name:hashlib.sha256(subprocess.check_output(['ar','p',str(path),name])).hexdigest()for name in names}
    objects=members(current)
    assert members(frozen)==objects, 'kernel object changed; audit replay requires revalidation'
    (OUT/'kernel-object-parity.json').write_text(json.dumps(dict(identical=True,objects=objects),indent=2)+'\n')
    groups={};selection=[]
    for index,inp in enumerate(inputs):
        for target in [1e-3,1e-4]:
            method='polar'if train.predict(model,features[inp['geom_id'],target])else'cartesian'
            data=json.loads((OLD/f'final-{method}-full'/f'{index:05d}.json').read_text())
            row=next(r for r in data if r['target']==target)
            assert row['id']==inp['geom_id']
            key=inp['profile']+':'+str(target)
            g=groups.setdefault(key,dict(conditions=0,polar_selected=0,accepted=0,exceptions=[],unmet=[],false_accepts=[],contract_failures=[]))
            g['conditions']+=1;g['polar_selected']+=method=='polar'
            if 'exception'in row:g['exceptions'].append(row['id'])
            elif not row['result']['estimated_target_met']:g['unmet'].append(row['id'])
            else:g['accepted']+=1
            if row.get('accepted_reference_disagreement'):g['false_accepts'].append(row['id'])
            if row.get('contract') is False:g['contract_failures'].append(row['id'])
            selection.append(dict(id=row['id'],target=target,selected=method,split=train.split(inp['case_id'])))
    result=dict(groups=groups,selection=selection,scope='Saved full numerical audits, unchanged static kernel archive; replay, not newly computed integrations or a blind holdout')
    (OUT/'full-accuracy-replay.json').write_text(json.dumps(result,indent=2)+'\n')
    for key,g in groups.items():print(key,{k:(len(v)if isinstance(v,list)else v)for k,v in g.items()})

if __name__=='__main__':main()
