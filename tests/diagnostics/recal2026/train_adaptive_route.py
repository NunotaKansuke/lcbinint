"""Small cost-sensitive trees, split by event before fitting or tuning.

Accuracy references are used for reporting only, never as router inputs or
training labels. A non-accepted Cartesian call is not counted as a cheap win.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'tests/diagnostics/results/recal2026/predictive-route-20261015'
NAMES=['log_s','log_q','log_rho','limb','log_tolerance','log_point_amp','image_count',
       'log_radial_stretch','log_angular_stretch','log_axis_stretch','log_perimeter',
       'log_curvature','log_min_radius','log_max_radius','negative_flux_fraction']

def read(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))

def split(case):
    bucket=int(hashlib.sha256(('route-v1:'+case).encode()).hexdigest()[:8],16)%10
    return 'train' if bucket<6 else 'validation' if bucket<8 else 'test'

def fit(x,y,w,features,depth,min_leaf=20):
    vote=float(np.sum(w*y));total=float(w.sum());leaf=dict(polar=vote>total/2,n=len(y))
    if depth==0 or len(y)<2*min_leaf:return leaf
    best=min(vote,total-vote);found=None
    for j in features:
        order=np.argsort(x[:,j],kind='stable');v=x[order,j];ww=w[order];yy=y[order]
        pos=np.cumsum(ww*yy);neg=np.cumsum(ww*(1-yy));n=len(y)
        valid=np.where((np.arange(n)>=min_leaf-1)&(np.arange(n)<n-min_leaf)&(v<np.roll(v,-1)))[0]
        if not len(valid):continue
        loss=np.minimum(pos[valid],neg[valid])+np.minimum(vote-pos[valid],total-vote-neg[valid])
        k=int(valid[np.argmin(loss)]);value=float(loss.min())
        if value<best-1e-12:
            best=value;found=(j,float((v[k]+v[k+1])/2))
    if found is None:return leaf
    j,t=found;m=x[:,j]<=t
    return dict(feature=j,threshold=t,n=len(y),left=fit(x[m],y[m],w[m],features,depth-1,min_leaf),
                right=fit(x[~m],y[~m],w[~m],features,depth-1,min_leaf))

def predict(tree,x):
    if 'ensemble' in tree:
        return int(tree['base']+tree['rate']*sum(regress(t,x)for t in tree['ensemble'])>0)
    while 'feature' in tree:tree=tree['left'] if x[tree['feature']]<=tree['threshold'] else tree['right']
    return int(tree['polar'])

def regress(tree,x):
    while 'feature'in tree:tree=tree['left']if x[tree['feature']]<=tree['threshold']else tree['right']
    return tree['value']

def fit_regression(x,y,depth):
    leaf=dict(value=float(y.mean()))
    if not depth or len(y)<24:return leaf
    best=float(np.sum((y-y.mean())**2));found=None
    for j in range(x.shape[1]):
        order=np.argsort(x[:,j],kind='stable');v=x[order,j];yy=y[order];n=len(y)
        sums=np.cumsum(yy);squares=np.cumsum(yy*yy)
        ix=np.where((np.arange(n)>=11)&(np.arange(n)<n-12)&(v<np.roll(v,-1)))[0]
        if not len(ix):continue
        count=ix+1
        loss=squares[ix]-sums[ix]**2/count+(squares[-1]-squares[ix])-(sums[-1]-sums[ix])**2/(n-count)
        k=int(ix[np.argmin(loss)]);score=float(loss.min())
        if score<best-1e-10:best=score;found=(j,float((v[k]+v[k+1])/2))
    if found is None:return leaf
    j,t=found;m=x[:,j]<=t
    return dict(feature=j,threshold=t,left=fit_regression(x[m],y[m],depth-1),right=fit_regression(x[~m],y[~m],depth-1))

def summarize(rows,pred):
    if not rows:return dict(n=0)
    pairs=[(r,p) for r,p in zip(rows,pred) if all(math.isfinite(t) and t>0 for t in r['times'])]
    timed=[r for r,p in pairs]
    selected=[r['times'][p]+r['overhead_ms'] for r,p in pairs]
    oracle=[min(r['times']) for r in timed]
    valid=[i for i,(r,p) in enumerate(zip(rows,pred)) if r['accepted'][p] and math.isfinite(r['mag'][p])]
    false=[r['id'] for r,p in zip(rows,pred) if r['accepted'][p] and r['errors'][p]>r['target']]
    cost=[r['costs'][p]+r['overhead_ms'] for r,p in zip(rows,pred)]
    return dict(n=len(rows),polar_selected=sum(pred),accepted=len(valid),
        false_accept_ids=false,not_accepted=[r['id'] for r,p in zip(rows,pred) if not r['accepted'][p]],
        latency_pairs=len(pairs),
        median_cartesian_over_selected=statistics.median(r['times'][0]/s for r,s in zip(timed,selected)),
        median_polar_over_selected=statistics.median(r['times'][1]/s for r,s in zip(timed,selected)),
        median_selected_over_oracle=statistics.median(s/o for s,o in zip(selected,oracle)),
        p90_selected_over_oracle=float(np.quantile([s/o for s,o in zip(selected,oracle)],.9)),
        total_ms=sum(selected),always_cartesian_ms=sum(r['times'][0]for r in timed),
        always_polar_ms=sum(r['times'][1]for r in timed),oracle_ms=sum(oracle),
        penalized_cost=sum(cost),
        geometric_penalized_regret=math.exp(statistics.mean(math.log(c/min(r['costs']))for r,c in zip(rows,cost))),
        geometric_selected_over_oracle=math.exp(statistics.mean(math.log(s/o)for s,o in zip(selected,oracle))),
        median_feature_us=statistics.median(r['overhead_ms']*1000 for r in rows))

def main():
    inputs={r['geom_id']:r for r in read(ROOT/'figures/recal2026/adaptive-vbm-six-panel-20261014/inputs.tsv')}
    refs={k:float(r['reference']) for k,r in inputs.items()}
    for r in read(ROOT/'figures/recal2026/polar-relative-matched-six-panel-20261015/reference-adjudications.tsv'):
        refs[r['geom_id']]=float(r['adjudicated_reference'])
    features={(r['id'],float(r['target'])):r for r in read(OUT/'features.tsv')}
    raw={}
    timing_paths=sorted(OUT.glob('timing-[0-9]*.tsv'))
    if not timing_paths:
        timing_paths=[OUT/'paired-timing-raw.tsv']
    for path in timing_paths:
        for r in read(path):raw.setdefault((r['sample_id'],float(r['target']),r['method']),[]).append(r)
    data=[]
    for key,target in sorted({(k,t)for k,t,m in raw},key=lambda z:(int(z[0]),z[1])):
        rr=[raw.get((key,target,m),[])for m in ['cartesian','polar']]
        if any(len(a)!=3 for a in rr):raise RuntimeError('incomplete timing '+key)
        inp=inputs[key];f=features[key,target]
        times=[statistics.median(float(a['cold_ms'])for a in group)for group in rr]
        accepted=[all(a['cold_met']=='1' for a in group)for group in rr]
        mags=[float(group[0]['cold_magnification']) for group in rr]
        penalty=max(t for t in times if math.isfinite(t))*10
        costs=[t if ok and math.isfinite(t) else penalty for t,ok in zip(times,accepted)]
        data.append(dict(id=key,case_id=inp['case_id'],profile=inp['profile'],target=target,
            split=split(inp['case_id']),x=[float(f['f'+str(k)])for k in range(15)],
            overhead_ms=float(f['feature_us'])/1000,root_us=float(f['root_us']),
            times=times,costs=costs,accepted=accepted,mag=mags,
            errors=[abs(m/refs[key]-1)if math.isfinite(m)else math.inf for m in mags]))
    x=np.array([r['x']for r in data]);y=np.array([r['costs'][1]<r['costs'][0]for r in data],dtype=float)
    absolute=np.array([abs(r['costs'][0]-r['costs'][1])for r in data]);absolute=np.minimum(absolute,np.quantile(absolute,.95))
    relative=np.array([abs(math.log(r['costs'][0]/r['costs'][1]))for r in data])
    train=np.array([r['split']=='train'for r in data]);validation=np.array([r['split']=='validation'for r in data])
    candidates=[];trees={}
    for name,ff,w in [(n+'-'+objective,ff,weights) for n,ff in [('point',[3,4,5]),('basic',list(range(7))),('geometry',list(range(15)))] for objective,weights in [('absolute',absolute),('relative',relative)]]:
        for depth in range(1,7):
            tree=fit(x[train],y[train],w[train],ff,depth)
            pred=[predict(tree,r['x'])for r in data]
            val=summarize([r for r in data if r['split']=='validation'],[p for p,r in zip(pred,data)if r['split']=='validation'])
            label=name+'-'+str(depth);trees[label]=tree
            candidates.append(dict(model=label,validation=val))
    chosen=min(candidates,key=lambda r:r['validation']['geometric_penalized_regret'])['model'];tree=trees[chosen]
    # Four independently fitted leaves account explicitly for LD and tolerance.
    # Tuning uses validation events only; no test-event labels enter fitting.
    forest={};forest_candidates=[]
    for profile in ['uniform','linear']:
        for target in [1e-3,1e-4]:
            subgroup=np.array([r['profile']==profile and r['target']==target for r in data])
            trials=[]
            for depth in range(1,5):
                for objective,w in [('absolute',absolute),('relative',relative)]:
                    t=fit(x[train&subgroup],y[train&subgroup],w[train&subgroup],list(range(15)),depth,8)
                    rows=[r for r in data if r['profile']==profile and r['target']==target and r['split']=='validation']
                    score=summarize(rows,[predict(t,r['x']) for r in rows])
                    trials.append(dict(profile=profile,target=target,depth=depth,objective=objective,tree=t,validation=score))
            selected=min(trials,key=lambda v:v['validation']['geometric_penalized_regret'])
            forest[profile,target]=selected['tree'];forest_candidates+=trials
    forest_tree=dict(feature=3,threshold=0.0,left=dict(feature=4,threshold=-3.5,
        left=forest['uniform',1e-4],right=forest['uniform',1e-3]),
        right=dict(feature=4,threshold=-3.5,left=forest['linear',1e-4],right=forest['linear',1e-3]))
    forest_val=summarize([r for r in data if r['split']=='validation'],[predict(forest_tree,r['x'])for r in data if r['split']=='validation'])
    global_val=next(c['validation']for c in candidates if c['model']==chosen)
    if forest_val['geometric_penalized_regret']<global_val['geometric_penalized_regret']:
        chosen='profile-tolerance-forest';tree=forest_tree
    boost_candidates=[]
    log_ratio=np.clip(np.array([math.log(r['costs'][0]/r['costs'][1])for r in data]),-4,4)
    best_score=summarize([r for r in data if r['split']=='validation'],[predict(tree,r['x'])for r in data if r['split']=='validation'])['geometric_penalized_regret']
    for depth in [2,3]:
        base=float(np.mean(log_ratio[train]));scores=np.full(len(data),base);ensemble=[]
        for iteration in range(80):
            t=fit_regression(x[train],(log_ratio-scores)[train],depth);ensemble.append(t)
            scores+=.12*np.array([regress(t,row)for row in x])
            if iteration+1 not in [20,40,80]:continue
            val_rows=[r for r in data if r['split']=='validation']
            val=summarize(val_rows,[int(v>0)for v in scores[validation]])
            boost_candidates.append(dict(depth=depth,iterations=iteration+1,validation=val))
            if val['geometric_penalized_regret']<best_score:
                best_score=val['geometric_penalized_regret'];chosen=f'boost-{depth}-{iteration+1}'
                tree=dict(base=base,rate=.12,ensemble=ensemble.copy())
    result=dict(chosen=chosen,feature_names=NAMES,split_rule='sha256(route-v1:case_id) mod 10; train<6, validation<8, test>=8',
                candidates=candidates,forest_candidates=forest_candidates,boost_candidates=boost_candidates,tree=tree,groups=[])
    for name,t in [('selected',tree),('point_best',trees[min((c for c in candidates if c['model'].startswith('point-')),key=lambda r:r['validation']['geometric_penalized_regret'])['model']])]:
        for partition in ['train','validation','test']:
            for profile in ['uniform','linear']:
                for target in [1e-3,1e-4]:
                    rows=[r for r in data if r['split']==partition and r['profile']==profile and r['target']==target]
                    result['groups'].append(dict(model=name,split=partition,profile=profile,target=target,**summarize(rows,[predict(t,r['x'])for r in rows])))
    result['splits']={s:dict(events=len({r['case_id']for r in data if r['split']==s}),conditions=sum(r['split']==s for r in data))for s in ['train','validation','test']}
    (OUT/'model.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
    def emit(t,indent='    '):
        if 'ensemble'in t:
            def expr(node):
                if 'feature'not in node:return f"{node['value']:.17g}"
                return f"(f[{node['feature']}] <= {node['threshold']:.17g} ? {expr(node['left'])} : {expr(node['right'])})"
            return indent+f"double score={t['base']:.17g};\n"+''.join(indent+f"score += {t['rate']:.17g} * {expr(node)};\n"for node in t['ensemble'])+indent+'return score > 0;\n'
        if 'feature' not in t:return indent+'return '+('true'if t['polar']else'false')+';\n'
        return indent+f"if (f[{t['feature']}] <= {t['threshold']:.17g}) {{\n"+emit(t['left'],indent+'    ')+indent+'} else {\n'+emit(t['right'],indent+'    ')+indent+'}\n'
    header='''#pragma once
#include <array>
namespace lcbinint::magnification::adaptive_detail {
// Empirical v1 model. True selects Polar. Trained without event overlap.
// Applicable to relative tolerance 1e-3/1e-4, uniform or linear LD only.
inline bool predict_adaptive_polar(const std::array<double,15>& f) {
'''+emit(tree)+'}\n}\n'
    (ROOT/'src/lcbinint/magnification/adaptive_route_model.hpp').write_text(header)
    print(chosen,result['splits'])
    for g in result['groups']:
        if g['split']=='test':print(g['model'],g['profile'],g['target'],'n',g['n'],'polar/selected',g.get('median_polar_over_selected'),'regret',g.get('median_selected_over_oracle'),'unmet',len(g.get('not_accepted',[])))

if __name__=='__main__':main()
