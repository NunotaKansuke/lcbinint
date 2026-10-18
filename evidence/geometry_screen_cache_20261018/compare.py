from pathlib import Path
import numpy as np,json,sys
p=Path(__file__).resolve().parent;old=p.parent/'easy_gradient_20261018/after';new=p/sys.argv[1]
b=np.loadtxt(p.parent/'easy_gradient_20261018/baseline-native-stages.txt').reshape(3,2,401,-1)[2,1];mask=(b[:,3]==7)&(b[:,19]==0)&(b[:,29]==0)
a=np.load(old/'notebook.npz');c=np.load(new/'notebook.npz')
for name,m in [('simple',mask),('hard',(b[:,3]==7)&~mask),('hex',b[:,3]==1)]:
 print(name,int(m.sum()),'gradient',np.median(a['gradient_seconds'],axis=0)[m].sum(),np.median(c['gradient_seconds'],axis=0)[m].sum(),'value',np.median(c['value_seconds'],axis=0)[m].sum())
for case in ['notebook','ordinary','crossing','point']:
 a=np.load(old/(case+'.npz'));c=np.load(new/(case+'.npz'))
 print(case,{k:bool(np.array_equal(a[k],c[k])) for k in ['values','fused','gradients','valid']},'maxscaledgradient',np.max(abs(a['gradients']-c['gradients'])/np.maximum(1,abs(a['gradients']))))
