"""Summarize matched final benchmarks and independently retained scalar references."""
import json,csv,math,statistics,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
status=json.loads((P/'status.json').read_text());assert status['status']=='COMPLETE'
refs=json.loads((P.parent/'triple_broad_api_audit/references.json').read_text())
assert (P/'cases.json').read_bytes()==(P.parent/'triple_broad_api_audit/cases.json').read_bytes()
rows=[];checks=[];gradient_checks=[];bad=invalid=mismatch=0
for key in status['rows']:
 z=json.loads((P/'rows'/(key+'.json')).read_text());c=z['condition'];eng=z['engines'];r={k:c[k]for k in ['name','limb','rtol']}
 for e,d in eng.items():
  for field in ['warm_median_ms','changed_median_ms','cold_total_ms']:r[e+'_'+field]=d[field]
 for e in ['value','gradient']:
  b=eng['before_'+e];a=eng[e]
  ratios=[x['cpu_ms']/y['cpu_ms']for x,y in zip(a['samples'],b['samples'])]
  r[e+'_paired_change_percent']=100*(statistics.median(ratios)-1)
  for v in [a['first'],*a['samples'],a['diagnostic'],*a['changed_samples']]:
   bad+=sum(x!=0 for x in v['output']['status']);invalid+=sum(not x for x in (v['output']['gradient_valid']or[]))
  ref=next(v for v in refs['rows']if v['name']==c['name']and v['limb']==c['limb'])
  out=a['diagnostic']['output'];mu=[out['mu'][i]for i in ref['indices']]
  errs={k:[abs(x/y-1)for x,y in zip(mu,v)]for k,v in ref['references'].items()}
  checks.append(dict(name=c['name'],limb=c['limb'],rtol=c['rtol'],mode=e,max_error=max(max(v)for v in errs.values()),above_request=sum(x>c['rtol']for x in errs['Singlepoly']),relative_errors=errs))
 a=eng['gradient']['diagnostic']['output'];b=eng['before_gradient']['diagnostic']['output']
 gaps=[]
 for i,(ga,gb)in enumerate(zip(a['gradients'],b['gradients'])):
  # Comparison diagnostic only: a small absolute floor, never a runtime rule.
  for j,(x,y)in enumerate(zip(ga,gb)):
   if x is None or y is None:continue
   gaps.append((abs(x-y)/(1+max(abs(x),abs(y))),i,j,x,y))
 g=max(gaps);gradient_checks.append(dict(name=c['name'],limb=c['limb'],rtol=c['rtol'],max_scaled_difference=g[0],epoch=g[1],parameter=g[2],after=g[3],before=g[4]))
 v=eng['value']['diagnostic']['output'];mismatch+=sum(x!=y for x,y in zip(v['routes'],a['routes']))
 for route,label in [(0,'point'),(1,'hex'),(7,'integral')]:r[label]=v['routes'].count(route)
 r['gradient_extra_percent']=100*(r['gradient_warm_median_ms']/r['value_warm_median_ms']-1)
 rows.append(r)
prov=json.loads((P/'provenance.json').read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in prov['hashes'].items())
summary=dict(status='COMPLETE',protocol=status['protocol'],conditions=len(rows),native_epoch_configurations=48*65*2,all_calls_nonzero_status=bad,all_calls_invalid_gradient_flags=invalid,route_mismatched_epochs=mismatch,reference_source_points=sum(len(v['indices'])for v in refs['rows']),reference_scalar_comparisons=sum(len(c['relative_errors']['Singlepoly'])for c in checks),reference_above_request=sum(c['above_request']for c in checks),max_reference_error=max(c['max_error']for c in checks),checks=checks,gradient_comparisons=gradient_checks,rows=rows)
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (P/'summary.csv').open('w')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
L=['# Triple通常APIの仕事量削減と最終比較','','## 採用した変更','','- 同時求根のAberth反発項で、根の組を一度だけ評価して正負を配る。係数絶対値も根ごとに再計算しない。収束後の残差・Vieta完全性検査、失敗時の再求根は維持。通常範囲の逆数だけ軽量化し、一般のNewton除算は標準のスケール付き複素除算を維持する。','- 通常支持領域の境界イベント探索を256 boxで打ち切らず、同じ探索途中データで1024まで継続できるようにした。少しの追加探索で高価な全域fallbackを回避する。全体のevent_budgetは引き続き尊重する。近い光源では前の境界イベントから初期角度分割を回転し、全周の証明をやり直す。古い証明を流用する処理ではない。','- 角度方向の曲率を既存の実成分演算で計算する。LDの局所quad fallbackでは、同一半径のGradientRay/ValueRayを必要時に一度構築し、端点・追加サンプルで共有する。微分方向ごとのquad除算を共通の随伴ベクトルにまとめる。','- 角度多項式で繰り返していた小さな二項係数を正確な整数表にした。値・微分の共通処理に適用。','- 要求精度、値によるrefinement、局所quadへのfallback条件を維持。微分誤差によるrefinementは追加していない。値のみ経路で微分用quadキャッシュは構築しない。','','## 比較条件','',status['protocol'],'','変更前は今回の作業開始時のdirty checkoutから保存したnative package。HEADだけの比較ではない。最終source/DSOと変更前DSOのhashはprovenance.json、開始時差分はbefore.diffに保存した。中間の一般複素除算簡略化案は極端な値で中間積のunderflowがありうるため不採用とし、general-division-trial/に隔離した。最終表には含めない。','', '## warm時間','','単位ms、65点の光度曲線全体。VBMは値のみ。native微分は値＋10パラメータの解析微分。「増減」は同じroundの変更前後比の中央値。CPU周波数は固定していないため、数%の差は慎重に解釈する。']
for limb in [0.,.5]:
 for tol in [1e-3,1e-4]:
  L+=['',f'### LD={limb}, relative={tol}','','|ケース|SinglePoly|MultiPoly|NoPoly|旧値|新値|値増減|旧微分|新微分|微分増減|新微分追加|PS/hex/積分|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
  for r in rows:
   if r['limb']!=limb or r['rtol']!=tol:continue
   vals=[r[e+'_warm_median_ms']for e in ['SinglePoly','MultiPoly','NoPoly','before_value','value']]
   s='|'+r['name']+'|'+'|'.join(f'{v:.3f}'for v in vals)+f"|{r['value_paired_change_percent']:+.1f}%|{r['before_gradient_warm_median_ms']:.3f}|{r['gradient_warm_median_ms']:.3f}|{r['gradient_paired_change_percent']:+.1f}%|{r['gradient_extra_percent']:.1f}%|{r['point']}/{r['hex']}/{r['integral']}|";L.append(s)
L+=['','## 初回・形状変更','','以下はrelative1e-3の値のみ、ms。各欄は旧→新。coldはコンストラクタ＋初回1回なので揺らぎに弱い。形状変更はsを交互に微小変更する4回の中央値。全条件・微分・VBMはsummary.csvに記録。','','|ケース|LD|cold|warm|形状変更|','|---|---:|---:|---:|---:|']
for r in rows:
 if r['rtol']!=1e-3:continue
 L.append('|'+r['name']+'|'+str(r['limb'])+'|'+'|'.join(f'{r["before_value_"+f]:.3f}→{r["value_"+f]:.3f}'for f in ['cold_total_ms','warm_median_ms','changed_median_ms'])+'|')
L+=['','## 精度と範囲','',f'- C++30テスト、API回帰27テスト通過。点光源の独立診断1792点で全status=0、VBM3方式との最大相対差5.3e-10。', '- 追加の旧point-source/safetyテストは18件通過・9件失敗。保存した変更前packageでも同じ9件が失敗し、新規失敗は0。古いinverse-ray既定値または削除済みwrapper属性を期待するテストであり、この作業では書き換えていない。詳細はlegacy-test-comparison.json。', '- 別の7配置・2半径・2LD・2要求精度・両native経路の1008点出力を、VBM SP/MP relative1e-7・absolute0と比較。全status=0、要求誤差超過0。matrix.jsonに保存。',f'- 上表48条件・両モードで6240 epoch出力。保存した初回・warm・形状更新を含む全呼出しの非ゼロstatus={bad}、無効微分フラグ={invalid}、値/微分の経路不一致={mismatch}。',f'- 前回独立に計算した厳しいVBM SP/MP参照を、同一cases.jsonを確認して再利用。{summary["reference_source_points"]}点、両精度・両経路で{summary["reference_scalar_comparisons"]}値を照合し、要求相対誤差超過={summary["reference_above_request"]}、最大相対差={summary["max_reference_error"]:.3g}。全6240点を厳しい参照で照合した意味ではない。','- GradientRayの全8方向は一般のpromoted方程式とのC++比較を通過。微分のbefore/after差はsummary.jsonに保存。gradient_validは精度保証ではない。新たな微分専用の精度契約は導入していない。','- 静的・水平軌道の限定検証。全レンズ配置、視差、軌道運動、全微分方向の精度を認定したものではない。','', '再現: build-safe-reciprocal.log、ctest-safe-reciprocal.log、regression-safe-reciprocal.log、point_check.py、validate_matrix.py、audit.py、summarize.py。全raw時間・値・解析微分・経路はrows/*.json。']
(P/'REPORT.md').write_text('\n'.join(L)+'\n')
print({k:v for k,v in summary.items()if k not in ['rows','checks','gradient_comparisons','protocol']})
print('gradient largest differences',sorted(gradient_checks,key=lambda x:x['max_scaled_difference'],reverse=True)[:8])
