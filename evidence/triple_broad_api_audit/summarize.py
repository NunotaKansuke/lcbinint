import json,csv,collections,statistics,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
names={'far':'遠方・非接触','miss':'非接触・hex主体','near_mixed':'近傍・複数経路','grazing':'円盤がかすめる','crossing':'横断','large':'大きい光源','equal':'等質量','close':'近接配置','wide':'広い配置','planetary':'惑星質量比','resonant':'共鳴付近','secondary_dominant':'第2レンズが最大質量'}
cases=json.loads((P/'cases.json').read_text());refs=json.loads((P/'references.json').read_text());profiles=json.loads((P/'profile-summary.json').read_text());scaling=json.loads((P/'scaling.json').read_text());status=json.loads((P/'status.json').read_text());assert status['status']=='COMPLETE';assert refs['status']=='COMPLETE'
rows=[];invalid=nonzero=outputs=mismatch=0;all_call_bad=all_call_invalid=0
for case in cases:
 for limb in [0.,.5]:
  for tol in [1e-3,1e-4]:
   d=json.loads((P/'rows'/f"{case['name']}-u{limb}-r{tol}.json").read_text());e=d['engines'];value=e['value']['diagnostic']['output'];grad=e['gradient']['diagnostic']['output'];counts=collections.Counter(value['routes'])
   for mode in ['value','gradient']:
    z=e[mode]['diagnostic']['output'];outputs+=len(z['mu']);nonzero+=sum(v!=0 for v in z['status']);invalid+=sum(not v for v in (z['gradient_valid']or[]))
    for sample in [e[mode]['first'],*e[mode]['samples'],e[mode]['diagnostic'],*e[mode]['changed_samples']]:
     all_call_bad+=sum(v!=0 for v in sample['output']['status']);all_call_invalid+=sum(not v for v in (sample['output']['gradient_valid']or[]))
   mismatch+=sum(a!=b for a,b in zip(value['routes'],grad['routes']))
   c=dict(name=case['name'],label=names[case['name']],limb=limb,rtol=tol,point=counts[0],hex=counts[1],integral=counts[7],gradient_extra_percent=100*(e['gradient']['warm_median_ms']/e['value']['warm_median_ms']-1))
   for mode in ['value','gradient','point_base','SinglePoly','MultiPoly','NoPoly']:
    for field in ['warm_median_ms','changed_median_ms','cold_total_ms']:c[mode+'_'+field]=e[mode][field]
   rows.append(c)
ref_above=sum(sum(c['above_requested'].values())for c in refs['checks']);summary=dict(status='COMPLETE',trajectories=len(cases),conditions=len(rows),native_epoch_configurations=outputs,nonzero_status=nonzero,invalid_gradient_flags=invalid,route_mismatched_epochs=mismatch,all_call_nonzero_status=all_call_bad,all_call_invalid_gradient_flags=all_call_invalid,reference_source_points=sum(len(r['indices'])for r in refs['rows']),reference_native_checks=len(refs['checks']),reference_native_epoch_values=sum(len(c['relative_errors']['Singlepoly'])for c in refs['checks']),reference_above_request=ref_above,max_reference_method_disagreement=max(r['max_reference_disagreement']for r in refs['rows']),rows=rows)
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with open(P/'summary.csv','w')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
L=['# Triple通常API・広範囲の速度／処理内訳監査','', '対象は現在のnative static tripleの通常 `LightCurve.magnification()`。本体・設定・要求誤差は変更せず、測定コードと証拠だけを追加した。相対誤差のみ1e-3／1e-4、一様／線形LD(c=0.5)、12軌道×65点。接触に限らず遠方、非接触、接触直前、横断、光源サイズ、質量比、近接・広い配置を含む。全ケースが水平の静的軌道であり、視差・軌道運動や全配置の網羅検証ではない。','', '## 結果と見つかった改善対象','', '1. **外側の経路選択だけが主因ではない。** warmのコースティック木判定は概ね数%。一方、像の支持領域・境界イベントの証明は、かすめるケースの値のみで約70%、惑星質量比で約62%、近接配置で約49%のサンプルを占める。これは外側のpoint/hex距離判定とは別の仕事。','2. **非接触・hex主体では求根と追跡が主因。** 約97%が点像計算。排他的なleafサンプルでは `__hypot_finite`/`hypotf32x` 合計約38%、複素除算 `__divdc3` 約16%。通常範囲用の複素絶対値・除算の軽量化が候補。値と微分の双方に効く。計測は重い演算を特定したもので、置換の正確性・速度改善はまだ未検証。','3. **大きい光源のLDでは積分本体が主因。** 値のみ約79%、微分あり約84%。角度トポロジーの継続、shape-gradientのradius評価、複素イベント、LD項の運搬を調べるべきで、Pythonの固定費削減だけでは解決しない。','4. **微分費用はケース依存。** 遠方は小さいが、近接配置1e-3は一様で約70%、LDで約84%増。かすめるLDも約77%増。全体で30%以内になったとは言えない。近接LD微分のleafではquad演算も約18%を占めるが、これはquadが不要であるという証明ではない。','5. **小さなレンズ形状更新には固定費が残る。** far更新では木の再証明が約42%、grazing更新では約28%。warmの数%とは分けて扱う。初回・更新・同一形状反復の全時間を下に記録した。','', '## 測定条件','', status['protocol'],'', '各条件内のエンジン順を回転・反転して比較した。異なる条件・補足scaling・perf間ではCPU周波数や呼出間隔が揃わないため、絶対時間をそのまま足し引きして処理内訳を作っていない。coldは各エンジン1回で、中央値のwarm／更新より揺らぎに弱い。`point_base`はmode=0の**点光源だけにする診断用ablation**であり、有限光源計算の同等な競合手法でも、中心求根だけを正確に分離したタイマーでもない。','', '## 通常APIのwarm時間','', '単位ms、65点の光度曲線全体。VBMは値のみ。lcbinint微分欄は値＋10パラメータの解析微分。経路数はPS/hex/積分で、値と微分の全3120 epoch条件で一致した。']
for limb in [0.,.5]:
 for tol in [1e-3,1e-4]:
  L+=['',f'### LD={limb}, rtol={tol}','','| ケース | SinglePoly | MultiPoly | NoPoly | 値のみ | 値＋微分 | 微分増 | PS/hex/積分 |','|---|---:|---:|---:|---:|---:|---:|---|']
  for r in rows:
   if r['limb']!=limb or r['rtol']!=tol:continue
   vals=[r[m+'_warm_median_ms']for m in ['SinglePoly','MultiPoly','NoPoly','value','gradient']]
   L.append('|'+r['label']+'|'+'|'.join(f'{v:.3f}'for v in vals)+f"|{r['gradient_extra_percent']:.1f}%|{r['point']}/{r['hex']}/{r['integral']}|")
L+=['','## cold・warm・形状更新の比較','','以下は1e-3の値のみ、ms。cold=コンストラクタ＋初回。更新はsを毎回1.0001／1.0002倍に交互に変更。差分は異なる入力・キャッシュ状態を含むので純粋な証明タイマーではない。全条件の微分あり・1e-4・VBMのcold/更新時間はsummary.csvおよび各raw rowに保存。','','| ケース | U cold | U warm | U 更新 | LD cold | LD warm | LD 更新 |','|---|---:|---:|---:|---:|---:|---:|']
for case in cases:
 u=next(r for r in rows if r['name']==case['name']and r['limb']==0 and r['rtol']==1e-3);ld=next(r for r in rows if r['name']==case['name']and r['limb']==.5 and r['rtol']==1e-3)
 L.append('|'+names[case['name']]+'|'+'|'.join(f'{r["value_"+f]:.3f}'for r in [u,ld]for f in ['cold_total_ms','warm_median_ms','changed_median_ms'])+'|')
L+=['','## サンプリングによる処理内訳','','各ケース・モードをwarm後、3 thread-CPU秒だけperfのuser CPUイベントを有効化。199 Hz、約600サンプル／条件。26条件を逐次測定し、コード変更・参照計算や速度計測と重ねていない。以下はコールスタックによる排他的分類の概算で、直接のstageタイマーではない。インライン化により分類粒度に限界があり、小さい差を精密な差と解釈しない。','','配布用nativeはstrip済みだったため、既存object/archiveからstrip前のシンボル用コピーを別ファイルへ再リンクした。productionの `.text` / `.rodata` / `.eh_frame` のバイト列と仮想アドレスが**完全一致**することを確認し、その関数シンボルで元のperf記録を解決した。本体の差し替え・再測定はしていない。51,017個の不明nativeフレームをすべて解決した。symbol-identity.json / symbolization.jsonに証拠を保存。','','| ケース・モード | LD | 求根/hex | 木の判定 | 支持領域/イベント証明 | 積分 | その他 |','|---|---:|---:|---:|---:|---:|---:|']
for z in profiles:
 st=z['stage_percent'];other=sum(v for k,v in st.items()if k not in ['point_hex_roots','caustic_geometry','image_support_events','integration']);vals=[st.get(k,0)for k in ['point_hex_roots','caustic_geometry','image_support_events','integration']]+[other]
 L.append('|'+z['key']+'|'+str(z['condition']['limb'])+'|'+'|'.join(f'{v:.0f}%'for v in vals)+'|')
L+=['','### 証明処理で再利用を検討する具体的な箇所','','ソース確認では、`PreparedRegularSupport::contact_plan`は前回のイベント候補を渡せる。一方、通常のcaustic-free支持領域が成立した経路は、継続した像の候補を使っていても最後に新しい `regular_support_detail::event_plan(...)` を構築する。grazingの高い証明費用に対応する候補は、**この通常経路にも分割構造とイベント候補を運び、変更された光源で再証明すること**。古い証明をそのまま使う提案ではなく、速くなるかは次の実装・検証課題。','','点像ソルバはwarm時もdegree-10の多項式根10個を継続し、物理像の候補へpolishする。hexのガード側には認証された像数が既にあるため、認証された像数と物理像追跡を組み合わせて不要な候補の仕事を減らす余地も検討対象。ただし、像の完全性・重複排除・残差を保つ必要があり、未実装・未検証の案。単純な演算置換と数学的な仕事量削減は区別する。','','## 遠方でのAPIサイズ依存','','別プロセスごとに3 warmups＋20回を連続して計時。上のIPCで交互に回す比較とはCPUの動作条件が違うため、上表との絶対時間差を速度改善とは呼ばない。N=1の値は同じ遠方軌道の先頭epoch。','','| 点数 | 値のみms | 値＋微分ms | point-only ablation ms |','|---:|---:|---:|---:|']
for n in [1,17,65,257]:
 L.append('|'+str(n)+'|'+'|'.join(f'{next(z for z in scaling if z["condition"]["n"]==n and z["engine"]==mode)["median_ms"]:.4f}'for mode in ['value','gradient','point_base'])+'|')
L+=['','## 検証と範囲','',f'- 12軌道×4条件×65点×値／微分の **{outputs} native epoch出力**。収束失敗 {nonzero}、無効微分フラグ {invalid}、値／微分の経路不一致 {mismatch}。cold/warm/更新の保存された全呼出しでも非ゼロstatus {all_call_bad}、無効微分フラグ {all_call_invalid}。',f'- 24軌道・LD組の計{summary["reference_source_points"]}点を、VBM SinglePoly／MultiPolyのrelative1e-7・absolute0で別途照合。各組で粗い比較の差が大きい3点と最初／最後の積分点を選択対象に追加した。native両モード×両要求精度で{summary["reference_native_epoch_values"]}値を比較し、要求誤差を超えた比較は {ref_above}。2つの厳しいVBM参照間の最大相対差は {summary["max_reference_method_disagreement"]:.3g}。全6240点を厳しい参照で検証したという意味ではない。','- gradient_validは有限／使用可能性のフラグで、今回新たに全微分方向の高精度を認定したものではない。微分専用の誤差要求・refinementは導入していない。','- 遠方・missの非接触ラベルはnativeの全点PS/hexクリアランス認証と、計時外のcausticサンプルによる位置確認に基づく。grazingはサンプル上の最大caustic高さより1.001*rho外側を通る。サンプル曲線そのものを本体の経路判定に使っていない。','- productionのsource／native hashは測定前後で固定。本体・既存テスト・設定は変更していない。性能改善の採用ではなく、改善対象を特定する監査である。','', 'Raw: cases.json, rows/*.json, references.json, profiles/*.data, profiles/*.symbolized.stacks, profile-summary.json, scaling.json, summary.csv, provenance.json. 実行は `audit.py --cases`, `audit.py`, `references.py`, `profiles.py`, `scaling.py` の順。シンボル解決はrelink.log、symbolize.py、profile_summary.py。']
cadence=json.loads((P/'cadence.json').read_text())
C=['','## 積分ケースのepoch間隔依存','','補足のnative-only診断。LD=.5、relative1e-3、2 warmups＋6回の連続計時。上のエンジン交互比較とは別の測定。各点数では異なる点をサンプルするため、経路比率も変わる。','','| ケース | N | 隣接移動/rho | 値のみms/点 | 微分ありms/点 |','|---|---:|---:|---:|---:|']
for name in ['crossing','grazing','close']:
 for n in [17,65,129,257]:
  v=next(r for r in cadence['rows']if r['condition']['name']==name and r['condition']['n']==n and r['engine']=='value')
  g=next(r for r in cadence['rows']if r['condition']['name']==name and r['condition']['n']==n and r['engine']=='gradient')
  C.append('|'+names[name]+'|'+str(n)+f'|{v["step_over_rho"]:.3f}|{v["milliseconds_per_epoch"]:.3f}|{g["milliseconds_per_epoch"]:.3f}|')
C += ['', '0.75rhoの接触イベント候補再利用ゲートをまたいでも、大幅な費用の段差は確認できなかった。小さいNの固定費償却やサンプルの違いは見えるが、このゲートを緩めるだけで大きく速くなる根拠にはしない。通常の非接触側のイベント証明が毎回構築されることと、接触側の候補再利用条件は区別する。全補足呼出しのstatusは0だった。']
L += C
(P/'REPORT.md').write_text('\n'.join(L)+'\n');print({k:v for k,v in summary.items()if k!='rows'})
