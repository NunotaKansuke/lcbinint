# Phase 54: root-pair midpointの有理式評価

基準69b2dc8。transport_pairs_validの中点inside判定で、mからatanを経てsin/cosを求め直していた。研究flag HOLO_MV_RATIONAL_MIDPOINTでcos=(1-m²)/(1+m²)、sin=2m/(1+m²)を直接作り、既存と同じlens equationでphiを評価する試作。

TMax、positive v、finite、root ordering/gap、両endpointのquartic residual、warm thin-arc certificate、後段arc生成とendpoint polishは変更しない。mの大きさは既存TMax guardで制限済み。数学的には同じ点だがbinary64丸めは変わる。新しいinterval sign certificateではなく、全入力で旧判定と一致する証明はない。production default/routerは変更しない。

## 初期A/B

CPU0、同じcompiler flags、baseline→candidate→candidate→baseline。V2Profileを含むwarm-timing7216 rows、先頭epochを除く5412 rowsの中央値。rawのcache1/cache2は候補を指す名前で、今回cacheを実装したわけではない。

|run|uniform whole ms|uniform rootpair ms|LD whole ms|LD rootpair ms|
|---|---:|---:|---:|---:|
|baseline1|0.244965|0.052897|0.302124|0.061231|
|candidate1|0.241968|0.051176|0.299373|0.059510|
|candidate2|0.241406|0.050955|0.299017|0.059618|
|baseline2|0.242410|0.052802|0.301637|0.061384|

全7216行mu/convergence/nodes一致。rootpair成功、cold移行、branch reject、Newton更新、quartic cold/warm呼出しも行ごとに一致。rootpairで約1.7usの減少が見えたため通常whole runnerの比較へ進めた。

## 数値検証

- adaptive unit 1484 checks 0 failures。
- reference 550行中466有効、observed violation 0。84行はreference未認証であり成功扱いしない。
- analytic5Jac 220行をPhase48と比較し全フィールド一致。
- parameter routing、gate緩和、新しいfallbackは追加していない。

## 再現

保存先 evidence/holonomic/adaptive_extreme_phase54。repository rootからbuild_trial.sh、run_trial.sh、summarize.pyで初期A/Bを再生成。validate_trial.shでunit/reference/5Jacとwhole候補binaryを作る。build_whole_baseline.shで同HEAD・flagなしのwhole baselineを作り、run_whole.shで両方式の3repeat比較と集計を行う。既存rawを上書きするので再実行前に保護すること。

入力は既存input_snapshot.tsv、座標変換/trajectory順序/precisionは既存runnerを維持。cold・warmともD14/topologyを含み、radial-onlyはtopology事前構築。通常whole runnerにはV2ProfileScopeなし。warm統計にはtrajectory先頭のcold epochも含む。

## 通常whole A/Bと判断

同HEAD・flag一個のみ相違、各3repeat、CPU0、baseline全行の後にcandidate全行。初期profileのABBAと異なり通常wholeは交互実行ではなく、小さい差には実行順の影響が残り得る。

|条件|warm baseline ms|warm candidate ms|paired baseline/candidate 中央値|
|---|---:|---:|---:|
|1e-3 uniform|0.257504|0.254916|1.00782|
|1e-3 LD|0.307570|0.305511|1.00655|
|1e-4 uniform|0.301067|0.298306|1.00885|
|1e-4 LD|0.358328|0.355254|1.00713|

両profileを合わせた1e-3: cold p50 0.376858→0.374658 ms、warm 0.287018→0.285048、radial 0.110562→0.108785。warm p90 0.570059→0.567426、p99 1.009464→1.006664。1e-4 warm p50 0.330066→0.327746、p99 1.271568→1.269148。詳細全quantile/maxはwhole_summary.jsonへ保存。

全14432行、cold/warm/radial全laneでconvergence 14432、status/node差0、mu差0。保存VBM referenceに対するtarget超過は1e-3が0、1e-4が既存の3行のままで、新規増加なし。1e-4 radial maxだけ1.034695→1.037956 msの増加がある。中央値改善を全行非回帰と表現しない。

研究flagとして保持し、最新候補の追加A/Bに利用可能とする。production defaultには昇格しない。初期profileと通常wholeの双方に小幅な利益があり、source差分も局所的なため、前2案のように完全revertはしない。ただし1e-3 uniformでVBMに勝つ目標は未達で、この約1%を大幅改善とは扱わない。
