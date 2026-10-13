# v32: 最終flux精度に角度積分の仕事量を結びつける

2026-10-13、`experiment/seeded-boundary-events`、HEAD f3547723b79828d9c501182746028e99f7bad116。変更前は未コミットの採用v31。全体の要求rtol=1e-3/1e-4、積分受理の総誤差予算は変更していない。最終計測は `summary.json`、原データは `forward_*` / `reverse_*` / `normal_*`。

## 採用した設計

従来のLD角度積分は、毎radius/arcで16点と8点を計算し、相対差が固定1e-8を超えるとchart再試行や64/32点の角度積分に移る。要求flux精度1e-3でもこの固定精度を要求していた。

value-only、0<u<=1の研究frontierでは、要求相対精度の5%を角度積分の観測誤差に割り当てる。3点と、その中に含まれる1点の規則の差をまず調べる。差の2倍が予算内なら3点で止める。足りなければ7→15→31点へ進め、以前の点は新規配列位置へ移して再利用する。各段階の比較はGauss-Chebyshev第二種の同じ入れ子規則。

角度積分の観測誤差は、実際のradial重みとLD係数で積分し、従来のinner ledgerに加える。幾何、event、radial、roundoffと足した最終誤差が元の要求精度に入る必要がある。local gateで通っただけでは全体を受理しない。線形LDで0<=u<=1ならuniform項とLD項は非負なので、LD項の相対予算を割り当てることでその観測誤差は局所総fluxの同じ割合以下になる。

31点でも不足する場合や物理的radicand/denominatorの判定に落ちる場合は、その円弧だけ既存のreciprocal/chart/angular rescueを使う。他のradial点やseed、topology、既存積分は再計算しない。31点までの段階では新しい点だけ評価する。chartを変更した場合のノードは座標が異なるため再使用しない。

新しい制御は専用関数に分離。既存 `k_rule_converged` / `v_times_K` の固定8/16ルールはそのまま保存。uniform、微分、外側の標準ライブラリ経路は新しい角度ルールを有効にしない。新しいAdaptiveConfigフィールドの既定値は0。frontierのLD/value-onlyだけ0.05を設定する。`HOLO_FRONTIER_DISABLE_ANGULAR_VALUE_BUDGET` で無効化できる。

## 最終速度

低増光A<1.1、uniform248入力、linear LD251入力。各入力5反復中央値の集団中央値。seed構築を含みcaustic構築を除外。単一thread。順序を反転してもLDの改善を確認。

|条件|順序|積分 before→after [µs]|全体 before→after [µs]|
|---|---|---:|---:|
|LD 1e-3|forward|52.666→49.275|82.876→79.668|
|LD 1e-3|reverse|52.450→49.823|82.108→78.597|
|LD 1e-4|forward|60.561→57.120|86.564→84.106|
|LD 1e-4|reverse|60.276→57.358|87.334→83.576|
|uniform 1e-3|forward|34.663→35.162|62.672→64.329|
|uniform 1e-3|reverse|34.657→35.432|62.834→63.666|

入力ごとの比率の中央値ではLD 1e-3積分4.8–5.7%、全体2.5–3.0%削減。LD 1e-4積分4.3–4.6%、全体2.4–2.6%削減。集団中央値の比とは集計が異なる。uniformは改善なし、全体の入力別比率で約0.9–1.3%遅い。uniformの値、点数、角度処理のアルゴリズムは同じだが、binary/code/layout変更の影響は完全には消えていない。この小さな費用も隠さない。

従来87組のrtol=1e-4では、各組の中央値の平均：

|順序|積分 before→after [ms]|seed込み全体 before→after [ms]|
|---|---:|---:|
|forward|0.154826→0.144960|0.271385→0.261139|
|reverse|0.153116→0.143379|0.269234→0.259428|

全局面でVBMに勝ったとはしていない。低増光LDの全体は今回約79µsで、保存済みの同じ入力・tolのVBM中央値約50µsよりまだ遅い。過剰精度を狙った設計変更の一部には効果があるが、radial全体の精度制御を安全に緩められたという結果ではない。

## 検証と限界

VBM RelTol=1e-6参照の7216入力×2許容誤差=14432条件で変更前と比較。新たな許容誤差超過0、新たな未収束0。rtol=1e-3では全収束7213条件が目標内。rtol=1e-4では既存の超過3条件が残る。未収束6条件も残る。既存問題が直ったとはしていない。uniformの計算値は変更なし。LDの計算値は変更するのでbit一致を要求せず、独立VBM参照との精度で評価する。

- budget angular: 1202 checks、0 failures。非正のradicand、不正分母、不正予算を拒否し、3/7/15/31点の各段階で評価点が重複しないことを確認。372 accepted arcs を独立long-double Simpson積分と比較し、要求するlocal budget内。
- adaptive radial: 1614 checks、0 failures。既存微分、budget interruption、primal snapshotを含む。
- seeded frontier: 321 checks、0 failures。既存のfold/map/arc/count/anchor/direct-transport auditフラグを使用。

積分規則の差は経験的な誤差推定で、厳密な剰余の上界ではない。これは以前の8/16差の判定も同様。support coverageや面積積分の普遍的な数学的保証を新たに主張しない。`coverage_certified=false` は維持。

## 不採用の実験も保存

1. decay条件が弱くても保守的detailをerror ledgerに載せ、budget内なら受理する案。代表1368条件でnode変化は1、計算値変化0。効果がなく不採用。`rejected_resolution`、`pilot_after.tsv`。
2. 点ごとの補間detailを除き、3段階のradial積分差だけで止める案。1368条件で新しい精度超過10件。相殺を見逃すため撤回。`rejected_embedded`、`integral_pilot.tsv`。今回の採用版はradial detailとそのdecay判定を変更していない。
3. 固定8/16ルールを保ったままlocal gateだけ予算化する案。精度は通ったが低増光の速度改善が小さかった。`angular_pilot.tsv`。
4. 最初から7点開始した中間案も保存。`nested_pilot.tsv`、`nested7_transport.hpp`。
5. 制御を大きなmapped-radius関数へinline展開した中間版ではuniformにも約1µsの遅延が出た。`pre_outline` / `outlined_wrapper` に中間計測を保存。最終版の数値・タイミングに混ぜない。

## 再現

この実験checkoutで以下を順に実行。buildとtestを完了してからtimingを行う。scratch/buildはproject内。単一thread、毎回seed構築を含め、caustic構築を除外する。

```bash
bash evidence/holonomic/value_budget_v32/build.sh
bash evidence/holonomic/value_budget_v32/build_normal_before.sh
bash benchmarks/holonomic/build_seeded_paper_qrho.sh
bash benchmarks/holonomic/build_seeded_radial_master.sh
bash evidence/holonomic/value_budget_v32/build_tests.sh
bash evidence/holonomic/value_budget_v32/run_accepted.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_value_budget_after_v32 \
 evidence/holonomic/seeded_paper_qrho_20261013/input_snapshot.tsv \
 evidence/holonomic/value_budget_v32/full_after.tsv 1
python3 evidence/holonomic/value_budget_v32/summarize.py
```

`before_include` は変更前header、`source` は採用header。ライブラリ、provider、入力、実行ファイル、結果はSHA256SUMS、環境と計測契約はprovenance.json。前からのdirty researchは保持。外側masterへ反映、commit/push、epoch間cache、複数コア化はしていない。
