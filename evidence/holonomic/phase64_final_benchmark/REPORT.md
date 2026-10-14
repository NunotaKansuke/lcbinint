# Phase64 最良候補：最終ベンチマーク

実装基準: 4e23537。production routerやsolverには今回変更なし。研究用最良フラグはrun.shに記録。

現在のsourceから再compileし、CPU0固定、各epoch3回の中央値。各laneの事前warm-up trajectoryは統計から除外。full-warmはtrajectoryごとにstateを新規作成するので、各trajectory先頭のcold epochは測定に含む。GradientPolicy::None、with_jacobian=false、Tol=1e-16、RelTol=1e-3/1e-4、adaptive radial（fixed64ではない）。

entry pointはcold: epoch_adaptive、warm: prepared_topology + flux_adaptive_integrate、radial-only: flux_adaptive_integrate。runnerはevidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp。LensParams構築・I/Oは計時外。cold/warmはD14/topologyを含む。radial-onlyはそれを除く診断用。VBMは従来のpure selected kernel時間を再利用し、追加計算なし。VBMとの測定日は異なる。

入力7216行はcanonical inputとSHA256完全一致。両tolで14432行、重複keyなし。各条件3608行。q–rhoはprotocolの12×12 bins、全144セルが8行以上。図の配置・色・カラーバーは既存phase10から保持。増光率との投影図で8行未満のセルは灰色。

速度比は行ごとのVBM/V2比の中央値（>1でV2が速い）。時間中央値の比とは異なる。誤差は入力に対応するVBM RelTol=1e-6参照との差で、VBM速度artifactのreference vectorとは混同しない。

|RelTol|LD|lane|p50 ms|p90 ms|p99 ms|max ms|VBM/V2 median|収束|参照誤差p95|要求超過|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0.001|uniform|cold|0.334873|0.601636|2.923969|51.684671|0.1836|3608/3608|1.127e-05|0|
|0.001|uniform|warm|0.235170|0.509059|0.900069|51.512401|0.2898|3608/3608|1.127e-05|0|
|0.001|uniform|radial|0.078321|0.139532|0.232146|0.397632|0.8322|3608/3608|1.127e-05|0|
|0.0001|uniform|cold|0.364669|0.642695|2.933302|51.749924|0.2933|3608/3608|8.947e-07|1|
|0.0001|uniform|warm|0.275667|0.552036|1.011417|51.585120|0.4233|3608/3608|8.947e-07|1|
|0.0001|uniform|radial|0.101725|0.182281|0.339328|0.664842|1.0498|3608/3608|8.947e-07|1|
|0.001|linear|cold|0.379823|0.657854|2.952468|51.833171|0.8144|3608/3608|1.321e-05|0|
|0.001|linear|warm|0.287185|0.564739|0.980559|51.659530|1.1633|3608/3608|1.321e-05|0|
|0.001|linear|radial|0.106983|0.219685|0.326084|0.621539|2.9795|3608/3608|1.321e-05|0|
|0.0001|linear|cold|0.420352|0.727591|3.004295|51.801263|3.1566|3608/3608|7.936e-06|2|
|0.0001|linear|warm|0.333564|0.633998|1.192575|51.729997|4.2993|3608/3608|7.936e-06|2|
|0.0001|linear|radial|0.145106|0.293577|0.483466|0.767279|9.4821|3608/3608|7.936e-06|2|

精度差は既存VBM参照との観測差。全入力への厳密保証とはしない。1e-4の既存3行の参照超過についてはPhase64 checkpointの扱いを維持。

再現: bash evidence/holonomic/phase64_final_benchmark/run.sh → python evidence/holonomic/phase64_final_benchmark/summarize.py → python evidence/holonomic/phase64_final_benchmark/make_figures.py → python evidence/holonomic/phase64_final_benchmark/write_report.py。

raw: results.tsv、対応付け後: joined.tsv、統計: summary.json、provenance/hash: provenance.json、図: figures/ のPNG/PDF（LD有無×両tol×cold/warmの8枚）。

依頼どおり、本ベンチと図の報告をもって一旦停止。追加の最適化実験は開始しない。

再計測とPhase64保存rawの比較: 全laneでmu完全一致、status/convergence変化0。図はuniform 1e-3 warmおよびLD 1e-4 warmを目視確認し、q–rho穴なし・凡例・単位・ラベルの表示を確認。
