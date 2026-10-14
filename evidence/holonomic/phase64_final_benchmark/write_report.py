import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=['# Phase64 最良候補：最終ベンチマーク',
'',
'実装基準: 4e23537。production routerやsolverには今回変更なし。研究用最良フラグはrun.shに記録。',
'',
'現在のsourceから再compileし、CPU0固定、各epoch3回の中央値。各laneの事前warm-up trajectoryは統計から除外。full-warmはtrajectoryごとにstateを新規作成するので、各trajectory先頭のcold epochは測定に含む。GradientPolicy::None、with_jacobian=false、Tol=1e-16、RelTol=1e-3/1e-4、adaptive radial（fixed64ではない）。',
'',
'entry pointはcold: epoch_adaptive、warm: prepared_topology + flux_adaptive_integrate、radial-only: flux_adaptive_integrate。runnerはevidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp。LensParams構築・I/Oは計時外。cold/warmはD14/topologyを含む。radial-onlyはそれを除く診断用。VBMは従来のpure selected kernel時間を再利用し、追加計算なし。VBMとの測定日は異なる。',
'',
'入力7216行はcanonical inputとSHA256完全一致。両tolで14432行、重複keyなし。各条件3608行。q–rhoはprotocolの12×12 bins、全144セルが8行以上。図の配置・色・カラーバーは既存phase10から保持。増光率との投影図で8行未満のセルは灰色。',
'',
'速度比は行ごとのVBM/V2比の中央値（>1でV2が速い）。時間中央値の比とは異なる。誤差は入力に対応するVBM RelTol=1e-6参照との差で、VBM速度artifactのreference vectorとは混同しない。',
'',
'|RelTol|LD|lane|p50 ms|p90 ms|p99 ms|max ms|VBM/V2 median|収束|参照誤差p95|要求超過|',
'|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in s:
 t=r['time_ms']
 lines.append(f"|{r['target']:g}|{r['profile']}|{r['lane']}|{t['p50']:.6f}|{t['p90']:.6f}|{t['p99']:.6f}|{t['max']:.6f}|{r['paired_speedup_median']:.4f}|{r['converged']}/{r['rows']}|{r['error']['p95']:.3e}|{r['reference_excess']}|")
lines+=['','精度差は既存VBM参照との観測差。全入力への厳密保証とはしない。1e-4の既存3行の参照超過についてはPhase64 checkpointの扱いを維持。',
'',
'再現: bash evidence/holonomic/phase64_final_benchmark/run.sh → python evidence/holonomic/phase64_final_benchmark/summarize.py → python evidence/holonomic/phase64_final_benchmark/make_figures.py → python evidence/holonomic/phase64_final_benchmark/write_report.py。',
'',
'raw: results.tsv、対応付け後: joined.tsv、統計: summary.json、provenance/hash: provenance.json、図: figures/ のPNG/PDF（LD有無×両tol×cold/warmの8枚）。',
'',
'依頼どおり、本ベンチと図の報告をもって一旦停止。追加の最適化実験は開始しない。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
