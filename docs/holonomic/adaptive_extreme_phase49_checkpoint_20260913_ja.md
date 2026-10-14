# Phase49: VBMとの同一行比較と残存費用の再確認

基準4af4110。Phase48までの研究候補について、保存済みVBM pure contour kernel時間へ14,432行を正確に対応させた。新しいtiming runではなく既存rawの再集計。production変更なし。

## Joinと計測境界

case_id/profile/d_bin_index/epoch_index/targetで1対1join。VBM配列のepochは0,1,2,3ではなくreference_indices=[0,7,15,23]なので、その実indexを使用する。初回のordinal joinはassertで拒否され、reference_indicesへ修正した。全14,432行が対応、重複なし。s/q/rho/x/yをrtol2e-15、atol0で照合し、VBM selected RelTolとV2 target一致、全VBM timing status completed/finite/positiveをassert。

V2 wholeはD14/topologyを含む。warmはtrajectory先頭coldを含む。VBMは既存pure kernel timingで、新たな再実行は行わない。保存済みbenchmark間の比較なので測定日時の差がある。V2 status/独立精度の検証はPhase48を参照。

## 結果

speedupは各行VBM_ms/V2_msの中央値。中央値同士の比ではない。各群3608行。

|RelTol/profile|VBM p50 ms|V2 cold p50 ms|V2 warm p50 ms|paired cold speedup|paired warm speedup|warm勝利行|
|---|---:|---:|---:|---:|---:|---:|
|1e-3 uniform|.061369|.354759|.257403|.174|.266|18/3608|
|1e-3 LD|.314456|.400419|.306822|.784|1.095|1909/3608|
|1e-4 uniform|.107503|.388700|.300161|.278|.393|98/3608|
|1e-4 LD|1.305374|.445124|.358556|3.010|3.998|3074/3608|

LDの1e-3 warmはpaired中央値では僅かに勝っている。しかしuniformは大幅に負けており、全目標達成とはしない。全profileを混ぜたp50だけではこの差が隠れる。

## topologyをゼロにしても足りるか

各rowのwarm_msからwarm_topology_msを引く思考実験をした。実装上その費用が消せることを意味せず、mesh/他費用を不変とした仮定。これは実測radial-onlyとも異なる。

1e-3 uniformのwarm_without_topology中央値は.111218 ms、paired speedup中央値.603。勝つのは354/3608行。physical_msだけを残す極端な仮定でも中央値.0855425 ms、paired speedup.788、勝利1070/3608行。stage timer overheadは含むので数学的下限ではないが、現行仕事配分のままD14だけ削れば解決、とは考えられない。

1e-3 LDではtopology除去仮定のpaired中央値2.310。LDに対してD14改善は有効だが、uniformにはradial側も必要。

## uniformのnode帯

|node数|行数|VBM p50 ms|warm p50 ms|physical p50 ms|panel中央値|
|---|---:|---:|---:|---:|---:|
|1–31|41|.082538|.385670|.059431|3|
|32–63|1513|.045440|.182478|.057196|3|
|64–127|1905|.070588|.286702|.109006|6|
|128–255|149|.182558|.564399|.223735|9|

1–31群を最もeasyと仮定しない。少nodeでもtopology tailを含む。主な母集団は32–127nodeであり、少数の高node難例だけを直して中央値の差を埋めることも難しい。

## 判断

ここまでの数us単位改善を積む価値はあるが、uniformに必要な差は数倍。今後の採否はuniform/LDを分離し、D14の削減だけでなくradial境界追跡の仕事自体を減らす案を検討する。精度契約を無視してnode数を減らすことはしない。LDだけの勝利へゴールを縮小しない。

## 再現

```
python benchmarks/holonomic/audit_vbm_remaining_cost.py --vbm /tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/results.json --v2 evidence/holonomic/adaptive_extreme_phase48/whole_candidate.tsv --output evidence/holonomic/adaptive_extreme_phase49
```

matched.tsvに各行のVBM/V2/stage時間、summary.jsonにprofile別p50/p90/paired speedup/勝利行数。uniform_nodes.jsonはwarm node帯集計。source_sha256.jsonに入力hash。入力Snapshotのxはsourceの基準座標であり、epochごとのtimeとは区別する。joinは既存benchmarkのreference_indicesを使用し、独自のepoch配置を作らない。
