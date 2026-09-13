# Phase 53: VBMとの差を埋める費用削減量とmeshの監査

基準e730496。Phase 50–52の局所改善はwhole利益を示せなかったため、より大きい変更を選ぶための費用感度を同一行データで監査した。新しいsolver変更・新しいtimingはない。

## 計算方法と制限

Phase 49のmatched VBM/V2 14432行とPhase 48のwhole rawをcase/profile/db/epoch/targetで一対一join。キーの重複なし、全行value convergedを確認。入力ファイルSHA256をsummaryに保存。

各行で `other = warm_ms - warm_topology_ms - warm_physical_ms` を作り、全行finite/nonnegativeを確認した。stageは各repeatの中央値なので、差を厳密なexclusive stopwatch時間とは解釈しない。

試算時間を `other + a*topology + b*physics` とし、a,bを0/0.25/0.5/0.75/1で動かした。a=bの場合、paired VBM/timeの中央値が1または1.15になる最大の保持率を二分探索した。**これは変更の実測でも達成可能な下限でもない。現在のmeshと残余費用を固定する費用感度の試算**。

## 結果

|条件|nodes中央値|panels中央値|二分割0回の行|同等になるtopology/physics共通保持率|15%速くなる保持率|
|---|---:|---:|---:|---:|---:|
|1e-3 uniform|67|5|3608/3608|17.47%|13.61%|
|1e-3 LD|74|5|3608/3608|100%（既に上回る）|94.63%|
|1e-4 uniform|91|5|3608/3608|30.93%|25.13%|
|1e-4 LD|98|5|3608/3608|100%（既に上回る）|100%（既に上回る）|

今回のcorpusにはpanel二分割が一件もない。adaptiveの仕事量増加は既存panel内のnested level増加であり、深い二分木の管理を削る案は対象を外している。D14の初期cell/panel構成と1nodeあたりのgeometry費用をまとめて考える必要がある。

1e-3 uniformのother中央値は0.0231285 ms。a=bという限定した試算ではtopology+physicsを約1/5.7へ減らさないと同等にならない。これは必ず両方を同率で減らす必要があるという主張ではない。stageの一方に偏った変更の結果はfrontier.tsvに保存した。

## 初期7nodeだけにする仮定も確認

全panelを初期levelの7nodeに留めると仮定し、physics費用がnode数に比例して `min(1,7*panels/nodes)` へ減る仮想試算も追加した。**そのmeshが精度条件を満たす証拠はなく、提案するsolver変更ではない**。nodeごとに費用が異なるため時間の線形比例も仮定。

1e-3 uniformではphysics保持率中央値56.76%。この仮定だけならpaired speedupは0.326。さらにtopologyをゼロと仮定しても0.920だった。したがって、単純な「全panelを7nodeで打ち切る」だけを本線にする根拠はない。精度gateを緩めて見かけの速度を作らない。

## 次の判断

endpointキャッシュや不要Newtonのような0.x us単位の変更を積み重ねるだけでは目標との隔たりが大きい。候補を選ぶ際は、node当たりのroot/geometry構築をまとめて省けるか、あるいはD14で既に証明したcell情報を重複して扱っていないかを優先する。具体的な安全な実装案の成立は未検証で、今回採用したsolver変更はない。

1e-3 uniformのVBM勝利は未達。production router・現行最速候補・精度契約は変更しない。

## 再現

```
python benchmarks/holonomic/audit_vbm_cost_frontier.py \
 --matched evidence/holonomic/adaptive_extreme_phase49/matched.tsv \
 --whole evidence/holonomic/adaptive_extreme_phase48/whole_candidate.tsv \
 --output evidence/holonomic/adaptive_extreme_phase53
```

summary.jsonは出典・mesh分布・保持率、frontier.tsvは100通りの費用感度、mesh.tsvは全14432行のnode/panel/splitと残余費用。数値solverを変更していないためunit/referenceの再実行は行っていない。
