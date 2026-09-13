# Phase35: 14次全根探索を避ける6+4 factor discovery（不採用）

ユーザーの「14次方程式を真正面から解かずevent情報を得る」提案を受けた小実験。基準d17c8ad。production solverには組み込まない。

## 過去の方式と区別

正実根PRSは包含計算とfallback費用が重く、stationary atlasは領域全体のno-contact証明が重かった。これらを再実装せず、D14の既存構造

D/4096 = F6*G4² + 8*C3*(2*C3²-9*v*G4)*Z3 - 432*v²*Z3²,
F6=C3²-4*v*G4

を出発点にした。補正項を除いた多項式は6個の単根と4個の重根。6次・4次を解き、G根gの周囲の2候補を

v = g ± sqrt(-D(g)/(4096*F(g)*G'(g)²))

で生成。その後0/2/4回の独立qf structured Newtonを比較した。これは初期候補の生成であり、Rouché/completeness certificateはまだない。独立Newtonの局所収束をsolver成功とは扱わない。

## 結果（7216 geometry）

|追加Newton|physical event個数不一致|physical event不足|最近傍対応の重複geometry|
|---|---:|---:|---:|
|0|7064|7060|6514|
|2|6750|6744|6650|
|4|6458|6398|6696|

全候補は有限だったが、有限であることと欲しいeventを保持することは別。全14根のscaled差<1e-8を満たすgeometryも4回補正後268/7216に留まる。

physical個数は現行の正実候補filterとdouble_root_is_realで判定し、保存済みqf全根経路のroot_work.physical_realと比較。Rmaxは現行と同じ式。adaptiveのphysical event保持に合わせ、物理根を一律dedupしない。個数一致したケースもradius/ordering/topologyまで合格した意味ではない。soft-cutと全体積分はこの段階では未検証。

## 低次数solver精度の切り分け

標準試作は6次/4次をbalanced double Aberth（最大80 sweep）で解き、各factor根にqf Newton3回。追加診断ではfactor全根探索をqf Aberth（tol1e-26、最大200 sweep）へ交換。
qf版でも上表の不一致件数、対応重複件数は同じ。qf版の合計sweep最大87なので200の上限による打切りではない。単なるdouble factor solverの精度不足では説明できず、Zを含む項を小さな摂動として扱う初期値が、多くのgeometryで適切でないと解釈する。全代替方式が不可能という意味ではない。

## 費用

標準版中央値: qf構造係数生成＋factor solve .0633ms、split .0432ms。2回の全候補qf Newton約.186ms、追加2回約.175ms。分類費用はclassify_msとして別記（3段階合計）。候補生成だけで約.1msを払い、coverageも不足するので、現行solverへfallbackを追加して採用する価値はない。
qf factor版はfactor段階だけ中央値.7413ms。これは精度切り分け用で、高速候補ではない。

計測はCPU0、単回診断。IOはstage timer外、構造係数生成はfactor_ms内。求根candidateの精度が不足しているためwhole benchmarkを追加実行せず棄却した。失敗をstatus低下や重量fallbackで埋めない。

## 判断

欲しい情報がeventであるという方向は維持。ただし今回の静的6+4分解は、解く多項式の次数を下げても失ったroot correspondenceを戻す費用が高い。次の候補はeventの存在/分離を安く証明できる構造とセットで検討する必要がある。現行D14 oracle/backstop、production default/routerは不変。

## 再現

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native benchmarks/holonomic/bench_d14_factor_seed.cpp -o /tmp/p35_probe -lquadmath
taskset -c 0 /tmp/p35_probe evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv /tmp/p35_roots.tsv /tmp/p35_timing.tsv
# qf factor診断は末尾に qf を追加
python benchmarks/holonomic/check_d14_factor_seed.py --input evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv --oracle evidence/holonomic/adaptive_extreme_phase34/roots.tsv.gz --candidate /tmp/p35_roots.tsv --timing /tmp/p35_timing.tsv --output /tmp/p35_summary
```

standard/qf variantのraw roots、timing、parity、physical countをevidence/holonomic/adaptive_extreme_phase35へ保存。oracleの根座標はbinary64 export、physicalラベルは既存qf経路での分類。任意精度の数学的真値や新しい証明と取り違えない。
