# Certified algebraic topologyによるD14置換実験（2026-09-13）

## 結論

旧algebraic branchの本来のcomponent discoveryを確認し、現行V2 adaptiveへ接続した。これは中心点像だけの`fast_bands`ではなく、cached caustic branch上の距離極値をnormal±/tangent±からprobeし、image-count straddleでsource disk内の全component supportを認証する既存`component_certificate`を使用する。

しかし、現状のwhole-epochではD14版より遅く、coverageとaccuracyも下がったためproductionへ採用しない。研究経路だけを残す。

## 実装した研究経路

1. `FiniteSourceMagnifier::cached_binary_image_seeds`でcentre/caustic/certified-component image seedsを取得する。
2. 旧algebraicと同じく、各seedからR方向へstep doublingし、3回のboolean bisection後に`P=P_t=0`の2変数Newtonでsupport endpointを求める。Newtonを両側のimage有無で検証し、失敗時だけ合計16回のbisectionを使う。
3. support band端で連続cell partitionを作る。D14 interior eventは持たないため、各adaptive nodeでdegree-4 Sturmにより0/2/4 crossingを認証し、現行`(m,v)`、K-rule、angular rescueを使う。

通常のD14 path、production router、fixed-n_r APIは変更していない。dynamic cellは`n_crossings < 0`の研究用表現だけで有効になる。

## 14,432 lane A/B

入力は既存の1,804 trajectory×4 epoch、uniform/linear、RelTol `1e-3`/`1e-4`。value-only、1 timed repeat、CPU core 0固定。`alg-cold`はcaustic cacheを毎epoch再構築し、`alg-warm`は同一trajectory内でcacheを共有した。D14比較も同じprocess・入力・compiler flagsで実行した。

| RelTol | lane | converged/7216 | p50 | p90 | p95 | p99 | topology p50 | node p50 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|1e-3|D14|7216|0.663|1.123|1.549|4.552|0.436|122|
|1e-3|algebraic cold|6945|2.187|10.784|17.176|33.835|1.639|204|
|1e-3|algebraic trajectory-cache|6946|1.656|8.035|11.909|23.572|0.490|204|
|1e-4|D14|7216|0.918|1.598|2.878|9.247|0.440|287|
|1e-4|algebraic cold|6907|2.738|23.587|34.979|52.230|1.628|494|
|1e-4|algebraic trajectory-cache|6906|2.225|15.686|21.236|40.060|0.496|494|

単位はms。single-repeatなので最終性能値ではないが、差は測定揺らぎより十分大きい。trajectory-cache時のtopology内訳中央値はseed/certificate約0.096/0.103 ms、radial band約0.305/0.306 ms（1e-3/1e-4）。D14 topology中央値約0.44 msと同程度まで来るが、D14 interior cutsとendpoint metadataがないadaptive本体の仕事が増える。

## correctness

- D14: 両toleranceで7216/7216 converged/OK。
- algebraic: `support_proven=false`が各tolerance 90行。
- supportがprovenでもTopologyUnresolvedが約178–179行、BudgetExceededが1e-3で2行、1e-4で41行。
- converged行にもreference violationが1e-3で66行、1e-4で163行あり、最大相対誤差は約0.123。

最大誤差の直接原因を追加A/Bした。旧`find_radial_bands`は、後続component seedの半径が既発見band内なら、そのseedをvalidateするだけでradial marchを省く。異なるimage componentのradial projectionは重なり得るが端は一致するとは限らず、この省略でprojectionの延長を失うケースがある。全certified seedをmarchする試作ではcase 30の最大誤差が約0.123から`3.28e-4`へ低下した一方、重複seedが最大数百あり、warm中央値は約13.9 ms、probe中央値は約1,790回となった。従ってこの修正は正しさの方向には効くが速度候補ではない。

support endpointをphysical foldとして現行fold mapへ仮接続すると、小標本のnode中央値は338から142へ下がった。しかしendpointの高精度radius/t-seed certificateが不足し、160 lane中10 laneがfail closedしたため採用していない。

## 判断

旧algebraicのcomponent certificateは、現行`fast_bands`が欠いていたcaustic-born componentを扱っている点で有効である。しかしD14はevent discoveryだけでなく、adaptive radialを少ないnodeで成立させるinterior fold/soft cutsと高精度endpoint metadataを同時に供給している。

今回の結果では、caustic cache共有後でもD14を外す利益より、band completionとradial workの増加が大きい。完全置換は停止する。再検討するなら、certified seedをcomponent ID付きで返して各componentのradial projectionを一度だけmarchし、同時に`P=P_t=0` endpoint enclosureを生成する必要がある。それでもcomplex soft cutsなしのadaptive accuracyを独立に解決する必要がある。

## 再現

```bash
cmake --build build -j4 --target lcbinint_magnification
PATH=/rogue1_8/nunota/local/gsl/bin:$PATH g++ -std=c++17 -O3 -march=native \
  -funroll-loops -ffp-contract=fast -fno-math-errno -fext-numeric-literals \
  -Isrc -Iinclude benchmarks/holonomic/bench_certified_algebraic_topology.cpp \
  -Wl,--start-group build/liblcbinint_magnification.a build/liblcbinint_lightcurve.a \
  -Wl,--end-group $(/rogue1_8/nunota/local/gsl/bin/gsl-config --libs) \
  -lquadmath -fopenmp -o /tmp/bench_certified_algebraic_topology
LD_LIBRARY_PATH=/rogue1_8/nunota/local/gsl/lib taskset -c 0 \
  /tmp/bench_certified_algebraic_topology \
  evidence/holonomic/d14_precision_warm_20260912/trajectory/input_snapshot.tsv \
  evidence/holonomic/certified_algebraic_topology_20260913/results_current_rep1.tsv 1
```

Rawとmachine-readable summaryは`evidence/holonomic/certified_algebraic_topology_20260913/`に保存した。
