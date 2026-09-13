# Phase36: warm event情報と直接disk認証

基準7ce3bdb。14次の全根探索を省き、前epochの根周囲に現在の根が存在することを直接認証できるかを調べた。production router/API/受理経路は変更しない。

## Oracleから見た変化

5412 warm遷移のうちphysical event個数不変5020、最小費用対応でphysical label不変4972。単に「同じevent数」と仮定できない。
未来のoracle座標を使い、前root中心からの移動量＋binary64 export用の32eps目安を半径とした円板が全部非重複なのは3604遷移、physicalに関わる円板に重なりがないのは4242。これは将来の正解を使う幾何診断で、本番のcertificateやroutingには使えない。root移動量/旧最近傍分離の各epoch内中央値について、全体median=.0692、p90=.234、p99=1.196。

## 候補生成と認証

前epoch oracle根（binary64 export）を現在のD14構造式で0/1/2回だけ独立D14Real Newton補正。その後point prefilter、通過した場合のみ外向きbinary128 intervalのRouché certificate。新epochのAberth全根探索は行わない。

現行margin版はstage0/1で認証0、stage2で1088/5412認証。stage2の候補作成＋screen＋intervalの平均.389ms。通過時interval平均.864ms。最終integratorには組み込まない。

## 厳密な不等式を使う研究variant

既存interval版に残る16倍marginを、本来のRouché条件に整理したcopyをbenchmarks/holonomic/d14_tight_disk_probe.hppへ置いた。既存production headerは変更しない。

円周でlinear項のmodulusの下界をLr、constant＋higher termsのmodulus上界をUとすると、外向き計算で **Lr > U** を満たせばlinear項と同じ1根が円板内にある。これは近似残差のthresholdを緩めて受理するものではない。14個の非交差円板でそれぞれ1根を認証すれば、degree14の全根を数えたことになる。

変更は初期radiusを32*constant/linearから2倍へ、判定をlhs>16*rhsからlhs>rhsへ。rhsは既存同様、各加算/積をupした上界、lhsはdownした下界。radius<separation/3、finite/degree/environment gate、現行のreal-center projectionは維持。前段point screenは受理authorityではなく、最終intervalで必ず検証する。

|独立補正回数|現行margin認証|strict interval認証|
|---|---:|---:|
|0|0|1156|
|1|0|2040|
|2|1088|2630|

全根探索も独立補正もせず認証できる遷移が存在することを確認できた。一方、strict variant stage2の平均費用は.746ms、通過時interval平均.879msで、現行solverより高価。point screen自体もstage2中央値.345msとなる。coverage向上は速度勝利ではない。

## 回帰確認

専用testでknown case0のqf-warm候補は14disk認証、wrong-topologyのcase49は8diskで拒否。人工的にrootを重複させた集合も拒否。failures0。

ただしこの検証は各円板の根の存在/個数であり、event radiusの精度、physical/soft分類の切替、cell topology、whole valueの品質まで確認したものではない。独立補正のlocal convergenceを受理条件にしていない。未認証ケースを成功扱いしない。既存backstopへのfallbackでcoverageを補ったベンチでもない。

## 判断

14次のroot solveを省いて情報を得る道は、一部のwarm遷移で数学的に成立する。ただし現行のqf Taylor shift/interval係数構築では認証が高価。次はこの存在証明の演算費用を下げられるかが主題。古いevent個数を信じるだけの省略や、重い認証を毎epoch追加する方式は採用しない。

全データは前epochのincumbent oracle seedから始める。研究candidateで連続trajectoryを走らせた結果ではない。exportのbinary64精度による制限もある。cold solveを省く証拠ではない。

## 再現

```
python benchmarks/holonomic/check_d14_event_motion.py --roots evidence/holonomic/adaptive_extreme_phase34/roots.tsv.gz --output /tmp/p36_motion
python benchmarks/holonomic/prepare_d14_warm_disk.py --input evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv --roots evidence/holonomic/adaptive_extreme_phase34/roots.tsv.gz --output /tmp/p36_input.tsv
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -DHOLO_D14_SCALAR_CONSTANTS benchmarks/holonomic/bench_d14_warm_disk.cpp -o /tmp/p36_probe -lquadmath
taskset -c 0 /tmp/p36_probe /tmp/p36_input.tsv > /tmp/p36_disks.tsv
taskset -c 0 /tmp/p36_probe /tmp/p36_input.tsv tight > /tmp/p36_tight.tsv
python benchmarks/holonomic/summarize_d14_warm_disk.py --input /tmp/p36_tight.tsv --output /tmp/p36_summary.json
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_D14_EVENT_CONTRACT_RESEARCH tests/holonomic_cpp/test_d14_tight_disk_probe.cpp -o /tmp/p36_test -lquadmath
/tmp/p36_test
```

単回diagnostic、CPU0。IOはstage timer外、qf構造係数/展開係数はprepare内。stagesのcorrection_msは累積、各stageを独立採用した場合の費用。stage間に測定・IOが挟まるため純粋な全epoch wallではない。非finite/未測定の平均はJSON nullで表す。
