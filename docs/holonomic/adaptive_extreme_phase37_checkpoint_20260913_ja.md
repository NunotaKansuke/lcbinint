# Phase37: 全根反復を省くwarm認証の費用削減

基準4764a4f。Phase36の前epoch rootからの直接Rouché円板認証を高速化した研究実験。production solver、router、固定n_r APIは変更していない。

## 結果と採否

5412 warm遷移を同一入力で測定。stage2は前epoch oracle rootに現在の構造式で独立D14Real Newtonを2回適用する。新epochの全根Aberth探索は行わない。

|方式|stage2全14disk認証|screen中央値 ms|認証実行時のinterval平均 ms|
|---|---:|---:|---:|
|Phase36 qf point screen + interval|2630|0.345|0.879|
|native screen + generic interval|2444|0.012925|0.873|
|native screen + point-specialized interval|2444|0.013207|0.695|

native screenは受理authorityではなく、通過候補を必ず外向きbinary128 intervalで認証する。native版が失った186件は認証機会の損失であり、誤った受理ではない。stage0/1の認証数は1046/1880、Phase36比の損失110/160。point-specialized版とgeneric版の認証結果差は全stageで0。

全遷移に対するstage2費用平均はnative .433 ms、point-specialized .353 ms。ただしscreen拒否が多く、中央値約.039 msは成功時のコストを表さない。認証だけで約.7 msを払うため、現行D14の代替としては未採用。whole-epoch/VBM勝利の証拠ではない。

## 実装

- reject-only screenをlong doubleへ変更し、係数シフトを三角Hornerで計算。現在のhostではnative binary80。低精度screenが通っても最終interval検証は省略しない。
- interval Taylor shiftの点定数×区間積を、定数の符号に応じた2端点積へ専用化。実中心では不要な虚部積を省略。各積の外向き丸めは維持する。既存interval shift自体も既に三角形の計算なので、これを新たに発見した最適化とはしない。
- strict Rouché不等式、非交差、degree、finite、environment gateは変更しない。

初回の専用積はzero区間の早期returnを欠き、正確なzeroが極小幅へ広がり、interval平均が約1.725 msまで悪化した。`b.zero()`を既存演算同様に保持して解消。失敗時rawはpoint_disks.tsv、修正版はpoint_zero_disks.tsv。現行pointモードは修正版を再現する。旧失敗版のbinaryは保存していない。非有限operandのgeneric演算への委譲でも、qf定数をdoubleへ暗黙変換しないよう2引数interval constructorを使う。

## 検証と限界

専用回帰test failures=0。known case0は14disk認証、wrong-topology反例case49は8diskで拒否。人工重複集合も拒否。整数定数と区間の積の包含、既知rootでの全Taylor係数の専用区間がgeneric区間に含まれることを確認した。

これはroot存在・分離の認証であり、event位置精度、physical/soft分類、cell topology、adaptive value/Jacの統合検証ではない。前epoch seedは毎回incumbent oracleのbinary64 export。候補だけを自己継続したtrajectoryでもcold solveでもない。未認証行をfallbackで補ってcoverageを報告していない。

## 再現

```
python benchmarks/holonomic/prepare_d14_warm_disk.py --input evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv --roots evidence/holonomic/adaptive_extreme_phase34/roots.tsv.gz --output /tmp/p37_input.tsv
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -DHOLO_D14_SCALAR_CONSTANTS benchmarks/holonomic/bench_d14_warm_disk.cpp -o /tmp/p37_probe -lquadmath
taskset -c 0 /tmp/p37_probe /tmp/p37_input.tsv native > /tmp/native.tsv
taskset -c 0 /tmp/p37_probe /tmp/p37_input.tsv point > /tmp/point.tsv
python benchmarks/holonomic/summarize_d14_warm_disk.py --input /tmp/point.tsv --output /tmp/summary.json
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_D14_EVENT_CONTRACT_RESEARCH tests/holonomic_cpp/test_d14_tight_disk_probe.cpp -o /tmp/p37_test -lquadmath
/tmp/p37_test
```

単回CPU0 diagnostic。準備費含むstage集計、補正時間は累積、IOはtimer外。全epoch wallではない。rawとsummaryはevidence/holonomic/adaptive_extreme_phase37/。

## 次の判断材料

14次を最後まで解かずに情報を得る方式は一部warm遷移で成立するが、探索を存在証明へ交換しただけでは速くならない。残存主費用はqf interval係数/Taylor shift。ここを大きく削れない限り本番統合へ進まない。root identityの証明とevent精度の保証を混同しない。
