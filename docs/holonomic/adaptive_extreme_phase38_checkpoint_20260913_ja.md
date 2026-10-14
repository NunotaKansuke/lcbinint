# Phase38: 全Taylor係数を省く剰余上界のA/B

基準8904224。Phase37の認証固定費を減らすため、次数14のTaylor係数を全部作る代わりに、m未満の局所係数だけinterval Hornerで生成し、残りをTaylor剰余で抑える研究試作。production pathは変更しない。

## 数式

P(z)=sum p_k z^k、中心c、半径rに対し、複素線分上のTaylor積分剰余から

```
|P(c+w) - sum_{j<m} P^(j)(c)/j! w^j|
 <= r^m sum_{k=m}^{14} binom(k,m) |p_k| (|c|+r)^(k-m), |w|<=r.
```

局所linear項の下界×rが、constant＋局所2..m-1次項＋この剰余上界を上回る場合だけ1根と判定。係数は元のbinary64物理パラメータから外向きqf intervalで再構築。各絶対値・積・和・除算の上界/下界、14円板の非交差、degree/finite/environment gateを維持する。前段native screenはreject-only。

## 実測

Phase37と同じ5412遷移、前epoch incumbent oracle seed、現在の式で独立D14Real Newton 2回。CPU0、単回stage計測。

|剰余次数m|全14disk認証数|screen通過時の認証平均 ms|
|---|---:|---:|
|2|0|0.112|
|4|1504|0.362|
|6|2348|0.553|
|8|2442|0.659|
|Phase37 全Taylor・点積専用化|2444|0.695|

全方式のscreen通過数は2444。費用平均は成功だけでなく認証に入って失敗した行も含む。m=2は計算量を減らしても認証0件なので不採用。展開係数の絶対値上界が局所相殺を失うことが原因と解釈する。m=6は96件を失う代わりに費用約20%削減、m=8は2件だけを失うが削減約5%。この交換をrawへ残した。

認証だけで0.55 msかかるため現行D14の代替にはまだ高価。whole-epochへ統合せず、VBMに勝ったとは判断しない。screen rejectを含む全行中央値を成功時性能と取り違えない。

## 検証

専用回帰test failures=0。人工重複集合は全mで拒否、known case49の誤ったqf-warm集合も全mで拒否。case0/49の正しいoracle集合について、各受理円板を独立の全Taylor interval展開で同じ半径にて再検査し、Rouché不等式を満たすことを確認した。test出力の`remainder`行はoracle集合であり、誤ったqf-warm集合の受理ではない。

これは全5412行の独立区間監査ではない。event radius精度・physical/soft classification・cell topology・value/Jacを保証する統合検証も未実施。自分の候補だけを連続再利用したtrajectoryではなく毎回前epoch oracleからの試作。coldには適用しない。

## 再現

Phase37と同じprepareコマンドで/tmp/p36_input.tsvを生成する。

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native -DHOLO_D14_SCALAR_CONSTANTS benchmarks/holonomic/bench_d14_warm_disk.cpp -o /tmp/p38_probe -lquadmath
for order in 2 4 6 8; do
 taskset -c 0 /tmp/p38_probe /tmp/p36_input.tsv remainder${order} > /tmp/remainder${order}.tsv
 python benchmarks/holonomic/summarize_d14_warm_disk.py --input /tmp/remainder${order}.tsv --output /tmp/summary${order}.json
done
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -DHOLO_D14_EVENT_CONTRACT_RESEARCH tests/holonomic_cpp/test_d14_tight_disk_probe.cpp -o /tmp/p38_test -lquadmath
/tmp/p38_test
```

m=2のraw名はremainder.tsv、summary.json。他は次数suffix付き。原データはevidence/holonomic/adaptive_extreme_phase38/。補正時間は累積、IOはstage timer外、prepareを含む。旧Phase37の測定との比は独立単回runであり、同時交互反復したwhole-epoch比較ではない。

## 判断

保証を保って演算を省く道はあるが、absolute expanded-coefficient上界では局所構造をかなり捨てる。低次にするだけでは認証率を失う。現在のデータからこの方式を通常warm epochへ追加する理由はない。次に認証を追うならC/G/Zの構造を保った局所上界など、大きく費用を変える案が必要。現行fast pathはそのまま残す。
