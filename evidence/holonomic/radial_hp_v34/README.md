# v34: 減衰の遅い重み付き区間の早期分割

## 変更と理由
採用dirty v33を基準に、要求rtol=1e-3/1e-4を変更せず、研究weighted treeのvalue-only controllerに早期h-refinementを追加。

case44 uniformの主像は、平方根の重みを除いても内部の変化が強い。7→15点でfull interpolation detailが0.425→0.311にしか減らず、既存decay testは不合格。それでも従来は255点までp-refinementしてからsplitしていた。15点以上かつ既存decay test不合格の場合は、先に同じ既存のh-refinementを選ぶ。level4は最低7点を経た次の15点段階であり、case IDや時間による分岐ではない。重み付き親の子は既存のone-fold mapを使う。普通の子もweighted treeの保守的ledgerを保持する。

誤差ledger、full interpolation detail、decay test、tol、全体の受理判定を変更しない。両childが評価できるまでparentを保持。滑らかなweighted区間は従来のp ladderを維持する。LD・微分・外側master/defaultへの変更なし。

## 精度・回帰
全7216入力×2tol=14432条件。新しい精度超過0、新しい未収束0。既存超過3条件と未収束6条件はそのまま。LDの値と点数は全て不変。値が変わった451条件、点数が変わった490条件を含めて全数比較した。

weighted tests334、adaptive1614、frontier321 checks、全2269 checksで0 failures。新規テストは急峻なquotientの早期splitと独立theta-domain Simpson参照、領域保存、および片側childの計算後にnode予算が尽きた際のparent保持を検証する。テストの最初のexp20モデルは15点でdecay testを満たしており早期splitの試験にならなかったため、exp100の急峻なモデルを使用した。独立参照との一致は両方で成功。

誤差推定はinterval enclosureではない。coverage_certified=falseを保持。新規全数比較の参照は保存済みVBM RelTol=1e-6。VBM自体は再実行していない。

## 通常87組の同時期3版比較
fresh seed込み、caustic構築除外、one thread、各組5反復中央値の平均、rtol=1e-4。順序v32→v33→v34と逆順。build/tests終了後にserialで計測。

|順序|v32 [ms]|v33 [ms]|v34 [ms]|
|---|---:|---:|---:|
|forward|0.258522|0.261594|0.254808|
|reverse|0.256889|0.264152|0.253854|

v34はv33から約2.6–3.9%短縮、v32から約1.2–1.4%短縮。この行列ではv33の全体平均regressionを解消。case44単体はv32よりまだ遅いので、全入力の一律改善を意味しない。

case44 uniform/1e-4はv33の1135→655点、約1.43–1.45→0.775 ms。v32は571点・約0.72–0.73 ms。1e-3の診断は539→321点。local topology repairは0で、今回削ったのはquadrature refinementの仕事量。

通常行列の各版はresolved D14 referenceに対するcandidate failure0。D14 unresolved14条件は精度成功の根拠に含めない。

## 低増光 A<1.1
v33/v34で同じ684入力×2tol×5反復、両順序。uniform248、linear251入力を集計。以下はseed込み全体の集団中央値。ステージ中央値同士は足さない。

|条件|順序|v33→v34 [µs]|点数中央値 v33→v34|
|---|---|---:|---:|
|uniform 1e-3|forward|63.581→62.202|22→22|
|uniform 1e-3|reverse|62.172→62.361|22→22|
|uniform 1e-4|forward|70.552→69.434|38→38|
|uniform 1e-4|reverse|69.105→69.695|38→38|
|LD 1e-3|forward|78.342→77.998|53→53|
|LD 1e-3|reverse|77.827→77.572|53→53|

この群の中央値はほぼ横ばいで、今回の追加改善を主張しない。特にLDは値・点数不変なのでtiming差をアルゴリズムの改善と解釈しない。保存済み同一入力/tol VBM timingとの比較ではuniform1e-3はVBM中央値46.85 µsに対してv34約62 µs、まだ届かない。詳細はsummary.json。

## 再現・来歴
checkout /rogue1_8/nunota/lcbinint/experiments/seeded-boundary-events、branch experiment/seeded-boundary-events、base f3547723b79828d9c501182746028e99f7bad116、host rogue1。dirty v33を引き継いだ。source控え、before header/test、provenance、SHA256SUMSを保存。commit/pushなし。

```bash
bash evidence/holonomic/radial_hp_v34/build.sh
bash evidence/holonomic/radial_hp_v34/build_normal.sh
bash evidence/holonomic/radial_hp_v34/build_tests.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_radial_hp_v34 evidence/holonomic/seeded_paper_qrho_20261013/input_snapshot.tsv evidence/holonomic/radial_hp_v34/full_after.tsv 1
bash evidence/holonomic/radial_hp_v34/run_timing.sh
python3 evidence/holonomic/radial_hp_v34/summarize.py
bash evidence/holonomic/radial_hp_v34/run_normal_three_way.sh
```

比較用v32/v33 binaryは元の保存済みbaselineを再利用。currentから上書き再buildしない。v33の元SHA256SUMSがcurrent sourceを指す項目は本変更後には変化するが、v33/sourceの控えと元binaryは保持されている。

## 残る課題
低増光のseed込み約62 µs対VBM約47 µsの差は残る。今回の改善は通常行列のtail対策。LDにはuniform平方根項とlinear項を別々の重みで扱う設計が必要。既存の精度超過・未収束とcoverage未認証も解決していない。
