# Phase 72 続き: rand035 gradient reference 収束確認

日付: 2026-09-14
基準: `dev/holonomic` commit `f83e30f398fa1a7918f28de0afa30165f47a5cdc`
範囲: Phase72で見つかったrand035のX/Y gradient降格について、p-onlyとh-split@4の実誤差を独立value積分の有限差分で比較。

## 結論

**h-split@4はdefault候補にしない。** rand035のu=0/.5、X/Yの全4成分で、h-split@4の実誤差はp-onlyより大きかった。p-onlyの誤差は `1.86e-7`〜`6.54e-7`、h-split@4は`0.1306`〜`0.4599`で、約70万倍大きい。従って `FiniteUncertified` はこの4成分ではledgerだけの保守性で起きたものではなく、実gradient自体が悪化している。

h-split@4ではraw corpusの`unique_nodes`が300から89へ70.3%減り、muは両armでbitwise一致したが、gradient精度を大きく損なった。依頼条件に照らし、**gradient error estimator/ledgerは変更しない**。scheduler閾値・gradient tolerance・production pathも変更しない。Hermiteは再開していない。

## 独立参照の強化と収束確認

Phase72の旧rand035 FD診断は、近接foldで独立reference側の積分が失敗する/有限FDを作れない点が残っていた。今回は参照側だけを修正・高解像度化した。

- row 99/100は同じLensParams、非barycentricで、u=0/.5。X=xs、Y=ysなのでuser-coordinateとprimary-frame座標の変換は恒等。
- 各FDの±h、±2hごとにwarm D14 rootを渡さず `classify_cells()` をcold再生成した。32 topologyすべてがStatus::OKで、全て11 cells/12 events。
- angular boundaryはbinary128 quartic係数とqf Sturm実根分離で求め、直接lens `phi=0` にsafeguarded Newtonをかけた。F0と`sqrt(phi)`値積分から5-point central FDを形成し、production analytic gradientとproduction K-ruleを参照には使っていない。
- FD stepは`rho×{1e-3,5e-4,2.5e-4}`。radial GL order/subdivisionとangular解像度を段階的に上げた。Xについては前回残っていたradial解像度差を解消するため160×32、1024 angular-nodeまで追加した。Yは128×16、1024 angular-nodeでstep/order安定性を確認した。
- 全144 direct value integralsが有限値で完了。FD値のstep差と直前解像度との差から得た最大観測幅は`4.72e-11`〜`8.74e-9`。これは収束診断であり、形式的な誤差包含ではない。topology/cell cutsとfold mapはcold再生成した既存geometryを使い、angular root/value積分をanalytic-gradient pathから独立させた数値参照である。

| u | 成分 | FD reference | 最大観測安定幅 | radial解像度 | step差 (5e-4対2.5e-4) |
|---:|:---:|---:|---:|:---:|---:|
| 0 | X | 17.04260957790129 | 8.74e-9 | 160×32, angular 1024 | 2.16e-11 |
| 0 | Y | 36.010206446936074 | 4.72e-11 | 128×16, angular 1024 | 4.72e-11 |
| 0.5 | X | 16.918289951078755 | 5.24e-9 | 160×32, angular 1024 | 2.95e-11 |
| 0.5 | Y | 35.747525136525304 | 5.90e-11 | 128×16, angular 1024 | 5.90e-11 |

Xの前解像度との差は、u=0で中step `8.74e-9`/小step`2.19e-9`、u=.5で`5.24e-9`/`1.31e-9`まで縮んだ。参照幅はh-split@4の誤差より7桁以上小さい。FD referenceの不安定性では結論は反転しない。

## 実gradient比較

whole-epoch raw `corpus.tsv`のRelTol=1e-3、ValueFirst、gradient 4-round armを利用した。各行はcold repeat 0の代表値。4 paired cold/warm/repeatでgradient値とqualityは同一だった。全110例のwhole-epoch A/B自体は再実行せず、Phase72の既存matched rawを使用した。このcorpusはimplementation commit `64bb1e3`で測定され、`f83e30f`がcheckpoint/rawを記録した。両commit間にproduction solver sourceの差はない。

| u | 成分 | p-only gradient | p-only実誤差 / quality | h-split@4 gradient | h-split実誤差 / quality |
|---:|:---:|---:|---:|---:|---:|
| 0 | X | 17.042609887299207 | 3.09e-7 / ToleranceMet | 17.26027913754309 | 0.217670 / FiniteUncertified |
| 0 | Y | 36.010207100782154 | 6.54e-7 / ToleranceMet | 36.47013162791261 | 0.459925 / FiniteUncertified |
| 0.5 | X | 16.91829013675337 | 1.86e-7 / ToleranceMet | 17.048891637616318 | 0.130602 / FiniteUncertified |
| 0.5 | Y | 35.74752552884675 | 3.92e-7 / ToleranceMet | 36.02348014099197 | 0.275955 / FiniteUncertified |

全4件でh-split@4の実誤差が増加。h-split/p-only実誤差比は約`7.0e5`。h-splitの相対誤差はu=0で約1.28%、u=.5で約0.77%。p-onlyは約`1.1e-8`〜`1.8e-8`相対。u=0のmuは両armで`6.3075026570247736`、u=.5は`6.292839505068458`でbitwise一致。

同じuではX/Yのh-split gradient/reference比も一致し、u=0で約`1.01277208`、u=.5で約`1.00771956`。これは全成分共通のsplit-path biasを示唆するが、この追試では原因箇所を特定していない。h-splitのu=.5相対実誤差は既定gradient rtol 1e-2内にある一方、p-onlyに比べ約70万倍悪い。品質を誤って昇格する根拠にはせず、指定どおりledger変更も行わない。

## gradient error ledger

単位は各gradient成分に対する絶対誤差推定値。roundoffはrawの値をそのまま表示。

| u | 成分 | arm | radial | inner | geometry | event | roundoff | ledger total |
|---:|:---:|:---|---:|---:|---:|---:|---:|---:|
| 0 | X | p-only | 6.54456e-3 | 0 | 3.33292e-9 | 4.55823e-6 | 2.19626e-14 | 6.54912e-3 |
| 0 | X | h-split@4 | 1.97637 | 0 | 5.40932e-11 | 4.55678e-6 | 2.18904e-14 | 1.97637 |
| 0 | Y | p-only | 1.38283e-2 | 0 | 7.04146e-9 | 9.63131e-6 | 4.64058e-14 | 1.38380e-2 |
| 0 | Y | h-split@4 | 4.17597 | 0 | 1.13436e-10 | 9.62826e-6 | 4.62534e-14 | 4.17598 |
| 0.5 | X | p-only | 3.93102e-3 | 1.80678e-8 | 1.99975e-9 | 2.75711e-6 | 2.31596e-14 | 3.93380e-3 |
| 0.5 | X | h-split@4 | 1.30281 | 1.80265e-8 | 3.24559e-11 | 2.85589e-6 | 2.31651e-14 | 1.30281 |
| 0.5 | Y | p-only | 8.30606e-3 | 5.15242e-8 | 4.22487e-9 | 5.82564e-6 | 4.89351e-14 | 8.31194e-3 |
| 0.5 | Y | h-split@4 | 2.75278 | 5.19206e-8 | 6.80617e-11 | 6.03437e-6 | 4.89468e-14 | 2.75278 |

全armでradialがledgerを支配する。p-only ledgerも実誤差より約`2.1e4`倍大きい保守値だが、h-split@4では実gradient誤差そのものが大きく、ledgerを過剰と判断してqualityを昇格させる根拠はない。今回の分岐条件「h-split実誤差が同等以上に良いのにFiniteUncertified」の前提は成立しないため、estimator調査はこのPhaseでは行わない。

## 検証・成果物

- cold topology: 32/32 Status::OK、11 cells / 12 events。
- direct reference value integrals: 144/144 finite、fail code 0。
- paired value comparison: p-only対h-split@4のmu bitwise mismatch 0。
- targeted CTest `holonomic_quartic_sturm`, `holonomic_quartic_local_bracket`, `holonomic_adaptive_radial`: 3/3 pass。
- Phase72からproduction sourceの変更なし。Phase72 adaptive unitは元checkpointの1535 checks/0 failuresを参照し、本追試ではtargeted CTestのみ再実行。

raw、machine-readable summary、再現コマンド、input hash、compiler/host情報は[`evidence/holonomic/gradient_radial_split_phase72_rand035_reference/`](../../evidence/holonomic/gradient_radial_split_phase72_rand035_reference/)に置いた。`run_all.sh`で参照2段階・summary・targeted CTestを再現できる。
