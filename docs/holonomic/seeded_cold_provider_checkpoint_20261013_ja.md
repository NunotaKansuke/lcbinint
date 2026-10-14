# Caustic構築を避けるcold seed providerのcheckpoint

日付: 2026-10-13。`experiment/seeded-boundary-events`、ベース `f3547723b79828d9c501182746028e99f7bad116`。
研究用APIのみ。標準dispatchの置換・昇格は行っていない。

## 変更

`seeded_point_epoch.hpp` に `epoch_adaptive_point_seeded` を追加。

- 各epochで新しいPointSourceMagnifierを作り、中心のnative quinticを**1回**解く。
- 得られた5候補を像側の局所境界探索に使う。ghostは物理像seedとは扱わない。
- Ghostの共役像の相方と中点を作り、既存物理像とともに局所critical-distance方程式を補正する。
  `|kappa|²-1=0` と `Re[(lensmap-source) exp(i arg(kappa)/2)]=0`。
  解析微分、残差減少のbacktracking、最大32反復。
  方程式残差を確認し、lens mapが光源内に写るcritical点だけ追加する。
- 全caustic / critical curve点列は構築しない。D14も新経路では求めない。
- 得られたevent間で3つの角度quartic交点数を比較する。
  異なる区間だけ局所bisectionとfold補正を追加し、event・追跡済み状態を保持する。
  全旧方式へのやり直しはしない。
- 積分は既存のroot-pair/適応積分。積分の許容誤差・tiny-arc gateは変更しない。

単純な中心候補のみでは、新holdout88件のうち4件のeventが不足した。
Ghostだけのcritical補正、反復64回、実軸反転の初期値追加も解決しなかった。
物理像にも局所補正を使い、残る動径折り返しを交点数変化区間で補正して88/88にした。
改善のなかった64回・反転guessは採用しない。失敗結果もevidenceに保存。

## 正しさ

`evidence/holonomic/seeded_cold_provider_v2/`。

| 入力 | Event/cell照合 | 既存適応積分との比較 |
|---|---:|---|
| 従来48 geometry | 48/48一致 | 192条件中D14収束178、全て候補も収束・差はtol以内 |
| 新holdout88 geometry | 88/88一致 | 352条件中D14収束288、全て候補も収束・差はtol以内 |
| 更に浅い接触88 geometry | **86/88一致** | 未受理。1014/1071でeventを2個ずつ見落とす |

新holdoutはq=1e-5～3、a=.7～2の10 geometry、critical位相.47、limb深さ.98rhoの80点と軸上等8点。
更に浅いholdoutは位相1.13、深さ.9999rho。入力生成だけでcritical quarticを使用し、runtimeには接続しない。

D14未収束の14/64条件を成功に数えない。strict adaptive auditはexit1。
固定128点比較は真値ではない。D14照合も一般的な数学的保証ではない。
`coverage_certified=false`。局所critical求根の全解性、有限3probeの完全性は保証しない。
開始queueが終了しても、全境界を発見したという意味ではない。
この理由で研究APIを一般利用の代替solverとしては受理していない。

解析微分・座標frameのunit test:72 checks、0 failures。
既存root-pair単体test:58 checks、0 failures。既存M7:10397 checks、0 failures。
監査入力が開けない/空ならexit2に修正し、0件の偽成功を防いだ。

## Matched cold全体計測

Host rogue1、Intel Xeon Gold6530、native O3、1 worker、OMP/BLAS1。
geometry毎に独立、workspaceとPointSourceMagnifierを毎回新規作成。
同一入力・u=0,.5・tol=1e-3,1e-4・同一積分実装。各条件3回の中央値。
seed構築、event/cell、積分を**実際のcold全体呼出し内**で分割計測。
D14と候補を交互の順序にし、最終計測中に別build/benchmarkは走らせない。
比較可能な条件のみ下表。収束失敗は除外数として上表に明記。

| 指標(us) | 従来48:候補 | 従来48:D14 | 新88:候補 | 新88:D14 |
|---|---:|---:|---:|---:|
| Seed中央値 | 14.716 | 不要 | 9.755 | 不要 |
| Event/cell中央値 | 360.816 | 466.994 | 338.119 | 338.534 |
| 積分中央値 | 217.942 | 206.588 | 260.428 | 200.495 |
| cold全体中央値 | 590.043 | 695.459 | 627.315 | 549.568 |
| 条件ごとの時間合計比 | .67323 | 1 | 1.04268 | 1 |
| 候補が速かった条件 | 87/178 | — | 110/288 | — |

中央値どうしの比では従来48で15.2%短縮、新88で14.1%増加。
**全ケースの高速化とは言えない**。各段階の中央値の和は全体中央値と一致しない。
過去cold seed snapshot約1219usとの比較は異なるsnapshot/timing境界なのでmatched倍速として扱わない。

積分部分も分割点・event初期値で実際のnode数が変わる。積分を外した数字だけで全体の改善としない。
本実験の成果はcaustic構築の固定費を外せたことと、従来48のcold改善。
残課題は浅い接触の2ケースと、難条件における局所追跡・積分の費用。
重い全域certificateや多数の追加quinticで覆い隠さず、このcheckpointで制約を残す。
