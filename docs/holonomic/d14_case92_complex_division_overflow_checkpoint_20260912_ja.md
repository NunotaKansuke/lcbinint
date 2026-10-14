# D14 case 92 の binary64 複素除算overflow修正

基準は `dev/holonomic` の `f643fd8029334cc56531d6f59bce96d22efab327`。変更はisolated holonomicのD14 solverと試験・evidenceに限り、production routerは変更していない。

## 結果

case 92 / `d_bin=0` / linear profile のD14+topology計測では、旧binary64除算のままqf coldへ落ちる経路から、補正後はD14Realで完了する経路になった。10 repeats、各lane 40行（epoch 0, 7, 15, 23）で比較した。

| lane | `whole_classify_ms` p50 | p90 | max | qf cold |
|---|---:|---:|---:|---:|
| cold・新 | 1.216 | 1.628 | 1.673 | 0 / 40 |
| cold・旧double除算 | 28.371 | 28.629 | 30.652 | 40 / 40 |
| warm・新 | 1.060 | 1.262 | 1.515 | 0 / 40 |
| warm・旧double除算 | 1.007 | 28.294 | 28.360 | 10 / 40 |

ここで `whole_classify_ms` はD14 solveとtopology構築の時間であり、radial integrationを含むwhole epochではない。warm laneのepoch 0は各repeatでroot seedなし、epoch 7以降はtrajectory内warm rootを利用する。epoch 0を除く30 warm行のp50は新1.049 ms、旧1.005 msで、新しい除算のfast-path検査分だけ中央値に小さな費用が見える。一方、epoch 0のqf cold spikeは消えた。

case 92 の80 root setsでstatus、cells、events、physical-real eventsはすべて一致し、root setのgreedy bijectionによる最大絶対差は `2.70e-47` だった。新経路ではD14Real有限・収束40/40、qf cold 0回。

同じsource・同じflagsでbinary64除算だけを旧式へ戻したmatched A/Bを、入力全7,216行についてcold/warm各1回実行した。各lane 7,216/7,216でstatus 0、topology status、cells、events、physical-real events、soft events、qf root countが全件一致した。全14,432 root setsで最大root差は `9.22e-14`、`max(1,max |v|)` で割った最大差は `1.74e-14`。

| lane | 構成 | `whole_classify_ms` p50 / p90 / p99 / max |
|---|---|---:|
| cold | 新 | 0.845 / 1.098 / 3.687 / 56.484 |
| cold | 旧double除算 | 0.805 / 9.963 / 19.357 / 56.516 |
| warm | 新 | 0.735 / 0.983 / 1.426 / 59.577 |
| warm | 旧double除算 | 0.709 / 0.963 / 14.005 / 54.620 |

全コーパスの中央値は新経路が約3–5%高い。一方、qf coldはcold laneで944→58回、warm laneで256→44回に減り、p90/p99は大きく改善した。maximumは改善していない。残る最大尾はcase 149等でqf coldが400 sweepまで回る行で、今回のcase 92修正とは別に残った課題である。

## 原因

観測した最初の失敗はdouble presearchのsweep 1、root 0。値は次のとおり。

| quantity | value |
|---|---:|
| `|P|` | `7.3864e159` |
| `|P'|` | `1.5093e149` |
| 最近傍root間隔 | `3.1938e11` |
| `|S|` | `5.2390e-12` |
| `|P' - P S|` | `1.2026e149` |
| 旧式での `|w|` | `Inf/NaN` |
| scaled reciprocalでの `w` | `5.1535e10 + 3.3419e10 i` |

最近傍root間隔は十分大きく、root collisionが原因ではない。旧複素除算は分母の二乗ノルムをbinary64で計算し、分子cross-productを掛け合わせてから割る。分母ノルムは有限だが、分子積の指数が1025まで達し、binary64の上限を超えてoverflowした。正しい商自体は有限だった。

generic Aberthはin-placeでrootを更新するため、最初のroot 0がNaNになると、同じsweepでroot 1以降が `1/(z_j-z_i)` にそのrootを読み、interaction sum全体がNaNになる。したがって「14 rootすべてが独立に壊れた」のではなく、最初の除算overflowをin-place更新が全rootへ伝播させた。

## 実装とfail-closed動作

`cdiv_fast<double>` は、従来の高速式を通常経路に保ちつつ、複素分子のcross-productまたは和が非有限になった場合だけ、スケール済み `crecip` を使って商を作る。DD/D14Real/qfの除算式は変えていない。旧double除算のA/B hookはcompile-timeだけで有効になり、通常buildにbranchを追加しない。

active presearchは各root updateをcommitする前に、polynomial値、derivative、interaction sum、correction、候補rootが有限であることを確認する。opt-in `HOLO_D14_LAST_FINITE_HANDOFF=1` は最後の完全なfinite sweepを保持してD14Realへ渡す。handoff自体はfast fixの代替ではなく、non-finite候補を部分更新のまま使わないための回復実験である。

全コーパス検証中、別の安全性問題も見つかった。D14Realの有限だが未収束のroot setが、残差だけ小さいためacceptedになり、case 6, 49, 152, 203の計36行でpositive-real topologyがqf cold基準と変わった。たとえばcase 6はphysical-real eventが4対6、cellsが12対13になった。D14Realが未収束なら、そのfinite setをseedとしてqf warmを最大24 sweep試し、qf convergenceと既存residual gateを満たせば採用する。qf warmが収束しないときだけ既存qf coldへ進む。case 6はqf warm 10 sweepで6 physical-real rootへ戻り、最終full A/Bでは全14,432行のtopology差が0になった。

qf residual、root count、Newton-sum completenessの閾値は緩めていない。非収束root setをresidualだけで採用しない。V2/V3 router、fixed-n_r経路、PF6/GM研究経路には触れていない。

## 採用しなかった試み

- last-finite checkpointからD14Realへ直接渡すだけではcase 92を救えなかった。二進64だけ旧除算に戻したhandoff再試験12行すべてで、sweep 1/root 0 failureからhandoffした後もD14Realはfiniteでなく、qf cold 198 sweepへ進み約28.6–28.8 msだった。原因は初期seedのbasin品質にもあり、checkpointを残すだけでは足りない。
- qf warm上限を64へ上げる9-case subsetはqf cold回数とp90を下げたが、最大約66.7 msが残り、warm側のmaximumも改善しなかったため採用しない。
- qf warmを400 sweepにするcase 149実験ではwarm側が最大106 msまで悪化した。
- D14Real上限を100 sweepに増やしたhard subsetでもqf cold fallbackが残り、最大73 msとなった。単純な反復上限増加は採用しない。

## 検証と再現

Release全build成功。isolated CTestは18/18 pass。case 92および全コーパスのraw timing/root rows、診断trace、A/Bコマンド、machine-readable summaryは[`evidence/holonomic/d14_case92_presearch_overflow_20260912/`](../../evidence/holonomic/d14_case92_presearch_overflow_20260912/)にある。checkpoint-only試験は3 repeats×4 epochs=12 rowsで、全行が同じsweep 1/root 0 failureからhandoffし、約28.55–28.79 ms、qf cold 198 sweepだった。再実行は同directoryの`run_commands.txt`を参照。

証拠ファイルのSHA-256一覧は同directoryの`sha256_manifest.txt`。`case92_safe_active_presearch_trace.log`は修正後binaryのfiniteなpresearch traceであり、旧除算のfailure証拠ではない。旧除算のfailureは`case92_first_bad_active_presearch.log`、`case92_first_bad_update_detail.log`、`case92_first_bad_generic_trace.log`とcheckpoint-only rawに記録した。
