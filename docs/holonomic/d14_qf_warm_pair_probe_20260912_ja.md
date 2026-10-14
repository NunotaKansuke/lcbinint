# D14 qf-warm pair 1-step viability probe — 2026-09-12

## 結果

qf warm が既存の global step gate に届かない24件を対象に、pair Newtonを1 stepだけ試して続行可否を判定した。probeは従来のpair-polish成功2件を両方通し、失敗22件をすべて早期に止めた。既存のglobal verifier、residual、completeness、topology certificateは変更せず、probe成功だけでroot setを採用しない。

pair-polishの総反復は59→18、backtrackは201→23、global verifier sweepは4→2になった。qf-coldへのfallback回数は変わらず22/24のまま。従ってこれはpair-polishの無駄試行を減らす改善であり、残るqf-cold tailを解消する修正ではない。

対象whole-epochでは数値とstatusは全行一致したが、速度差は小さく、warm p50はprobe側が約0.164 ms遅い。p90以降は約0.14–0.18 ms短かったものの、radial-only controlにも同程度の揺れがあり、このsubsetからwhole-epochの速度改善とは結論できない。productionへの昇格はしない。pair-probeは既存のcompile-time research macro内に留め、さらに実運用候補へするなら別corpusでの再現性が必要。

## Probeの条件

qf warmが `1e-20` のglobal step gateに失敗し、既存のscaled residual gateを満たす場合に限って、従来どおり最大step rootとそのnearest partnerからpair-polishを試す。probeの最初のdamped Newton stepでは、line searchを最大3回backtrackまでに制限する。

以下をすべて満たした場合だけ続行する。

1. pair endpoint residual `max(|P(a)|, |P(b)|)` が有限で、受理step後に初期値の半分以下になる。
2. driver pairが維持された場合、pair外12根の最大Aberth correctionが `1e-20` 以下。
3. driver pairが切り替わった場合、全14根の最大Aberth correctionが `1e-20` 以下。

pairが維持されたときは既存のpair polishを残りのiteration budget内で続ける。pairが切り替わり、全根のcorrectionがgate内なら、pair polishを続けずglobal verifierを1 sweep実行する。いずれもglobal verifier・residual・scalar completeness・後段topology gateが最終受理権限を持つ。probe拒否時は候補を採用せず既存qf-cold経路へ進む。

半減条件は「1 stepで明確に縮む」という今回の実験上の判定、3 backtrack上限は観測された既存成功2件の最大値、外側correctionの `1e-20` は既存global step toleranceをそのまま使った。物理parameterに依存する分岐はない。Jacobian determinant fraction、step/separation、driver identityもrawに記録したが、24件では単独の安定した判別条件にならないため、採否条件へは加えていない。成功したcase9ではprobe後にdriver pairが変わったので、pair identityの固定も必須条件にしていない。

## 観測値

診断corpusは `case149/uniform/dbin1`、`case0/linear/dbin0`、`case64/linear/dbin0`、`case9/uniform/dbin2` の4ケース、4 epoch、cold/warm laneで構成する。32行のD14 rootwork中24行でpair-polishを試行した。

| 指標 | 旧pair polish | 1-step probe |
|---|---:|---:|
| 試行数 | 24 | 24 |
| probe viable / pair accepted | — / 2 | 2 / 2 |
| false positive / false negative | — | 0 / 0 |
| pair Newton iteration | 59 | 18 |
| backtrack合計 | 201 | 23 |
| global verifier sweep | 4 | 2 |
| qf-cold call（同じ32 rootwork行） | 22 | 22 |

拒否理由は、3 backtrack以内に残差減少なし7件、残差半減条件未達7件、pair外correction超過5件、driver切替後の全root correction超過3件。残る2件だけ通過した。

no-pair baselineは同じ32 rootwork行でqf-cold 24回、旧pair版とprobe版はともに22回だった。つまりprobe版は従来pair版に対してcold restartを新たに減らしておらず、前段のpair polishが救っていた2行をそのまま保っている。

probeで受理され、no-pair qf-cold baselineのroot setとの差を確認できたwarm行は次の2つ。

| case / dbin / epoch | max absolute root delta | role mismatch | completeness / count / conjugacy / Vieta failure |
|---|---:|---:|---:|
| 149 / 1 / 15 | `1.20e-16` | 0 | 0 |
| 9 / 2 / 7 | `5.72e-16` | 0 | 0 |

両行で14根数が一致し、候補の全既存certificateはpass。残り22件ではqf-cold call数もsweep数も旧pair polishから変わらない。probeはそのcold restartを避ける解法ではない。

## Whole-epoch matched A/B

比較は旧full pair-polish候補対1-step probe候補。`pair_whole_epoch_input.tsv` の2 trajectory×4 epochを、value-only / adaptive radial、RelTol `1e-3` と `1e-4` で実行した。各processは各rowを3回測り、そのmedianを取り、2 processのmedian同士を比較した。full-coldはepochごとにD14/topologyをcold生成、full-warmはtrajectory stateを使い、radial-onlyはtopologyをtimer外で共有する。CPU affinityはcore 0で固定。

時間単位はms。各lane 16行のpercentileはlinear interpolation。

| lane | arm | p50 | p90 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|
| full-cold | old pair | 2.352 | 55.710 | 56.658 | 56.719 | 56.735 |
| full-cold | probe | 2.366 | 55.538 | 56.470 | 56.555 | 56.576 |
| full-warm | old pair | 4.755 | 55.413 | 56.392 | 56.534 | 56.570 |
| full-warm | probe | 4.890 | 55.249 | 56.271 | 56.365 | 56.389 |
| radial-only | old pair | 0.289 | 0.438 | 0.513 | 0.611 | 0.635 |
| radial-only | probe | 0.295 | 0.439 | 0.516 | 0.611 | 0.635 |

全laneで16/16が`OK`かつvalue converged。topology、status、stop reason、nodes/evaluations/panels/splitsが全て一致し、scaled mu differenceとreported error differenceは0。warm p50はprobeが約0.135 ms遅く、p90以降は約0.12–0.18 ms短かった。一方radial-only controlも中央値で約0.006 ms、p95で約0.003 ms動いており、このtail-targeted subsetだけでは明確なwhole-epoch勝利とは断定しない。全trajectory母集団の速度主張でもない。

この比較はvalue-onlyであり、value+5Jacは測定していない。probeはD14候補生成の局所制御だけを変えるが、Jacobian経路のwhole-epoch parityをこの結果から主張しない。

## 検証・再現

通常buildでholonomic CTest 18/18 pass。pair-probe有効のrootwork harnessとwhole-epoch runnerも再build・実行し、上記rawからsummaryを再生成した。

主要成果物:

- raw trace / rootwork / whole-epoch rows: [`evidence/holonomic/d14_qf_pair_probe_20260912/`](../../evidence/holonomic/d14_qf_pair_probe_20260912/)
- machine summary: [`pair_probe_summary.json`](../../evidence/holonomic/d14_qf_pair_probe_20260912/pair_probe_summary.json)
- reproduction commands and binary/source hashes: [`run_commands.txt`](../../evidence/holonomic/d14_qf_pair_probe_20260912/run_commands.txt), [`binary_hashes.txt`](../../evidence/holonomic/d14_qf_pair_probe_20260912/binary_hashes.txt)

実装箇所は `d14_structure.hpp` と `radial_events.hpp`。利用には `HOLO_D14_QF_PAIR_POLISH_RESEARCH=1` に加えて `HOLO_D14_QF_PAIR_PROBE=1` が必要で、production router/defaultは変更していない。
