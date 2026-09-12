# Vmax D14-free portfolio checkpoint (2026-09-13)

## 結論

旧mainstreamのcertified inverse-ray gridを、trajectory warm-upで決めた固定planとして再利用すれば、`RelTol=1e-3` のwhole epoch中央値は約 `0.11 ms` まで下がる。しかし、このplan単独は独立VBM `1e-6` referenceに対する要求誤差を全行では満たさず、tailも現行adaptive V2より大きい。このためproduction候補には採用しない。

同じimage seedを再利用したhalf-grid checkを追加し、不一致時だけ現行のD14 + `(m,v)` + `vK` K-rule adaptiveへ渡すhybridも実装した。coverageは全行へ戻ったが、約78--84%がboundary backstopへ移り、gridを先に実行した費用が丸ごと追加された。中央値・tailとも現行V2より遅く、このgateも不採用とした。

今回の結果は、D14を外す方向そのものを否定しない。必要なのは、grid積分を完了してからD14へfallbackする構成ではなく、旧flood traversalが既に得ているcomponent boundaryを安価なevent/band witnessとして取り出し、その同じ探索中に `(m,v)` / K-ruleへ渡す単一passである。現在の公開seed集合にはcomponent identityがなく、radial projectionが重なる別componentを区別できないため、seed radiusだけからbandを再構築する既存prototypeではこの条件を満たせない。

## 実装したresearch harness

[`bench_vmax_preplanned.cpp`](../../benchmarks/holonomic/bench_vmax_preplanned.cpp) は次を同一timer内で測る。

1. binary point-image solve
2. certified augmented image seed / cache reuse
3. warm-upで選択済みのCartesianまたはPolar inverse-ray kernel
4. nested check版ではhalf-grid evaluation
5. check不成立時は現行equation-derived adaptive boundary/K-rule backstop

Lens parameter construction、I/O、warm-upでのresolution探索はtimer外。magnification値そのものは再利用しない。production routerは変更していない。

## 14,432-row結果

入力はcanonical q--rho corpus `v2_vbm_pure_kernel_20260911/input_snapshot.tsv`。各数値は1 repeatのwhole epochで、point-image solveを含む。

### warm-up planを直接実行

| profile / target | p50 ms | p90 ms | p99 ms | max ms | reference violation |
|---|---:|---:|---:|---:|---:|
| uniform / 1e-3 | 0.114 | 2.071 | 19.202 | 215.752 | 61 / 3608 |
| linear / 1e-3 | 0.111 | 2.285 | 21.659 | 117.015 | 47 / 3608 |
| uniform / 1e-4 | 0.700 | 14.484 | 180.771 | 946.601 | 97 / 3608 |
| linear / 1e-4 | 0.660 | 15.742 | 207.296 | 1094.929 | 75 / 3608 |

plan内訳はCartesian 9,501、Polar 4,930、既存planが得られずadaptiveへ送ったものが1行。低tolで高いNbinを要求するtailが大きく、特に`1e-4`ではD14を外した利益よりgrid解像度費用が大きい。

### quarter-budget nested check + boundary backstop

| profile / target | p50 ms | p90 ms | p99 ms | backstop | converged |
|---|---:|---:|---:|---:|---:|
| uniform / 1e-3 | 0.736 | 4.008 | 36.095 | 3041 / 3608 | 3608 / 3608 |
| linear / 1e-3 | 0.800 | 4.177 | 38.181 | 2940 / 3608 | 3608 / 3608 |
| uniform / 1e-4 | 1.548 | 23.889 | 243.904 | 2866 / 3608 | 3608 / 3608 |
| linear / 1e-4 | 1.544 | 24.770 | 280.390 | 2799 / 3608 | 3608 / 3608 |

独立referenceの少数の既存不一致は現行adaptive backstopにも残るため、nested checkで新しく解消できるものではなかった。gateを緩めれば高速になるが、grid errorの非単調性があるためaccuracy contractを満たす根拠にならない。

## A/Bで棄却した変更

- adaptive radialの初期Fejer levelを3から2へ下げる案: p50は実質不変。3,463行でnode数が変わり、高増光caseでは値が最大`2.76e-2`動いたためrevertした。
- certified seed radiusだけから全radial bandを作る案: radial projectionが重なる別image componentのidentityを失い、全seedをmarchすると約1,790 probes / 13.9 msとなる。D14の代替にならない。
- grid完了後のnested check + adaptive fallback: 正しいが二重払いとなり、上表の通り遅い。

## 次に必要な実装

本命は旧Cartesian flood traversal内部から、各connected image componentについて次を直接返すAPIである。

- component stable ID
- certified seed
- traversal中に観測した最小・最大image radius
- source-limb crossing cellと局所境界seed
- support completeness certificate

このcomponent witnessを受け取れば、重なるradial projectionを落とさず、境界近傍だけexact quarticでpolishし、区間内部は現行 `(m,v)` continuationと`vK` K-ruleで積分できる。grid面積積分とradial band探索を別々に完走する必要がなくなる。これは旧mainstreamの強みとV2の強みを同じ探索で共有する設計であり、次phaseで実装すべき最小の構造変更である。

## Evidence

- raw direct plan: `evidence/holonomic/vmax_preplanned_phase1_20260913/raw_rep1.tsv`
- raw nested hybrid: `evidence/holonomic/vmax_preplanned_phase1_20260913/raw_nested_rep1.tsv`
- machine summaries: `summary.json`, `summary_nested.json`
- exact plan: `plan.tsv`

