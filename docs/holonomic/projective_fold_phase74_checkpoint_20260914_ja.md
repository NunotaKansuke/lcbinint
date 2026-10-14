# Phase 74: `chart_p4` 上の projective physical fold

実施日: 2026-09-14
基準: `dev/holonomic` @ `f928e3b42d25635195c7fba985cb72b811c383f3`
判断: **caustic-cross の θ=π 接触2件を shadow adaptive 経路で fold map へ接続できた。Phase 9 corpusでは topology/value/status の非回帰を確認した。production default/routerへの昇格はしていない。**

## 判定と実装

`chart_p4` は引き続き通常は表現切替eventである。eventを physical fold 扱いするのは、reciprocal quartic

```text
Q(u;R) = u^4 P(-1/u;R)
       = p4 - p3 u + p2 u^2 - p1 u^3 + p0 u^4
```

について、`u=0` の `Q=0`, `Q_u=0`、非零 `Q_uu`、非零 `Q_R` がbinary128の厳しい残差・conditioning screenを通り、さらに既存D14 positive-real eventのradiusと局所 `p4(R)=0` refineが一致する場合だけとした。接触が未解決、角度方向が高次、または半径方向の横断が数値的に判定できない場合は `chart_p4` のまま残す。

局所計算はevent半径を `u=0` の reciprocal `Q` で最大12回Newton補正するだけで、新しいglobal root solveは追加していない。`RadialEvent` に split radius (`radius`, `radius_lo`)、uncertainty、reciprocal seed `u=0`、projective-fold flagを保存し、既存 adaptive event/map契約へ渡す。eventの種類・順序・数とcell planは変更しない。候補経路は `HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH` compile defineでのみ有効である。

reciprocal係数のlocal derivative対応をunit testで確認し、高次接触・半径横断なし・接触未解決をgateがfail-closedにすることも直接検査した。実ケースで高次degeneracyに遭遇した例はPhase 9 corpusにはなく、その実例を検証したとは主張しない。

## Shadow結果: caustic-cross `a`, row 17, `u=0`

cell 5/7 の左endpointに、既存physical foldと同じradial fold mapが付いた。cell 7右端に元からあったphysical fold mapは維持された。Topology status `OK`、16 events、14 cellsはbaseline/candidateで同じ。

Phase 73の独立binary128 angular + radial Gauss–Legendre監査では、mapなしのwhole gradient系列は `N=128/256/512/1024` で `-2.148665978/-2.276430381/-2.340497476/-2.372577583` と遅く収束した。cell 5/7のθ=π端へmapを強制した独立積分は `N=128: -2.404689176456`、`N=256: -2.404689191247`。cell寄与もそれぞれ `44.586302642541/-62.509115037124`、`44.586302627505/-62.509115045339` と独立referenceへ収束し、Phase 73で観測した主cell差が消えることを再確認した。rawはPhase 73の `caustic_fold_map_diagnostic.tsv` と `caustic_direct_high.tsv` に保存している。

Phase 74の同一topology adaptive A/Bは以下。

| lane | baseline | projective fold | 結果 |
|---|---:|---:|---|
| value-only `mu` | 31.943948575450037 | 31.943882703752212 | 差 `-6.58717e-5` |
| value-only unique nodes | 195 | 91 | candidateの方が少ない |
| value + 5Jac `dmu/da` | -0.1052900480898 | -2.4046891773126 | candidateが独立FDへ一致 |
| value + 5Jac unique nodes | 345 | 139 | candidateの方が少ない |
| value + 5Jac quality for `a` | FiniteUncertified | ToleranceMet | candidateで改善 |

独立whole-value central-difference midpoint proxyは `31.943882689839185`。baselineとの差は `6.58856e-5`、candidateとの差は `1.39130e-8`。これは `a±2e-6` の高解像度独立積分の平均で、形式的な誤差上界ではない。

独立gradient FDは `-2.4046891642370833`、step系列の観測spreadは `8.26e-8`。candidateとの差は `1.31e-8` でspread内、baselineとの差は `2.2994`。このため、前回の `-2.40468918 ≈ -2.40468916` を現在のadaptive whole-epoch経路でも再現した。

全epoch関数を通した1回ずつのcold観測時間は value-only `0.914/0.763 ms`、value+5Jac `1.608/0.953 ms` (baseline/candidate)。分布測定ではないため、速度向上の一般的主張には使わない。

## 否定例と全corpus監査

`reference_cases.tsv` の110行で70個のchart-event記録を走査した。8記録 (4つのcase/radius locationをu重複込みで記録) が条件を満たし、62記録をrejectした。Phase 9 corpusでprojective foldとして受理されたのは `on-axis-off` の2半径と `caustic-cross` の2半径だけ。`rand006` row 41のchart event 2件は `Q_u(0)/scale=0.1536, 0.1478` で、D14 eventとの距離も `0.0273, 0.0174`。いずれも `no_coincident_d14_event` でrejectされ、fold mapへ誤昇格しなかった。corpus内で誤昇格は0件。

同じPhase 9入力をbaseline/candidate両方でfull adaptive epoch評価した (`None` と `ValueFirst`, `mu_rtol=1e-4`, gradient rtol `1e-3`)。計220出力/variant。value convergenceは双方200/220で一致し、未収束20件は全て同じ `TopologyUnresolved`。数値status変更0、value stop変更0、新規value convergence regression 0。event count・event identity・cell plan変更は全て0。最大 `mu`差は `6.58717e-5` で、全比較行においてbaseline/candidate報告value-error推定値の合計を超える差はなかった (推定値はformal boundではない)。ValueFirstの5成分quality変更はすべて `FiniteUncertified -> ToleranceMet` の改善で、悪化はなかった。

per-row wall timeは各binary一回・固定順のため、性能比較には使わない。candidateはprojective foldのradial map効果だけを見るresearch buildであり、production router/defaultは変更していない。

## 限界・残件

- binary128残差・条件判定と既存D14 eventの一致を使う数値certificateであり、outward interval arithmeticによる形式証明ではない。promotion条件はすべての曖昧さをrejectするが、production昇格の前に必要な証明強化は残る。
- high-order projective contact / `Q_R=0` の実レンズfixtureはcorpus中にない。gateのunit testはあるが、実物理caseでの挙動を確認したわけではない。
- independent FD/GLは観測収束でありformal error boundではない。`R` binary64分解能のため、Phase 73の最高GL orderが低orderより悪くなる既知driftも残してある。
- `rand035` cell 3 radial quadrature問題は別件として変更していない。Hermite、gradient h-split、gradient tolerance、quality gate、schedulerも変更なし。
- したがって今回はshadow/A/B成功の段階まで。production defaultへの採用は次判断とする。

## 再現

```bash
bash evidence/holonomic/projective_fold_phase74/run_phase74.sh
cmake --build build-holonomic-m7 --target test_projective_fold -j2
ctest --test-dir build-holonomic-m7 -R 'holonomic_(quartic_sturm|quartic_local_bracket|root_pair|adaptive_radial|projective_fold)' --output-on-failure
sha256sum -c evidence/holonomic/gradient_cell_attribution_phase73/SHA256SUMS
```

raw TSV/JSON、compile/run script、環境とflagsは `evidence/holonomic/projective_fold_phase74/` にある。独立参照は `evidence/holonomic/gradient_cell_attribution_phase73/` とそのPhase 73 checkpointを参照する。
