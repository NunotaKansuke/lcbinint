# Phase 75: projective fold候補の最終trajectory A/B

実施日: 2026-09-14
基準: `dev/holonomic` @ `65330381491f175d54da1af431c7e6b1a2d337da`
結論: **合成ordinary-fold fixtureと全22 CTestはpass。14,432-row corpusでは値・status・gradient品質・topologyの非回帰を確認したが、projective foldの受理は0件で、whole-epoch medianは約1–3%遅くなった。production defaultには昇格せず、研究compile gateを維持する。**

## 判定

Phase 74のcaustic-cross `a` では、cell 5/7の未map `chart_p4` endpointをphysical projective fold mapへ接続すると、独立FDと整合し、value nodesが195→91、value+5Jac nodesが345→139へ減った。今回の最終trajectory corpusはこの幾何イベントを含むかを別scanし、速度結果と修正対象の有無を分けて評価した。

production router/defaultは変更していない。現candidateを全入力で有効にすると、今回のcorpusでは修正対象がないまま追加費用だけが見えたため、現状のままproductionへ上げる根拠はない。

## 合成接触fixture

`test_projective_fold` に、productionのnormalized contact gateへ実際の合成 `Q(u,R)` 係数を通すfixtureを追加した。`Q=(R-1)+u^2` のordinary projective foldは受理し、次はfail-closedでrejectする。

- `Q=(R-1)+u^4` とほぼ零の `Q_uu`: `DegenerateAngularContact`
- `Q=(R-1)^2+u^2` とほぼ零の `Q_R`: `NoRadialCrossing`
- `Q=(R-1)+u`: `ProjectiveContactUnresolved`

既存の実レンズnegative controlである`rand006` row 41もPhase 74と同じくD14 coincidenceなしでrejectした。合成fixtureは局所contact gateを検査するもので、物理レンズ全域を形式証明するものではない。

## matched trajectory A/B

入力は論文用trajectory benchmarkと同じ凍結snapshotで、SHA-256は `6646492658ed416e75cb8c6c821ee853f79cdf7761fc41441186aa67b81fa05b`。物理入力7,216行、1,804 trajectory × 4 epoch。`RelTol=1e-3,1e-4`を掛けて各arm/policy 14,432結果行とした。

baselineとcandidateは同じ`trajectory_ab.cpp`、同じGCC 11.5.0 flags、同じ入力・policy・precision、CPU affinity `0-7`でserialに実行した。反復はarm/policyごとに3回、同一rep内でarm順を交互にし、各processは各laneに未計測warmup trajectoryを1回実施した。`LensParams`生成はtimer外。

- `full-cold`: `epoch_adaptive`。D14/topologyとadaptive integrationをwhole timerへ含める。
- `full-warm`: trajectory内のL2 D14 root warm-start。`allow_topology_reuse=false`, `l2_drift=1e18`。topology再構築とadaptive integrationをwhole timerへ含み、各trajectoryの最初のepochはcold。
- `radial-only`: topologyをtimer外で事前生成し、adaptive integrationだけを計時。全epoch比較ではなく、追加されたprojective処理の局所cost確認用。
- policyはvalue-only `None` とvalue+analytic 5Jac `ValueFirst`。valueは`mu_atol=1e-16`, targetごとに`mu_rtol=1e-3/1e-4`。Jac laneは`grad_rtol=1e-3`, `grad_atol`は現行default。

candidateは`HOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH` compile defineだけをbaselineへ追加した。D14設定、production router、adaptive tolerance/gateは同一。

### whole / radial latency

下表は3反復から得た各rowの中央値を集計したp50/p90/p99（ms）。各セルはbaseline→candidate。

| policy | value RelTol | lane | p50 | p90 | p99 |
|---|---:|---|---:|---:|---:|
| value-only | 1e-3 | full-cold | 0.4165→0.4230 | 0.7179→0.7344 | 3.2451→3.2858 |
| value-only | 1e-3 | full-warm | 0.3416→0.3484 | 0.6393→0.6583 | 1.1884→1.1575 |
| value-only | 1e-3 | radial-only | 0.1241→0.1267 | 0.2448→0.2624 | 0.3861→0.4147 |
| value-only | 1e-4 | full-cold | 0.4600→0.4659 | 0.7860→0.8032 | 3.2343→3.2649 |
| value-only | 1e-4 | full-warm | 0.3911→0.3966 | 0.7143→0.7310 | 1.4220→1.4722 |
| value-only | 1e-4 | radial-only | 0.1610→0.1650 | 0.3222→0.3400 | 0.5588→0.5830 |
| value+5Jac | 1e-3 | full-cold | 0.5759→0.5931 | 1.0589→1.0944 | 3.4657→3.4489 |
| value+5Jac | 1e-3 | full-warm | 0.5093→0.5251 | 0.9809→1.0191 | 2.3105→2.3563 |
| value+5Jac | 1e-3 | radial-only | 0.2623→0.2725 | 0.5802→0.6141 | 1.3872→1.4235 |
| value+5Jac | 1e-4 | full-cold | 0.6128→0.6292 | 1.2351→1.2700 | 3.5586→3.5991 |
| value+5Jac | 1e-4 | full-warm | 0.5446→0.5621 | 1.1561→1.1962 | 2.7942→2.8089 |
| value+5Jac | 1e-4 | radial-only | 0.2933→0.3043 | 0.7804→0.8200 | 1.7382→1.7738 |

paired per-row candidate/baseline p50 ratioは、value-only full-cold/full-warmで `1.0089–1.0133`、value+5Jacで`1.0232–1.0288`。steady trajectory warmでもcandidateはp50で約1.5–3.0%遅かった。p95/maxを含む全分布、profile別、steady/first splitは`evidence/holonomic/projective_fold_final_ab_phase75_20260914/summary.json`を参照。

### correctness / status / event coverage

- value-only: 両target・cold/warm/radial全て7,216/7,216 value-converged、status `OK`。
- value+5Jac, `1e-3`: 両armで全lane 7,216/7,216 value-converged、status/gradient quality一致。
- value+5Jac, `1e-4`: 両armで全lane 7,215/7,216 value-converged。共通の未収束1行は`case 86 / linear / d_bin 0 / epoch 7`、`InnerAccuracyLimited`。candidate起因ではない。quality/statusの行一致も維持した。
- 1e-4のValueFirstにある共通`GRADIENT_UNRELIABLE`/`Invalid`成分もbaseline/candidate間で変化なし。5成分のgradient数値差は全行0、gradient quality regression 0。
- 全policy/target/laneでvalue mu、value stop/status、node/panel/split数、event/cell counts・topology hashの不一致0。独立VBM `RelTol=1e-6` snapshotに対する観測誤差は両armで同じ。1e-4の既存3件のreference相対誤差超過は残るが、新規超過は0。参照との相対誤差は観測比較で、厳密誤差上界ではない。
- full corpusのphysical input 7,216行を別にcold topology scanしたところ、`chart_p4`記録3,376件すべてが`no_coincident_d14_event`でrejectされ、projective fold受理は0件。従って本A/Bは実際のfold修正効果をtrajectory上で測っていない。candidateの速度差は対象foldの効果ではなく、no-hit条件での追加処理を含む。

Phase 74のcaustic-cross targeted controlでは実際に2箇所のprojective foldが受理され、独立FDへgradientが一致し、node削減も観測済み。この局所correctness/利益と今回のno-hit corpus結果を混同しない。

## 採否と残件

合成ordinary-fold/high-order/nontransverse gateとCTestsはpassし、Phase 74 caustic controlは独立参照に整合する。一方、現行candidateをこのまま全epochで有効にした場合のtrajectory A/Bでは受理イベントがなく、median固定費が増えた。したがって**projective foldの数学的修正は有望なcorrectness fixだが、この実装のproduction昇格は保留**とする。compile gateは維持し、router/defaultは変更しない。

数値gateはbinary128、D14 event coincidence、fail-closed判定を組み合わせたもの。outward-rounded interval proofは未実装であり、形式証明済みとは扱わない。今回のtrajectory corpusだけではproduction環境におけるtrue-fold頻度・全体利益も結論できない。次に進める場合は、(1) true projective foldを含むtrajectory集合でのmatched whole A/B、(2) no-hit chart eventのcertificate固定費削減を分けて評価する。

## 再現・検証

```bash
bash evidence/holonomic/projective_fold_final_ab_phase75_20260914/run_phase75.sh
cmake --build build-holonomic-m7 --target test_projective_fold test_d14_native_residual -j2
ctest --test-dir build-holonomic-m7 --output-on-failure
sha256sum -c evidence/holonomic/projective_fold_final_ab_phase75_20260914/SHA256SUMS
```

12個のgzip raw run、frozen input、scan TSV/log、manifest、CTest log、summary JSONとchecksumを`evidence/holonomic/projective_fold_final_ab_phase75_20260914/`へ保存した。
