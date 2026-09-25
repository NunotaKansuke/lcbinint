# Phase 77: physical fold近傍のcomplex soft cutとgradient積分

## 結果

rand035のX/Y gradient誤差の主因は、fixed-R微分やfold式そのものではなく、D14の非physical `physical_complex` soft cutがphysical fold直後にradial積分を分断し、fold mapのない側panelに平方根型の急変を残していたことだった。

adaptive積分用のspanだけを、次の全条件を満たす場合に統合した。

* 隣接する両セルがcertified `kArcs` で、crossing数が同じ正の偶数
* 共有境界にはnonphysical `physical_complex` eventだけがある
* 両外端が既存のcertified physical fold
* `fold_maps` が有効
* soft cutからfoldまでの距離が、隣接セルに設定したfold map上での初期Fejér-II最近傍node距離より短い

最後の条件は、`t0=(1+x_near)/2` として最近傍nodeの実radial stride `w(2t0-t0²)` を使う。lens parameterや固定角度/radius閾値には依存しない。初期則が十分細かくcutとfoldの間をサンプルできる場合は、元の積分分割を維持する。

rand035ではfold / soft cut / foldが`0.88716938242769516 / 0.88723369079252823 / 0.95630209345982764`。fold-to-cut gapは`6.43e-5`、隣接cell幅は`6.91e-2`。level 3のFejér則で計算した隣接fold-mapの最初のradial node strideは`5.16e-3`なので、gapはその約1/80であり、統合条件を満たす。

変更はadaptiveの評価spanだけに限定した。`TopologyResult.events`、D14判定、`restore_physical_cuts()`が作るsource cells、quartic crossing countは書き換えない。physical event、`chart_p4`、crossing数が異なる境界は統合しない。V2 fixed-`n_r`経路とproduction routerにも変更はない。

## 数値検証

独立参照はfresh topologyを用いた5-point finite difference of direct physical-value integral。Xはradial order 160 / 32 subdivisions、Yは128 / 16 subdivisions、angular 1024。step/resolutionの観測安定幅はXで約`5–9e-9`、Yで約`5e-11`。これは独立FDの安定性観測であり、形式的な誤差上界ではない。

同一のrand035入力で、value RelTol 1e-3/1e-4、None/ValueFirstを比較した。ValueFirstの明示設定はgradient atol `1e-4`、rtol `1e-2`、round budget 64、node budget 8192。4条件すべてでnode数は300から69、panel数は4から3へ減り、X/Yは既存gateの`ToleranceMet`を維持した。

| u | value RelTol | baseline `|Δμ|` | 修正後 `|Δμ|` | baseline X/Y abs error | 修正後 X/Y abs error |
|---:|---:|---:|---:|---:|---:|
| 0 | 1e-3 | 7.27e-5 | 1.68e-8 | 3.09e-7 / 6.54e-7 | 1.96e-11 / 6.19e-11 |
| 0 | 1e-4 | 3.03e-6 | 2.66e-15 | 3.09e-7 / 6.54e-7 | 1.96e-11 / 6.19e-11 |
| 0.5 | 1e-3 | 4.35e-5 | 1.01e-7 | 1.86e-7 / 3.92e-7 | 2.41e-11 / 5.09e-11 |
| 0.5 | 1e-4 | 1.82e-6 | 2.66e-15 | 1.86e-7 / 3.92e-7 | 2.41e-11 / 5.09e-11 |

また、同じ110入力・2 tolerance・2 policyの440 cold rowsをmatched A/Bした。統合はrand035のu=0/.5だけで8行に発生し、ほかの108入力はnode数不変。各設定のvalue coverage/statusはbaseline/candidateとも100/110、既存の10 `TopologyUnresolved`は同じ入力で残り、新しいvalue stop/status差は0。全5成分でgradient quality downgradeは0、upgradeは12件だった。総node数は43,613から42,537へ2.47%減った。これはwork proxyであり、whole-epoch速度勝利の主張には使っていない。

targeted runではvalue-onlyとValueFirstのmuは各ケースでbitwise同一。110-row corpusには従来から丸め差だけのvalue-only/ValueFirst不一致があり、baseline 10件に対しcandidate 9件で、新しい差は増えていない。

## 検証

`test_adaptive_radial`: 1,567 checks, 0 failures。追加regression testはrand035のu=0/.5、両tolerance、source cell/event不変、初期則の解像度条件、fold map無効時、balanced interior cut、chart event/crossing変更のnegative controlを確認する。

CTest selected 5/5 passed: `holonomic_adaptive_radial`, `holonomic_quartic_sturm`, `holonomic_quartic_local_bracket`, `holonomic_root_pair`, `holonomic_projective_fold`。

## 再現とraw evidence

再現コマンド:

```bash
bash evidence/holonomic/gradient_softcut_fold_phase77/run_ab.sh
cmake --build build-holonomic-m7 --target test_adaptive_radial test_quartic_sturm test_quartic_local_bracket test_root_pair test_projective_fold -j2
ctest --test-dir build-holonomic-m7 -R 'holonomic_(adaptive_radial|quartic_sturm|quartic_local_bracket|root_pair|projective_fold)$' --output-on-failure
```

AB scriptはbaseline `cd1343cc2bd7fc8bfde17c5556772ed6420f14b4`を`git archive`で展開し、同じC++17 / `-O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno` flagsとCPU 0でbaseline/candidateを実行する。wall-time列はrawに残すが、単発計測のため性能判定には使わない。

入力、両armのraw TSV、独立FD raw、machine-readable `summary.json`、runner/sourceは`evidence/holonomic/gradient_softcut_fold_phase77/`にある。推定誤差式や停止gateは変更していない。

## 残る範囲

確認したのはPhase 72のrand035 hard caseと同じ110-row gradient corpus。10件の既存topology未解決は未修正であり、この変更の対象外。ほかのケースで同じ幾何条件が出た場合も形式上は同じresolution testを通るが、今回の独立FD参照があるのはrand035 X/Yだけ。速度の全epoch再評価はここでは採択根拠にしていない。
