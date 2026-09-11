# V2 adaptive radial Phase 9.2 checkpoint

実施日: 2026-09-11
基準: `dev/holonomic` `7b296fe` (`Fix adaptive Phase 9.1 coverage and timing diagnostics`)
対象: isolated holonomic の adaptive V2 経路

## 判定

Phase 9.1で修正済みのvalue単調性とValueFirstのvalue成功を維持したまま、adaptive setupでtopologyが既に計算したphysical event情報を再利用する経路を実装した。setup timerはintegration開始前に停止し、`setup_ms`からadaptive integrationの時間を除いた。

None / ValueFirst の value coverage は、110ケース、cold/warm、RelTol=1e-3/1e-4の全組合せで110/110だった。独立referenceのobserved violationはreuse-on/offともcold/warm 0件だった。ValueFirst warmの `paper_highA, u=0.5` は全体stopが `TopologyUnresolved` になる行を残しているが、`value_stop=Converged` と `value_converged=1` は維持され、gradientは`Invalid`として返っている。topology unresolvedを`FiniteUncertified`へ格下げしていない。

## 実装

`RadialEvent`に、D14のqf rootから作った`radius_lo`、physical-fold分類時のstationary root `fold_t_seed`、precision tier、D14 `D/D'` conditioning estimateを保持させた。adaptive用にだけ`classify_cells(..., retain_adaptive_metadata=true)`を使い、通常のfixed-`n_r`／reference／GM呼び出しは既定値falseのままにした。これにより、fixed-`n_r` production callerへadaptive metadata生成費を追加していない。

adaptive event setupは次の局所ladderになった。

```text
D14 qf radius + topology stationary seed
        ↓
DD coupled Newton: P(R,t)=0, P_t(R,t)=0
        ↓ fail/ambiguous
direct qf coupled Newton
        ↓ fail
panel guard rejects (fail closed)
```

DDは同じ`P=0, P_t=0`の2変数Newtonを実行する。qf fallbackも`PolyFamilyR`の構築やderivative cubicの再探索をせず、`boundary_polynomial.hpp`の明示的な`P, P_t, P_R, P_tt, P_tR`だけを評価する。`local_fold_quantities<T>`はdouble、DD、`__float128`で入力精度を維持する。

`RootPairWarm`はquarticで最大2組であることを利用して、heap-backed vectorから2要素のinline storageへ変更した。`arc_intervals()`が返したquartic係数は同じnodeのmapped-radius計算でも再利用する。

`eval_poly5`のascending coefficient評価も通常のHorner順へ修正した。旧実装は`c[4]x+c[3]x^2+...`となっており、event residual probeを誤ったpolynomialで評価していた。

## event A/B

`HOLO_ADAPTIVE_EVENT_REUSE=1`がD14/topology metadata reuse、`=0`が従来のadaptive側double probeの比較である。rand033 diagnosticでは、reuse-onの128 event recordsは全て`TopologySeedReused`、DD accepted、precision tier 1だった。reuse-offでは全て`DoubleBudgetAccepted`、precision tier 0だった。両方ともqf family constructionは0だった。

代表としてu=0、value-only、cache-on、RelTol=1e-3の110ケースを示す。timingは各case 1 repeatのため、性能の最終主張ではなく固定費の確認用である。

| 構成 | cold whole p50 | warm whole p50 | setup p50 | setup_event p50 |
|---|---:|---:|---:|---:|
| reuse-on | 0.820 ms | 0.661 ms | 0.0134 ms | 0.0071 ms |
| reuse-off | 0.826 ms | 0.663 ms | 0.0222 ms | 0.0155 ms |

Phase 9.1の同じ代表groupではsetup p50が約0.419 ms、setup eventが約0.308 ms、qf family constructionが1、qf event refinementが4だった。今回のsetup計測はintegrationを含まないので、旧setup値との差はtimer修正とevent重複除去を合わせた結果である。

Value+5Jacは、今回もvalueとgradientのcontractを分離したまま測定した。Jacobian strict pathの失敗行を速度比較から除外していない。ValueFirstのgradient tolerance未達は`FiniteUncertified`、topology/root/branch/nonfiniteの失敗は`Invalid`として既存の意味を維持した。

## correctness / measurement

- `test_adaptive_radial`: **1470 checks, 0 failures**
- relevant CTest: **15/15 passed**
- None value: cold/warm × 1e-3/1e-4 = **110/110**
- ValueFirst value: cold/warm × 1e-3/1e-4 = **110/110**
- Strict value stop: 全組合せで value stopは`Converged`; gradient側の既存制約は維持
- independent reference observed violation: reuse-on/offとも **0件**
- timing checker: reuse-on/offとも **4400 rows, 4400 finite, bad 0**
- `contracts.csv`: 1320行、全行39列
- `paired.csv`: 4400行、全行54列
- setup componentの和は各rowで`setup_ms`を超えず、全stageはfinite/non-negativeだった

直接qf局所Newtonはunit testで実際に実行し、全physical eventについて`QfRefined`を確認した。今回の110-case end-to-end corpusでは、topologyから渡したDD tierが全eventで通ったため、direct qf escalationの実測件数は0である。したがって、qf fallbackの速度分布は未測定であり、これを「qfが常に不要」とは解釈しない。

## 採否と範囲

今回の変更は全て採用した。generic `PolyFamilyR`をadaptive setupから外す変更は、直接local formulaとfail-closed guard、qf局所Newtonのunit parityを追加してから採用した。固定storageはquarticの上限を超える候補を拒否するため、黙ってroot pairを切り捨てない。

V2/V3 production router、fixed-`n_r` production API、PF6/GM research pathは変更していない。metadata reuseはadaptiveの内部A/Bであり、`HOLO_ADAPTIVE_EVENT_REUSE=0`で旧probeとの比較を再現できる。

今回の最大の残存costは、通常ケースではD14/topology、gradient付き難例ではadaptive physicsとestimatorである。event setupは縮小したが、これだけでwhole epoch全体をV2より速くするという判定はしていない。次段で性能を主張する場合は、repeat数を増やしたcold/warm benchmarkと、direct qf escalationを含む難例を別に測る必要がある。

## 再現コマンド

```bash
HOLO_ADAPTIVE_EVENT_REUSE=1 bash checks/holonomic/run_adaptive_radial.sh \
  evidence/holonomic/adaptive_radial_phase92_20260911/reuse_on 1
HOLO_ADAPTIVE_EVENT_REUSE=0 bash checks/holonomic/run_adaptive_radial.sh \
  evidence/holonomic/adaptive_radial_phase92_20260911/reuse_off 1
python3 checks/holonomic/check_adaptive_timing.py \
  evidence/holonomic/adaptive_radial_phase92_20260911/reuse_on/paired.csv
python3 checks/holonomic/check_adaptive_timing.py \
  evidence/holonomic/adaptive_radial_phase92_20260911/reuse_off/paired.csv
python3 checks/holonomic/summarize_adaptive_radial.py \
  evidence/holonomic/adaptive_radial_phase92_20260911/reuse_on
python3 checks/holonomic/summarize_adaptive_radial.py \
  evidence/holonomic/adaptive_radial_phase92_20260911/reuse_off
ctest --test-dir build-holonomic-m7 --output-on-failure
```

raw benchmarkとmachine-readable summaryは次に保存した。

- `evidence/holonomic/adaptive_radial_phase92_20260911/summary.json`
- `evidence/holonomic/adaptive_radial_phase92_20260911/reuse_on/`
- `evidence/holonomic/adaptive_radial_phase92_20260911/reuse_off/`
- `evidence/holonomic/adaptive_radial_phase92_20260911/ctest.txt`
