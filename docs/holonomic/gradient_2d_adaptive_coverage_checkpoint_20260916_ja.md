# 半径・角度2D adaptiveによる微分失敗ケースの検証 checkpoint（2026-09-16）

## 結論

半径方向だけをadaptiveにする経路へ、同じradial sample上で角度ruleを必要な点だけ昇格する2D controllerを組み合わせると、微分失敗の一部を実際に解消できた。

通常budgetの108ケース比較で、全5成分が `ToleranceMet` になる行は次のように増えた。

- cold Strict: radial-only 44/108 -> dual nested15 48/108
- cold ValueFirst: radial-only 44/108 -> dual nested15 48/108
- warm Strict: radial-only 43/108 -> dual nested15 46/108
- warm ValueFirst: radial-only 43/108 -> dual nested15 46/108

改善したケースは次のとおり。

- cold: `close-binary, u=0.5`、`rand033, u=0.5`、`resonant, u=0.5`、`very-close, u=0.5`
- warm: `close-binary, u=0.5`、`rand033, u=0.5`、`resonant, u=0.5`

これらはradial-onlyでは `InnerAccuracyLimited` だったが、角度誤差を局所的に下げたことで `Converged` へ移った。したがって、角度方向を追加したadaptiveは実際に有効な経路である。

## 実装と比較方法

研究用ベンチ `bench_gradient_2d_modes` を追加し、同一case・同一tolerance・同一warm条件で次を比較した。

- `radial_only`: 半径panelだけをrefine
- `dual_gc2`: 半径panelと既存16/24/32/...角度ruleをadaptiveに昇格
- `dual_nested15`: 半径panelとnested 15/31/63/...角度ruleをadaptiveに昇格

全経路で `Strict` と `ValueFirst`、coldとwarmを測った。角度昇格は、global ledgerとpanel内の `inner`/`radial` 比較で角度側が律速と判断されたsampleだけに適用し、同じRのroot-pair・geometry・analytic 5Jacを再利用する。角度のquality gate、value契約、gradient品質分類は変更していない。

通常budgetは既存診断と合わせて `max_level=8`、`max_depth=12`、`max_node_evals=65536`、ValueFirstのgradient追加budget `32768 evaluations / 64 rounds`。角度は既存dual nestedの初期15点、最大511点とした。

## 通常budgetの結果

値の収束は全armで同じ `98/108`。共通の `TopologyUnresolved` 10行を含むため、今回の目的であるvalue validityを変えずに微分側を比較している。

### cold

| policy / arm | value converged | 全5成分 ToleranceMet | gradient stop の主な内訳 | 中央ms | 中央evaluation |
|---|---:|---:|---|---:|---:|
| Strict / radial-only | 98/108 | 44/108 | Converged 44, InnerAccuracyLimited 4, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 0.7100 | 93 |
| Strict / dual GC2 | 98/108 | 47/108 | Converged 47, InnerAccuracyLimited 1, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 0.6940 | 93 |
| Strict / dual nested15 | 98/108 | 48/108 | Converged 48, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 0.6876 | 93 |
| ValueFirst / radial-only | 98/108 | 44/108 | Converged 44, InnerAccuracyLimited 4, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 0.6913 | 93 |
| ValueFirst / dual GC2 | 98/108 | 46/108 | Converged 46, EventLocationLimited 46, BudgetExceeded 6, TopologyUnresolved 10 | 0.6713 | 93 |
| ValueFirst / dual nested15 | 98/108 | 48/108 | Converged 48, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 0.6675 | 93 |

### warm

| policy / arm | value converged | 全5成分 ToleranceMet | gradient stop の主な内訳 | 中央ms | 中央evaluation |
|---|---:|---:|---|---:|---:|
| Strict / radial-only | 98/108 | 43/108 | Converged 43, InnerAccuracyLimited 4, EventLocationLimited 46, BudgetExceeded 5, TopologyUnresolved 10 | 0.5958 | 93 |
| Strict / dual GC2 | 98/108 | 46/108 | Converged 46, EventLocationLimited 46, BudgetExceeded 6, TopologyUnresolved 10 | 0.5952 | 93 |
| Strict / dual nested15 | 98/108 | 46/108 | Converged 46, EventLocationLimited 46, BudgetExceeded 6, TopologyUnresolved 10 | 0.6170 | 93 |
| ValueFirst / radial-only | 98/108 | 43/108 | Converged 43, InnerAccuracyLimited 4, EventLocationLimited 46, BudgetExceeded 5, TopologyUnresolved 10 | 0.5818 | 93 |
| ValueFirst / dual GC2 | 98/108 | 46/108 | Converged 46, EventLocationLimited 46, BudgetExceeded 6, TopologyUnresolved 10 | 0.5929 | 93 |
| ValueFirst / dual nested15 | 98/108 | 46/108 | Converged 46, EventLocationLimited 46, BudgetExceeded 6, TopologyUnresolved 10 | 0.6070 | 93 |

中央値は1回測定のdiagnostic値なので性能採用の根拠にはしない。coldではdual nested15がradial-onlyよりわずかに速いが、warmではGC2が同程度、nested15は角度cacheと昇格回数の固定費で遅い。今回の主目的は速度ではなく、同一のvalue mesh契約を保った微分coverageの確認である。

## 何がまだ失敗しているか

失敗の中心は角度ruleではない。

- `EventLocationLimited`: 46/108行前後。角度orderを増やしてもevent uncertaintyが支配するため変化しない。
- `BudgetExceeded`: 通常budgetでcold 4行、warm 6行前後。`caustic-cross`、`on-axis-off`、`very-close` などでradial/inner refinementが長く続く。
- `TopologyUnresolved`: 10行。これは角度積分の品質ではなく、value/topology validityの問題である。

角度方向の差が実際に現れたのは `InnerAccuracyLimited` の群である。`radial_only` では `close-binary`、`rand033`、`resonant`、`very-close` の一部が止まるが、dual nested15では角度orderを上げて収束した。角度昇格は無条件ではなく、各sampleのorder histogramと`angular_upgrades`に記録されている。

失敗case集合に対して、budgetだけを拡大したdual nested15も試した。`max_node_evals=262144`、ValueFirst gradient追加budget `262144 evaluations / 256 rounds`、`max_depth=32`、`max_panels=8192`、`max_bytes=256 MiB` としたが、event-limited群は残った。これは「単に両方向へさらにnodeを追加すれば全て解決する」状態ではなく、event/geometryまたは別のinner representationが律速であることを示す。

## 判断

今回の2D adaptiveは、微分失敗を実際に減らす研究経路として有効である。特に、角度ruleを増やす前にradial-onlyで止めていた `InnerAccuracyLimited` を、値を別meshへ置き換えずに救えることを確認できた。

一方で、角度adaptiveを入れれば失敗ケース全体が収束するわけではない。`EventLocationLimited` を角度ruleで解決しようとすること、またbudget上限を単純に増やすことは本質的な対策にならない。production router/defaultは変更しない。次の候補は、event/geometry ledgerが支配するpanelの原因特定と、`caustic-cross` などのinner-limitedなcaseでradial・angularの寄与をさらに個別に記録することである。

## 保存した証拠

- 通常cold: `evidence/holonomic/gradient_2d_modes_20260916_cold.csv`
- 通常warm: `evidence/holonomic/gradient_2d_modes_20260916_warm.csv`
- generous cold dual nested: `evidence/holonomic/gradient_2d_modes_20260916_cold_generous_dual_nested15.csv`
- generous warm dual nested: `evidence/holonomic/gradient_2d_modes_20260916_warm_generous_dual_nested15.csv`
- 集計: `evidence/holonomic/gradient_2d_modes_20260916_summary.json`
- radial-only失敗case入力: `evidence/holonomic/gradient_2d_failure_cases_20260916.tsv`
- 診断ベンチ: `tests/holonomic_cpp/bench_gradient_2d_modes.cpp`

## 再現コマンド

```text
cmake -S tests/holonomic_cpp -B /tmp/lcbinint_gradient_fold_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/lcbinint_gradient_fold_build --target bench_gradient_2d_modes -j2
/tmp/lcbinint_gradient_fold_build/bench_gradient_2d_modes evidence/holonomic/gm_coverage_cases.tsv 108 0 1 0 > evidence/holonomic/gradient_2d_modes_20260916_cold.csv
/tmp/lcbinint_gradient_fold_build/bench_gradient_2d_modes evidence/holonomic/gm_coverage_cases.tsv 108 1 1 0 > evidence/holonomic/gradient_2d_modes_20260916_warm.csv
/tmp/lcbinint_gradient_fold_build/bench_gradient_2d_modes evidence/holonomic/gradient_2d_failure_cases_20260916.tsv 58 0 1 1 3 > evidence/holonomic/gradient_2d_modes_20260916_cold_generous_dual_nested15.csv
/tmp/lcbinint_gradient_fold_build/bench_gradient_2d_modes evidence/holonomic/gradient_2d_failure_cases_20260916.tsv 58 1 1 1 3 > evidence/holonomic/gradient_2d_modes_20260916_warm_generous_dual_nested15.csv
```
