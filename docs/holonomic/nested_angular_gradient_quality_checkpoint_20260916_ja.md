# Nested角度昇格による微分失敗の診断 checkpoint（2026-09-16）

## 結論

角度ruleを `7 -> 15 -> 31 -> 63 -> 127 -> 255 -> 511` とnestedに昇格させても、既存108ケースで観測された微分失敗の大半は解消しなかった。失敗の主因は `EventLocationLimited`、`BudgetExceeded`、`TopologyUnresolved` であり、角度積分の次数不足ではない。

初期7点は既存7点を再利用して15点へ進められるが、7点だけでは安全な角度誤差を持てないため、実際には15点以上への昇格管理が必要になる。初期15点に対して速度上の利益はなく、warmの `close-binary, u=0.5` では微分品質と停止状態が悪化した。したがって初期7点を既定値へ昇格せず、nested実装とcacheは研究用比較資産として残す。

## 対象と実装

対象は研究用adaptive経路のみで、production router、固定 `n_r` 経路、PF6/GM経路、gradient tolerance、quality gate、schedulerは変更していない。

角度評価は固定のnested GC2 ladderを使う。各昇格では既存nodeの `phi` と解析5Jacを再計算せず、新規nodeだけを評価し、保存済み点から低次数の重み付き和を再利用する。今回の診断では初期次数15と初期次数7を比較した。

診断設定は次のとおり。

- 入力: `evidence/holonomic/gm_coverage_cases.tsv` の108ケース
- policy: `Strict`, `ValueFirst`
- 実行: cold / warm
- 角度order: 初期15または初期7、最大511
- `mu_rtol=1e-5`
- gradient tolerance: `atol=1e-6`, `rtol=1e-4`
- gradient追加budget: `32768` evaluations / `64` rounds
- 反復: 各組み合わせ1回（品質・停止原因の診断用）

初期7点の安全性について、7点と比較する下位ruleがない間は有限の未確定マーカーをledgerへ入れ、15点への昇格を必須にした。これにより、未評価の角度誤差をゼロとして早期停止する経路は入れていない。

## 結果

`value_valid_rows` は `TopologyUnresolved` ではない98行。`quality_components` は5成分×108行の集計で、`Invalid` は主にtopology未解決行に対応する。

### cold

| policy / arm | value-valid行の非Tolerance行 | ToleranceMet / FiniteUncertified / Invalid | gradient stop | 中央evaluation | 中央ms |
|---|---:|---:|---|---:|---:|
| Strict / nested15 | 50/98 | 375 / 115 / 50 | Converged 48, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 93 | 0.6575 |
| Strict / nested7 | 50/98 | 375 / 115 / 50 | Converged 48, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 153 | 0.6671 |
| ValueFirst / nested15 | 50/98 | 376 / 114 / 50 | Converged 48, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 93 | 0.6474 |
| ValueFirst / nested7 | 50/98 | 374 / 116 / 50 | Converged 48, EventLocationLimited 46, BudgetExceeded 4, TopologyUnresolved 10 | 153 | 0.6713 |

初期7点はStrictで品質・停止状態が初期15点と完全一致したが、evaluation数は増えた。ValueFirstでも改善行はなく、`caustic-cross, u=0.5` と `on-axis-off, u=0.5` で `ToleranceMet` の成分が `FiniteUncertified` へ変わった。

### warm

| policy / arm | value-valid行の非Tolerance行 | ToleranceMet / FiniteUncertified / Invalid | gradient stop | 中央evaluation | 中央ms |
|---|---:|---:|---|---:|---:|
| Strict / nested15 | 52/98 | 366 / 124 / 50 | Converged 46, EventLocationLimited 46, BudgetExceeded 6, TopologyUnresolved 10 | 93 | 0.6174 |
| Strict / nested7 | 53/98 | 365 / 125 / 50 | Converged 45, EventLocationLimited 46, BudgetExceeded 6, InnerAccuracyLimited 1, TopologyUnresolved 10 | 159 | 0.6829 |
| ValueFirst / nested15 | 52/98 | 367 / 123 / 50 | Converged 46, EventLocationLimited 46, BudgetExceeded 6, TopologyUnresolved 10 | 93 | 0.6109 |
| ValueFirst / nested7 | 53/98 | 364 / 126 / 50 | Converged 45, EventLocationLimited 46, BudgetExceeded 6, InnerAccuracyLimited 1, TopologyUnresolved 10 | 159 | 0.6776 |

warmでは `close-binary, u=0.5` が初期7点で悪化した。Strictでは該当成分が `ToleranceMet` から `FiniteUncertified` になり、gradient stopも `Converged` から `InnerAccuracyLimited` になった。ValueFirstでも同じ成分品質の悪化が残った。

### 角度次数で解決できた範囲

- 共通して46行前後ある `EventLocationLimited` は、角度orderを上げても変化しない。event位置の不確かさが支配しているため、角度nodeの追加では解決しない。
- `BudgetExceeded` の難例（cold 4行、warm 6行）は、角度orderを上げても収束しなかった。代表例は `on-axis-off`、`caustic-cross`、warmの `very-close`。
- `TopologyUnresolved` 10行はvalue-validではなく、角度ruleの問題として扱えない。
- 今回観測された角度order感度はwarmの `close-binary, u=0.5` が中心で、初期7点は初期15点を改善せず悪化させた。初期15点で全成分が `ToleranceMet` になっているため、初期7点を導入する根拠にはならない。

以上から、角度方向のnested cacheは「昇格時の再計算を避ける」実装としては機能しているが、現在の微分失敗群を広く解決する手段ではない。7点開始で最終的に同じ15点へ進むケースでは、初期15点の埋め込み誤差推定を最初から持つ方が安く、制御も単純になる。

## 検証

実装後の既存検証は次のとおり。

- `test_adaptive_radial`: 1608 checks, 0 failures
- `test_holonomic_m7`: 10397 checks, 0 failures
- 関連CTest: 9/9 pass
- 初期7と初期15のfull A/Bで、value coverageと既存stop状態の追加非回帰なし

今回のgradient-quality診断は1 repeatで、独立gradient referenceとの数値誤差受入判定を行うベンチではない。したがって、ここでの結論は「次数昇格でどの停止理由・quality状態が動くか」の原因切り分けであり、gradient全成分の独立accuracy証明ではない。

## 保存した証拠

- raw cold: `evidence/holonomic/nested_angular_gradient_quality_20260916/raw_cold.csv`
- raw warm: `evidence/holonomic/nested_angular_gradient_quality_20260916/raw_warm.csv`
- 集計: `evidence/holonomic/nested_angular_gradient_quality_20260916/summary.json`
- 診断ベンチ: `tests/holonomic_cpp/bench_nested_gradient_quality.cpp`
- 初期7点のfull A/B: `evidence/holonomic/nested_angular_initial7_20260916/`

## 再現コマンド

```text
cmake -S tests/holonomic_cpp -B /tmp/lcbinint_gradient_fold_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/lcbinint_gradient_fold_build --target bench_nested_gradient_quality -j2
/tmp/lcbinint_gradient_fold_build/bench_nested_gradient_quality evidence/holonomic/gm_coverage_cases.tsv 108 0 1 > /tmp/gradient_nested_quality_cold_cmake.csv
/tmp/lcbinint_gradient_fold_build/bench_nested_gradient_quality evidence/holonomic/gm_coverage_cases.tsv 108 1 1 > /tmp/gradient_nested_quality_warm_cmake.csv
cp /tmp/gradient_nested_quality_cold_cmake.csv evidence/holonomic/nested_angular_gradient_quality_20260916/raw_cold.csv
cp /tmp/gradient_nested_quality_warm_cmake.csv evidence/holonomic/nested_angular_gradient_quality_20260916/raw_warm.csv
```

## 次の扱い

初期7点への変更、gradient toleranceの緩和、quality gateの変更、Hermite再導入は行わない。微分失敗の次の調査対象は、角度次数ではなく `EventLocationLimited` のevent/geometry誤差、`BudgetExceeded` のradial・inner ledger、及び既存の `FiniteUncertified` と独立gradient referenceの差である。角度ruleを二次元adaptiveへ拡張する場合も、角度誤差が支配しているpanelだけを対象にし、今回の結果を根拠に全体の初期次数を7点へ下げない。
