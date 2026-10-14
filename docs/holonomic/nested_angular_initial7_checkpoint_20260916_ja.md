# Nested angular rule: 初期7点 A/B checkpoint (2026-09-16)

## 目的

角度方向の nested GC2 rule で、初期評価を15点から7点へ下げられるかを確認した。対象は研究用 `nested_angular_adaptive` 経路だけで、既存の production V2、固定角度 rule、PF6/GM 経路は変更していない。

## 実装

`7 -> 15 -> 31 -> 63 -> 127 -> 255 -> 511` の固定 ladder を追加した。7点開始時は7点の値と解析5Jacを保存するが、7点と比較する下位ruleがないため、角度誤差を推定済みとは扱わない。有限の未確定マーカーを ledger に入れて、必要な場合は15点へ昇格する。15点への昇格では、既存の7点を再計算せず、新しい8点だけを評価する。その後の昇格も既存点を再利用する。

初期7点の未確定マーカーが残っている間に event 停止が先に発火すると、必要な7→15昇格へ到達しないことが分かった。そのため controller の順序を、角度誤差の昇格を試してから event/inner の fail-closed 判定へ進む形に修正した。昇格できない場合は従来どおり停止し、誤差を小さく見せる処理は入れていない。

## 検証

入力は `evidence/holonomic/gm_coverage_cases.tsv` の108行、各行5回、`None` と `ValueFirst`、cold/warmを測定した。各rawファイルはヘッダを除いて4320行（108ケース × 2 policy × 4 arm × 5 repeat）である。比較armは incumbent、既存 angular adaptive、nested初期15、nested初期7とした。

単体・回帰検証:

- `test_adaptive_radial`: 1608 checks, 0 failures
- `test_holonomic_m7`: 10397 checks, 0 failures
- 関連CTest: 9/9 pass
- 初期7と初期15の stop/value_converged のcase単位差: cold 0、warm 0
- corpusで両方式に共通する既存 `TopologyUnresolved`: 50/540 rows/policy/arm。初期7による追加失敗は0

## whole epoch 結果

単位はms。`row p50` は全repeat行、`case median` は各caseの5回中央値を集めた中央値。rawは同じディレクトリの `raw_cold.csv` と `raw_warm.csv`、machine-readable集計は `summary.json` に保存した。

### cold

| policy | arm | row p50 | p90 | p95 | p99 | max | case median | value convergence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| None | nested 15 | 0.498224 | 0.732937 | 0.889233 | 1.199364 | 1.561031 | 0.494524 | 490/540 |
| None | nested 7 | 0.504050 | 0.768670 | 0.897017 | 1.233535 | 1.691141 | 0.504258 | 490/540 |
| ValueFirst | nested 15 | 0.585788 | 1.982417 | 12.011447 | 31.728325 | 41.898201 | 0.584963 | 490/540 |
| ValueFirst | nested 7 | 0.618737 | 2.121837 | 12.279941 | 39.795335 | 54.114503 | 0.617707 | 490/540 |

初期7−初期15のcase median時間差は、Noneで +0.005265 ms、ValueFirstで +0.008501 ms。ValueFirstでは初期7が段階昇格を多く発生させ、追加の同一node angular callbackが増えた。

### warm

| policy | arm | row p50 | p90 | p95 | p99 | max | case median | value convergence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| None | nested 15 | 0.389166 | 0.647955 | 0.753153 | 11.872178 | 53.964348 | 0.389963 | 490/540 |
| None | nested 7 | 0.396319 | 0.691740 | 0.823036 | 11.966211 | 53.778931 | 0.394425 | 490/540 |
| ValueFirst | nested 15 | 0.494739 | 3.603231 | 19.736647 | 29.614106 | 54.142538 | 0.490417 | 490/540 |
| ValueFirst | nested 7 | 0.522418 | 3.605395 | 20.651112 | 34.155621 | 53.782442 | 0.520622 | 490/540 |

初期7−初期15のcase median時間差は、Noneで +0.006733 ms、ValueFirstで +0.008781 ms。warmでも初期7の方が速くならなかった。

## 解釈と判断

今回の結果では、初期7点は値・status coverageを壊さず実装できたが、速度上の利益は出なかった。LD寄与がある `u=0.5` では、7点だけで安全な角度誤差を作れないため、結局15点へ昇格する。初期15方式は同じ15点を一度に評価して7点との差を持てるので、初期7方式の段階的な昇格管理費だけが追加される。

したがって、初期7を production/default や nested研究経路の既定値へ昇格しない。初期7 ladderとcacheは比較用の research arm として残す。7点開始を速くするには、7点だけから安全に使える独立した角度誤差上界、または7点で終了できる数学的 certificate が必要であり、今回それを仮定して停止条件を緩めることはしなかった。

## 再現コマンド

```text
cmake --build /tmp/lcbinint_gradient_fold_build --target test_adaptive_radial bench_adaptive_2d test_holonomic_m7 -j2
/tmp/lcbinint_gradient_fold_build/test_adaptive_radial
/tmp/lcbinint_gradient_fold_build/test_holonomic_m7
ctest --test-dir /tmp/lcbinint_gradient_fold_build -R 'holonomic_(adaptive_radial|quartic_sturm|quartic_local_bracket|root_pair|projective_fold|radial_atlas|d14_positive|d14_root_scheduling|d14_event_contract)$' --output-on-failure
/tmp/lcbinint_gradient_fold_build/bench_adaptive_2d evidence/holonomic/gm_coverage_cases.tsv 108 5 0 > evidence/holonomic/nested_angular_initial7_20260916/raw_cold.csv
/tmp/lcbinint_gradient_fold_build/bench_adaptive_2d evidence/holonomic/gm_coverage_cases.tsv 108 5 1 > evidence/holonomic/nested_angular_initial7_20260916/raw_warm.csv
```

このcheckpointは初期7点のA/Bを記録するもので、production router/defaultの変更を意味しない。
