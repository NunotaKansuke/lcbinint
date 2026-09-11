# V2 adaptive radial Phase 9.1 checkpoint

実施日: 2026-09-11
基準: `dev/holonomic` `765229e0a6952e13a67cab91974f4b2d13b1665d`
対象: isolated holonomic の adaptive V2 経路

## 判定

RelTol=1e-3 で value-only / ValueFirst が 108/110 になっていた非単調回帰を修正し、Phase 9 corpus の None / ValueFirst は cold と warm、RelTol=1e-3 と 1e-4 の全組合せで 110/110 に戻った。独立 reference の observed violation は cold/warm とも 0 件だった。

benchmark の `physics_ms` 異常値は、solver の物理計算ではなく `bench_adaptive_radial.cpp` の `printf` 型列数不一致だった。stage timing の出力列を修正し、修正後の paired raw 4400 行を全て機械検査できた。

## value coverage の回帰

失敗した論理ケースは次の2つだった。

| case | u | cold/warm | policy | RelTol=1e-3（修正前） | RelTol=1e-4（修正前） |
|---|---:|---|---|---|---|
| rand033 | 0.0 | cold, warm | None, ValueFirst | `EventLocationLimited`, value未収束 | `Converged` |
| rand033 | 0.5 | cold, warm | None, ValueFirst | `EventLocationLimited`, value未収束 | `Converged` |

修正前の失敗は `diagnostics_before/epochs.csv` に保存した。各 logical case は cold/warm と2 policyを含むため、失敗行は8行になる。修正前は全て85 nodeで event stop になり、厳しいtolでは166 nodeまで進んで収束していた。

根本原因はevent precision ladderの value budget依存だった。`double_event_estimate` は従来、

```text
uncertainty <= max(256 * ULP, 0.05 * value_budget * max(1, |R|))
```

ならdouble eventをphysical endpointとして許可していた。rand033 の event index 6（`R = 1.1259749488987669...`）では、RelTol=1e-3 の粗いdouble推定が約 `1.00961e-5` の不確実性を持ったまま許可され、fold mapの最初のFejer nodeがevent uncertainty内に入って `EventLocationLimited` になった。RelTol=1e-4ではvalue budgetが小さくなるため同じeventがqf refinementへ進み、逆に成功していた。

したがって主因はpanel schedulingやwarm cacheではなく、event location uncertaintyを緩いvalue toleranceだけで許可したprecision ladderである。warmでも同じevent機構が再現しており、warm state単独が原因ではない。

修正ではphysical endpointに対して次の局所anchor条件を追加した。

```text
ulp_budget    = max(256 * ULP(R), 64 * eps * (1 + |R|))
anchor_budget = min(ulp_budget, 0.05 * value_budget * max(1, |R|))
```

double/DDのeventを採用するには、residual・derivative条件に加えて `uncertainty <= anchor_budget` を要求する。満たさないeventだけを既存のDD/qf局所経路へ送るため、全epochのqf化や新しい重量fallbackは入れていない。rand033 event index 6 は修正後、RelTol=1e-3でも qf tier 2、uncertainty約 `2.94e-16` になった。

修正後のcoverageは以下の通り。

| policy | cold 1e-3 | cold 1e-4 | warm 1e-3 | warm 1e-4 |
|---|---:|---:|---:|---:|
| None | 110/110 | 110/110 | 110/110 | 110/110 |
| ValueFirst | 110/110 | 110/110 | 110/110 | 110/110 |

ValueFirstのdiagnostic fixtureでは、valueは全行で収束し、gradientは独立した小budgetの範囲で判定された。gradient tolerance未達をvalue failureへ昇格する変更はしていない。`FiniteUncertified` と `Invalid` の意味も維持した。Strictは従来のgradient品質制約を維持し、value convergenceだけを緩めていない。

詳細な修正後rawは次にある。

- `evidence/holonomic/adaptive_radial_phase91_20260911/diagnostics/epochs.csv`
- `evidence/holonomic/adaptive_radial_phase91_20260911/diagnostics/events.csv`
- `evidence/holonomic/adaptive_radial_phase91_20260911/diagnostics/refinements.csv`
- `evidence/holonomic/adaptive_radial_phase91_20260911/diagnostics_before/`

修正後diagnosticは16 epoch行、128 event行、304 refinement行を記録し、case/u/policy/warm/tol、stop、value stop、mu、error、node/evaluation数、event precision tier、double/DD/qf理由、radius/uncertainty、panel/refinement履歴を含む。集計は `summary.json` の `phase91_diagnostics` にある。

## stage timing の回帰

`bench_adaptive_radial.cpp` のCSV headerは、panel後に8個のsize counterを宣言していたが、format stringは `%zu` を9個並べていた。そのため次の `double physics_ms` を `%zu` で読み、`physics_ms` に桁違いの値が出ていた。

修正前rawに対して新checkerを実行すると、4400/4400行が `stage_scale` で検出された。代表例は次の通り。

| 状態 | whole_ms | physics_ms |
|---|---:|---:|
| 修正前 `paired.csv` | 0.967630 | 1,943,936,557 |
| 修正後 `plan15` value-only cold | 0.951779 | 0.132369 |

修正後の `paired.csv` では、各行について有限・非負、各stageが `10 * whole + 0.1 ms` 以下、stage sumが `2 * whole + 0.1 ms` 以下であることを検査した。結果は次の通り。

```json
{
  "rows": 4400,
  "finite_rows": 4400,
  "bad_rows": 0,
  "passed": true
}
```

checkerの結果は `evidence/holonomic/adaptive_radial_phase91_20260911/timing_sanity.json`、修正前の検出結果は `timing_sanity_before.json` に保存した。`summary.json`にも同じ結果を埋め込んでいる。

## 検証

- `test_adaptive_radial`: **1383 checks, 0 failures**
- relevant CTest: **10/10 passed**（point images、quartic/Sturm、M7/reference、transport、finite-source、root-pair、ODE、chart、reciprocal chart、adaptive）
- reference: cold 550行 / warm 550行、observed violation **0 / 0**、reference unusable 4 / 4
- contract raw: 1320行、None / ValueFirst の value coverageは全4条件で **110/110**
- timing sanity: **4400/4400 finite、bad 0**

再現コマンド:

```bash
bash checks/holonomic/run_adaptive_radial.sh \
  evidence/holonomic/adaptive_radial_phase91_20260911 1
python checks/holonomic/check_adaptive_timing.py \
  evidence/holonomic/adaptive_radial_phase91_20260911/paired.csv
python checks/holonomic/summarize_adaptive_radial.py \
  evidence/holonomic/adaptive_radial_phase91_20260911
```

production router、fixed-`n_r` production path、PF6/GM research pathは変更していない。速度最適化やrouter昇格はこのcheckpointの判定対象外で、今回の変更はadaptive valueの単調性とbenchmark計測の信頼性を戻すことに限定した。
