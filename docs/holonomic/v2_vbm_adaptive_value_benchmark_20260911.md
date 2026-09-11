# V2 adaptive value-only benchmark checkpoint — 2026-09-11

前回の固定 `n_r=64` pure-kernel結果ではなく、現行V2の adaptive value path を同じ論文用入力へ適用した。対象は既存VBM benchmarkの1,804 geometry/profile/epoch行で、各行について VBM speed target `RelTol=1e-3` と `1e-4` をV2の `mu_rtol`へ対応させ、uniform/linearの4条件、合計14,432 timed V2 callsを作成した。

V2は `flux_adaptive_integrate(p,u,topo,cfg,workspace)` を測定した。`LensParams`、`PrimaryFrame`、`classify_cells`（D14/topology）、入力処理、出力処理はtimer外、adaptive setup、event handling、panel refinement、physics、estimator、schedulerはtimer内である。`mu_atol=1e-16`、gradient policy `None`、Jacobianなし、workspace warm-up後3回の中央値を使った。従って固定64ではなく、要求RelTolに応じてnode数を決めるvalue-only測定であり、既存VBMのwarm direct-kernel timingと同じpure-kernel境界である。full epochのD14/topology時間は含めていない。

| profile | target | value-converged | V2 p50 ms | V2 p95 ms | median VBM/V2 |
|---|---:|---:|---:|---:|---:|
| uniform (LD off) | 1e-3 | 3451/3608 (95.6%) | 0.1166 | 0.2601 | 0.535 |
| uniform (LD off) | 1e-4 | 3447/3608 (95.5%) | 0.1480 | 0.3578 | 0.753 |
| linear (LD on, c=0.5) | 1e-3 | 3451/3608 (95.6%) | 0.1555 | 0.3592 | 2.076 |
| linear (LD on, c=0.5) | 1e-4 | 3447/3608 (95.5%) | 0.1917 | 0.5027 | 7.049 |

速度比は `VBM selected_seconds * 1000 / V2 ms` で、1より大きいとV2が速い。LDなしでは今回のadaptive pathはVBMより遅く、LDありでは速い。`RelTol=1e-4`のuniformだけは中央値で0.753倍、linearでは7.049倍だった。

既存VBM `RelTol=1e-6` referenceとの相対誤差は、value-converged行で次の通りだった。

| profile | target | p50 | p95 | max |
|---|---:|---:|---:|---:|
| uniform | 1e-3 | 6.56e-8 | 3.32e-6 | 1.15e-5 |
| uniform | 1e-4 | 2.71e-8 | 3.11e-7 | 2.71e-4 |
| linear | 1e-3 | 8.08e-7 | 9.22e-6 | 2.18e-4 |
| linear | 1e-4 | 6.42e-7 | 7.82e-6 | 2.18e-4 |

全体で636行がvalue non-convergedになった。これらはすべて `topology_status=OK`、`topology_uncertain_cells=0`、node/panel生成済みであり、topology classifierの失敗ではない。adaptive sample evaluatorの`reliable=false`が現在のAPIで`TopologyUnresolved`へ集約されている。したがって、このベンチでは636行を成功扱いへ変えていない。より細かい内部reject理由は現行evaluatorのraw schemaにないため、今回の結果からは個別原因を推定していない。

この636行のうち624行（98.1%）は、topologyのcell planに少なくとも1つ`kFull` cellを持つgeometryだった。`kFull`はその半径で円周全体が有効で、arcの端点が存在しない状態である。固定V2の`radius_terms()`には`full_circle_terms()`処理がある一方、adaptiveの`mapped_radius()`は端点arcを前提にしており、`kFull`/`kDegenerate`を`reliable=false`として停止する。このため大きな`rho`領域の斜線は、value estimatorの収束不足というよりadaptive側のfull-circle表現未対応を示している。残り12行はtiny-source/near-axisの2 geometry（`case_id=9, d_bin=2, epoch=7` と `case_id=92, d_bin=0`）で、現行evaluatorが内部arc guardの細分類を出していないため、個別reject条件は未確定である。詳細は`topology_diagnostic.tsv`と`arc_reliability_diagnostic.tsv`に保存した。

成果物は [REPORT.md](../../evidence/holonomic/v2_vbm_adaptive_phase92_20260911/REPORT.md)、[summary.json](../../evidence/holonomic/v2_vbm_adaptive_phase92_20260911/summary.json)、[joined_v2_vbm.tsv](../../evidence/holonomic/v2_vbm_adaptive_phase92_20260911/joined_v2_vbm.tsv)、[v2_results.tsv](../../evidence/holonomic/v2_vbm_adaptive_phase92_20260911/v2_results.tsv) と `figures/` に保存した。VBM speed sourceと1e-6 referenceは前回ベンチの固定artifactを再利用し、今回追加計算はV2 adaptiveのみである。
