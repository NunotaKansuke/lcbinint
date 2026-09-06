# ATPT holonomic solver — milestone 一覧

branch `claude/lcbinint-holonomic-solver-b7d3cd`。設計書 14 節の M0–M8 を、本リポジトリの
実状 (baseline は inverse-ray `finite_source_magnifier.cpp`、`ForwardJet` 不在、
`algebraic_boundary.cpp` 不在) に合わせて具体化したもの。

各 milestone は `docs/holonomic/checkpoint_M<n>.md` に短い報告を残す:
実装内容 / 数式根拠 / correctness 結果 / benchmark 結果 / 未解決リスク。

| M | 目的 | 成果物 | 完了条件 (採用ゲート) | 状態 |
|---|---|---|---|---|
| M0 | baseline 固定・計器化 | `benchmarks/holonomic/baseline_probe.py`: 現行有限光源 (uniform + linear-LD) の median/p95、失敗率、精度 | 現行ソルバ cost の分解が再現可能 | TODO |
| M1 | 係数・判別式・追加イベントの数式検証 + radial event 列挙 | `checks/holonomic/symbolic_checks.py`, `holonomic_ref/{polynomial_family,radial_events}.py`, `tests/holonomic/test_{symbolic_identities,radial_event_completeness}.py` | 全 boxed 恒等式が exact/複数特殊化で一致。dense/random/caustic stress で crossing-count のジャンプが全て列挙イベント近傍 | **完了** (`checkpoint_M1.md`) |
| M2 | radial トポロジー・セル・incidence graph | `holonomic_ref/topology.py` | 各セルの円周交差 0/2/4・内部円弧 <=2 を分類、`CellPlan` 生成、代表角符号で empty/full 判定 | TODO |
| M3 | 周期還元 reference (7形式→留数ゼロ6形式→観測形式) | `holonomic_ref/period_reduction.py`, `connection.py` | 係数恒等式・留数条件・7D/6D 周期値・直接角度積分が rtol ~1e-10 で一致 | TODO |
| M4 | 通常セルの輸送 + flux (Python reference) | `holonomic_ref/transport.py`, `seed.py`, `root_pair.py` | 複数セルで F0,F_{1/2} が直接二重求積と一致。条件数記録 | TODO |
| M5 | 特異パッチ + 全 epoch reference | `holonomic_ref/singular.py`, `solver.py` | 監査領域で精度と失敗率を別々に報告、silent miss なし | TODO |
| M6 | value/JVP 整合 (5成分 Jacobian) | forward-mode jet, `tests/holonomic/test_value_jvp_consistency.py` | 中心差分収束、チャート変更前後で整合、既存経路との勾配比較 | TODO |
| M7 | C++ production 実装 + 高速化 | `src/lcbinint/magnification/holonomic/*.hpp` + `solver.cpp`, versioned FFI | 同じ精度・被覆で linear-LD value+Jacobian の end-to-end median >= 2x, p95 悪化 <= 25% | TODO |
| M8 | 高リスク上積み最適化 | seed-only tangency、rank-change 解析接続、epoch topology continuation | M7 通過が前提 | TODO |

## 判断済みの設計修正 (理由は各 checkpoint に記録)

1. baseline 差し替え: `algebraic_boundary.cpp` は存在しない →
   `finite_source_magnifier.cpp` (inverse-ray) を A/B 相手にする。
2. `ForwardJet<5>` → 自前 `Jet<double,5>`: 既存の同名型が無いため。
   パラメータ順は JAX backend の `(xs, ys, rho, q, a)` に合わせる。
3. Python reference を先に完成させる (M1–M6)。C++ は M7 で導入。
