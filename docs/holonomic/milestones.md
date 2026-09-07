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
| M2 | radial トポロジー・セル・incidence graph | `holonomic_ref/topology.py` | 各セルの円周交差 0/2/4・内部円弧 <=2 を分類、`CellPlan` 生成、代表角符号で empty/full 判定 | **完了** (`checkpoint_M2.md`) |
| M3 | 周期還元 reference (7形式→留数ゼロ6形式→観測形式) | `holonomic_ref/period_reduction.py`, `connection.py` | 係数恒等式・留数条件・7D/6D 周期値・直接角度積分が rtol ~1e-10 で一致 | **完了** (`checkpoint_M3.md`) |
| M4 | 通常セルの輸送 + flux (Python reference) | `holonomic_ref/{transport,seed,root_pair,flux,direct_quadrature}.py` | 複数セルで F0,F_{1/2} が直接二重求積と一致。条件数記録 | **完了** (`checkpoint_M4.md`) |
| M5 | 特異パッチ + 全 epoch reference | `holonomic_ref/singular.py`, `solver.py`, `benchmarks/holonomic/audit.py`, `tests/holonomic/test_solver_audit.py` | 監査領域で精度と失敗率を別々に報告、silent miss なし | **完了** (`checkpoint_M5.md`) — 監査 36 点: 精度 median rel 3.1e-5 / p90 2.4e-4 / max 5.9e-4 (OK_VALIDATED 27点)、失敗率 8/36 = 22.2%、silent miss 0、GATE PASS。full suite 252 passed / 3 skipped |
| M6 | value/JVP 整合 (5成分 Jacobian) | forward-mode jet, `tests/holonomic/test_value_jvp_consistency.py` | 中心差分収束、チャート変更前後で整合、既存経路との勾配比較 | TODO |
| M7 | C++ production 実装 + 高速化 | `src/lcbinint/magnification/holonomic/*.hpp` + `solver.cpp`, versioned FFI | 同じ精度・被覆で linear-LD value+Jacobian の end-to-end median >= 2x, p95 悪化 <= 25% | TODO |
| M8 | 高リスク上積み最適化 | seed-only tangency、rank-change 解析接続、epoch topology continuation | M7 通過が前提 | TODO |

## 判断済みの設計修正 (理由は各 checkpoint に記録)

1. baseline 差し替え: `algebraic_boundary.cpp` は存在しない →
   `finite_source_magnifier.cpp` (inverse-ray) を A/B 相手にする。
2. `ForwardJet<5>` → 自前 `Jet<double,5>`: 既存の同名型が無いため。
   パラメータ順は JAX backend の `(xs, ys, rho, q, a)` に合わせる。
3. Python reference を先に完成させる (M1–M6)。C++ は M7 で導入。
4. `xs = ys = 0` (原点上の対称ソース) は Q が全 R で二重根を持つ特異軌跡。
   接続は fail closed し、M4/M5 は輸送で通さず singular patch として扱う
   (`checkpoint_M3.md` §4)。
5. M4 flux は seed を 1 点輸送しない。ψ 基底の cell 条件数は最大 ~1e12
   (plan §15 の "monomial 基底の悪条件" を実測で確認) なので、reference flux
   は各動径求積ノードで周期を再アンカーする。well-conditioned な
   flux-priority 基底と単一 seed 輸送は plan 通り M7/M8 送り
   (`checkpoint_M4.md` §4)。
6. `cell_conditioning` は既定で exact sympy 接続を使う。float 版
   `connection_matrix_numeric` は θ→π 次数落ち近傍で偽の ~1e20 条件数を返す
   ため、benign-config 専用の高速プレビュー扱い (`checkpoint_M4.md` §4-5)。
7. `connection_matrix_numeric` は変数バランス (`t = c τ`, `c=(|q0/q8|)^{1/8}`)
   と次数落ち時の oracle ハンドオフ (`|q8| < 1e-6·max|q|`) で硬化。
   小 ρ で係数が 6 桁以上分散する問題への対策 (`checkpoint_M4.md` §5)。
8. M5 の value cross-check は原則 `binary_ray_shooting`。ソース中心が
   レンズ点に厳密に乗る時のみ (`min(|ζ|,|ζ-a|) < 1e-9`) 無限倍率で
   adaptive refinement が止まらないため guard して None を返す
   (`checkpoint_M5.md` §4-1)。
9. `source_plane_flux` は caustic 近傍でも disk がレンズを含む時でも
   `A_pt` 発散で収束しない。この 2 領域では corroborator を
   `image_plane_flux_grid` (有界 `√φ≥0` マスク和) に切替える。
   planetary caustic `ρ~√q` では grid も ~5e-3 ノイズがあるため
   ここは非採用領域 (`checkpoint_M5.md` §4-2)。
10. `xs=ys=0` 特異パッチは周期機構を使わず、`classify_cells` の
    セル毎に画像面 flux を直接求積 (QAWSE)。global adaptive quad は
    薄い arc band を数万回探索するため必ずセル分割し、solver が既に
    計算した `topo` を渡す (14s→6.5s、`checkpoint_M5.md` §3)。
11. linear-LD blend `μ(u) = ((1-u)F0 + u F_{1/2})/(πρ²(1-u/3))` は
    `d/du = πρ²(F_{1/2} - 2F0/3)` = u に依らず一定 → 常に単調だが
    符号は幾何依存 (caustic 近傍のソースでは u とともに増加)。
    テストは単調性 + 正値のみ主張 (`checkpoint_M5.md` §7a)。
