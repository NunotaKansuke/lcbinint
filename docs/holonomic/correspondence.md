# 既存コードとの対応表 (ATPT holonomic solver)

作成 2026-09-07 / branch `claude/holonomic-solver-m1`

## 0. 重要な発見: 設計書が前提とする baseline は本リポジトリに存在しない

`implementation_plan_ja.md` は「既存の algebraic-boundary solver
(`algebraic_boundary.cpp` / `algebraic_boundary_autodiff.cpp`, commit `54781b3`)」に
対して A/B するとしているが、本 worktree の全履歴を検索しても該当ファイル・
シンボル・コミットは **存在しない**。設計書自身も 1.3 節で
「参照された相対パスと短縮コミット 54781b3 は取得できなかった」と認めている。

したがって A/B 比較の baseline は、実在する有限光源経路
`src/lcbinint/magnification/finite_source_magnifier.cpp` (inverse-ray /
hexadecapole, ~350 KB) とする。設計書の "algebraic-boundary" という語は
以後「現行 production 有限光源ソルバ」と読み替える。

## 1. 接続点マップ

| 設計書のモジュール (13節) | 役割 | 既存コードの対応 | 備考 |
|---|---|---|---|
| `polynomial_family.hpp` | 境界四次 P, T, B の係数生成 | 新規。既存に境界四次の生成器は無い | root solver の正規化を流用しない |
| `radial_events.hpp` | D14 実根列挙, 追加イベント | 新規。既存「band search」は `finite_source_magnifier.cpp` 内の適応 radial 格子 (`caustic_bins`, `source_bins`) | |
| `topology_cells.hpp` | 円弧トポロジー分類 | 新規。`component_certificate.{hpp,cpp}` が連結成分の証明を持つ — incidence graph の参考 | |
| `angular_charts.hpp` | t=u/v チャート | 新規 | |
| `period_reduction/connection_builder/period_seed/taylor_packet` | 周期輸送コア | 新規、既存対応なし | |
| `root_pair_transport.hpp` | (m,v) 一様項輸送 | 新規。点光源根は `point_source_magnifier.cpp` (5次) | |
| `singular_patch.hpp` | 特異セル | 新規 | |
| `flux_and_jacobian.hpp` | F0,F1/2,gradF 組立 | 新規。LD 積分の現行は `finite_source_magnifier.cpp::limb_darkening_table_brightness` + inverse-ray grid | c,d 係数 (2次 LD 則) |
| `diagnostics.hpp` | ステータス/counters | 既存 `FiniteSourceResult` (`finite_source_magnifier.hpp:96`), `probe_diagnostics.hpp` | 新 `EpochResult` を versioned に追加 |
| `solver.cpp` エントリ | FFI | `python/bind_lc.cpp`, `python/bind_jax_ir.cpp`; JAX 経路 `python/lcbinint/jax_backend.py` | versioned new entrypoint |

## 2. autodiff / Jacobian

- 設計書は `ForwardJet<5>` を前提とするが、本リポジトリに `ForwardJet` は無い。
  微分経路は JAX 側 (`python/lcbinint/jax_backend.py`, `python/lcbinint_jax/`) と
  C++ 側の解析的 Jacobian が混在。5成分 = `(xs, ys, rho, q, a)` は
  `jax_backend.py:163` 付近のパラメータ順と一致。
- 新 solver の初期版は C++ 内に固定長 `Jet<double,5>` を自前で用意する。
  value lane が返却値と一致することを契約とする。

## 3. LD (limb darkening)

- 現行: `FiniteSourceSettings.limb_darkening_c/_d` — 2次(平方根+線形)則。
  設計書の初期スコープは linear LD (`u`, sqrt モーメント F_{1/2}) のみ。
- 対応: 新 solver は `F0`, `F_{1/2}` を返し、
  `mu_u = ((1-u)F0 + u F_{1/2}) / (pi rho^2 (1-u/3))`。

## 4. テスト/ベンチ基盤

| 種別 | 既存 | 新規配置 |
|---|---|---|
| unit | `tests/unit/` | `tests/holonomic/` |
| regression | `tests/regression/` | `tests/holonomic/test_reference_grid.py` |
| diagnostics/sweep | `tests/diagnostics/recal2026/` | `benchmarks/holonomic/` |
| JAX IR | `tests/jax_ir/` | `tests/holonomic/test_value_jvp_consistency.py` |

## 5. ビルド

- `CMakeLists.txt` + `Makefile` + `pyproject.toml`。
  新ヘッダは header-only で追加し、M1-M4 は Python reference 実装で進め、
  C++ は M4 以降。
