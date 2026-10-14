# ATPT holonomic solver — milestone 一覧

branch `claude/lcbinint-holonomic-solver-b7d3cd`。設計書 14 節の M0–M8 を、本リポジトリの
実状 (baseline は inverse-ray `finite_source_magnifier.cpp`、`ForwardJet` 不在、
`algebraic_boundary.cpp` 不在) に合わせて具体化したもの。

各 milestone は `docs/holonomic/checkpoint_M<n>.md` に短い報告を残す:
実装内容 / 数式根拠 / correctness 結果 / benchmark 結果 / 未解決リスク。

| M | 目的 | 成果物 | 完了条件 (採用ゲート) | 状態 |
|---|---|---|---|---|
| M0 | baseline 固定・計器化 | `benchmarks/holonomic/baseline_probe.py`、`evidence/holonomic/baseline_M0.json` | 現行ソルバ cost の分解が再現可能 | **完了** (`checkpoint_M0.md`) — inverse-ray `binary_ray_shooting` の value+Jacobian (central FD ×11) cost は **強く bimodal**: median 15 ms / p90 0.12 s / **p95 4.38 s**、遅い裾 9.3% は全て extreme-q (`q ≲ 1.4e-3`, planetary caustic をグリッドで解像)。jax 微分 backend は polar epoch FFI 不在で使用不可 → baseline は native inverse-ray の有限差分。bimodal は評価データとして記録するが採用 gate は変更しない (設計判断 18, 20) |
| M1 | 係数・判別式・追加イベントの数式検証 + radial event 列挙 | `checks/holonomic/symbolic_checks.py`, `holonomic_ref/{polynomial_family,radial_events}.py`, `tests/holonomic/test_{symbolic_identities,radial_event_completeness}.py` | 全 boxed 恒等式が exact/複数特殊化で一致。dense/random/caustic stress で crossing-count のジャンプが全て列挙イベント近傍 | **完了** (`checkpoint_M1.md`) |
| M2 | radial トポロジー・セル・incidence graph | `holonomic_ref/topology.py` | 各セルの円周交差 0/2/4・内部円弧 <=2 を分類、`CellPlan` 生成、代表角符号で empty/full 判定 | **完了** (`checkpoint_M2.md`) |
| M3 | 周期還元 reference (7形式→留数ゼロ6形式→観測形式) | `holonomic_ref/period_reduction.py`, `connection.py` | 係数恒等式・留数条件・7D/6D 周期値・直接角度積分が rtol ~1e-10 で一致 | **完了** (`checkpoint_M3.md`) |
| M4 | 通常セルの輸送 + flux (Python reference) | `holonomic_ref/{transport,seed,root_pair,flux,direct_quadrature}.py` | 複数セルで F0,F_{1/2} が直接二重求積と一致。条件数記録 | **完了** (`checkpoint_M4.md`) |
| M5 | 特異パッチ + 全 epoch reference | `holonomic_ref/singular.py`, `solver.py`, `benchmarks/holonomic/audit.py`, `tests/holonomic/test_solver_audit.py` | 監査領域で精度と失敗率を別々に報告、silent miss なし | **完了** (`checkpoint_M5.md`) — 監査 36 点: 精度 median rel 3.1e-5 / p90 2.4e-4 / max 5.9e-4 (OK_VALIDATED 27点)、失敗率 8/36 = 22.2%、silent miss 0、GATE PASS。full suite 252 passed / 3 skipped |
| M6 | value/JVP 整合 (5成分 Jacobian) | `holonomic_ref/jacobian.py` (有限数値再構成 jet)、`solve_epoch(with_jacobian=True)`、`tests/holonomic/test_value_jvp_consistency.py` | 中心差分収束、チャート変更前後で整合、既存経路との勾配比較 | **完了** (`checkpoint_M6.md`) — value と grad_mu を単一の per-cell Gauss–Chebyshev 再構成から生成。`epoch_flux` 中心差分との比較: plan15/resonant 全成分 rel ≤ 1e-3、benign 小 ρ の ∂μ/∂ρ (激しい相殺) と ∂μ/∂a (~0) のみ oracle 律速で ~1e-2。independent `image_plane_flux(+evs)` Richardson とも plan15 ≤ 1.6e-4。n_r 収束 O(1/n_r²) (48→192 で 6.6e-3→4.5e-4)。IFT-θ vs IFT-t チャート整合 2.8e-14。特異パッチ / degenerate quartic / full circle は `GRADIENT_UNRELIABLE` で fail closed。test_value_jvp_consistency.py 17 passed |
| M7 | C++ production 実装 + 高速化 | `src/lcbinint/magnification/holonomic/*.hpp` + binding `_lcbinint_holonomic_m7`、standalone `tests/holonomic_cpp/CMakeLists.txt` | 同じ精度・被覆で linear-LD value+Jacobian の end-to-end **median >= 2x faster**、かつ p95/p99 non-regressing (理想的には大幅高速化)、analytic Jacobian 品質 (設計判断 20) | **完了** (`checkpoint_M7.md`) — best-of-200 / `taskset -c 0-7` / load ~13、108 (config,u) 点: **median 4.77 ms** (incumbent 14.89 ms、3.1×、gate <= 7.44 ms PASS)、p90 8.15 / p95 8.37 / p99 8.44 ms (incumbent 145 / 4392 / 4628 ms、~18–548× 改善、non-regress PASS)。μ は M6 Python と bit-identical (OK 104 点)、status 一致 (108 点、fail-closed 4 点 = rand008/rand031 の near-tangency・extreme-q、M6 と同一)、grad_mu は M6 の oracle bar 内 (median 5.4e-10)。10397 checks / 0 failures。D14 event solve の silent-failure (wide binary で μ=0 を status OK で返す) を発見・修正 (monomial-basis balancing + NaN-safe residual gate)。設計判断 21 参照 |
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

12. 端点微分は chart-free な IFT-in-θ 形 `∂θ*/∂P = -(∂φ/∂P)/(∂φ/∂θ)` を採用。
    `t = tan(θ/2)` 形は代数的に同一だが θ*=π に可除極を持つため cross-check
    専用 (`checkpoint_M6.md` §2.3)。
13. 各動径ノードの arc 端点は `boundary_quartic(R)` の実根 (`np.roots`) から
    列挙する。`topology.arcs_at` の角度グリッド走査は tangency セル端近傍の
    薄い newborn arc を取りこぼし ∂F0/∂P に ~10% 誤差 (value は ~1e-4 で不感)
    (`checkpoint_M6.md` §3.1、設計判断 13)。
14. jax A/B 勾配経路は polar epoch FFI 不在で使用不可 → native
    `binary_ray_shooting` の有限差分を A/B 相手にする (~1e-2、粗い)。
15. `∂μ/∂u = πρ²(F_{1/2} - 2F0/3)/D²` の分子は u に依らない → flux 一組で
    全 u の `∂μ/∂u` が得られる。
16. `solve_epoch(with_jacobian=True)` は value/JVP 整合のため jet 再構成の
    `F0`/`F_{1/2}`/`μ` を採用 (plan §11)。この再構成は adaptive-quad `epoch_flux`
    と ~1e-4 (最悪の小 ρ 構成で 2e-3) で一致 (`checkpoint_M6.md` §3.6)。
17. `∂μ/∂ρ` は `((1-u)dF0+u dF_{1/2})/D` と `2μ/ρ` (各 O(1/ρ)) の差で
    小 ρ で激しく相殺する。n_r=64 で絶対精度 ~1e-2、O(1/n_r²) で収束。
    C++ jet では `∂(F/ρ²)/∂ρ` を直接組む (`checkpoint_M6.md` §5-1)。

18. M7 baseline は native inverse-ray `binary_ray_shooting` の central FD
    (jax 微分 backend は `polar_epoch_directional_ffi` 不在 + 共有 .so の
    再ビルド禁止で使用不可)。実測で incumbent の value+Jacobian cost は
    強く bimodal (median 15 ms / p95 4.38 s、遅い裾は extreme-q のみ)。
    この bimodal 性は **評価データ** として p90/p95/p99 を必ず併記する
    根拠になるが、採用 gate は plan §14 の「median ≥ 2x」を維持する
    (設計判断 20 で確定、`checkpoint_M0.md` §4 の再フレーム提案は撤回)。
19. M7 C++ は `build-holonomic-m7/` に直接 cmake でビルドし
    `sys.modules["lcbinint._lcbinint"]` 事前投入で読み込む。共有
    `site-packages/_lcbinint*.so` は絶対に上書きしない
    (`project_editable_install_serves_stale_so.md`、`checkpoint_M7.md` で詳細)。

20. **M7 gate 再フレームの撤回** (ユーザ指示 2026-09-07)。M7 の目的は
    bounded-tail 化だけでなく、通常の binary-scale regime を含む
    end-to-end 高速化。採用条件は:
    (1) median >= 2x faster than incumbent (元の採用条件、維持);
    (2) p95/p99 non-regressing、理想的には大幅高速化;
    (3) 同一 accuracy / failure coverage;
    (4) analytic Jacobian 品質。
    Python reference が 0.4 s、C++ 見積もりが 3–12 ms であることを理由に
    目標を下げない。M7 の仕事はその 3–12 ms をさらに削り、通常ケースでも
    既存 ~2 ms を明確に下回る設計にすること。必須最適化対象:
    D14 event solve/certificate の amortization; seed/re-anchor 削減;
    Gauss–Manin connection matrix の precompute / factor reuse;
    full-cell transport 安定化; fixed-size SIMD 化;
    heap allocation / dynamic dispatch 排除; branchless hot path;
    trajectory 内 geometry reuse / warm-start;
    value+Jacobian の fused evaluation。
    median 2x 未達の場合は目標変更ではなく、profiling を分解して支配項を
    特定し M8 最適化候補を具体提示する。

21. **D14 two-stage solve の silent-failure 修正** (2026-09-08、`checkpoint_M7.md` §4)。
    M7 の D14(v) 判別式求解を「balanced double Aberth 事前探索 → 113-bit
    warm polish → residual/finiteness-gated cold __float128 fallback」の
    二段構成にした。当初の plain double 探索は wide binary (`a` 大、係数
    spread ~1e11) で `|z|^14` が double range を超えて全 NaN を返し、
    `worst > 1e-12` gate が `NaN > 1e-12 == false` で cold fallback を
    起動せず、全 D14 event を silent に落として **μ=0 を status OK で
    返した**。修正: (a) `v = s·w`, `s = |a_n/a_0|^{1/deg}` で monomial
    basis を balancing し Cauchy bound を O(10) に保つ (fast path 維持、
    p99 8.4 ms); (b) seed の `std::isfinite` チェックと `!(worst <= tol)`
    形式の NaN-safe gate。standing 指示「fail closed、silent
    approximation 禁止」に直接対応。

22. **Phase 2 — 3-way 比較 (algebraic-boundary / holonomic M7 / inverse-ray)**
    (ユーザ指示 2026-09-08、`checkpoint_algebraic_vs_holonomic.md`)。
    M7 の decision-20 gate は **inverse-ray baseline 比** で達成 (median
    4.78 ms vs 14.89 ms = 3.1x)。本来の比較対象であるローカル
    algebraic-boundary 実装 (`algebraic-boundary-ld-moment-recurrence`
    @ `cc5e55d`) を **同一 config・同一 frame (Mapping B)・同一 LD 規約・
    同一 trusted reference (M0 μ)・analytic-vs-analytic Jacobian** で
    同じ harness (`tests/holonomic_cpp/bench_three_way.cpp`) に載せた。
    共有 `.so` は触れず、detached worktree `algebraic-bench-cc5e55d` の
    private static-lib build を使用。central FD を algebraic 側の本番速度
    として扱っていない (analytic forward-mode `ForwardJet<5>` を使用)。
    結論 (holonomic の速度は **inverse-ray baseline 比** としてのみ記載):
    - algebraic は bimodal。74/108 は `binary_mag` の routing が
      multipole shortcut を選び boundary integral を回さない
      (~0.05 ms、精度は exact hexadecapole)。32/108 のみ deep solve。
    - **value-only: algebraic が明確に速い** (deep-solve ~1 ms vs
      holonomic ~7 ms)。holonomic に value-only path は無い
      (5-Jacobian は同じ radial pass に fuse)。
    - **value+Jacobian median (ordinary-binary deep-solve): algebraic が
      やや速い** (~5.2 ms vs ~7.2 ms、約 1.4x)。
    - **value+Jacobian tail: holonomic が bounded で勝つ** (p95 8.5 vs
      10.5 ms、max 8.5 vs 12.5 ms)。node budget 固定のため。
    - **Jacobian 可用性: holonomic が決定的に上** (reliable 96% vs 57%。
      algebraic は near-caustic deep-solve でこそ fail-close する)。
    - **robustness: holonomic が上**。algebraic は `rand002` (wide planet,
      q~1e-4) で **非終了** (無限 allocation、return も fail も返さない
      = fail-closed policy 違反)。holonomic は同 config で status OK。
    - **accuracy: 両者とも modelling tolerance 内**。algebraic は median
      ~3x tight。holonomic は `tiny-rho` (rel 7.7e-3)・`very-wide`
      (rel 4.8e-3) で ~0.5-0.8% bias、status は OK (楽観的) → M8 で
      tolerance/status を締める follow-up。
    - **wall-clock 支配項 (fresh 実測)**: holonomic は `classify_cells`
      (arcs_at(3072) probe ×3/cell + radial_events) が **58-83%**。
      M7 §7 の ~30% 見積もりを **改訂**。per-cell radial×angular pass が
      17-42%。D14 solve ~6%、chain rule <4%。
    - **M8 目標 (well-posed)**: algebraic-boundary の ordinary-binary
      **deep-solve median** に明確に勝つ (7.2 → <5 ms、理想 <3 ms) かつ
      p95/p99 bounded-tail と Jacobian 可用性の優位を維持。優先:
      `arcs_at(3072)` / `classify_cells` (quartic_topology 置換 +
      trajectory-level cache) → radial×angular の SIMD 化。D14 solve は
      過剰最適化しない。

