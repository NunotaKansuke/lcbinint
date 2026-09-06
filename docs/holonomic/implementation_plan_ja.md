---
title: "連星有限光源の代数トポロジー・周期輸送ソルバ"
subtitle: "実装計画書 | 半径候補の完全性、周縁減光、微分、性能検証"
date: "2026-09-06 / version 1.0"
lang: ja-JP
---

> このファイルはユーザー提供の設計仕様 (source of truth) の要約である。
> 実装で判明した矛盾・誤り・修正は本ファイルを書き換えず、
> `docs/holonomic/design_addendum.md` と各 `checkpoint_M*.md` に記録する。
> 完全な逐語コピー (LaTeX 込み) は `implementation_plan_ja.SOURCE.md`。

# 要点

- **スコープ (節1)**: 連星・円形有限光源・uniform + linear LD。
  `p = (xs, ys, rho, q, a)`。レンズ 0 と a、質量 `m0`, `m1 = 1 - m0`。
  `mu_u = ((1-u) F0 + u F_{1/2}) / (pi rho^2 (1 - u/3))`。
- **境界四次 (節2)**: `f(z) = z - m0/zbar - m1/(zbar - a)`、
  `phi = 1 - |f - zeta|^2 / rho^2`。
  `P(t;R) = rho^2 R^2 A(t) B(t;R) - T(t) Tbar(t)`、`phi = P / (rho^2 R^2 A B)`。
  `A = 1 + t^2`、`B = (R-a)^2 + (R+a)^2 t^2`、
  `n0 = -zeta R^2`、`n1 = R(R^2 - 1 + a zeta)`、`n2 = a(m0 - R^2)`、
  `T = n0 (1-it)^2 + n1 (1+t^2) + n2 (1+it)^2`。
- **半径候補 (節3)**: `Disc_t P = R^4 D14(R^2)`、`deg_v D14 <= 14`。
  追加イベント `R = a`、`R = sqrt(m0)`。
  `Res(P,A) = 256 a^2 R^4 (m0 - R^2)^2 |zeta|^2`、
  `Res(P,B) = 256 a^2 R^4 m1^2 [L^2 + ys^2 vR^2 (vR - a^2)^2]`、
  `L(vR) = (a - xs) vR (vR - a^2) - a vR + a^3 m0`、`vR = R^2`。
  `Disc A = -4`、`Disc B = -4 (R^2 - a^2)^2`、`Res(A,B) = 16 a^2 R^2`。
  探索上限 `R_max = (a + W + sqrt((a - W)^2 + 4)) / 2`、`W = |zeta| + rho`。
- **トポロジー (節4)**: 円周交差 0/2/4、内部円弧高々2。
  境界根 0 本 = 空 or 全周 (代表角で符号判定)。`topology_complete` ステータス。
- **LD 還元 (節5-6)**: `J_{1/2}(R) = rho^{-1} oint_gamma P/(A Y) dt`、
  `Y = sqrt(Q)`、`Q = P A B`。
  `P/(A Y) dt = H/Y dt + d_t(S Y / A)`、`S = -t/(4 a R)`、
  `H = (2 (R+a)^2 P + t (B P_t + B_t P)) / (8 a R)`、`deg_t H <= 6`。
- **周期基底 (節7)**: `eta_k = t^k dt / Y` (k=0..6)。留数ゼロ6形式
  `psi = (eta0, eta1, eta2, eta4 - b1 eta3, eta5 - b2 eta3, eta6 - b3 eta3)`、
  `b1 = -q7/(2 q8)`、`b2 = -q6/(2 q8) + 3 q7^2/(8 q8^2)`、
  `b3 = -q5/(2 q8) + 3 q7 q6/(4 q8^2) - 5 q7^3/(16 q8^3)`。
  接続 `Pi_R = C6 Pi`、`J_{1/2} = rho^{-1} c^T Pi`、
  `c = (h0, h1, h2, h4, h5, h6)`。
  接続構築: `S_k = t^k Q_R (Q_t)^{-1} mod Q`、
  `T_k = (S_k Q_t - t^k Q_R) / (2 Q)`、`C_k = T_k - S_{k,t}`。
  exact identity `-1/2 t^k Q_R = C_k Q + S_{k,t} Q - 1/2 S_k Q_t`。
- **高速輸送 (節8)**: Taylor packet `(n+1) Pi_{n+1} = sum_k C_k Pi_{n-k}`。
  CRT 因子分解 `Q = P A B` (4x2x2)。flux 優先基底 `G`。
  根ペア `(m, v)`: `m = (t+ + t-)/2`、`v = ((t+ - t-)/2)^2`、
  `E = P(m) + v/2 P''(m) + v^2/24 P''''(m) = 0`、
  `O = P'(m) + v/6 P'''(m) = 0`、
  `det ∂(E,O)/∂(m,v)|_{v=0} = -1/2 P''(m)^2`。
  `Δθ = 2 atan2(2 sqrt(v), 1 + m^2 - v)`。
  通常状態数 ~10-12 (周期6 + 根ペア2-4 + フラックス2)。
- **特異点 (節9)**: 通常接点全周期級数
  `Pi_k^{(7)} -> 2 pi m_*^k / sqrt(C(m_*, R_*))` at `v -> 0`。
  `R = sqrt(m0)`、`R = a`、因子衝突は前後に正則アンカーを置き橋渡し。
- **Jacobian (節11)**: value/JVP 契約 — 返す value と Jacobian は同じ有限数値
  再構成から。`ForwardJet<5>` replay。接点位置微分は 2 変数 IFT
  (`[[P_R, P_t],[P_tR, P_tt]] [dR*, dt*]^T = -[P_p, P_tp]^T`)。
  端点微分は初期版で省略しない。
- **誤差予算 (節12)**: value/gradient 別管理。`fast` / `validated` モード。
  ステータス: `OK_ESTIMATED / OK_VALIDATED / LOCAL_REFERENCE_USED /
  EVENT_UNRESOLVED / ARC_TOPOLOGY_INVALID / BASIS_DEGENERATE /
  CONNECTION_ILL_CONDITIONED / TRANSPORT_TOLERANCE_FAILED /
  GRADIENT_UNRELIABLE / RESOURCE_LIMIT`。
- **モジュール (節13)**: `src/lcbinint/magnification/holonomic/`、`tests/holonomic/`、
  `benchmarks/holonomic/`。`Event` / `CellPlan` / `EpochResult` 構造体。
- **試作結果 (節15)**: `a = 6/5`, `m0 = 2/3`, `zeta = 1/5 + i/7`, `rho = 1/8`。
  判別式からの正候補半径 8 個 + `sqrt(2/3) = 0.8165` + `a = 1.2`。
  短区間輸送の相対差 `< 5.15e-9`、最悪線形系条件数 `~1.89e9`
  (= monomial 基底の悪条件、要対策)。
- **監査 (節16)**: `q = 1e-8..1`、`a = 0.1..10`、`rho = 1e-6..0.3`。
  caustic/cusp/軸上/`zeta=0`/`R=a`/`R=sqrt(m0)` 近傍を重点。
- **やらないこと (節17-18)**: 三重レンズ、偏平光源、任意輝度、2階微分、
  分母の下限クリップ、巨大 global Picard-Fuchs、未知角度区間の打ち切り。

# 成功基準

| 軸 | 基準 |
|---|---|
| correctness | trusted reference に対し現行 production solver 同等以上の精度と robustness |
| completeness | stress test で像バンド取りこぼしゼロ |
| differentiation | accepted points で value と Jacobian が数値的に整合 |
| performance | linear-LD value+Jacobian の end-to-end median: 対現行 backend で **2x 最低採用**、3x 強い成功、5x 最終目標 |
