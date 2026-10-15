# 近接 caustic 接触での勾配追跡 — 2026-10-15

作業場所は `/rogue1_8/nunota/lcbinint/experiments/algebraic_boundary`、ブランチは `experiment/algebraic_boundary`、開始時 HEAD は `1d6984f7501066f22ff0143d69c1cd1c45b04d06`。既存の未コミット変更とノートブックは保持した。

## 修正した不具合: case_012

設定は `s=1.000001`, `q=1e-7`, `rho=0.8448456160095342`, `u0=0`, `alpha=1.2431195471839853`, LD=0。接触時刻 `t=-rho*tE=-8.448456160095342`、`tol=reltol=1e-9`。

修正前の `dmu/drho=11.93419648` に対し、値APIの中央差分は刻みを小さくすると `12.76934` に安定した。修正後は `12.77100128` で、差分とのずれは約0.013%。差分側の各点は `value_converged=true` だが、`fully_converged=false` / `support_certified=false` のため、これは独立した数学的認証ではなく同一エンジン内の再現性確認である。

根は半径 `R=5.9182426631757277e-8` の物理折り返しイベント。イベントの停留角 seed `t=0.716268018...` は残っていたが、root-pair座標の微分は `pair_motion_valid=false`。従来の `MaterialMap` はこの状態をイベント移動微分全体の無効と扱い、そのspanだけ端点を固定していた。今回、停留角seedがあれば `phi=0` の暗黙微分 `dR/dp=-phi_p/phi_R` を計算し、同時に既存の `phi` と `phi_theta` 残差ゲートを通すようにした。イベント位置の差分検証は別の `event_fd.cpp` にある。

## 追加スクリーニング

ストレス集合から14個の厳密接触条件を拾い、`dmu/drho` と同一値APIの中央差分を比較した。case_058、064、088、114は細かい刻みでよく一致。case_012は上記の通り修正後に一致した。case_001、022、045、097、098などの一部はその高精度条件で値積分が未収束のため、差分を正解とは扱わない。

### case_033 regression recheck (2026-10-16)

The historical stress artifact reported `dmu/drho=-150.42` at `t=-0.15` and `+110.93` at `t=+0.15`, while strict finite differences were `+1.1235` at both epochs. With the current public API build, the gradients are `+1.123497` and `+1.124066` at each of `tol=1e-3,1e-5,1e-7`; strict same-engine differences over three steps span `1.123475..1.123521` and `1.124057..1.124075`. These agree with the independent source-plane reference in `stress_cause_20261015/source_plane.json`. The prior diagnosis localized the old error to derivative replay on an unsplit depth-zero radial panel; the existing replay-gate correction in `benchmarks/holonomic/fused_gradient/build_geometry_overlay.py` is consistent with this regression disappearing. No pre-fix binary A/B was rebuilt in this turn, so attribution is strong but not a new controlled A/B. `support_certified=false` remains; this is a regression check, not a formal support certificate. `case33_fd.py` reproduces the current public-API check.

### Formerly invalid symmetric rows: case_004 and case_049

The saved stress arrays had all seven gradient components NaN at `t=0` for both cases. In the current public API build (`_lcbinint` SHA-256 `b6ba4c1a395452784ac1a287986a2cc8dcd9372b32c13405775fc7a02499258d`), all seven components are finite and marked usable. At `tol=1e-3`, `dmu/drho` is `0.0332534706` for case_004 and `0.0023480176` for case_049. Same-engine central differences at `tol=1e-7` agree to better than `5e-7` relative over three source-radius steps; the rows remain `support_certified=false`. This verifies the rho component and confirms the old NaN no longer occurs in the current build; it does not establish full gradient accuracy for every component. `case004_049_fd.py` reproduces the check.

### case_025 の修正と残る精度候補

- **case_025 (修正済み)** (`s=.75602239`, `q=1.12897e-4`, `rho=.14869772`, LD=.5): 幾何ペアの存在だけで depth-0 パネルまで Gaussian replay していたため、隣接する広いspanのrho寄与が `694.82958` から `698.92706` に変わり、値を保ったまま勾配に約3.2%のずれを作っていた。`build_geometry_overlay.py` のreplay条件を実際に値適応で分割されたパネルに限定した。高精度の同一エンジン差分（`tol=1e-5`, 刻み比 `1e-7,1e-8,1e-9`）は `128.24465998,128.24465903,128.24466739`。修正後の勾配は `tol=1e-3` で `127.92594`（修正前 `132.34222`）、`1e-4` で `128.13397`、`1e-5` で `128.24462930` と収束し、差分に対するずれはそれぞれ約0.25%、0.086%、2.4e-7。`tol=1e-3` の同じ粗い値APIを差分すると `150.26` となった（両側の値収束フラグはtrueだが、細かい値許容誤差で得た差分と大きく異なり、粗い値計算からの差分は勾配精度の参照にできない）。`case25_fd.py` が tolerance sweep と高精度差分を再現する。`support_certified=false` は維持しており、同一エンジン内の照合で外部精度認証ではない。
- **case_109** (`s=.18626282`, `q=5.12486e-7`, `rho=.46850782`, LD=.5): `tol=1e-8` で解析 `30.92782918`。値API差分は刻み比 `1e-9` と `1e-10` で `30.83178902`, `30.82971743` に安定（両側 `value_converged=true`）。ずれは約0.32%。小イベント2個の `dR/drho` はイベント半径を直接摂動した差分と相対 `3e-10` 以下で一致し、端点微分の欠落では説明できない。初期半径次数を3から6に上げても勾配は `30.92782917` のまま。近接複素根 span の固定Gauss則も64点から128点に増やしたが `30.92782918` と不変（評価数4313→4889）。双曲線座標を外す診断では `31.19999` へ悪化した。従って単純なGauss次数不足や座標変換の有無だけでは説明できない。Strict勾配適応は32,768-node上限で `BudgetExceeded` となり、通常経路の代替として不適切。
- **case_117 follow-up (2026-10-16)**: At `t=-rho*tE=-1.671007233467463`, `u0=1e-9` puts the source center effectively at the source-plane origin when the source limb passes through it. The case remains `support_certified=false`. Public-API gradient values at `tol=1e-3,1e-4,1e-5,1e-6` are `dmu/dt0=(-18.4818,-21.6389,-21.3550,-21.3550)`, `dmu/drho=(140.9901,172.5695,169.7276,169.7276)`, and `dmu/dq=(33.9735,64.4754,60.9369,60.9369)`. At `tol=1e-7`, finite differences for `q` stabilize near `60.9742` across relative steps `1e-2` to `1e-3`, a 0.06% difference from the tighter analytic gradient. For `t0` and `rho`, decreasing steps move the finite differences toward the `tol=1e-5` gradients, but the values continue to drift; all tested sides report `value_converged=true` and `support_certified=false`. The `tol=1e-3` gradient differs substantially from the tighter result (about 13–17%). This is a near-contact stress candidate, not a confirmed implementation bug: the same-engine reference is not support-certified and its step study is not yet asymptotic. `case117_fd.py` reproduces the public-API tolerance/step study. No source fix is claimed.

これらの結果は、勾配を見てtolを自動調整する根拠にはしない。特にcase_109では小さなradial event間隔が見えるが、必要な追加処理の形はまだ確定していない。

## 検証

- `bash benchmarks/holonomic/fused_gradient/build_algebraic.sh` 成功。
- `build-algebraic-boundary/fused-gradient/api_check` 成功。通常APIとの共存、解析極限、無効入力を確認。
- `git diff --check` 成功。
- `case12_fd.py` と `case25_fd.py` が、それぞれ修正後の接触勾配と中央差分を公開APIで再計算する。C++イベント診断のソースは `event_probe.cpp` と `event_fd.cpp`。

`gradient_valid` は有限で使用可能な出力を示し、勾配精度保証ではない。該当行は `support_certified=false` なので、この検証結果だけで科学的受理や全条件での精度保証を主張しない。
