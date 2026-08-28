# lcbinint algebraic-boundary: 値と JVP の整合化（MB21333 HMC 発散対策）

対象ブランチ: `feature/algebraic-boundary-autodiff`（`algebraic-boundary-cpp`
から分岐、master へマージ・origin へ push はしない）。

autodiff-vbm 側の同種修正
(`/rogue1_8/nunota/autodiff-vbm/docs/value-jvp-consistency-fix.md`, MB21333
レポート設計案 (3)) を lcbinint に持ち込んだ記録。

---

## 1. 何が問題だったか

HMC の勾配は `python/lcbinint_jax/cpp_backend.py` の
`_algebraic_boundary_epoch_jvp` が返す。この JVP は **Jacobian FFI
(`lcbinint_algebraic_boundary_autodiff_jacobian_f64_v1`) だけ** を呼び、
その戻り値の `magnification` を primal、`jacobian` 行を tangent として返す
（forward FFI は JVP 経路では呼ばれない）。

Jacobian FFI (`algebraic_boundary_epoch_jacobian_ffi_impl`,
`bind_jax_ir.cpp:5579`) は
`FiniteSourceMagnifier::experimental_algebraic_boundary_binary_jacobian` の
`AlgebraicBoundaryJacobian` をそのまま流す。その `.magnification` は、
2 つのドライバのどちらでも **「オーソリテイティブなプライマル solve の値」**
だった:

- deep: `experimental_algebraic_boundary_jacobian` —
  `out.magnification = primal.magnification;`（別の adaptive パネル分割
  quadrature を通った値）。derivative は別に `ForwardJet<5>` 再構成
  (`mag_j`) を frozen 半径バンド上で走らせて作る。
- multipole: `experimental_algebraic_boundary_multipole_jacobian` —
  `out.magnification = authoritative_magnification;`。derivative は
  `binary_point_mag_jet` / hexadecapole Jet コンボ (`mag_j`)。

つまり `out.jacobian = d(mag_j)/dp` なのに `out.magnification ≠ mag_j.value`。
caustic 近傍では primal quadrature と ForwardJet 再構成が別の有効関数に
なり、`J_h ≠ d(value)/dθ`。**値は滑らかなのに微分が飛ぶ** → Hamiltonian
error → HMC divergence（尤度バイアスではない）。autodiff-vbm と全く同型。

---

## 2. どう直したか

`src/lcbinint/magnification/algebraic_boundary_autodiff.cpp` の両ドライバで、
**返す `out.magnification` を「微分している再構成そのものの値」
`mag_j.value` に差し替えた**（`std::isfinite && > 0` でガード、
使えないときだけ従来のプライマル値にフォールバック）。tol / retol /
grad_tol は一切変更なし。

1. `experimental_algebraic_boundary_jacobian`、`if (out.grad_reliable)` 内:
   Jacobian コピー直前に `out.magnification = mag_j.value;`。
   grad が unreliable のときは行が NaN なので値は従来どおりで無害。
2. `experimental_algebraic_boundary_multipole_jacobian` の
   非 hexadecapole（point-source）枝: `binary_point_mag_jet` の直後に同処理。
3. 同 hexadecapole 枝: `mag_j` 合成の直後に同処理。

forward FFI (`algebraic_boundary_epoch_forward_ffi_impl`) は JVP 経路に
現れないので変更不要。ルーティングプローブ
(`experimental_algebraic_boundary_binary_jacobian`,
`finite_source_magnifier.cpp:6152`) はドライバ結果を無改変で返すので、
ドライバを直せば全経路で一致する。

### テスト

`tests/unit/test_algebraic_boundary_autodiff.cpp` check 4 / 5 の `mag_at`
ラムダを、`experimental_algebraic_boundary_binary_mag`（プライマル値経路）
から `experimental_algebraic_boundary_binary_jacobian(...).magnification`
（Jacobian ドライバの報告値）へ変更。**Jacobian がいま微分している値そのもの
を中心差分する**形。許容値は不変（check 4: `1e-4*(1+|fd|)+1e-6`、
check 5: `2e-4*(1+|fd|)+1e-5`）。

---

## 3. 検証（値の変更なので数値のみ、HMC はユーザ側）

- `test_algebraic_boundary_autodiff`（CTest `unit_algebraic_boundary_autodiff`）
  全行 `ok` → "all autodiff-core checks passed"。check 4/5 の caustic 近傍行
  は ~1e-4（`|fd|` にスケールする相対許容内、再構成自身の値対微分の残差 =
  FD step ノイズであって値/JVP の分裂ではない）。clear ケースは
  ~1e-10..1e-12。
- `test_algebraic_boundary`（プライマル）green。
- `tests/jax_ir/test_algebraic_boundary_epoch.py` 19 pass + 1 skip
  （小 rho=0.008 + two-coeff LD 行のフラグ → jvp NaN、契約どおり）。
- 広域回帰 `tests/jax_ir/`: **260 passed, 1 skipped**（14m26s）。

---

## 4. 残っている限界

- **物理精度**: 返す値が frozen-support / stencil 再構成側になったので、
  adaptive パネル分割ぶんの精度は手放した方向。整合が取れたので leapfrog
  は安定するはずだが、事後分布バイアスは tol/retol で管理。
- **非収束 caustic epoch**: 従来どおり Jacobian 行 = NaN（honest refusal）。
  値/微分整合とは別問題。
- forward FFI の `magnification` は依然プライマル solve 値。JVP では未使用
  なので HMC には無害だが、`binary_algebraic_boundary_polar_ffi` を
  微分なしで呼ぶ診断コードは forward と JVP で僅かに違う値を見る。

---

## 5. autodiff-vbm レポート section 5 の 3 チェックへの回答

1. primal が微分している輪郭 walk と同一経路か → **NO だった。両ドライバとも
   別のプライマル solve 値を返していた。** → `mag_j.value` に差し替えて解消。
2. grad_tol 由来の追加細分後の輪郭に値も乗るか → deep ドライバは
   `mag_j` が frozen 半径バンド上の ForwardJet 再構成。その `.value` を
   返すので定義上一致。
3. point / hexadecapole ショートカットで値と微分が別枝に落ちないか →
   multipole ドライバの両枝とも `mag_j.value` を返すよう修正済み。
   ルーティングプローブはドライバ結果を無改変で通す。
