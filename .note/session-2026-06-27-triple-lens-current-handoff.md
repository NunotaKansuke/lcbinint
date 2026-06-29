# 2026-06-27 triple lens handoff

## 状態

triple lens は「点源」「旧 lcbinint 座標系」「直交座標 finite source」「hexadecapole」まで入っている。
polar は後回し。adaptive は効果が薄い前提で、今後は固定 bin と速度・一致性の詰めを優先する。

主な変更ファイル:

- `src/lcbinint/model/triple_lens_geometry.hpp`
- `src/lcbinint/model/triple_lens_geometry.cpp`
- `src/lcbinint/magnification/point_source_magnifier.*`
- `src/lcbinint/magnification/finite_source_magnifier.*`
- `src/lcbinint/model/lens_model.cpp`
- `python/lcbinint_pybind.cpp`
- `tests/unit/test_core.cpp`
- `tests/regression/test_triple_lens_point_source.py`
- `tests/diagnostics/triple_reference_compare.py`
- `tests/diagnostics/triple_finite_sweep.py`

未追跡だが今回作ったもの:

- `src/lcbinint/model/triple_lens_geometry.*`
- `tests/regression/test_triple_lens_point_source.py`
- `tests/diagnostics/triple_reference_compare.py`
- `tests/diagnostics/triple_finite_sweep.py`

既存由来っぽい未追跡:

- `build_new/`
- `example/compare-vbm/.ipynb_checkpoints/`
- `example/compare-vbm/quickstart_compare_vbm.png`

これらは消さない。

## triple geometry

`make_triple_lens_geometry(s, q, q2, sep2, ang)` は旧 `lcbinint.c` の `amp_point3` / `amp_hexadecapole3` 系に合わせた。

```cpp
eps2 = q / (1 + q + q2)
eps3 = q2 / (1 + q + q2)
eps1 = 1 - eps2 - eps3
eps4 = eps2 + eps3

z1 = -eps4 * s
z2 =  eps1 * s + eps3 / eps4 * sep2 * exp(i * ang)
z3 =  eps1 * s - eps2 / eps4 * sep2 * exp(i * ang)
```

ここで `q` と `q2` は primary に対する質量比。`sep2` は secondary と third の距離、`ang` は binary 軸に対する角度。

`coordinates="vbm"` は triple ではまだ実質的な VBM convention switch ではない。内部では旧 lcbinint convention を使っている。

## point source

`PointSourceMagnifier::triple_mag0(TripleLensGeometry, SourcePosition)` を追加済み。

実装内容:

- 一般 N-lens 方程式から degree-10 polynomial を構成。
- polynomial root から lens equation residual で物理解を選別。
- Jacobian は `1 - |sum eps_i / (z - z_i)^2|^2`。
- Python 側は `LightCurve(lens="triple_lens")` の dict params で動く。
- triple では `q2`, `sep2`, `ang` が必須。
- `q2 <= 0` は `ValueError`。
- 古い positional binary-only overload で triple を使うと `ValueError`。
- parallax / dynamic triple はまだ unsupported。

旧 `lcbinint.c` の `amp_point3` との比較は `tests/diagnostics/triple_reference_compare.py` で行う。

## finite source

triple finite source は以下の選択にしている。

- caustic から十分遠い: point source
- 中間域: triple hexadecapole
- 近傍: cartesian image-plane inverse ray
- source-plane quadrature は fallback / diagnostic 寄り

hexadecapole は旧 `amp_hexadecapole3` に寄せた。

- 中心
- `rho` の 4 点
- `rho / 2` の 4 点
- diagonal の 4 点
- limb-darkening correction は binary と同じパターン

caustic 距離:

- critical curve は `sum eps_i / (z - z_i)^2 = exp(i phi)` の degree-6 polynomial。
- 6 branch を tracking。
- geometry / `caustic_bins` ごとに thread-local cache。
- segment distance と golden phase refinement を併用。

cartesian image-plane inverse ray:

- binary の Cartesian flood-fill template を mapper 型で generalize。
- triple mapper は `std::complex` を避けて real array 化。
- `map_triple_lens_real` と fast Jacobian/evaluation を追加。
- seed は以下から作る:
  - source center の point-source images
  - caustic branch candidates
  - boundary probe samples
- 現在の目安:
  - triple seed candidates max: 64
  - caustic candidates max: 12
  - boundary probe samples: 64
  - near-caustic では seeds が約 70 台まで減った

`LCBININT_AREA_DIAGNOSTICS=1` で `TRIPLE_AREA_DIAGNOSTICS` が出る。

## VBM との比較

VBMicrolensing は env 内に入っている。

- version attr: `5.5`
- source: `/home/nunota/.miniconda3/envs/myenv/lib/python3.10/site-packages/VBMicrolensing/lib/`

VBM `TripleLightCurve` の params:

```text
[log(s12), log(q2), u0, alpha, log(rho), log(tE), t0, log(s13), log(q3), psi]
```

VBM の triple geometry はこうだった。

```cpp
q[3] = {1, exp(pr[1]), exp(pr[8])}
s[0] = exp(pr[0]) / (q[0] + q[1])
s[1] = s[0] * q[0]
s[0] = -q[1] * s[0]
s[2] = exp(pr[7]) * complex(cos(psi), sin(psi)) + s[0]
SetLensGeometry(3, q, s)

y1 = u0 * sin(alpha) - tn * cos(alpha)
y2 = -u0 * cos(alpha) - tn * sin(alpha)
```

つまり VBM は lens1-lens2 の COM を原点に置き、lens1-lens2 を x 軸にする。
`s13` は 1-2 COM から third ではなく、lens1 から lens3。

旧 lcbinint convention から VBM への正しい変換は `legacy_edges`:

```python
z1, z2, z3 = old_lens_positions(case)
v12 = z2 - z1
v13 = z3 - z1

s12 = abs(v12)
s13 = abs(v13)
psi = arg(v13) - arg(v12)
q2 = q
q3 = q2

com12 = (eps1 * z1 + eps2 * z2) / (eps1 + eps2)
source_vbm = (source - com12) * exp(-1j * arg(v12))
```

VBM の trajectory は `alpha=0` で source が `(-time, -u0)` になるので、
point-source 比較では `time=-source.real`, `u0=-source.imag` を渡す。

この変換後の VBM point-source 比較:

```text
planetary_subsystem_left      rel_vs_vbm = -1.981517e-04
planetary_subsystem_high_mag  rel_vs_vbm = -2.741533e-04
planetary_subsystem_right     rel_vs_vbm = -1.768850e-04
moderate_inner_pair           rel_vs_vbm =  3.937505e-05
wide_primary                  rel_vs_vbm = -4.501186e-05
```

high-mag small-q 周辺の 2e-4 程度の差は、旧 `amp_point3` との差も同程度。
根の解法精度か root filtering 由来の可能性が高い。

VBM finite triple はケースによって interpreter を落とすことがある。
`triple_reference_compare.py` はデフォルトでは VBM finite を走らせず、`--vbm-finite` の時だけ試す。

## 診断スクリプト

### triple_reference_compare.py

旧 `/moao38_7/nunota/binfit/integral/lcbinint.c` を wrapper compile して `amp_point3` と比較する。

旧依存:

- `zroots.c`
- `laguer.c`
- `complex.c`
- `nrutil.c`
- `option.c`

実行例:

```bash
PYTHONPATH=build_new python -u tests/diagnostics/triple_reference_compare.py --vbm-convention legacy_edges
```

### triple_finite_sweep.py

固定 bin の finite-source sweep。

実行例:

```bash
PYTHONPATH=build_new python tests/diagnostics/triple_finite_sweep.py \
  --source-bins 8,12,16,24,32 \
  --reference-bins 64 \
  --caustic-bins 1400 \
  --repeat 2 \
  --csv .note/diagnostic_runs/triple-finite-sweep.csv
```

quick smoke:

```bash
PYTHONPATH=build_new python tests/diagnostics/triple_finite_sweep.py \
  --source-bins 8,12 \
  --reference-bins 24 \
  --repeat 1
```

quick smoke の主な結果:

- `resonant_tiny_source`: bins 8 max_rel 約 0.302、bins 12 max_rel 約 0.0132 vs ref24
- `resonant_small_source`: bins 8/12 max_rel 約 0.048 vs ref24
- hex / point 領域は bins にほぼ依存しない

## 直近の検証

最後に通したもの:

```bash
cmake --build build_new -j2
ctest --test-dir build_new --output-on-failure
PYTHONPATH=build_new pytest -q tests/regression/test_triple_lens_point_source.py
PYTHONPATH=build_new python -u tests/diagnostics/triple_reference_compare.py --vbm-convention legacy_edges
git diff --check
```

結果:

- build passed
- ctest passed
- pytest は 10 passed
- VBM point-source 比較は上記の範囲
- `git diff --check` passed

このノート作成後には再テストしていない。

## 現在の性能と診断値

near-caustic 25 点、`rho=1e-3`, `source_bins=12`, `caustic_bins=1400` の目安:

```text
triple_finite_ir_best_ms_per_point = 11.5764
```

`LCBININT_AREA_DIAGNOSTICS=1` で見た固定 bin sweep:

```text
bins 8   mag 1342.6211527392857  err 7.126142683856315
bins 12  mag 1371.1330976486024  err 5.691310724049011
bins 16  mag 1372.2015909905058  err 4.270831578142925
bins 24  mag 1373.2352691886080  err 2.8497662403355943
bins 32  mag 1373.4819800737002  err 3.4150171524873225
```

`maxjump` diagnostic は inactive な `row + 1` を見て巨大値を出していたので修正済み。
binary / triple の gap repair loop で条件を以下にした。

```cpp
diagnostics != nullptr && scratch.ax[row + 1] > 0.0
```

これで同じケースの bins12 中央は `maxjump=584` 程度になり、以前の `1.24e+04` のような値は出なくなった。
ただし gap repair 自体はまだ重い。

例:

- bins12 central で `gaps=22139`
- `rows=57139`
- processed images 約 4
- seeds 約 72

duplicate repair seed guard を `unordered_set` で試したが、gap は減らず速度が約 12.8 ms/pt まで悪化した。
この試行は revert 済み。残したのは `maxjump` diagnostic 修正だけ。

## 次に詰める場所

優先度高:

1. triple の gap repair / row scan の構造を見直す。
2. binary 側の component tracking を triple に移植または共通化する。
3. fixed-bin の deterministic diagnostics を基準にして速度と値を追う。
4. 旧 `amp_point3` / `amp_hexadecapole3` との差分を高倍率 small-q 周辺でさらに切る。
5. VBM は point-source の sanity check として使い、finite triple は過信しない。

binary には以下のような component tracking がある。

- `ProcessedComponent`
- `seed_to_component_id`
- `subtracted_component_ids`
- `component_union_area`

triple はまだ `areaimage` と overlap vector 中心で、binary より単純。
複雑な caustic crossing では correctness と speed の両方に効いていそう。

adaptive は今は主戦場にしない。
ユーザー方針としても、adaptive は廃止寄りで、固定 bin と mode4/mode6 相当の直交座標・hex を詰める。

## 注意

- polar は後回し。
- VBM finite triple は落ちることがあるので自動回帰の基準にしない。
- VBM triple の parameterization は old lcbinint と違う。比較時は `legacy_edges` 変換を必ず使う。
- `coordinates="vbm"` は triple ではまだ本当の convention switch ではない。
- unrelated な未追跡ファイルや既存 note / diagnostic logs は消さない。
- `build_new` は現在の検証で使っている build dir。
