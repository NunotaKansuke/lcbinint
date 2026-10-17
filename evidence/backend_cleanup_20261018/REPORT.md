# Algebraic-boundary backend: final source cleanup

対象: `experiment/algebraic_boundary`。基点は `a66ebfa2`。
既存の experiment checkout 内で実施し、親の feature checkout は編集していない。

## 現行の入口と計算

```python
curve = lcbinint.LightCurve(
    lens="binary",
    options=lcbinint.Options(finite_source_backend="algebraic_boundary"),
)
values = curve(times, params)
result = curve(times, params, with_gradient=True, with_diagnostics=True)
```

共通の point/hex selector → caustic-aware seeds → seeded radial frontier / nested Fejér。
値だけなら tangent は実行しない。微分付きなら値と tangent を同じ controller から返す。
source-disk cancellation repair と接触時の regular-chart / Gaussian replay は保持する。
これらは同じ積分系の座標・安定化処理であり、別の旧積分器への切替ではない。

静的 single-source binary、uniform / linear LD が対象。二次 LD は明示的なエラー。
`value_converged`、support の認定、`gradient_valid` は別々の診断であり、
微分精度の要求値を保証する仕組みは追加していない。
有限の値・微分候補は診断が false でも保持する。診断で NaN の数値配列に置き換えない。
実際に有限候補がなければ明示的なエラーとし、ゼロ・1・有限差分で埋めない。

## 整理したソース

- 通常ソースと名前空間を `src/lcbinint/magnification/algebraic_boundary/` と
  `lcbinint::algebraic_boundary` に統一。
- 微分側の既に検証済み overlay を正式な `tangent/` ソースへ採用。
  名前空間は `lcbinint::algebraic_boundary::tangent`。
  ビルド時の Python によるソース書換え・別 extension・互換 helper は廃止。
- 独立した旧 Holonomic preplanned solver、rolling prepared geometry ABI、
  旧 image-boundary / Gauss–Kronrod 積分器とその radial AD を削除。
  point/hex の解析微分は保持して通常名に変更。
- 旧ソルバーだけに必要だったテスト・ベンチマーク・検証プログラムを削除。
  歴史的資料と計算済み evidence は保持し、削除した実装は Git 履歴に残る。
- 通常 inverse-ray、Cartesian/Polar、flood fill、triple-lens 機能は保持。
  inverse-ray の warmup テストはそのバックエンドを明示するよう修正。
- method ID 6 は廃止。algebraic-boundary の診断 ID 7 は保持し、古い plan の ID 6 は
  他の方式と取り違えず明示的に拒否する。

通常ビルドの compiler dependency と native symbols に、削除した holonomic namespace、
旧 integral、旧 preplanned solver、generated overlay への参照がないことを確認した。

## 検証

- Python regression **61 passed** (`tests-final.log`)。
  case 107、case 117 limb/contact handoff、case 081 cancellation、未証明 seed support、
  value/Jacobian の座標変換、JAX/eager の一致、非有限候補の明示エラー、
  inverse-ray と warmup を含む。
- C++ `unit_core` **passed** (`core.log`)。
- 保存済み120条件、各33 epochs、atol=rtol=1e-3 の整理前後比較:
  **3960 epoch の値・微分付きの値・27720微分成分・有効性フラグが全て完全一致**。
  全て有限。32 epoch の seed support 未証明フラグも保持。
  結果は `numerical_parity.json`、比較元は `baseline/`、hash は `manifest.json`。
  これは整理による数値変更がないことの確認であり、微分精度の独立認定ではない。
- 比較 notebook の全 Python code cell を順番に実行し、図を確認した。
  socket を使用した Jupyter kernel 実行ではなく in-process 実行。
  401 epochs、linear LD=0.5、atol=rtol=1e-4。
  VBM との最大相対差は 6.24223e-5。
  value/fused の最大相対差は 3.49595e-9。
  数値収束と利用可能な微分は各401/401。積分47点のglobal support認定は0/47で、
  未認定という診断は維持している。
  notebook は VBM の `BinaryLightCurve` を呼び、値の時間に微分を含めない。

速度・精度の過去の詳細比較は [一本化レポート](../route_unification_20261018/REPORT.md)。
今回の整理を一律の高速化や全ケースの微分 overhead 1.3倍以内とは主張しない。

## 再現

```sh
cmake -S . -B build-route-unified \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/nunota/.miniconda3/envs/myenv \
  -DLCBININT_BUILD_TESTS=ON -DLCBININT_ENABLE_JAX_FFI=ON \
  -DPython3_EXECUTABLE=/home/nunota/.miniconda3/envs/myenv/bin/python
cmake --build build-route-unified --target lcbinint_python test_core -j2
/home/nunota/.miniconda3/envs/myenv/bin/python evidence/backend_cleanup_20261018/run_tests.py -q \
  tests/regression/test_algebraic_boundary_unified.py \
  tests/regression/test_binary_caustic_enclosure.py \
  tests/regression/test_light_curve_wrapper.py \
  tests/regression/test_binary_ray_shooting.py \
  tests/regression/test_model_finite_source.py \
  tests/regression/test_light_curve_warmup.py
ctest --test-dir build-route-unified -R unit_core --output-on-failure
/home/nunota/.miniconda3/envs/myenv/bin/python evidence/backend_cleanup_20261018/validate.py
```

この環境の editable install は別 checkout を参照するため、test runner と notebook は
明示的にこの build を読み込む。グローバルな install は変更していない。
