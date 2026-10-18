# 微分付き LightCurve の重複費用削減 — 2026-10-18

対象: `/rogue1_8/nunota/lcbinint/experiments/algebraic_boundary`、
`experiment/algebraic_boundary`、基点 `e740210b`。既存 checkout で実施。

## 変更

1. **採用されない接触再評価を省く。**
   fused frontierの後、従来のsource-disk修復をregular-chart再評価より先に試す。
   同じcaustic clearance証明、同じquadrature、同じ要求精度を使う。
   source-diskで認定できない点は、従来の接触処理をそのまま実行する。
   kernel内部のoptional callbackでcallerの既存修復処理を呼び、値と微分の組を返す。
   公開Pythonオプションや新しい幾何の閾値は追加していない。
2. **hexadecapoleの根を引き継ぐ。**
   batch値評価で求めた12点と中心の物理像を、必要な微分呼出しだけで保持する。
   解析微分はその像へIFTを適用し、同じ13点の根を解き直さない。
   値だけのcallは像handoff用の配列を要求しない。
3. **形状認定の共通計算を共有する。**
   同じepochの同じD14微分多項式に対するinterval/exact Sturm chainを
   各occupied bandで作り直さず、必要時に一度だけ構築する。
   exact符号評価では巨大整数係数vectorのコピーを避け、rationalの分母・分子も
   chain内で共有する。正確にゼロの場合の片側符号は従来通り微分次数で判定する。

要求精度、seed生成器、supportの認定条件、接触時の数学的修復は保持。
通常のpoint/hex選択と値専用積分アルゴリズムは変更していない。
診断falseでも有限候補を返し、非有限候補は明示エラーにする契約も保持する。

## matched timing

同じ入力、atol=rtol、float64、同じCPU affinity 1 core、OMP/OpenBLAS 1 thread。
変更前のPython/native packageを保存し、別processで変更後と比較した。
別lensでコード・quadratureを初期化し、各対象LightCurve callはtarget epoch cachesをclear。
値・微分の順序を交互に7回計測し、中央値を比較した。target geometry/selector/seedsを含む。
後→前の順の最終計測。前→後の中間計測でも同じ改善傾向を確認した。

| 条件 | 値: 前 → 後 | 値＋微分: 前 → 後 | 後の微分/値 |
| --- | ---: | ---: | ---: |
| notebook 401点、LD=.5、1e-4 | 18.47 → 18.54 ms | **63.02 → 50.47 ms** | 2.72 |
| 同じ401点、LD=.5、1e-3 | 16.12 → 16.14 ms | **40.39 → 37.40 ms** | 2.32 |
| 同じ401点、uniform、1e-4 | 17.72 → 17.82 ms | **50.41 → 40.07 ms** | 2.25 |
| crossing LD 101点、1e-3 | 3.60 → 3.57 ms | 5.12 → 4.83 ms | 1.35 |
| pointのみ201点、1e-3 | .236 → .236 ms | .356 → .354 ms | 1.50 |

notebook条件の微分付きは約20%短縮。値専用はほぼ同じ時間で、全保存入力でbitwise一致。
一律1.3倍以内という目標は達成していない。close-pairの認定とGaussian replayの費用は残る。
値専用の表の小差をゼロと断定したり、全条件での速度不変を保証する主張はしない。

## 回帰・精度確認

- **Python 62 passed**。case117のexact limb / handoffとqのstrict-value FD、
  case107、case081の桁落ち修復、座標変換、JAX/native一致、inverse-ray/warmupを含む。
- **C++ unit_core passed**。新しいbatch画像handoffについて、q<1とq>1で
  物理lens equationの残差、像からのflux和、像を要求しない値との完全一致も確認。
- exact符号評価を独立したrational Horner評価と比較。
  **9184条件**（重根、正負のleading coefficient、両側極限を含む）で一致。
- 元の120条件×33epochsを全件再実行。
  **3960/3960の値専用結果が変更前と完全一致**。
  値・微分付きの値・27720微分成分は全て有限。
  微分付きの数値収束は3960/3960、gradient_validは3928/3960。
  残り32点は従来のseed support未証明で、**フラグ変更は0件**。
- 保存微分からの最大変化は `abs(delta)/max(1,abs(old)) = 9.17e-9`。
  主にhexで引き継ぐrootセットと加算順が変わることによる小差。
  値専用/微分付きの最大相対差は従来と同じ **9.25178e-4**。
- 新たにt=.076のsmooth contact修復のq/rho微分をstrict-value中央差分と比較。
  q: -6292.9636 vs -6292.9645、rho: 505.9647 vs 505.9695。
  derivative refinementや新しい精度判定ルールは追加していない。

これらは保存ケースの回帰確認であり、全パラメータでの微分精度認定ではない。

## notebook

通常のLightCurve APIを使う比較notebookも全Python code cellを再実行し、図を確認した。
Jupyter kernelではなくin-process順次実行。401点、LD=.5、atol=rtol=1e-4。
今回の1回計測は値38.67 ms、微分付き61.00 ms、VBM BinaryLightCurve 133.26 ms。
初回利用の処理を含むため上の反復中央値とは混ぜない。
VBMとの差の最大相対値6.24223e-5、value/fused差3.49595e-9は従来通り。

## 再現・証跡

```sh
cmake --build build-route-unified --target lcbinint_python test_core -j2
/home/nunota/.miniconda3/envs/myenv/bin/python evidence/backend_cleanup_20261018/run_tests.py -q \
  tests/regression/test_algebraic_boundary_unified.py \
  tests/regression/test_binary_caustic_enclosure.py \
  tests/regression/test_light_curve_wrapper.py \
  tests/regression/test_binary_ray_shooting.py \
  tests/regression/test_model_finite_source.py \
  tests/regression/test_light_curve_warmup.py
ctest --test-dir build-route-unified -R unit_core --output-on-failure
/home/nunota/.miniconda3/envs/myenv/bin/python evidence/gradient_optimization_20261018/stress.py
/home/nunota/.miniconda3/envs/myenv/bin/python evidence/gradient_optimization_20261018/benchmark.py \
  --build build-route-unified \
  --output evidence/gradient_optimization_20261018/recheck-timing.json
```

package/nativeは明示的にこのcheckoutのbuildから読み込み、別checkoutを指すeditable installは使わない。
`before-final-timing.json` / `after-final-timing.json` にnative hashと設定、
`stress.json` / `stress-results/` に全件の結果を保存。
変更前のpackageは `before-package/`（ローカル実行artifact）に保存。
