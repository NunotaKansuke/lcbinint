# Light-curve API benchmark: value-only vs value + gradient

This is the historical 2026-10-16 measurement. The lcbinint value-only mode
explicitly selected `algebraic_boundary`, while its gradient mode called the
separate `holonomic_gradient()` implementation. It is not a measurement of the
current unified algebraic-boundary `with_gradient` API. Use the
[current API runner](../README_algebraic_boundary_api.md) for a fresh run; the
saved raw data below remains unchanged and is not mixed with that result.

測定日: 2026-10-16
対象 checkout: `experiment/algebraic_boundary` at `40e66e510799f9d3d15b459be2b38694daad52ee`

## 測定方法

全候補で時刻列と物理パラメータを受け取るベクトル／light-curve APIを呼び、Pythonの時刻ごとのforループは使っていない。12ケース、合計3,317時刻を測定した。入力、結果、環境、各ライブラリのrevisionとnative build hashは同じディレクトリの `inputs.json`、`metadata.json`、`raw/` に保存してある。

比較モードは `value_only` と `value_plus_gradient`。後者ではmicrolux、microjax、autodiff-vbmがベクトル出力の `jax.jacfwd(..., has_aux=True)` で値とヤコビアンを一度に計算し、lcbinintは `holonomic_gradient()` の一括APIで値と勾配を返す。通常のVBMicrolensingには勾配APIがないため、値のみを測った。

全入力はVBMicrolensingの軌跡規約で作られている。microlux／microjaxだけαにπを加えて同じ物理軌跡へ変換し、VBM、autodiff-vbm、lcbinintには元のαを渡した。最初の比較ではこの規約差を見落としていたため、そこに保存された結果は精度比較に使わない。

CPUは1コアに固定し、x64・各thread数1、共通許容誤差は1e-3。microjax CPU APIにはtol引数がないため、その経路は固定設定で測定した。各ケース3回の定常呼び出しの中央値を集計し、最初の呼び出し（JIT compileを含む場合あり）は別記録とした。

## 定常速度

下表の時間は12ケースそれぞれの定常中央値を足したもの。throughputは3,317時刻をその合計で割った値で、このケース構成に限った集計値であり一般的な速度保証ではない。

| API | 値のみ: 合計時間 | 値のみ: 時刻/秒 | 値＋勾配: 合計時間 | 値＋勾配: 時刻/秒 |
|---|---:|---:|---:|---:|
| VBMicrolensing 5.6.1 | 1.528 s | 2,171 | APIなし | — |
| microlux develop | 5.539 s | 599 | 27.523 s | 121 |
| microjax develop CPU | 3.008 s | 1,103 | 8.681 s | 382 |
| autodiff-vbm integration | 1.368 s | 2,425 | 1.281 s | 2,590 |
| lcbinint algebraic_boundary | 0.715 s | 4,642 | 3.160 s | 1,050 |

JAX系は起動直後のcompile費用が大きい。たとえば `ordinary_point` の初回はmicrolux約29.8秒、microjax約10.9秒だったが、表には含めていない。反復回数の少ない用途では初回費用も含めて選ぶ必要がある。

## 値と勾配の一致

真値として高解像度計算は使っていない。値の一致度は、各時刻でmicrolux、microjax、lcbinint、そしてVBMとautodiff-vbmの中央値を1つにまとめた4系統の中央値を基準にした記述統計であり、正しさの認定ではない。全3,317点に対する基準からの相対偏差は次の通り。

| 実装 | 相対偏差中央値 | 95 percentile | 最大 |
|---|---:|---:|---:|
| VBMicrolensing | 0.000894% | 0.02879% | 27.57% |
| microlux | 0.000221% | 0.29681% | 0.3198% |
| microjax | 0.000262% | 0.03588% | 0.1703% |
| autodiff-vbm | 0.000861% | 0.09989% | 0.4785% |
| lcbinint | 0.000214% | 0.03596% | 0.09773% |

最大の系統間の開きは `near_limb_contact_117` の13.934%。そこでVBMが他系統から外れている。合意中央値を真値とはみなせず、この点は別の独立基準で未解決のまま。

勾配の中央値合意からの相対偏差（有限な値をプールし、合意値の絶対値が1e-8より大きい成分）は、中央値でmicrolux 0.0226%、microjax 0.00766%、autodiff-vbm 0.0678%、lcbinint 0.00633%。95 percentileはそれぞれ259%、100%、100%、71%まで上がる。勾配がゼロに近い成分では相対誤差が発散するため、これは精度保証や微分可能性の判定ではない。パラメータ別・点別の内訳は `summary.json` にある。

値＋勾配の一括APIが返す値と値のみAPIの値との差は、最大相対差でmicrolux 2.20e-16、microjax 6.46e-16、autodiff-vbm 1.39e-13、lcbinint 4.42e-4。最後の差は共通tol 1e-3以下だが、他3実装と同じ丸め一致ではない。

## 既知の難ケースと制限

- `stratified_00`: microjaxは27点中2点がNaN。他4実装は全点有限。
- `stratified_02`: 全候補で値は25/25点有限、最大系統間相対差は0.0177%。lcbinintの勾配valid flagは175/175。これは勾配精度の保証を意味しない。
- `stratified_09`: 全候補の比較から除外。別のlcbinint測定では27/27値は有限だが勾配validは182/189で、値のみ中央値17.96 ms、値＋勾配中央値165.09 ms。VBM/autodiff-vbmの値計算も非常に遅く、autodiff-vbm値＋勾配は最初の呼び出しが5分以内に完了せず、比較可能な定常値は得られていない。生の全候補出力は未保存なので、このケースの精度比較は未完了。
- VBMicrolensing単独は5.6.1。autodiff-vbmの現行checkoutは `eco-system/integration` の指定HEADだが、内部のVBMは5.5系。完全に同じVBM solver版同士の比較ではない。

主データは `summary.json` と `raw/*.npz`。計測スクリプトは親ディレクトリの `benchmark.py`。この測定のためにライブラリ本体は変更しておらず、コミット／pushもしていない。

## APIごとの勾配時間について

値のみと値＋勾配の時間比は、勾配の純粋な追加コストではない。lcbinintでは値APIがpoint-source fast pathを選ぶ一方、勾配APIは別のseeded finite-source経路を全epochに適用する。autodiff-vbmは15反復の追試で値のみと値＋勾配がほぼ同速となり、勾配側の小さな勝ちは一般的な優位とは確認できなかった。本家VBMとのケース別追試では、autodiff-vbmが大光源で速い一方、接触ケースでは遅く、値にも大きな差がある。詳細は[API timing audit](../api_timing_audit.md)。
