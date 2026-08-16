# 合成 binary light-curve benchmark 引き継ぎ資料

作成日: 2026-08-15／最終更新: 2026-08-16  
目的: 論文用の `lcbinint` / `VBMicrolensing` 速度・精度比較を、同じ条件で再利用できる形に残す。

## 現在の状態

- 比較対象は NASA の実データではなく、意図的に選んだ synthetic binary-lens cases。
- 最終図は binary の6ケースだけを使用している。triple の結果は今回の図・結論には使用しない。
- `lcbinint` 側の作業ブランチは `master`。warm-up 実装・テスト・ベンチマーク・作図スクリプトには未コミット変更がある。
- 今回は前回 run を上書きせず、狭い時間窓の新 run を作成した。

## 最終成果物

出力ディレクトリ:

`tests/diagnostics/results/recal2026/synthetic_lightcurve_benchmark_narrow_windows_20260816/`

- [最終 light-curve 比較図（PDF）](../results/recal2026/synthetic_lightcurve_benchmark_narrow_windows_20260816/paper_binary_c0_warmup_grid.pdf)
  - PNG 版も同じディレクトリに残している。
  - 2行3列、パネル番号 1--6。
  - `lcbinint` は warm-up 後の結果、比較線は `VBMicrolensing`。
  - 下段に相対誤差、各イベントに caustic/source-trajectory の inset。
- [最終速度比較図（PDF）](../results/recal2026/synthetic_lightcurve_benchmark_narrow_windows_20260816/paper_binary_speed_selected.pdf)
  - PNG 版も同じディレクトリに残している。
  - uniform source と linearly limb-darkened source の2パネル。
  - `lcbinint` の warm-up なし／ありと `VBMicrolensing` を表示。
  - 縦軸は対数の milliseconds per epoch。
- [人間向けベンチマークレポート](../results/recal2026/synthetic_lightcurve_benchmark_narrow_windows_20260816/REPORT.md)
- [機械可読な全結果](../results/recal2026/synthetic_lightcurve_benchmark_narrow_windows_20260816/benchmark.json)
- [ベンチマーク本体](benchmark_synthetic_warmup.py)
- [論文用作図スクリプト](plot_paper_binary_comparison.py)

## 比較条件

### `lcbinint`

各ケースについて新しい `LightCurve` を作り、次の設定を共通にした。

```python
lcbinint.Options(
    coordinates="vbm",
    nbin="auto",
    tol=1.0e-3,
    reltol=1.0e-3,
)
```

- lens: binary
- `C0_uniform`: uniform source (`LimbDarkening.none()`)
- `C1_linear_ld`: linear limb darkening with coefficient `0.5`
- `rho`: normalized source radius
- `alpha`: source-trajectory angle in the code convention
- `tE=1` for all six cases
- default epoch count: 240
- `close_secondary_caustics`: 400 epochs
- `wide_planet`: 600 epochs
- time grid: each case の `t_min`--`t_max` を等間隔にサンプル

### `VBMicrolensing`

`VBMicrolensing.VBMicrolensing().BinaryLightCurve(...)` を使って、`lcbinint` と同じ physical time grid の light curve 全体を評価した。

- `Tol = 1.0e-3`
- `RelTol = 1.0e-3`
- uniform source: `a1=0`, `a2=0`
- linear limb darkening: `a1=0.5`, `a2=0`
- binary parameter array は `vbm_parameters()` で `s`, `q`, `rho`, `tE` を log parameter として変換

つまり、VBM は単発の点源呼び出しではなく、今回の light-curve 比較では `BinaryLightCurve` API を使って全 epoch を評価している。

### timing protocol

- in-tree Release build を使用。
- `OMP_NUM_THREADS=1`。
- 各測定は first call を cold call として別記録し、その後5回の steady call を測定。
- 図の `milliseconds per epoch` は5回の steady call の中央値を epoch 数で割った値。
- warm-up ありでは、fresh curve に対して先に `curve.warmup(times, params, grid_timing_repeats=1)` を実行してから測定。
- warm-up のセットアップ時間は `warmup.extra_ms` に保存し、速度図の steady bar には混ぜていない。
- したがって、速度図は「同じ light curve を繰り返し評価する場合の1回あたり速度」を示す。warm-up の元を取る回数は別途検討が必要。

## warm-up の数値的な意味

`nbin="auto"` の出力をそのまま正解値として採用しているわけではない。現在の warm-up は次の流れになっている。

1. 通常の automatic dispatcher で、point source / hexadecapole / inverse-ray の route と初期 resolution hint を得る。
2. inverse-ray の epoch については、Cartesian と polar をそれぞれ高解像度側へ進め、3点の stable tail と追加の確認点で self-convergence を確認する。
3. Cartesian/polar の共通 reference と tolerance budget に対して候補 resolution が妥当かを再確認する。
4. 両方が使える場合は、その epoch で速い method と resolution を warm-up plan に保存する。
5. 以後の light-curve 評価では、epoch ごとの method/resolution を再利用する。

今回の最終6ケース×2 source profiles、計12レコードでは、全 epoch が `all_calibrated=true` になっている。したがって、以前出ていた「warm-up が全 epoch calibration に失敗して通常経路へ戻る」という状態は、今回の最終データには残っていない。

## 最終的に採用した6ケース

パネル番号は最終 light-curve 図と対応する。

| panel | case | `s` | `q` | `rho` | `u0` | `alpha` | `t0` | `tE` | time range | epochs | 狙い |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 1 | `resonant_high_mag` | 0.95 | 0.01 | 0.005 | -0.001 | 0.5 | 0 | 1 | [-0.4, 0.4] | 240 | resonant caustic 近傍の高増光・急変 |
| 2 | `resonant_large_source` | 0.95 | 0.01 | 0.020 | -0.01 | 0.5 | 0 | 1 | [-0.4, 0.4] | 240 | 同じ resonant topology で source を大きくした有限光源効果 |
| 3 | `close_binary` | 0.65 | 0.005 | 0.003 | 0.03 | 1.1 | 0 | 1 | [-0.5, 0.5] | 240 | close binary の基本形 |
| 4 | `high_q` | 1.00 | 0.10 | 0.010 | 0.05 | 1.3 | 0 | 1 | [-1.0, 1.0] | 240 | 高質量比・強い二体性 |
| 5 | `wide_planet` | 2.50 | 0.01 | 0.002 | 0.294 | 3.0 | -2.07 | 1 | [-2.8, 0.8] | 600 | wide planetary caustic と host-side の主増光を同時に表示 |
| 6 | `close_secondary_caustics` | 0.65 | 0.02 | 0.004 | 0.214 | 3.75 | 0 | 1 | [-1.2, 1.2] | 400 | close binary の上下に分かれた secondary caustics と斜め軌道 |

### なぜこの6つか

この6つは、代表的な天体イベントを統計的に抽出したサンプルではなく、数値計算の挙動と図上の信号が両方見えるように選んだ説明用の synthetic set である。

- resonant high magnification と resonant large source で、同じ topology に対する高増光と有限光源サイズの差を出す。
- close binary と close-secondary-caustics で、close topology の基本形と secondary caustic を分けて見せる。
- high-q を入れて、planetary limit に寄りすぎない二体性の強いケースを含める。
- wide planetary では、planetary caustic だけを切り出さず、主増光から planetary caustic までを同じ light curve に含める。
- `small_q` は信号が弱く図で差が見えにくいため最終図から外した。
- `cusp_small_source` は resonant/high-magnification 系と narrow caustic の情報が重なり、2×3 の紙面での多様性を増やしにくいため外した。

したがって、本文では「6ケースで一般性能を証明した」と書かず、「異なる caustic topology・source size・mass ratio・source profile を見せる代表的な stress-test cases」と説明するのが安全。

## wide case の `t0` の定義

`wide_planet` のパラメータは `t0=-2.07` である。ここでの `t0` は planetary caustic の通過時刻ではなく、VBMicrolensing の parameter convention における host/origin に対する source trajectory の closest-approach time である。

図の横軸は全ケースで

```text
(t - t0) / tE
```

に統一している。そのため wide case では host-side の主増光が表示上およそ0に来て、physical time でおよそ `t=0` に置いた planetary caustic の通過は表示横軸でおよそ `2.07` に現れる。planetary caustic を `t=0` と定義しているわけではない。

また、wide case の光度曲線の計算範囲は physical time `[-2.8,0.8]` と広めに取り、host-side の主増光から planetary caustic までを含めている。caustic/source-trajectory の inset 用には geometry-only の trajectory grid を別に作っており、geometry 用の点を magnification の評価に混ぜていない。

## 数値結果

`max rel. err` は、steady light curve と VBM steady light curve の各 epoch について

```text
abs(lcbinint - VBM) / abs(VBM)
```

を計算した最大値。`no/warm` は `no-warm-up time / warm-up time`、`warm/VBM` は `warm-up time / VBM time` である。したがって `warm/VBM < 1` は warm-up 後の `lcbinint` が VBM より速いことを意味する。

| panel | profile | no warm [ms/epoch] | warm [ms/epoch] | VBM [ms/epoch] | no/warm | warm/VBM | warm setup [ms] | max rel. err no / warm |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | C0 uniform | 0.6570 | 0.1390 | 0.05055 | 4.73 | 2.75 | 2924 | 3.82e-4 / 9.43e-4 |
| 1 | C1 linear LD | 0.8640 | 0.1300 | 0.2484 | 6.64 | 0.52 | 3437 | 5.57e-4 / 1.21e-3 |
| 2 | C0 uniform | 0.9829 | 0.1966 | 0.06611 | 5.00 | 2.97 | 3818 | 4.11e-4 / 1.01e-3 |
| 2 | C1 linear LD | 1.189 | 0.2134 | 0.5927 | 5.57 | 0.36 | 4245 | 5.58e-4 / 1.39e-3 |
| 3 | C0 uniform | 0.08299 | 0.01993 | 0.03003 | 4.16 | 0.66 | 285.4 | 7.55e-5 / 9.10e-5 |
| 3 | C1 linear LD | 0.1207 | 0.01822 | 0.02538 | 6.63 | 0.72 | 338.1 | 3.13e-4 / 3.43e-4 |
| 4 | C0 uniform | 0.06397 | 0.02307 | 0.04303 | 2.77 | 0.54 | 319.3 | 1.08e-3 / 9.65e-4 |
| 4 | C1 linear LD | 0.05028 | 0.01592 | 0.06878 | 3.16 | 0.23 | 287.6 | 7.01e-4 / 9.62e-4 |
| 5 | C0 uniform | 0.01987 | 0.006752 | 0.005583 | 2.94 | 1.21 | 796.2 | 9.22e-5 / 5.93e-4 |
| 5 | C1 linear LD | 0.02272 | 0.006971 | 0.009521 | 3.26 | 0.73 | 946.2 | 3.79e-4 / 6.63e-4 |
| 6 | C0 uniform | 0.04690 | 0.01228 | 0.02229 | 3.82 | 0.55 | 680.1 | 3.23e-4 / 6.70e-4 |
| 6 | C1 linear LD | 0.03470 | 0.009371 | 0.05202 | 3.70 | 0.18 | 566.9 | 5.20e-4 / 9.83e-4 |

### 結果の読み方

- 1--3番の時間窓を狭めたことで、主増光・有限ソース構造にサンプルを集中させた。狭い窓では1--3番の per-epoch cost と warm-up setup が前回より大きくなる。
- warm-up による steady throughput の改善は、C0 で 2.77--5.00倍、C1 で 3.16--6.64倍だった。
- C0 では warm-up 後の `lcbinint` は VBM と同程度から遅いケースが混在し、`warm/VBM` は 0.54--2.97。
- C1 では今回の6ケースすべてで `warm/VBM<1`、すなわち warm-up 後の `lcbinint` が VBM より速い。`warm/VBM` は 0.18--0.72。
- C0 の light-curve 比較図では、warm-up 後の最大相対誤差は `9.65e-4`。一方、12レコード全体では C1 の1番・2番でそれぞれ `1.21e-3`・`1.39e-3` となり、`1e-3` を超えた。
- したがって、今回の資料では「warm-up は全12レコードで VBM に対して `1e-3` 以下」とは書かない。warm-up 内部の全 epoch calibration は成功しているが、VBM との cross-code 差は別に確認が必要である。
- warm-up setup は約0.29--4.24秒。1回だけの評価ではなく、同じ trajectory 近傍を複数回評価する用途で初めて総時間のメリットが出る。

## 論文での主張の範囲

この図と数値から安全に言えるのは次の範囲。

> 同一の binary light curve を繰り返し評価する条件では、epoch ごとの数値 method/resolution を warm-up で再利用することで、`lcbinint` の steady evaluation cost が明確に低下した。今回の狭い時間窓の synthetic stress-test set では、linearly limb-darkened source の全6ケースで VBM より短い per-epoch time が得られた。一方、resonant cases の cross-code 相対差には `1e-3` を超える値もあるため、精度主張はケース別に記載する。

逆に、次のような一般化はまだ避ける。

- 全パラメータ空間で必ず VBM より速い。
- warm-up setup を含めた1回限りの総時間でも必ず速い。
- synthetic 6 cases が実イベント母集団を代表する。
- VBM の内部 method を固定した比較である。
- 狭い時間窓に変えても全ケースで cross-code 誤差が `1e-3` 以下である。

## 再現コマンド

現在のベンチマーク JSON/REPORT と同じ保存先に binary/triple の全測定を再生成する場合:

```bash
OMP_NUM_THREADS=1 python tests/diagnostics/recal2026/benchmark_synthetic_warmup.py \
  --output-dir tests/diagnostics/results/recal2026/synthetic_lightcurve_benchmark_narrow_windows_20260816
```

最終6ケースの light curve 図と速度図を再生成する場合:

```bash
OMP_NUM_THREADS=1 python tests/diagnostics/recal2026/plot_paper_binary_comparison.py
```

ベンチマークは wall-clock timing なので、再実行時に値は多少変動する。論文の表に採用する場合は、CPU・thread 数・build type を固定し、再実行した `benchmark.json` と図を同じ run として保存すること。

## Overleaf／GitHub への引き継ぎ

論文プロジェクト `lcbinint_paper` の GitHub remote を Overleaf 同期先として使用する。今回の狭い時間窓版では、以下の3つを更新対象にする。

1. `paper_binary_c0_warmup_grid.pdf`
2. `paper_binary_speed_selected.pdf`
3. この Markdown の「比較条件」「最終的に採用した6ケース」「数値結果」を本文または補足資料へ反映

PNG 版 (`paper_binary_c0_warmup_grid.png`, `paper_binary_speed_selected.png`) も paper repository には残しているが、本文への挿入は PDF 版を優先する。

図中のパネル番号とケース表はこの資料で固定する。特に wide case の `t0=-2.07` は planetary caustic の時刻ではなく host/origin closest approach という点を、本文・キャプションの両方で明記する。
