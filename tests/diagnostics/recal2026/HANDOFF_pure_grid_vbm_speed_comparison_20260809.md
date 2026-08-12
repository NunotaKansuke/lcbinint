# 引き継ぎ資料：純グリッド LCB-in と VBM の速度比較

作成日：2026-08-09
最終更新日：2026-08-12
対象ブランチ：`master`
対象データ：`near_caustic_pure_grid_large_equal_timeout_20260809`

## 1. このテストの目的

有限光源の積分処理だけを取り出し、次の二つを同じ精度条件で比較する。

- `lcbinint` の有限光源グリッド積分（Cartesian と Polar の両方）
- VBMicrolensing の直接有限光源積分

本テストは本番ディスパッチ全体の比較ではない。point-source、hexadecapole、source-plane quadrature などのショートカットは比較から外し、`lcbinint_auto` が純粋な有限光源 grid route に入るサンプルだけを対象にしている。

## 2. 速度比の定義

```text
R = t_VBM / t_LCB-in
```

- `R > 1`：LCB-in の方が速い
- `R < 1`：VBM の方が速い

各 reference epoch では Cartesian と Polar を両方測定し、精度条件を満たす方のうち実測時間が短い方を `lcbinint` 側の代表値に採用している。

測定値はオブジェクト生成・最初のウォームアップ呼び出しを除いた、積分呼び出しの限界時間に近い値である。各点2回測定し、中央値を使っている。

## 3. 比較条件

- 1418 jobs、5672 reference epochs
- `q >= 1e-4`
- サンプリング因子は要求値として `0, 0.25, 0.5, 0.8, 0.95, 1.0, 1.1, 1.35, 1.7, 2.0`
- 実測比較の対象は要求値 `d/rho < 2.01`
- Uniform source
- Linear limb darkening、`c=0.5`
- 相対誤差目標：`1e-2`, `1e-3`, `1e-4`
- reference epoch：各 block の `(0, 7, 15, 23)`
- Grid と VBM に共通の reference-epoch timeout：300秒
- job-level timeout：無効。点単位で監視し、タイムアウト点も分母から落とさない

結果は5671点が測定済みで、1点だけ VBM が指定精度に到達せず unresolved になった。grid/VBM の片方だけがタイムアウトした点を勝敗に含める処理は残しているが、今回の大規模結果では timeout 勝敗は発生していない。

## 4. VBM API の注意点

Linear limb darkening の呼び出しは次の形で固定している。

```text
vbm.a1 = c
vbm.SetLDprofile(vbm.LDlinear)
vbm.BinaryMagDark(s, q, -x, y, rho, 1e-12)
```

`BinaryMagDark` の第6引数は limb-darkening coefficient ではなく絶対精度である。`c` を第6引数に渡すと、エラーにはならず誤った精度指定になるので注意すること。

## 5. 主要結果

| profile | target | points | LCB-in wins | 勝率 | median R |
|---|---:|---:|---:|---:|---:|
| uniform | `1e-2` | 1132 | 0 | 0.0% | 0.024 |
| uniform | `1e-3` | 1132 | 0 | 0.0% | 0.040 |
| uniform | `1e-4` | 512 | 0 | 0.0% | 0.036 |
| linear LD | `1e-2` | 1124 | 2 | 0.2% | 0.134 |
| linear LD | `1e-3` | 1124 | 160 | 14.2% | 0.532 |
| linear LD | `1e-4` | 647 | 435 | 67.2% | 1.506 |

Uniform source では、この積分単体比較の範囲では VBM が一貫して速い。LCB-in が勝つのは主に linear LD かつ厳しい精度要求の領域である。

## 6. A_finite との関係

`A_finite` は各 reference epoch の有限光源 magnification である。結論として、`A_finite` 単独の単調な速度則は見つかっていない。

Linear LD、`epsilon=1e-4`、`A_finite >= 1000` の内訳は次の通り。

| 実測 d/rho | points | LCB-in wins | 勝率 | median R |
|---|---:|---:|---:|---:|
| `0–0.1` | 4 | 4 | 100.0% | 2.983 |
| `0.1–0.3` | 9 | 5 | 55.6% | 1.414 |
| `0.3–0.8` | 6 | 0 | 0.0% | 0.082 |

したがって「高増光率なら常に LCB-in が勝つ」ではなく、少なくともこのサンプルでは、

```text
高 A_finite + 十分小さい実測 d/rho
```

が LCB-in の勝ち領域に対応している。ただし高 `A_finite` のサンプル数自体は19点なので、境界を固定則にするには holdout が必要である。

## 7. rho との関係

同じ Linear LD、`epsilon=1e-4` で source radius ごとに集計すると、rho も大きく効いている。

| rho | points | LCB-in 勝率 | median R |
|---|---:|---:|---:|
| `3e-5–1e-4` | 40 | 50.0% | 0.920 |
| `1e-4–3e-4` | 52 | 34.6% | 0.573 |
| `3e-4–1e-3` | 79 | 57.0% | 1.235 |
| `1e-3–3e-3` | 136 | 55.1% | 1.182 |
| `3e-3–1e-2` | 64 | 92.2% | 2.233 |
| `1e-2–3e-2` | 72 | 80.6% | 3.325 |
| `3e-2–1e-1` | 132 | 84.8% | 1.880 |
| `>=1e-1` | 48 | 62.5% | 2.530 |

このため、速度則を作る場合は `A_finite` だけでなく、少なくとも実測 `d/rho` と `rho` を同時に見る必要がある。さらに `q`、binary topology、source profile も交絡している。

## 8. d/rho の扱いに関する重要な訂正

benchmark の raw result にある `d_over_rho` は、サンプル生成時に指定した **intended distance factor** である。geometry の法線方向ステップや cusp 近傍の別 branch の影響で、実際に到達した距離とは一致しないことがある。

最終図では、各 unique geometry に対して `lcbinint.LightCurveInfo.caustic_distances` を再計算し、

```text
actual d/rho = caustic_distance / rho
```

を色に使っている。これは速度測定そのものをやり直す処理ではなく、既存の速度結果に対する geometry の後処理である。

後処理の設定は次の通り。

- `coordinates="vbm"`
- `caustic_bins=1400`
- `nbin=16`
- cheap route の threshold は0にして距離計算を実行
- 283 unique geometries を再計算

## 9. 再現方法

### 既存の merged 結果から図を再生成

```bash
MPLBACKEND=Agg python \
  tests/diagnostics/recal2026/plot_pure_grid_afinite_vs_speed.py \
  --results \
  tests/diagnostics/results/recal2026/near_caustic_pure_grid_large_equal_timeout_20260809/merged/results.json \
  --output \
  tests/diagnostics/results/recal2026/near_caustic_pure_grid_large_equal_timeout_20260809/merged/afinite_vs_speed_ratio_actual
```

この処理は実測 d/rho を再計算するため、通常の plot 生成より少し時間がかかる。出力は次の通り。

- `REPORT_Afinite_vs_speed_ratio.md`
- `figures/Afinite_vs_speed_ratio_colored_d_over_rho.png/pdf`
- `figures/Afinite_vs_speed_ratio_colored_rho.png/pdf`
- `Afinite_vs_speed_ratio_summary.json`
- `actual_geometry_summary.json`

### part 結果を merge

```bash
cd tests/diagnostics/recal2026
python merge_benchmark_parts.py \
  --parts \
  "../../diagnostics/results/recal2026/near_caustic_pure_grid_large_equal_timeout_20260809/part*/results.json" \
  --output \
  ../../diagnostics/results/recal2026/near_caustic_pure_grid_large_equal_timeout_20260809/merged
```

実際の part は、重複を避けるため要求値の factor を分割している。

| part | factors |
|---|---|
| part01 | `0, 0.25` |
| part02 | `0.5, 0.8` |
| part03 | `0.95, 1.0` |
| part04 | `1.1, 1.35` |
| part05 | `1.7, 2.0` |

### フル benchmark の主要引数

各 part は `bench_grid_vs_vbm_dark.py` に対して、おおむね次の条件で実行した。

```text
--input speed_discovery
--case-count 160
--factors <part-specific factors>
--profiles uniform linear
--q-min 1e-4
--d-max 2.01
--route-filter grid
--seed 20260809
--repeats 2
--point-timeout 300
--job-timeout 0
--targets 1e-2 1e-3 1e-4
```

`--search-missing` は、既存の speed discovery ladder にない Nbin を本当に探索し直したい場合だけ付ける。通常の再集計では、既存 ladder を current build で再検証するだけでよい。

## 10. 関連ファイル

- Benchmark harness：[`bench_grid_vs_vbm_dark.py`](bench_grid_vs_vbm_dark.py)
- Part merge：[`merge_benchmark_parts.py`](merge_benchmark_parts.py)
- A_finite / speed / d/rho / rho plot：[`plot_pure_grid_afinite_vs_speed.py`](plot_pure_grid_afinite_vs_speed.py)
- 大規模速度レポート：[`REPORT_pure_grid_large_equal_timeout.md`](../results/recal2026/near_caustic_pure_grid_large_equal_timeout_20260809/merged/REPORT_pure_grid_large_equal_timeout.md)
- 最終 A_finite / speed レポート：[`REPORT_Afinite_vs_speed_ratio.md`](../results/recal2026/near_caustic_pure_grid_large_equal_timeout_20260809/merged/afinite_vs_speed_ratio_actual/REPORT_Afinite_vs_speed_ratio.md)
- merged raw result：[`results.json`](../results/recal2026/near_caustic_pure_grid_large_equal_timeout_20260809/merged/results.json)

## 11. 次にやるべきこと

1. 独立した parameter holdout で、`A_finite`、実測 `d/rho`、`rho`、`q` を同時に層別化する。
2. 高 `A_finite` かつ `d/rho < 0.3` の Linear LD、`epsilon=1e-4` のサンプルを増やす。
3. 速度比較を production dispatcher 全体にも拡張し、point/hex fallback を含めた実際の LightCurve API の勝敗を別に集計する。
4. 固定された経験則をすぐ実装せず、まず warmup の実測 grid 選択を holdout で検証する。
5. `d/rho` の intended 値と actual 値を benchmark raw result に最初から保存するよう、次回 harness を改善する。

このテストから安全に言えるのは、LCB-in の優位性は高増光率だけでは決まらず、source profile、要求精度、実測 caustic distance、source radius の組み合わせで決まる、というところまでである。

## 12. 2026-08-12：direct-XY 計時経路の整理

速度ベンチの lcbinint 側は、現在 `_evaluate_preplanned_xy(source_x,
source_y, ...)` を使う。source 座標は内部の lens frame の `(x, y)` を直接渡し、
epoch ごとの時刻からの軌道再構成・座標変換を計時区間から外した。

preplanned Cartesian inverse-ray では、seed 生成時にすでに得ている point-source
magnification を walk の hint として再利用する。同じ point-lens solve を hint の
ためにもう一度実行する処理は削除した。ただし point-image/seed 生成自体は積分に
必要なので残している。

実装箇所は次の通り。

- `src/lcbinint/model/lens_model.cpp`：直接 source を受ける
  `LensModel::magnification_source()`
- `src/lcbinint/lc/light_curve.cpp`：
  `evaluate_preplanned_xy_diagnostic()`
- `python/bind_lc.cpp`：`_evaluate_preplanned_xy` binding
- `tests/diagnostics/recal2026/bench_grid_vs_vbm_pure_kernel.py`：新 API を使用

数ケースの A/B では direct-XY による短縮はおおむね 0--3% 程度で、支配的な
高速化ではない。Cartesian/Polar、uniform/linear limb darkening の値は旧 time
経路と一致した。したがって速度の主因は依然として point-image/seed 生成と
有限光源積分本体であり、座標変換を抜いただけで大幅に速くなるわけではない。

確認済みコマンド：

```bash
cmake --build build -j 8
ctest --test-dir build --output-on-failure
python -m py_compile \
  tests/diagnostics/recal2026/bench_grid_vs_vbm_pure_kernel.py \
  tests/diagnostics/recal2026/report_pure_kernel.py
```

今回の direct-XY 実装後に大規模コーパス全体を再計時したわけではない。既存の
大規模結果は過去の計時記録として保持し、新経路の全件再計時は必要になった時に
別 run として保存する。

## 13. ブランチ整理方針

- `master`：今回の direct-XY 計時経路と資料を含む本線。
- `algebraic-boundary-cpp`：新しい algebraic-boundary 積分アルゴリズム。master
  にはマージしない。比較用の独立ブランチとして保持する。
- `codex/tile-approach-archive`：tile/JAX 系の別アプローチのアーカイブ。master
  にはマージしない。
- `backup/full-featured`：推論機能の退避ブランチ。今回の速度比較には混ぜない。
- `final-testing` と `codex/warmup-execution-plan`：master に完全マージ済みのため、
  整理時に削除可能。
