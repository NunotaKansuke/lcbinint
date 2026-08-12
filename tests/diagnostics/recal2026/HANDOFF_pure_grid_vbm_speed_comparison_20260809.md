# 引き継ぎ資料：純グリッド lcbinint と VBM の速度比較

作成日：2026-08-09
最終更新日：2026-08-13
対象ブランチ：`master`
対象データ：`controlled_pure_kernel_20260813`

## 1. このテストの目的

有限光源の積分処理だけを取り出し、次の二つへ同じ公称 tolerance
を独立に要求して比較する。

- `lcbinint` の有限光源グリッド積分（Cartesian と Polar の両方）
- VBMicrolensing の直接有限光源積分

本テストは本番ディスパッチ全体の比較ではない。point-source、hexadecapole、source-plane quadrature などのショートカットは比較から外し、`lcbinint_auto` が純粋な有限光源 grid route に入るサンプルだけを対象にしている。

ここでいう「同じ公称 tolerance」は、両実装を共通の正解値へ合わせるという意味ではない。
lcbinint は自身の grid 自己収束、VBMicrolensing は自身の `RelTol` という独立した
停止規則で計算し、その kernel time を比較する。両者の値の差は診断情報として保持するが、
速度勝敗から点を除外する条件にも、どちらか一方を不正確と判定する根拠にも使わない。
精度そのものを裁定する場合は、各実装の高解像度自己収束列や別の独立計算を追加で調べる。

## 2. 速度比の定義

```text
R = t_VBM / t_lcbinint
```

- `R > 1`：lcbinint の方が速い
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

| profile | target | points | lcbinint wins | 勝率 | median R |
|---|---:|---:|---:|---:|---:|
| uniform | `1e-2` | 1132 | 0 | 0.0% | 0.024 |
| uniform | `1e-3` | 1132 | 0 | 0.0% | 0.040 |
| uniform | `1e-4` | 512 | 0 | 0.0% | 0.036 |
| linear LD | `1e-2` | 1124 | 2 | 0.2% | 0.134 |
| linear LD | `1e-3` | 1124 | 160 | 14.2% | 0.532 |
| linear LD | `1e-4` | 647 | 435 | 67.2% | 1.506 |

Uniform source では、この積分単体比較の範囲では VBM が一貫して速い。lcbinint が勝つのは主に linear LD かつ厳しい精度要求の領域である。

## 6. A_finite との関係

`A_finite` は各 reference epoch の有限光源 magnification である。結論として、`A_finite` 単独の単調な速度則は見つかっていない。

Linear LD、`epsilon=1e-4`、`A_finite >= 1000` の内訳は次の通り。

| 実測 d/rho | points | lcbinint wins | 勝率 | median R |
|---|---:|---:|---:|---:|
| `0–0.1` | 4 | 4 | 100.0% | 2.983 |
| `0.1–0.3` | 9 | 5 | 55.6% | 1.414 |
| `0.3–0.8` | 6 | 0 | 0.0% | 0.082 |

したがって「高増光率なら常に lcbinint が勝つ」ではなく、少なくともこのサンプルでは、

```text
高 A_finite + 十分小さい実測 d/rho
```

が lcbinint の勝ち領域に対応している。ただし高 `A_finite` のサンプル数自体は19点なので、境界を固定則にするには holdout が必要である。

## 7. rho との関係

同じ Linear LD、`epsilon=1e-4` で source radius ごとに集計すると、rho も大きく効いている。

| rho | points | lcbinint 勝率 | median R |
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

このテストから安全に言えるのは、lcbinint の優位性は高増光率だけでは決まらず、source profile、要求精度、実測 caustic distance、source radius の組み合わせで決まる、というところまでである。

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

この時点では direct-XY 実装後に大規模コーパス全体を再計時していなかった。
その後の制御済み 160 構成 rerun は §16 に記録する。

## 13. ブランチ整理方針

- `master`：今回の direct-XY 計時経路と資料を含む本線。
- `algebraic-boundary-cpp`：新しい algebraic-boundary 積分アルゴリズム。master
  にはマージしない。比較用の独立ブランチとして保持する。
- `codex/tile-approach-archive`：tile/JAX 系の別アプローチのアーカイブ。master
  にはマージしない。
- `backup/full-featured`：推論機能の退避ブランチ。今回の速度比較には混ぜない。
- `final-testing` と `codex/warmup-execution-plan`：master に完全マージ済みのため、
  整理時に削除可能。

## 14. 2026-08-13：high-`A_finite` 監査と polar frontier 最適化

`optimize_irs` の160構成再測定で見えた high-`A_finite` の負けは、主として
VBM の外部 timeout や polar routing の失敗ではない。Linear LD の
`A_finite >= 1000` では VBM の全145点が完走し、VBM 5.5 の内部計数でも今回の
`epsilon=1e-3,1e-4` では `maxannuli=100` 到達は0点だった。uniform も
`NPSmax=10000` 到達は0点だった。

VBMに異常が全く無いわけではない。case 88、`d/rho=0.2`、Linear LDでは
`epsilon=1e-3` が `minannuli=1` のまま約0.3 msで停止して `A=563.61` を返す一方、
`epsilon=1e-4` は69 annuliで `A=590.51`、`minannuli=2` を強制した
`epsilon=1e-3` は `A=590.46` となった。またcase 10のtight条件にはfresh callと
warm後で `1e-4` を超えて値が変わる6点がある。ただし、これらは少数の実在する
VBM精度問題であり、high-`A_finite` 全域の速度差やtimeout傾向を説明しない。

主要因は計算量の違いである。inverse ray は polar でも像面内のほぼ
`O(A_finite Nbin^2)` セルを訪問する。VBM は source limb/caustic の輪郭複雑度に
依存するため、source が caustic の外側にあり輪郭が滑らかな high-mag arc は
非常に安い。Linear LD、`epsilon=1e-3`、`A_finite >= 100` では intended
`d/rho=0.2,0.6,1.0,1.4,1.8` に対する VBM 中央値が概ね
`5.00,3.07,1.60,0.64,0.47 ms` と外側ほど短くなる。polar 選択率は逆に上がるので、
polar は Cartesian に対して機能しているが、面積スケーリングを消せていない。

一方、polar flood fill には一般的な重複処理が残っていた。旧実装は長い radial
run の各inside cellから左右の角度列へ1件ずつqueueし、隣列の最初のcellがrun全体を
埋めた後、残りのqueue entryをvisitedとして破棄していた。現在は同じ4近傍 flood
fill のfrontierをradial区間1件として渡す。受信列は区間中の全未訪問cellを検査する
ので、複数のdisjoint runや別成分を取りこぼす特例則ではない。

同時に polar のradial source-limb crossingを、inside/outside endpointの線形補間
だけでなく、追加1回のlens-map評価によるbracketed secantで補正した。これは
Bennett型のsub-cell boundary correctionをpolar方向へ一般化したもので、積分方式は
inverse rayのままである。

小規模A/Bの結果：

- 保存済み旧結果と同じgrid/Nbinを使ったLinear LD 24点で、polar kernelは中央値
  `1.65x` 高速化（min `1.19x`, p90 `1.98x`）。
- uniform 12点では中央値およそ `1.7--2.0x`。
- case 92、`d/rho=1.8`、Linear LD、`epsilon=1e-3` は同じpolar/Nbin=27のまま
  `19.07 -> 9.89 ms`。
- 同じcaseの `epsilon=1e-4` では、旧実装で4/4点ともNbin=400確認時に
  `self_timeout` だったpolarが4/4点で自己収束し、2/4点でCartesianより速い候補に
  なった。ただしVBMには依然負ける。これは純inverse-rayの面積コストによる限界で
  あり、timeoutをVBM勝利の説明には使えない。

確認済み：`ctest`、polar関連regression 12件、triple polarのfocused regression。
この節の時点では全160構成の再測定は未実施だったが、後述の §16 で制御済み
rerun を完了している。

## 15. 2026-08-13：native triple と JAX binary/triple の同系統最適化

native triple は binary と同じ `inverse_ray_polar_core` を使うため、前節の
interval-frontier、写像距離の再利用、polar radial boundary correction はそのまま
適用される。旧buildとの固定解像度比較では、高倍率8点、`Nbin=32` の中央値が
約 `1.25x`、`A_point` が約 `1.96e4` の `Nbin=64` 点が約 `1.48x` 速くなった。

triple 固有では、caustic polyline の全 segment を歩く seed 生成が、最近点までの
距離に平方根を取り、採用後に二乗距離を再計算していた。閾値比較と候補順序を最初
から同じ二乗距離で行うようにし、start vertex の距離も1回だけ計算するようにした。
固定 `Nbin=1,8,32,128` および凍結24点の値は変更前と bitwise 一致した。速度差は
seed 支配の `Nbin=1` で約 `1.06x`、実用的な `Nbin=32,128` で約 `1.02x` だった。

自動 triple 解像度は、case/regime 分岐を増やさず、連続則
`N = 32 (epsilon/1e-3)^(-0.6)` と `N>=32` にした。以前の初期値は
`epsilon=1e-3` で80、`1e-4` で319だったが、nested half-grid check の大部分が
refinement level 0 で終了しており、通常コーパスを大幅に過積分していた。現在は
概ね32/128から開始し、既存のembedded errorとnested checkが必要な点だけ増やす。

inverse-ray を使う24構成、uniform/Linear LD各48 epochの `epsilon=1e-3` 監査では、
保存済み独立3-witness基準に対する最大相対差が uniform `8.35e-4`、Linear LD
`7.17e-4` で、全構成が要求内だった。代表3構成の48-epoch blockを旧初期解像度と
同じ固定80/319に対して直接比較すると、method列を変えず、`epsilon=1e-3` で
`1.40--3.23x`、`1e-4` で `4.02--5.46x` 速かった。新旧高解像度値の最大相対差は
それぞれ `1.17e-4`、`8.76e-6` だった。

`epsilon=1e-4` は保存済み3-witness自体のspreadが要求値を超える点があるため、古い
中央値との差だけでは判定しなかった。保存結果との差が大きい側から5構成を選び、
現行autoの全inverse-ray epochを同じgridの現行固定`Nbin=400` tailと比較したところ、
uniform/Linear LDの全10 blockが要求内で、最大相対差は `9.53e-5` だった。

caustic-clear disk で64個のtriple boundary probeを省く案も試したが、連続像成分の
証明とは別に、細い像弧上の polar lattice cellを見つける役割があった。
`d/rho=3.55`, `Nbin=64` の反例で値が `1.16e-3` 動いたため、案は完全に撤回した。
この短絡は現在のsourceには残っていない。

JAX C++ backendにも同じ一般的な削減を入れた。triple polar はcell queueから
radial interval-frontierへ変更し、flood中の写像距離をLD積分と境界補正に再利用、
support探索で不要なJacobian/shearを計算しない。binary polarではdouble経路の
classificationを再利用し、binary/triple共通で使わないLaplacianをcompile-timeに
省略した。triple Cartesianのセル分類もrow SIMD化した。

single-thread/cache-warmの旧extension比較は次の通り。

- triple polar far uniform：`6.25 -> 2.28 ms` (`2.74x`)
- triple polar extreme Linear LD：`2.10 s -> 0.515 s` (`4.08x`)
- triple Cartesian 16 epoch：`89.4 -> 75.2 ms` (約 `1.19x`)
- binary polar high-`A` Linear LD、`Nbin=27`：`18.04 -> 14.22 ms` (`1.27x`)
- binary polar high-`A` Linear LD、`Nbin=53`：`32.93 -> 27.98 ms` (`1.18x`)

同一入力のJAX値は旧extensionと17桁表示で一致した。focused確認は native
`ctest`、triple auto/polar tests、JAX polar 7件、discovery/fused/ladder 6件、
triple polar/active-support 8件、public moment-mode 3件で通過した。全規模テストは
この節の変更単独では実施していないが、最終状態のリポジトリ全体 pytest は
`483 passed, 3 skipped` で完了している。

## 16. 2026-08-13：controlled balanced log-uniform rerun

論文用の比較条件を揃えるため、独立対数一様な `s`, `q`, `rho` と、測定した
`d/rho` の等幅5層を使う160構成の純 kernel rerun を完了した。各構成から5位置を
採用し、uniform と linear limb darkening (`c=0.5`)、`epsilon_rel=1e-3,1e-4` を
評価した。`lcbinint` は VBM と基準値を共有せず、3点自己収束で最小 `Nbin` を決め、
Cartesian/Polar の実測時間が短い方を採用した。VBM は指定 `RelTol` の一回計算で、
両者の値の不一致は診断フラグに留め、勝敗から点を除外していない。

速度比は `R=t_VBM/t_lcbinint` で、`R>1` が `lcbinint` の勝ちである。最終集計は
次の通り。

| profile | `epsilon_rel` | measured | lcbinint wins | VBM wins | unresolved | win rate | median `R` |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 3200 | 2 | 3198 | 0 | 0.1% | 0.199 |
| uniform | `1e-4` | 3199 | 0 | 3199 | 1 | 0.0% | 0.105 |
| linear LD | `1e-3` | 3200 | 1740 | 1460 | 0 | 54.4% | 1.132 |
| linear LD | `1e-4` | 3198 | 1899 | 1299 | 2 | 59.4% | 1.432 |

linear LD では measured `d/rho` が `[0,0.4)` のときの勝率が `67.8%` (`1e-3`)、
`71.9%` (`1e-4`) で、`[1.6,2]` ではそれぞれ `38.8%`, `45.6%` に下がる。一方、
uniform は全層でほぼ VBM 優位だった。この結果から、単独の `A_finite` や `rho` だけ
ではなく、source profile、要求精度、実測 `d/rho` の組み合わせが支配的だと整理する。

測定は Intel Xeon Gold 6530（2 socket、各32 physical core）上で、各サンプルを
1 worker、`OMP_NUM_THREADS=1` として実施した。全最終測定に job-level timeout はなく、
自己収束未確定の3点は結果から黙って落とさず unresolved として保持した。

論文用の要約と最終図は次に保存した。

- [`REPORT_controlled_pure_kernel_20260813.md`](../results/recal2026/REPORT_controlled_pure_kernel_20260813.md)
- [`controlled_parameter_vs_R_2x3_profiles_20260813.pdf`](../results/recal2026/figures/controlled_parameter_vs_R_2x3_profiles_20260813.pdf)

35 MB の raw merged JSON はリポジトリへ入れず、ローカル diagnostics workspace に保持
している。
