# 最新: light curve autoのデフォルトへ接続

通常のbinary `LightCurve`で `inverse_ray_grid="auto"`, `nbin="auto"`が使われると、点源/hex判定後の積分に本選択器を適用する。中心像をseedと予測に共有し、要求された誤差予算と接線ガードを保つ。デフォルトの最大解像度を400→512へ変更し、16からの倍増で校正時の上限へ到達できるようにした。明示的な上限は維持。

- 最終nativeでlight curve入口188条件、予測先の一致・要求精度達成188/188。直接診断APIとの差の最大は 6.81e-07。Polar診断APIはyを反転正規化するためbit一致ではなく要求精度と参照値で確認した。
- 暗黙の既定精度、絶対精度のみ（1e-3）、明示的な上限400による未達、反復評価、点源への短絡、固定Cartesian/Polarも確認。別プロセスで通常のadaptive-caustic設定8例も確認し全てok。
- 最終nativeのCartesian132・Polar189内部チェック通過。
- 既存light curveの精度対応範囲ゲートは保持している。絶対精度のみ1e-4は既存の未対応境界であり、この作業で対応したとは主張しない。
- 下の速度比と全件replayはデフォルト接続前の校正・監査であり、新しいlight curve全体の速度比や全件再実行ではない。最終識別は `lightcurve-default-identity.json`。

---

# Cartesian / Polar 事前選択の試作（2026-10-15）

## 結果

**事前選択器を実装した。** 点源増光率だけの分類、浅い決定木、輝度・精度別の木、速度比を予測する小さな木の集合を比較。検証用イベントで選んだ最終モデルは **深さ3・40本のboosted regression trees**。両積分を走らせてから選ぶ方式ではない。通常の既定経路は変更せず、明示的な `binary_adaptive_auto_flux` とnative helperから使う。

## 学習と評価

- 現行カーネル、Cartesian16 / Polar2、seed cache miss込みcold3回中央値、中心像・caustic・LD表の準備は計時外。CPU0、1 thread。768入力×2精度×2手法。
- case_idをSHA256で分割。学習124イベント930条件、調整31イベント228条件、評価48イベント378条件。同一イベントの近接時刻・Uniform/LDが分割をまたがない。
- 速度ラベルは実測時間と積分器の受理状態。非受理を安い成功として学習しないため、その条件の最大有限時間の10倍を学習時の罰則にした。参照増光率・実誤差・case_id・epoch・保存caustic距離・積分後の評価数は特徴に含めない。参照値は精度集計だけに使用。
- 各条件の速度比を重視する学習と総時間重視の学習を比較し、調整イベントの平均log regretで選択。最終モデルの学習・ハイパーパラメータ選択に評価イベントのラベルは使わない。ただし開発中に評価側の診断も閲覧しており、封印された盲検試験とは呼ばない。
- sklearn依存なし。NumPyの小さな学習コードからC++の比較式を生成。点源のみの基準も保存した。

## 実際に選択処理を含めた再計時

各入力で直接指定版／事前選択版の実行順を交互にし、選択版は中心像の特徴計算・40本の評価・選んだ一方の積分を計時。数値・推定誤差・受理状態・評価数・停止理由は378条件で直接指定版と一致。

| 輝度 | 要求 | 条件数 / 計時ペア | Polar固定の総時間 / 選択版の総時間 | oracleに対する時間比の幾何平均 |
|---|---:|---:|---:|---:|
| uniform | 1e-03 | 91 / 89 | 1.427 | 1.082 |
| uniform | 1e-04 | 91 / 89 | 1.112 | 1.116 |
| linear | 1e-03 | 98 / 98 | 1.274 | 1.125 |
| linear | 1e-04 | 98 / 98 | 1.192 | 1.124 |

**上表は総時間の比であり、ケース別比の中央値ではない。** UniformのCartesian例外2入力×2精度には有限の両手法計時がなく、その4条件を速度比から除外した。これらも選択版はPolarを選び受理した。全378条件で選択版の未達0・精度外受理0。oracleは同じ入力の両手法の実測で速い方を事後選択した基準で、実行時には利用しない。

native選択版 / 選んだ直接指定版の追加費用は中央値で約1%（4群の直接/選択比 0.9912, 0.9891, 0.9858, 0.9892）。特徴生成単独は約0.4µsだが、木の評価・分岐も含めた実呼び出しで判断した。

## 使用する情報

中心像の既存rootとJacobianから、点源増光率、像数、半径方向・角度方向・Cartesian軸方向の伸び、周長の指標、局所曲率、原点からの最小/最大像半径、負パリティの寄与を計算。s、q、rho、LD係数、要求相対精度も入力する。局所曲率はrhoと写像の二階微分から計算し、causticへの近さの間接的な指標にする。連続caustic距離を新たに求める追加計算はしない。

浅い2段木は軽いが1e-4の総時間がPolar固定より悪い群があったため、最終採用しなかった。最終モデルでは局所曲率、点源増光率、原点に近い像、半径/角度の伸びが使われ、要求精度とLDにも依存する。

## API

```python
result = lcbinint.binary_adaptive_auto_flux(
    x, y, s=s, q=q, rho=rho,
    limb_darkening=lcbinint.LimbDarkening(0.5, 0.0),
    relative_tolerance=1e-3,
)
print(result["selected_method"], result["estimated_target_met"])
```

- Pythonのqは既存のadaptive APIと同じ意味。保存ベンチ入力TSVのqから渡す場合は従来どおり `1/input_q`。
- 結果辞書は選んだ既存APIのまま保持し、`selected_method`, `route_model`, `route_within_training_domain`, `route_reuses_center_roots`を追加。
- 校正した範囲はc=0または0.5、相対精度1e-3/1e-4、s≈0.202〜3.949、正規化q≈1.08e-4〜0.922、rho≈3.14e-5〜0.951。範囲外はPolarを明示的に選び、校正内フラグをfalseにする。これは速度予測の適用範囲であり、精度保証ではない。
- native側は `adaptive_detail::choose_adaptive_route(..., center_images)` を既存のpreplanned呼び出し前に使い、同じ中心像を渡せる。選択後に反対経路を試す救済は入れていない。
- Python便利APIは既存入口を呼び出すため中心像を追加で解く。nativeのroot共有版と速度境界が異なる。
- Python入口は96条件で選択先と数値・誤差・状態一致、無効入力6・範囲外2も確認。全準備込みの直接/選択時間比中央値 0.9802、追加時間中央値 4.47µs。このAPI計測から上表の速度比を主張しない。

## 残る制限と広い精度確認

- 既存Cartesianの未達・精度外受理を修正する選択器ではない。学習/調整側の選択結果に精度外受理は各1件ある。
- 広い見落としを調べるため、保存済みの両手法全14,432条件の数値結果へ最終選択を適用した。静的ライブラリの全object hashが監査時と一致することを確認し、数値結果を再利用した。これは積分の再実行でも独立評価でもない。

| 輝度 | 要求 | Polar選択 | 未達 | 例外 | 精度外受理 |
|---|---:|---:|---:|---:|
| uniform | 1e-03 | 1841/3608 | 1 | 0 | 8 |
| uniform | 1e-04 | 1926/3608 | 10 | 0 | 6 |
| linear | 1e-03 | 1942/3608 | 1 | 0 | 3 |
| linear | 1e-04 | 2023/3608 | 9 | 0 | 5 |

全件replayでは**未達21、例外0、精度外受理22**。内部契約違反0。`certified=false`, `complete_error_model=false`のまま扱う。条件を満たさない値を成功に書き換えていない。既存要求精度や停止条件を変更せず、ケースID救済も追加していない。

- 216の特徴/選択単体チェックで鏡映不変性、有限差分JacobianによるCartesian軸伸びの照合、校正範囲外の扱いを確認。
- 6パネルの再生成やVBMの再計時は今回行っていない。未コミット・未push。

## 保存先・再現

`identity.json`が計時開始時、`final-identity.json`が最終source/native/modelの識別。`timing-*.tsv`は768入力の両手法計時、`automatic/`は評価イベントの直接/選択ペア計時、`model.json`は選択根拠と比較候補、`measurements.json`は学習表。失敗行・NaN計時を含むrawを保持。`full-accuracy-replay.json`には全選択と失敗IDを保持。

project rootで実行（compilerのTMPDIRはproject内）:

```bash
python tests/diagnostics/recal2026/predictive_route_campaign.py
OPENBLAS_NUM_THREADS=1 python tests/diagnostics/recal2026/train_adaptive_route.py
TMPDIR="$PWD/build-adaptive-production/task-tmp" cmake --build build-adaptive-production -j2
# predictive_route_runner.cppを既存compile_runnerで再リンクしてから:
OPENBLAS_NUM_THREADS=1 python tests/diagnostics/recal2026/validate_predictive_route.py
OPENBLAS_NUM_THREADS=1 python tests/diagnostics/recal2026/check_predictive_route_api.py
OPENBLAS_NUM_THREADS=1 python tests/diagnostics/recal2026/replay_predictive_route_accuracy.py
```

選択モデルを再生成したらPython extensionとnative runnerの両方を再ビルドし、識別と再計時を更新する。nativeをロードする検証中にはextensionを上書きしない。
