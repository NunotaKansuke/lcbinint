# V2 adaptive radial Phase 9 checkpoint — 2026-09-11

基準は `dev/holonomic` の `50f39729407d1687af0aab10c283bbaeafd4a0b8`。変更範囲は isolated holonomic に限定した。production router、既存 fixed-`n_r` API、PF6/GM経路、未追跡 `.claude/` は変更していない。

## 判定

今回の主目的である value と gradient の契約分離は実装した。ValueFirst では、value が要求精度へ到達した後に value mesh を固定し、gradient だけを独立 budget で追加処理する。gradient の誤差推定だけが未認証の場合は `FiniteUncertified` として value を保持し、非有限値・topology/root/branch不確実性は `Invalid` として分離する。

setup 固定費については、無条件 qf family 構築を lazy 化し、event の double → DD → qf 局所 ladder、epoch内 event cache、setup内訳計測、workspace capacity 再利用を入れた。ただし今回のケース集合では全epochで少なくとも一つのqf event refinementが必要だったため、lazy化だけによるwhole-epoch短縮は観測していない。value/gradientの精度契約と実装検証は前進したが、adaptive pathをproduction defaultへ昇格する性能・gradient coverageの根拠にはまだ届いていない。

要求誤差は全成分で次を使う。

```text
Eabs <= max(Tol, RelTol * abs(Q))
```

絶対許容誤差と相対許容誤差を加算していない。gradient には value と独立した `grad_atol[j]` / `grad_rtol[j]` を持たせた。

## 実装した契約

`adaptive_radial.hpp` に次を追加した。

- `GradientPolicy::None`、`Strict`、`ValueFirst`。
- `GradientQuality::NotRequested`、`ToleranceMet`、`FiniteUncertified`、`Invalid`。
- gradientごとの `GradientReason` と `grad_error`、`grad_quality`、`grad_reason`。
- `value_converged`、`value_stop_reason` と gradient側の状態を独立に保持。
- `ValueFirst` の node/round budget。value snapshot とその error ledger を保存し、後段のgradient refinement結果でvalueを置き換えない。
- nonfinite gradient を成分ごとに記録する処理。ある成分が `Invalid` でも、別の有限成分を一括で `Invalid` にしない。

`Strict` は従来どおり要求gradient全成分の契約成立を全体の成功条件にする。ただし `value_converged` は独立に返す。`ValueFirst` は有限だが指定gradient toleranceを認証できない場合に `stop=Converged`、`numerical_status=OK`、gradient `FiniteUncertified` を返す。gradient自体が非有限、またはtopology/root/branchが壊れた場合は `Invalid` とし、品質を格下げして隠さない。

Strict/ValueFirstのvalue phaseではpanelの優先度をvalue errorだけで決める。value snapshot後にだけgradient errorをschedulerへ渡すため、gradient toleranceのためにvalue meshを先に細分化しない。代表 `plan15` では None/Strict/ValueFirst のvalue snapshotが同一であることを確認した。

## setup と event precision

`adaptive_epoch.hpp` では次を実装した。

- `PrimaryFrame`、physical cuts、event preparation、panel preparationを個別計時。
- `double_event_estimate` で quartic root、`P`/`Pt`/`PR` residual、ULP、Newton shift、value budgetを見てdoubleで止めるか判定。
- ambiguity時だけ既存 `DD` で `P`/`Pt`/`PR` を再評価し、条件を満たせばDD tierで止める。
- それでも必要なeventだけ `PolyFamilyR` を初回一度だけlazy構築してqfの `P=Pt=0` correctionを実行。
- 同一epoch内のevent radius/uncertainty/tierをcache。
- qf補正の `radius_hi` と `radius_lo` を `AdaptivePanel` に分離保持し、fold mapの局所計算ではlong doubleで合成する。
- `setup_event_ms` と `setup_panel_ms` はevent call時間をpanel totalから差し引き、同じ時間を二重に足さない形式にした。
- `AdaptiveWorkspace` のsamples/panels/cellsは `clear()` 後もcapacityを保持する。

DD tierの評価は、現段階では既存double係数をDDへ持ち上げて評価する局所cancellation対策である。係数構築そのものをDD精度へ戻すものではない。そのため、qfが必要な場合に係数精度まで置き換えたとは主張しない。

既存V2の折りたたみmapと、map Jacobianを掛けてからendpoint derivativeを扱うnear-foldの有限化は維持した。fixed-Rの巨大な微分を作ってから小さいmap Jacobianを掛ける新しい経路は追加していない。inner derivativeが不足する場合は `InnerAccuracyLimited` として残す。

## 再現コマンドとraw

```sh
bash checks/holonomic/run_adaptive_radial.sh \
  evidence/holonomic/adaptive_radial_phase9_contracts_20260911 1
python checks/holonomic/summarize_adaptive_radial.py \
  evidence/holonomic/adaptive_radial_phase9_contracts_20260911 \
  > evidence/holonomic/adaptive_radial_phase9_contracts_20260911/summary.txt
```

rawは次に保存した。

- [contracts.csv](../../evidence/holonomic/adaptive_radial_phase9_contracts_20260911/contracts.csv): 110ケース × 3 policy × cold/warm × RelTol 1e-3/1e-4。
- [paired.csv](../../evidence/holonomic/adaptive_radial_phase9_contracts_20260911/paired.csv): whole-epoch value/value+5Jacとfixed64対照。
- [controls.csv](../../evidence/holonomic/adaptive_radial_phase9_contracts_20260911/controls.csv): adaptive/cache/fold mapの対照。
- [summary.json](../../evidence/holonomic/adaptive_radial_phase9_contracts_20260911/summary.json): machine-readable集計。
- [provenance.json](../../evidence/holonomic/adaptive_radial_phase9_contracts_20260911/provenance.json): HEAD、入力、timing条件、compiler。

timed callの外で `LensParams` を作り、warmでは準備epochを統計から除外した。cold/warmとも同じケース・同じ tolerance で比較した。contract benchmarkは1 repeat、whole-epoch paired benchmarkはrunner設定に従う。

## 観測結果

独立referenceはcold/warm各550行。各4行はreference自身の解像度差が要求値以上で `reference_usable=0`、観測violationは0件だった。これは独立solverによる形式的な誤差上界ではなく、reference self-differenceを使った監査である。

### contract benchmarkのvalueとwhole time

`whole_ms` は失敗を含むattempt time。順に p50 / p90 / p95 / p99 / max、単位はms。

| policy | mode | RelTol | value converged | whole time |
|---|---|---:|---:|---:|
| None | cold | 1e-3 | 108/110 | 1.214 / 1.651 / 1.745 / 17.051 / 19.156 |
| None | warm | 1e-3 | 108/110 | 0.996 / 1.504 / 1.706 / 11.796 / 12.742 |
| None | cold | 1e-4 | 110/110 | 1.221 / 1.688 / 1.810 / 17.261 / 19.040 |
| None | warm | 1e-4 | 110/110 | 1.041 / 1.438 / 1.979 / 11.930 / 13.203 |
| Strict | cold | 1e-3 | 110/110 | 2.364 / 19.531 / 24.899 / 58.642 / 64.779 |
| Strict | warm | 1e-3 | 110/110 | 2.521 / 15.723 / 29.161 / 44.302 / 57.040 |
| Strict | cold | 1e-4 | 110/110 | 2.295 / 14.227 / 21.208 / 36.233 / 45.590 |
| Strict | warm | 1e-4 | 110/110 | 2.370 / 15.914 / 29.049 / 44.408 / 56.865 |
| ValueFirst | cold | 1e-3 | 108/110 | 1.440 / 2.019 / 2.425 / 18.055 / 20.233 |
| ValueFirst | warm | 1e-3 | 108/110 | 1.213 / 1.903 / 2.367 / 12.389 / 14.086 |
| ValueFirst | cold | 1e-4 | 110/110 | 1.465 / 2.049 / 2.321 / 18.064 / 20.108 |
| ValueFirst | warm | 1e-4 | 110/110 | 1.268 / 2.068 / 2.785 / 12.552 / 14.326 |

ValueFirstのgradient qualityは、cold/1e-3で `ToleranceMet=210`、`FiniteUncertified=340`、cold/1e-4で `247/303`、warm/1e-3で `216/334`、warm/1e-4で `248/302`（各550成分中）だった。Strictはcold/1e-3で `ToleranceMet=168`、`FiniteUncertified=297`、`Invalid=85`。未認証とInvalidを同じ成功数に数えていない。

### setup内訳

contract benchmarkのRelTol=1e-3で、`setup_ms` の中央値は次のとおり。`frame/cuts/event/panel` も中央値で、各行の合計が全体中央値になるとは限らない。

| policy | mode | setup | frame | cuts | event | panel |
|---|---|---:|---:|---:|---:|---:|
| None | cold | 0.461 | 0.00003 | 0.00169 | 0.318 | 0.00226 |
| None | warm | 0.479 | 0.00003 | 0.00122 | 0.326 | 0.00088 |
| Strict | cold | 1.429 | 0.00003 | 0.00146 | 0.311 | 0.00155 |
| Strict | warm | 1.592 | 0.00022 | 0.00305 | 0.381 | 0.00444 |
| ValueFirst | cold | 0.716 | 0.00003 | 0.00273 | 0.319 | 0.00242 |
| ValueFirst | warm | 0.737 | 0.00003 | 0.00126 | 0.325 | 0.00098 |

全policy・cold/warm・tolのcontract rowsを合計すると、event check 2408、DD check 344、DD accept 344、qf family construction 440、qf refinement 1980（Strict 1984）だった。つまりDD tierはこの集合で344/2408 eventを受理した。一方、440/440 epochでqf familyが少なくとも一度必要だったため、lazy familyだけで全体を速くしたとは言えない。

## 既存テスト

- adaptive unit: **1375 checks / 0 failures**。
- isolatedの既存関連CTest 10件: **10/10 passed**。point images、quartic/Sturm、M7 reference、transport、finite-source binary、root pair、ODE、chart_p4、reciprocal chart、adaptive radialを含む。
- production routerとfixed-`n_r`経路の変更はない。

## 採用・保留・未解決

### 採用したもの

- value/gradientの状態分離とValueFirst。
- `Eabs <= max(Tol, RelTol*abs(Q))` の独立component判定。
- gradient invalidityのcomponent単位管理。
- qf familyのlazy constructionとepoch内event cache。
- eventのdouble/DD/qf ladderとsetup計時。
- event hi/lo保持、workspace capacity再利用。

### 採用しなかったもの

- 全event・全nodeの一律qf化。
- gradient未認証をvalue failureへ変換する旧一括処理。
- 5Jacのためにvalue meshを先に細分化するscheduler。
- 全K-ruleを高node数へ置換する変更。
- production default/routerへの昇格。

### 残る制約

1. `radius_lo` はpanel mapの局所計算では保持するが、現行V2 evaluatorの入口はbinary64 `R` である。したがってevent offsetをroot/phi/IFT全体へDD/qfで一貫伝播したわけではない。near-fold gradientの完全な解決とは言えない。
2. DD tierはdouble係数の再評価であり、物理parameterから高精度係数を再生成していない。
3. adaptive estimatorは収縮モデルに基づく `Estimated` であり、`require_bound=true` の厳密boundは未実装のまま `BoundUnavailable` とする。
4. ValueFirstの有限gradientは返すが、`FiniteUncertified` をanalytic sensitivityの精度保証とは解釈しない。
5. setupを測れるようにはしたが、value-onlyでもwhole epochはsetup/eventが支配的であり、今回のrawだけからproduction速度勝利を主張しない。

次に進むなら、event offsetを実際にroot-pair/endpoint/IFTへ局所的に渡すAPIを作り、near-foldのgradientだけを独立に検証するのが先である。全体qf化やfallbackの重量化ではこの契約問題を解決できない。
