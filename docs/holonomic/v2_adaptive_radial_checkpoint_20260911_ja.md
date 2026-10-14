# V2 adaptive radial checkpoint — 2026-09-11

基準は `dev/holonomic` の `3f35991a1c22f11de60f26a4c88c3628e5aa5556`。
変更は isolated holonomic のみ。既存の fixed `n_r` API、production router、共有 `.so`、未追跡 `.claude/` は変更していない。

## 判定

**明示的な experimental adaptive APIを実装した。production採用は保留。**
値の高増光難例はfold変換とradial refinementで改善した。一方、要求5Jacを全部満たすcoverageとwhole-epoch速度は、採用条件に達していない。失敗をfixed64への置換や判定緩和で隠していない。

許容誤差はユーザー訂正どおり、各要求出力 `Q` について

```
Eabs <= max(Tol, RelTol * abs(Q))
```

である。**加算式ではない。** 誤差の「寄与」同士は加算するが、絶対許容誤差と相対許容誤差はmaxで選ぶ。ゼロ近傍、負のJacobian、両項が等しい場合、および加算なら誤って通るケースをテストした。

## 実装

- `nested_fejer2.hpp`: open Fejér-II 7→15→31→63→127→255。dyadic IDを正規化し、旧点の座標をbitwiseに保持する。各段階で全sampleへ新しい重みを掛け直す。
- `adaptive_radial.hpp`: epoch内のsample/root snapshot cache、同一cellの近いanchor、p refinementとh分割。h子は親の物理値を流用せず、親snapshotはrootの初期値としてだけ利用する。no-cache A/Bは明示的なconfigで区別する。
- 誤差推定は補間差のweighted L2 normと収縮モデル。収縮条件は `detail_N <= 0.5*detail_previous`、またはdetailが既存の数値誤差floor以下。これは経験的モデルであり、将来の収縮を証明するものではない。
- 値・解析5Jacの最終正規化後にradial / inner / geometry / event / roundoffを合算する。ρ正規化の減算と誤差増幅も含む。
- `adaptive_epoch.hpp`: V2のD14/Sturm、quartic/root tracking、F0、vK、angular rescueを再利用。foldには片側二乗・両側sin² mapを使う。無条件endpoint insetや角度samplingによるtopology上書きは入れない。
- 解析微分ではmap Jacobianを掛けてから小さいendpoint微分で割る。K-ruleのpair微分は既存の正則な `E=O=0` の2×2 IFTを利用し、t/reciprocal両chartを試す。既存TMax/VFloor/K-rule gateを緩めない。controllerやnode位置をAD/FDに掛けていない。
- 既存K-ruleの8/16（既存24点gateが有効なvalueでは16/24）差を数値として返す小変更。5Jacには別途8/16微分差、angular rescueには32/64差を付ける。value gate通過を微分誤差の保証として転用しない。
- physical eventは元のquartic familyの `P=Pt=0` をqfで局所補正する。根の分離・有限性をscreenし、cell幅を跨ぐ補正は拒否する。doubleへ戻した際の不確かさを消したことにはしない。
- 既存plannerが近接eventをmergeした結果、保持済みphysical eventがcell内部に残る場合だけ、そのcellを再分割しSturm再分類する。108ケース中22件でこの状況があった。従来empty扱いの微小区間も検査対象になる。ただし、この修正で5Jacの収束件数は改善しなかった。
- 既存plannerがskipした区間を検出した場合は、無条件のゼロ寄与とせず `TopologyUnresolved`。D14の上流で既に失われたeventを復元できると主張しない。

## APIと保証の意味

```cpp
#include "lcbinint/magnification/holonomic/adaptive_epoch.hpp"
using namespace lcbinint::holonomic;
AdaptiveConfig cfg;
cfg.tol.mu_atol = 1e-8;  // Tol
cfg.tol.mu_rtol = 1e-4;  // RelTol
AdaptiveWorkspace workspace;
auto result = epoch_value_adaptive(p, u, cfg, workspace);
// 使用側は result.stop == AdaptiveStop::Converged を確認する。
```

5Jacは `epoch_jacobian_adaptive`。許容誤差は `grad_atol[j]` / `grad_rtol[j]` に成分ごとの単位で指定する。warmは `epoch_adaptive_prepared` と既存 `PreparedEpochGeometry` を使う。再利用するのは認証付きD14 stateで、physical sampleはepochごとにresetする。

`AccuracyAssurance::Estimated` は事後推定。厳密な区間上界ではない。`require_bound=true` は `BoundUnavailable` を返す。任意の関数に対するaliasingの不可能性を証明しておらず、同梱の `U15^2` 反例も残した。off-grid sentinelやanalytic envelopeによる厳密保証は未実装。

予算超過・event/inner/roundoff制限・非有限値・topology不確実性は別stop reason。現在は複合要求の一つでも失敗したら収束flagを保守的に全てfalseにする。返る最後の近似値を「成功値」として使わない。`max_bytes` はsample/panel/root snapshotの保持領域に対する予算で、静的tableやtopology一時領域を含むprocess RSS上限ではない。

## 再現・測定範囲

```sh
bash checks/holonomic/run_adaptive_radial.sh evidence/holonomic/adaptive_radial_20260911 3
```

入力は既存108ケースと、高増光難例のuniform/LD各1件、計110ケース。tol ladderは `1e-3,1e-4,1e-5,1e-6,1e-8`、値のatolは `1e-8`、各Jacのatolは `1e-6`。u=0と0.5を分離。

- `paired.csv`: cold/warm、value/5Jac、cache有無、3反復、計13200行。caseごとの3反復中央値から集計する。実行順は交互。LensParams初期化、静的table初期化、warm-up epochは測定外。
- coldは毎回D14/topologyを再生成。warmは双方とも同じ前epochから `xs += 0.01*rho`。未認証sampleのepoch間再利用はしない。
- `controls.csv`: mapped fixed63 / adaptive affine / adaptive fold、計3960行。fixed63は解像度比較であり、tolerance成功を主張しない。
- `reference*.csv`: 独立GL128/256 radial + direct angular256/512。Sturm分類・必要時のqf root isolationをgeometryに用い、Fejér/vKを参照積分には使わない。topology/lens式は共有するので完全独立のsolverではない。
- 参照の128/256差が要求誤差以上、または参照不能の行は `reference_usable=0`、`violation=-1`。差が小さいことも厳密な包含証明ではない。
- `jacobian_reference.csv`: 通常例と高増光難例、全5成分、3つのFD step、2つの独立求積解像度。非有限参照もそのまま残す。
- `limits.csv`: 全110ケース・全tolについて、5Jac失敗を最終出力成分ごとの誤差ledgerで診断。
- `summary.json` / `provenance.json`: 集計、source hash、compiler、CPU、flags、測定条件。

`ladder_initial_unmatched_warm.csv` は初期診断で、warm adaptiveとcold fixed64を比較してしまっている。**性能判断から除外した。** `ladder.csv`、`restored_cuts_ladder.csv` は1反復の開発診断で、最終の対等な性能表は `paired.csv` を使う。初回runnerは実行中のscript更新で末尾の集計に失敗したため、最終runはscriptを固定してやり直した。

## 残る原因と未実装事項

1. **厳しいρ微分のevent誤差。** qfでeventを補正しても、評価半径をdoubleに戻すULPが残る。foldの微分は局所的に `1/sqrt(R-a)` となり、欠落寄与推定は `sqrt(delta_R)` に比例する。小さいρの正規化・減算がこの誤差を増幅する。局所Newtonの追加だけでは解消しなかった。次の根本対応はRとevent offsetを分離して保持し、必要なroot/φ/IFTをDD以上で一貫評価すること。単に閾値を緩めたり、rescueを重くして終える案は採用していない。
2. **内側微分誤差。** 8/16または32/64の微分差が要求値を超える場合、radial refinementだけでは直らない。現在は制限を返す。全難例での安定なnear-fold微分とinner精度昇格は未完了。
3. **幾何誤差の推定。** endpoint/lens-map cancellation・event残差のledgerは数値モデルであり、coefficient/branch/global topologyの区間保証ではない。特にFhalf幾何感度の包括的上界は未実装。
4. **性能。** node数削減がwhole-epoch勝利に直結していない。event補正・family生成・plan setupが追加される。root snapshotのvector領域やschedulerにも固定費がある。成功しない5Jacを速い結果として評価しない。
5. **検証範囲。** 通常例の5Jacにはstep studyがあるが、全108ケースの5Jacを独立高精度oracleで認証したわけではない。難例のFD参照不能も残る。productionのtol APIに昇格させる根拠はまだ不足している。

以下の数表は最終raw結果から生成する。

## 最終実測

単体テストは **1369 checks / 0 failures**。既存V2関連とadaptiveを含む7 CTestも通過。

値の独立監査はcold/warm各550条件。参照精度で判定可能なのは各546条件で、収束扱いの要求誤差超過は観測0件。残る各4条件はon-axis-off / caustic-crossの最厳tolで、参照の解像度差が要求値以上のため合否に含めない。これは全誤差の形式保証ではない。

### 値・5Jacの収束件数（cacheあり、uniform+LD、110ケース）

| RelTol | cold value | warm value | cold value+5Jac | warm value+5Jac |
|---|---:|---:|---:|---:|
| 0.001 | 110/110 | 110/110 | 68/110 | 67/110 |
| 0.0001 | 110/110 | 110/110 | 48/110 | 47/110 |
| 1e-05 | 110/110 | 110/110 | 34/110 | 34/110 |
| 1e-06 | 110/110 | 110/110 | 14/110 | 13/110 |
| 1e-08 | 106/110 | 105/110 | 10/110 | 10/110 |

### whole epoch（RelTol=1e-3、ms）

**全attemptの時間**。失敗を含むので5Jacについて同精度の速度勝利を意味しない。各caseの3反復中央値を母集団にする。

| u | output | mode | adaptive p50 / p90 / p99 | fixed64 p50 / p90 / p99 | adaptive success |
|---|---|---|---|---|---:|
| 0.0 | value | cold | 1.129 / 1.461 / 9.396 | 0.807 / 1.260 / 8.831 | 55/55 |
| 0.0 | value | warm | 0.959 / 1.400 / 7.630 | 0.753 / 1.276 / 7.429 | 55/55 |
| 0.0 | value+5Jac | cold | 1.259 / 2.504 / 34.252 | 0.968 / 1.898 / 10.006 | 34/55 |
| 0.0 | value+5Jac | warm | 1.117 / 4.339 / 36.837 | 0.963 / 1.931 / 7.716 | 33/55 |
| 0.5 | value | cold | 1.193 / 1.525 / 9.725 | 0.855 / 1.662 / 9.685 | 55/55 |
| 0.5 | value | warm | 1.009 / 1.623 / 7.700 | 0.827 / 1.681 / 7.553 | 55/55 |
| 0.5 | value+5Jac | cold | 1.369 / 4.340 / 40.904 | 0.993 / 1.994 / 9.786 | 34/55 |
| 0.5 | value+5Jac | warm | 1.301 / 5.667 / 49.522 | 0.972 / 1.940 / 8.426 | 34/55 |

### LD valueのstage費用（RelTol=1e-3、各stageの中央値、ms）

各中央値は別集計なので和はwhole中央値と一致しない。setupにはqf family/event補正とplan準備を含む。

| mode | topology | setup | physical nodes | estimator | scheduler |
|---|---:|---:|---:|---:|---:|
| cold | 0.7215 | 0.3285 | 0.1150 | 0.0042 | 0.0008 |
| warm | 0.5598 | 0.3203 | 0.1465 | 0.0040 | 0.0008 |

### cacheとfold mapの対照実験

RelTol=1e-4、LD value。no-cacheでも旧座標を再評価する以外の誤差条件は同じ。fixed mapped63にはtolerance判定はない。

| mode | variant | p50 ms | median physical evaluations |
|---|---|---:|---:|
| cold | adaptive no-cache | 1.270 | 113 |
| cold | adaptive cached | 1.168 | 69 |
| cold | mapped_fixed63 | 1.342 | 189 |
| cold | adaptive_affine | 1.278 | 197 |
| cold | adaptive_fold | 1.111 | 69 |
| warm | adaptive no-cache | 1.165 | 113 |
| warm | adaptive cached | 1.058 | 69 |
| warm | mapped_fixed63 | 1.461 | 189 |
| warm | adaptive_affine | 1.153 | 197 |
| warm | adaptive_fold | 0.968 | 69 |

### 高増光難例（cold value、RelTol=1e-6）

| u | adaptive μ | fixed64 μ | independent GL256/angular512 μ | adaptive nodes |
|---|---:|---:|---:|---:|
| 0 | 12363.402166976 | 12392.519642933 | 12363.402166972 | 480 |
| 0.5 | 12152.917846268 | 12182.273604358 | 12152.917845189 | 480 |

### 解析Jacobianの独立検証の限界

- `paper_highA`: 60 FD/reference行中、有限な参照は34行。
- `plan15`: 60 FD/reference行中、有限な参照は60行。

通常例は3stepの中央差分と整合。高増光難例では、微小にパラメータを動かした参照側のtopology/root解決にも非有限行が残る。その行は正解値として使わない。高増光のvalue成功を5Jac成功へ読み替えない。

**採用判断:** 値の誤差制御と高増光改善は確認できたが、現在の追加setup費用と5Jac coverageではproduction採用条件に届かない。固定64点が要求精度を満たさないケースを除いたvalueの公平な速度比も `summary.json` に保存した。今後はevent offsetを保った微分算術と内側誤差を先に解決し、その後にsetup再利用を検討する。
