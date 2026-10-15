# 最新追記: 両経路の仕事量調査・7案比較（2026-10-15）

正本は [REPORT.md](../results/recal2026/refinement-work-optimization-20261015/REPORT.md)、識別は同ディレクトリ `final-identity.json`、sourceは `final-source/`。**今回の大幅高速化は未達**。採用はscalar-v3の小さな算術処理削減と共通の任意trace。過去の408→61件修正は今回開始時点で済んでおり、今回の成果として数え直さない。

- 同じseed込み256入力の速度比（開始時/採用版）: Cartesian Uniform **0.9919/0.9968**、LD **1.0044/1.0032**。Polar Uniform **1.0081/1.0095**、LD **1.0099/1.0072**（1e-3/1e-4）。Cartesian横ばい、Polar約1%の微小差。
- native内訳: Cartesian初期fillが中央値約73%、全fill83%、誤差集計7%、管理0.5%。Polar細分化57%（LD43〜48%）。seed通常1〜2%。Python側の準備費用を除いて計測した。Cartesian1e-3は約4分の3が細分化なし、Polarの最後の試行は細分化時間の約2%。共通の順序変更だけで大幅改善する仮説は支持されなかった。
- deficit/費用の共通優先順位、高次角度則、axis制限緩和、画像内部block、2種類のSIMD実装を試して不採用。管理・判定費用が勝つ。全source/raw/単体結果と不採用理由を保存。SIMDの100万本のbit一致やblock実点照合の成功を、速度改善として扱わない。
- SIMD試作で未使用block hookの先頭隣接点の問題を専用2例で修正したが、経路全体を外したため**本採用には含めていない**。再開するなら `candidate-v7-source/src/lcbinint/magnification/adaptive_cartesian_run_fill.hpp` の修正を先に適用する。productionではinterior callbackは未設定。
- 最終版で両手法各14,432条件を再実行し、開始時と全result/例外が一致。Cartesian既知の精度外受理61・例外312形状は残り、Polar全件受理・精度外受理0。独立Cartesian216条件、Polar882条件も開始時と数値/状態一致。Polarは既知の1e-5難例1未達（881受理）を保持し、driverは新旧ともexit1。882全成功とは数えない。
- 最終132+189内部チェック通過。native SHA **127f31962354a866d10fc8d6946266deb80b6d65b275c1bcfe6035630cb3ce11**、全件検証版とbyte一致。再リンクしたrunnerも計時済み版とbyte一致。`certified=false`, `complete_error_model=false`。
- 全ての実験スケジューラ/内部block/SIMDフラグはproductionから除去。`LCBININT_DIAGNOSTIC_REFINEMENT_WORK_TRACE=1` だけが共通の計測用sidecarとして残る。初期格子・要求精度・停止条件は変えていない。`1.09e-4`を狙った保守化なし。

未commit・未push。stage済みindexのhashを開始時と照合し不変。6パネルは今回再生成していない。

---

# 最新追記: Cartesian誤差処理修正と採用版の確定（2026-10-15）

今回の採用は **Cartesian v7・初期16×16**。正本は [REPORT.md](../results/recal2026/cartesian-final-redesign-20261015/REPORT.md)、source/native/input識別は同ディレクトリの `final-identity.json`、source写しは `final-source/`、開始時dirty sourceとの差分は `cartesian-before-this-turn.patch`。

- seed込み256入力のケース別速度比中央値（今回開始時/採用版）: Uniform **1.0788 / 1.0770倍**、LD **1.0544 / 1.0592倍**（要求1e-3 / 1e-4）。5〜8%程度の改善であり、大幅な高速化とは呼ばない。
- 全14,432条件の要求精度外受理 **408→61**（Uniform12/26、LD4/19）。以前の誤受理318条件は正確な受理に改善、33は未達として返す。以前正確だった条件から新しい超過4。例外312形状は残る。
- capだけの修復で内部誤差が消える不具合、外側補正の重複、急変行での差分相殺、端点位相の偏り、欠落画像補修の二重加算を修正。Newton評価と保持領域チェックの重複を削減。
- `1.09e-4`の軽微な超過は追わず、一律安全係数・ケースID救済・楕円モデルは追加していない。残る61にはそれより大きい超過もあり、`certified=false`, `complete_error_model=false`を保持。
- 独立216条件は誤受理4→0、114受理・72例外・30未達。最終132内部チェック、全有限13,808条件のflux合計確認、固定格子9比較を通過。Polar全14,432の値/推定誤差/評価数/受理状態は前版と完全一致、189内部チェック通過。
- 縦8開始と高次境界モデルは試して外した。必要な粗格子チェック後の縦8版はLD1e-4が **1.019倍** に留まり、採用版1.059倍を下回る。パイロットの1.57/1.28倍を採用版の性能として使わない。不採用版のsource/native/raw/全件検証も保持。
- 最終再ビルドのnative SHA `cbb4f10d7725786de0656e57bcccc64c1e041ddd0e474d2a1b2a662c088fd278` は検証済みv7とbyte一致。最終archiveから再リンクしたrunnerも計時済みrunnerとbyte一致。archiveの違いはmtime欄のみ。
- **既存6パネルは今回のCartesian修正前**。今回、図は再生成していない。下記旧節のCartesian違反408や実装完了等の記述は履歴として読む。

未コミット・未push。元のstaged作業を保持し、masterへmergeしていない。

---

# 最新追記: Polar誤差推定の再設計（2026-10-15）

下記の旧「実装・検証完了」節は以前の版の記録。現在の採用版は **v43**。今回の正本は [REPORT.md](../results/recal2026/angular-estimator-redesign-20261015/REPORT.md)、source/native/input識別は同ディレクトリの `final-identity.json`、ソース写しは `final-source/`。

- 同じ初期16/2・seed込み256入力のケース別速度比中央値: uniform **1.238 / 1.335倍**、LD **1.262 / 1.358倍**（要求1e-3 / 1e-4、今回開始時比）。
- 独立比較の推定誤差/実誤差中央値は約 **151→29 / 143→12**。
- 全14,432条件は全受理・要求精度外受理0。独立882条件は881受理、極端1e-5で未達1、要求精度外受理0。以前の「独立882全受理」は現在版には適用しない。
- 新しい144条件のうち独立参照が安定した138で誤受理0、6は参照が時間切れ。189内部契約は通過。
- **LD1e-3のVBM勝利は未達**。同じ128 LDの再計時VBM/Polar中央値 **0.8705 / 3.1578**。1e-4では勝つが1e-3はまだ遅い。
- 角度則・LD端点重み・適応精度の根判定・平坦索引を採用。楕円画像近似、ケース専用救済、手法ルーターは追加していない。最後のv47/v48は正確だが追加速度改善が確認できず外した。
- `certified=false`, `complete_error_model=false`を保持。今回未コミット、元のstaged/dirty作業保持。

## 6パネル全件再計時の追記

`figures/recal2026/estimator-redesign-fastest-six-panel-20261015/` にCartesian/Polarの実測で速い方を選ぶ4枚を保存・表示確認済み。VBMも全7,216入力で再計時。比中央値(VBM/採用): Uniform1e-3/1e-4 **0.2731 / 0.4125**、LD **1.3396 / 4.6332**。これは実測後の選択で実行時ルーターではない。選択結果の誤受理はCartesian由来で **44 / 100 / 26 / 82件**。Polar単独は14,432条件全て精度達成、偽受理0。計時raw・失敗・除外・選択・精度集計を保持。全条件で少なくとも一方の結果があり除外0。

---

# Cartesian / Polar 最適化（2026-10-15）

実装・検証完了。branch `feature/boundary-error-and-discriminant-support`、HEAD `ee4d1c4e`。今回の変更は未コミット。実行したdirty sourceとnativeの正本は `final-source/`、`final-identity.json`。

## 採用した変更

- adaptive Polarの初期半径グリッドを4→2へ変更。Python APIとseed込みrunnerの既定値へ接続した。明示的な初期4も使用可能。Cartesian初期16と固定グリッド経路は維持。
- Polar LDの内部半径予算をabsolute-density 2%→8%、relative-column 8%→32%へ再配分。独立した物理評価点、8倍の半径指標、境界不確かさ、角度＋半径の最終受理条件と総評価上限を保持。
- 相対要求の半径予算に、指定のrelative_toleranceを直接使う。角度制御が更新した全体フラックス予算を古い粗い増光率で割り続ける不整合を修正。absolute要求は更新後の初期増光率を配分の目安に使う。
- Cartesian adaptive linear LDは端の明るさも解析式にし、不要な20001点の明るさ表の初期化を省く。legacy・固定グリッド・非linear経路の表は保持。Uniformではこの判定用の環境変数参照も実行しない。

## 最終の速度比較

256形状、profileごと128。CPU0・1 thread、同じ入力で実行順を交互にしcold3回中央値を比較。coldはseed cache miss＋積分であり、中心像・caustic構築・LD表の事前準備は含まない。両版が受理し参照精度を満たす組の比を集計し、失敗・誤受理はJSONに全て保持。

|光源|手法|要求|精度達成した比較数|基準時間 / 最終時間 中央値|基準ms中央値|最終ms中央値|
|---|---|---:|---:|---:|---:|---:|
|uniform|cartesian|0.001|117|0.9943|0.2891|0.2868|
|uniform|cartesian|0.0001|114|0.9915|0.3101|0.3126|
|uniform|polar|0.001|128|1.0758|0.5380|0.4563|
|uniform|polar|0.0001|128|1.0623|0.6536|0.5713|
|linear|cartesian|0.001|112|0.9928|0.3388|0.3458|
|linear|cartesian|0.0001|109|0.9964|0.3913|0.3923|
|linear|polar|0.001|128|1.1203|0.6410|0.5256|
|linear|polar|0.0001|128|1.0644|0.7571|0.6524|

Cartesianの継続計算はほぼ同等で、中央値の時間比0.991〜0.996を速度改善と主張しない。Cartesianで双方が有限だった全入力は増光率と写像評価数がbit一致。

初回費用は別の128形状で、LD表を事前準備せずwarmupなしの最初のcold呼び出しを比較。processごとの共通static初期化も両版に含む。Cartesian LD1e-3は基準/最終の中央値1.047、精度達成59組。LD1e-4は1.008。通常のcold3中央値とは異なる計時範囲である。

## 精度・契約の検証

- 最終nativeの保存コーパス: Polar14432条件、全件受理、裁定後の精度超過・例外・契約違反・新規未達0。
- 独立参照・接触・極端形状: 882条件、全件受理、偽受理・新規未達・例外・契約違反0。要求1e-3/1e-4/1e-5、極端q/rho下限1e-6を含む。旧版で評価上限に達した極端1条件も達成した。
- 既存回帰1194条件、全件受理、偽受理・例外・契約違反0。旧行列の参照不確か3条件は不確かなまま保持し、精度確認済みに含めない。
- 内部契約189、API・解析極限38チェック、失敗0。再試行テストは初期4を明示して再試行そのものを検証した。初期2は再試行せず直接粗い格子を使う。
- 最終版の256形状・初回128形状の比較に新しい精度の後退0。

保存参照はVBM RelTol1e-6と独立裁定3参照であり、全域の数学的保証ではない。`certified=false`、`complete_error_model=false`、`CALIBRATED=false`を保持。

## 修正した実際の予算不整合

形状4295のLDは粗い増光率約1.95に対し、積分値約242.58。内部半径配分4倍の最初の候補では、更新後のtargetを古い約1.95で割ったため後続列の配分が過大になった。半径誤差が支配的なのに角度を細分化し続け、両要求で100万評価上限へ到達した。失敗した `selected-full-accuracy/` は保存した。

相対値を直接使った最終版は、1e-3で7566評価・参照差5.59e-6、1e-4で11551評価・参照差1.20e-7。両者とも受理。ケースIDによる分岐は実装していない。

## 不採用・残る問題

- Cartesian初期8: 全7216形状の確認で、以前達成した157条件が新しく精度を失ったため一律採用しない。小規模で1.5倍という結果を全体へ外挿しない。
- 初期Boole/Gauss次数比較、独立求積の収縮による局所係数変更: 128形状で追加列の費用が効果を消したため不採用。元の角度モデル・係数・実測証人を保持。
- 写像/微分の分母共有: 定常時間の有意な改善を確認できず、変更を残さない。
- Cartesianの既知の偽受理はUniform81/155、LD54/118条件（1e-3/1e-4）を保持。既知の評価予算例外312形状も残る。この作業でCartesian全体の精度問題を解消したとは主張しない。
- Polar角度推定の保守性も残る。今回の最大の全体改善を2倍などと呼ばず、計測した256形状の中央値を報告する。

## 成果物・再現

- 最終全域: `final-native-polar-full/`、独立参照: `final-native-independent/validation.json`。
- 回帰: `final-native-replay/`、契約: `final-native-contracts.json`。
- 時間: `selected-v3-final-paired/summary.json`、初回: `selected-v3-first-call-paired/summary.json`。
- 最終検証ledger: `final-validation.json`、実装identity: `final-identity.json`、実装コピー: `final-source/`。
- 以前の候補・不採用・失敗したharnessやCPU競合を含む測定は残し、採用版の時間として使用しない。

全域: `python3 tests/diagnostics/recal2026/optimization_grid_accuracy_20261015.py --name NEW_OUTPUT --current-native --cartesian-bins 16 --polar-bins 2 --radial-allocation 4 --methods polar`。
時間: `python3 tests/diagnostics/recal2026/optimization_paired_benchmark_20261015.py --name NEW_OUTPUT --polar-bins 2 --radial-allocation 4 --count 256`。既存出力の再利用にはnative/sourceの同一性を要求する。

今回の変更をGit stage/commit/pushしていない。既存のステージ済み6ファイルを保持し、masterへmergeしていない。
