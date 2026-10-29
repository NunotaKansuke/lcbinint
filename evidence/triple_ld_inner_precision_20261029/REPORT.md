# Triple LightCurve LD精度の最終修正

`experiment/algebraic_boundary`、基点 `53335b9a`。残っていたLDの小さな角度微分の反例を修正し、最終DSO `863253398bab41c1d9fbc3018282234fbad67ad3ac57ad9d649ea9b32cb5f7df` で再検証した。元の値rtol=1e-3/1e-4、atol=0を維持する。

|最終検証|結果|
|---|---:|
|LightCurve条件 / epoch|1880 / 8280|
|値のみ / 全10微分付きの成功|1880 / 1880|
|以前の成功からの分類回帰|0|
|参照を採用できる値比較|4800、要求誤差超過0|
|最大値誤差 / 要求誤差|0.816735|
|独立微分参照|90、最大scaled差2.198e-7|
|不変性|170曲線 / 510検査、超過0|
|最大不変性scaled微分差|2.378e-5|
|C++ / Python対象テスト|34 / 260成功|
|legacyテスト|8成功、以前と同じ9失敗ID|

Python260件は、258件の対象一式と追加後の8件の新規ファイルを検証したもの（6件重複）であり、全repo pytest成功の主張ではない。計算予算は最大8 CPU / RAM16GiB、各ライブラリ1 thread。native worker最大観測RSSは70584KiB。全計算はforegroundで完了を確認した。

## 1e-3で微分がずれた原因と修正

値のrtolは、打ち消しで小さくなる微分の相対精度を保証しない。今回は二つの数値的な問題を分離した。

* **半径方向の不足**：LDのfluxがほぼ平坦なchartでは、値の収束が早くても角度のmaterial transportが未解像だった。unweighted LD Pattersonに固定31点の下限を置き、940/945の小さなalpha/ang微分を改善した。値だけによる適応制御は維持する。
* **critical角度領域の係数微分の桁落ち**：1070ではordinary momentを除いても直らず、critical係数の微分結合を除くと直った。同じsinh chartの同じ積分点で元の物理Phiの全parameter jetを評価し、係数微分同士の大きな打ち消しを避ける。fluxは既存の因数分解を保ち、物理端点近くのscalar subtractionを避ける。局所的な丸め誤差guardは既存の1e-5を使い、必要なsampleのみbinary128へ昇格する。

値と全IFTを同じfrontier・同じ点で運ぶ。微分誤差による適応、受理済み値積分の再実行、微小像や微分列の省略、source-disk production shortcutは追加しない。optional material-map compile modeの従来の係数transportは維持する。

940/945/1070の6角度微分は、値rtol=1e-9/1e-10、複数の中心差分刻みと比較して最大scaled差1.423e-6。これは厳しいnative値経路による差分検証で、独立エンジンの微分参照とは区別する。6数値は `ld-angular-accuracy.json` に保存した。新規8テストは旧採用DSOで6失敗・2成功、最終DSOで8成功した。

1070の周期変換の最大scaled差は4.106e-7、時刻原点移動では3.099e-6。新規単体テストは変換ごとに新しいmodelを作り、campaign不変性検査は同じmodelを順に使う。両方を確認した。

より厳しい1e-9の局所guardと追加per-cell floorを組み合わせた中間候補は、788で60 CPU秒上限に達したため棄却した。最終版では788が約1.8秒で正常終了し、元の1856条件3712実行も全件成功した。試作の失敗を最終成功の中へ混ぜず保存している。

## 速度

公開API全曲線。import/constructor除外、CPU0固定、cold1回、warmup2回、thread CPU9回の中央値。前は `53335b9a` のDSO `9e14a915…`。同じ入力、rtol、LD、thread数で比較した。外部負荷・CPU周波数は固定していない。

|case|全微分 前→最終 ms|
|---|---:|
|21|0.815→0.819|
|439|2.119→2.117|
|499|5.781→5.813|
|940|257.618→284.079|
|972|1396.881→1060.582|
|1508|863.147→992.882|
|1070|48.352→50.687|

軽い経路と修復済み439/499の費用を維持した。重いLDでは今回の固定下限と物理jet評価により増減があり、全ケースの高速化とは主張しない。全15曲線の値/微分測定、cold、全sample、DSO hashは `timing-summary.json` とarchive内のrawにある。

VBM SinglePoly / MultiPoly / NoPolyの値・失敗分類・独立参照による誤値除外、および以前のpole partition / rho微分の速度改善は、[前報](../triple_component_completion_20261029/REPORT.md) を引き継ぐ。VBMは変更していない。今回の4800比較でも同じ参照採用gateを維持し、単にVBMが正値を返したことを正解の根拠にしない。新しいnative warm時間と以前のVBM fresh時間から速度倍率を作らない。

## 再現と適用範囲

`SUMMARY.json`、`PROVENANCE.json`、`raw-evidence.tar.gz`、`raw-evidence-manifest.json.gz` を保存した。archiveには採用/棄却候補のraw、全実行の入力hash、source/wrapper/test snapshot、campaign/worker/入力、参照、検証スクリプトを含め、全memberのSHA-256を照合した。native binaryはローカルに保ち、Gitにはhashと再build用sourceを保存する。前報の独立参照archiveのhashも記録する。

現在のcheckoutで再検証する入口はarchive内の `physical_final_regression.py`、`physical_final_check.py`、`physical_final_check_prior.py`、`physical_final_invariants.py`、`physical_final_summarize.py`。build後に `physical-final-package/lcbinint` へ5wrapperとDSOを配置し、DSO hashを確認する。大規模再実行は保存済みcampaignへ `WORKERS=4 TAG=<new-tag> VARIANT=<package-relative-path> INCLUDE_HOLDOUT=1` を指定して `--full` で実行し、元のrawを上書きしない。

今回の完成範囲は、static・single-source・一様/線形LDの公開triple LightCurveで発見した停止、誤値、rho微分欠落、小さなLD角度微分の反例の修復と、この有限の検証集合での回帰解消である。任意の配置や全微分精度を認証したわけではない。特に極小質量比の個々の小さな質量微分まで独立に認証していない。`gradient_valid` は有限で利用可能な微分を意味し、値rtolから任意の微分誤差保証を導かない。今回確認された未修復の公開API失敗は残っていない。
