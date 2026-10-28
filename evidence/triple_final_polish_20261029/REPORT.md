# Triple LightCurve: contact recovery, certified annular bounds and LD work reduction

## 結果と適用範囲

既存 `experiment/algebraic_boundary` checkoutで、base `09f5261d` から継続した検証。新規worktree・branch・pushはない。別作業のdirtyファイルは変更しない。最終native DSO SHA256は `85ef8301e253b0c8ff65b9de7d6cd66dbef8c76ac4aa2218b504d79e00313c37`。source hashesは `mass-bound-provenance.json`。別作業のdirty JAX関連変更は今回に含めず、そのhashとbuild flagsを `environment.json` に記録した。DSO hashは実際に検証したバイナリの識別子で、clean checkoutからbit-identicalに再ビルドできるとは主張しない。static/single-source triple、一様・linear LD、通常の `LightCurve(lens="triple", param_type="vbm")` が対象。rtol=1e-3/1e-4、atol=0を維持した。

1528条件・6520 epoch、wide_random400、near_collinear_or_clustered120、rho=0の48、sampled_caustic_contact960を同じ入力で再計算した。

| 全曲線の成功 | base09f5261d | 今回最終 |
|---|---:|---:|
| 値 | 1460/1528 | 1503/1528 |
| 値＋微分 | 1448/1528 | 1501/1528 |
| 以前成功→今回失敗 | — | 0 |

61条件・96 engine-curvesが回復。接触近傍960条件とrho=0の48条件は両経路とも全件成功。全3056 raw結果に入力・DSO hashを保存した。最終全件結果は `mass-bound-status.json`、比較は `annulus-comparison.json`。

残件は値25例外、微分26例外＋1 invalid_gradient。すべてwide_randomまたはnear_collinear_or_clusteredで、`remaining-failures.json`に個別入力と公開stopを記録した。失敗やgradient_validを精度保証に読み替えない。本検証は一般のtriple配置全体やfitへの無条件な使用を認証しない。

## 採用した修正

1. Angular root bracketがtan/atanのbinary64往復で広がり、互いに重なる場合だけ、物理方程式・符号・高精度区間Newton/二分法で再収縮する。根の完全性・個数・区間の検証を維持し、距離による根の統合は行わない。
2. double端点Newtonの最後の局所修復が拒否された場合に限り、元の候補からquadで再検証する。不確定な負の中点phiもquadで正を確認できた場合だけ採用する。許容誤差や像数チェックは緩めない。
3. complex chart候補の分割点がdoubleで一致すると、material motionに0/0が入り、値は収束しても微分がNaNになっていた。物理半径・asinh座標の全区間が表現可能かを置換前に確認し、不可なら既存のreal-radius cellsを保つ。
4. 元のpolar frameで積分が拒否された場合だけ、別の物理レンズを原点にして同じimage-area積分を再試行する。新しいsupport proofと同じ許容誤差を使い、完全に収束した結果だけを採用する。VBMやsource-disk積分へのproduction fallbackではない。元のframeの成功結果は置換しない。
5. レンズ順序の変更では質量制約 m0=1−m1−m2も微分する。順序を戻した質量adjointをhとすると、元の質量微分はh1−h0、h2−h0。全6順序・11物理座標・2差分幅のC++検証を追加した。
6. LDのquad評価で、同じray上のquad端点をすでに保持している場合は、その値を再利用する。丸めたdouble端点から同じNewtonを再実行しない。この変更単独では、3056全件の分類、成功点の値・全微分がbitwiseで一致した（`reuse-comparison.json`）。
7. イベント半径の不確定性を常に全円環で評価すると、rhoの小さい配置で、その上限だけが予算を超え、他の積分誤差を減らしても収束しなかった。物理的に囲めた角度範囲で、この上限を保守的に狭めた。

最後の修正の根拠：R∈[lo,hi]がレンズ極を横切らないとき、固定thetaでのレンズ写像の変化率は L=1+Σ m_i/d_i² 以下（d_iは円環と各極との最小距離）。したがって元の光源円内に写る全ての角度は、中間半径における半径rho+L(hi−lo)の拡大光源内に入る。既存のcertified angular topologyから角度測度の上限Mを得て、元の全円環誤差上限にM/(2π)を掛ける。拡大半径の算術は外向きbinary128、arcの丸め分も加える。質量補数が丸め境界で0をまたぐ場合にも、Lには区間質量の絶対値上限を使う。極を横切る・証明が通らない場合は従来の全円環上限1へ戻す。許容誤差、元のsupport、実際の積分値は置換しない。

`test_annular_area_bound.cpp`で単レンズの解析的角度測度と20002半径点で比較し、全角・空集合・極のfallback、質量補数が0をまたぐ境界での解析的下限も検査した。境界補強の前後でも、全3056件の分類・成功点の値・全微分はbitwise一致した（`mass-bound-comparison.json`）。これは直接の積分誤差を隠す上限切り捨てではない。予備診断ではcase314の誤差評価0.001596中0.001562、case361の0.00018492中0.00018467がイベント全円環上限だった。新方式で値の非収束6条件はすべて回復した。

## 参照の監査と精度

VBM5.5のSinglePoly/MultiPoly/NoPolyによる同一入力の元比較は、`triple_stress_expanded_20261028/rows`を再利用。回復点にはMultiPoly/NoPolyの厳しい参照を追加した。

今回、旧LD参照など4076 method-evaluationsを、VBMの診断用コピーでRelTol=1e-7・maxannuli=100にして監査した。3226件は分割上限に達し、therrが指定精度を満たしていなかった。これは直ちに要求精度1e-3/1e-4を超える誤値という意味ではない。しかし「1e-7で収束した参照」とは扱えない。MP/NPの一致だけではこの問題を検出できない。

そのため旧精度集計の3180比較を除外し、独立LD積分を追加した。VBMの一様diskを別のGL積分で線形LDへ合成し、少なくとも3段階の積分次数、最後2回の変化、MP/NP間差を確認した。234対象点中195点が両参照の収束gateを通り、39点は未収束または外部solver異常終了として除外した。この195点でnativeの要求誤差超過は0。

回復した小光源6条件には、VBM点光源をsource disk上で別途積分する診断も行った。これは検証scriptだけであり、production LightCurveからは一切呼ばない。GL4/8/16、角度16/32/64点、linear LDの重みを使う。MP/NPなど2手法でmesh収束・相互一致が通った点だけを厳密な比較集計へ加えた。

case189/index2では、厳しいVBM MultiPoly有限光源値が約7.27e11、NoPolyが約4.34753546。MultiPolyの点光源disk積分にもmesh依存があり、SinglePolyの別積分も約2.9893で一致しなかった。nativeはNoPoly有限光源・NoPoly点光源disk積分と一致するが、2手法gateを通さないので、この点は1931比較には含めない。巨大な正の返値を正常値として扱わない。

最終の収束・参照gateを通した精度集計は212条件・972source points・1931比較、要求誤差超過0。最大相対誤差/要求誤差=0.81673459。`final-validated-reference-summary.json`と全比較行に参照の種類を記録した。この数は全6520点の精度認証ではない。

## 微分と回帰検査

- 現行契約のPython回帰90件pass。元の16接触失敗、回復した円環上限6条件、各値/微分のparity、独立参照を含む。
- 最終sourceで再ビルドしたC++テスト31/31 pass。
- 接触case988の10パラメータをVBM MultiPolyの独立中央差分・3段階h・Richardsonで検証した。NoPolyは摂動でSIGSEGVになり、参照成功には数えない。
- 再frameが必要なLD case972のt0微分は差分幅への依存が強かった。独立LD積分256/512/1024点、h=1e-8/5e-9/2.5e-9を保存。Richardsonは−302805.68→−302939.95→−302957.86、nativeは−302985.14。診断許容差3e-4で比較し、単一の粗い差分値で微分バグとは判定しない。
- 旧backendを期待する9テストの失敗は、修正前09f5261dの固定DSOでも同じ9件を再現した。期待値を書き換えて隠していない。旧テスト8件は両buildでpass。

最終DSOで154曲線・462件について、alpha/angの2π変更、時刻原点の+7変更、epoch逆順＋重複の不変性も再確認した。status/valid/値の許容誤差超過は0。最大相対値差2.94994e-09、最大scaled微分差4.12011e-06（wideのみ1.41648e-06）。これは物理的な独立微分精度の証明ではない。

微分のfinite/gradient_validと、全点・全パラメータの物理的微分精度は別物。微分誤差を使うmesh制御は導入していない。

## 修正に使わなかった追加240条件

seed=2026102907で新規geometryを生成した。moderate128、wide80、near_collinear16、rho=0のpoint16、各5 epoch、linear LD=0/.5/1、rtol=1e-3/1e-4。入力は `holdout/cases.json`、6 worker・CPU20s/worker・RAM2GiB AS/workerで、通常APIの値・微分とVBM3手法、さらにMultiPoly/NoPoly RelTol=1e-7の参照を計1680 worker-evaluationsで比較した。

nativeは値240/240・微分240/240で全件成功。厳しい両参照のerror estimate・相互差を通した239条件・1179点・2358比較で要求誤差超過0、最大誤差/要求誤差=0.15002173。参照のerror estimate未達16点と参照solverが停止した1曲線5点は精度集計へ入れない。`holdout/validation-summary.json`に除外理由を含めて記録した。

VBM MultiPoly/NoPolyの通常tolでの呼出しは各240件がfiniteで完了し、参照gateを通した点では要求誤差超過0。SinglePolyは229件がfinite返値、CPU上限9件・SIGABRT1件・invalid value1件。finite返値でも48点が要求誤差を超え、最大例は約7.49e16対参照約1.20152だった。finite completionを精度合格と扱わない。

既存matrixと合わせると1768条件・7720 epoch、native値1743/1768・微分1741/1768。既知matrixの1931比較と未使用条件2358比較、計4289比較で要求誤差超過0。精度を確認した点は計2151点で、全7720点の認証とは分ける。新しい240条件での微分について独立に検証したのは値とfinite/valid契約であり、全微分の物理精度を保証しない。

## profileと速度

専用header-only profileは同じ物理kernelのO2 buildで、nested scopeをexclusive CPU時間に分けた。通常LightCurve全体やO3 DSOの直接profileとは区別する。case1508の中央epochでは約329.5msのうちangular約198.2ms、endpoint約74.8ms、chart約28.4ms。quad端点再利用後は約296.0ms、endpoint約42.3ms。同じ値・packet数を維持した。case940でも約298.2→263.4ms。scope計測の固定費・外部負荷は含まれる。

別frameで回復するcase972では、拒否される元frameに多くの時間を使う。公開node数は採用frameの値であり、全retryの合計ではない。物理frameの選択をcall履歴のcacheで変えるとepoch順序依存を再導入し得るため、そのような短絡は採用していない。

通常APIの同条件逐次速度比較は下記に追記する。全計算・build終了後、CPU0・1 thread、2 warmup＋9 thread-CPU samplesの中央値。import/constructorを除く全epochのmagnification呼出し。比較前は今回の端点再利用・円環上限の直前DSO276a1e…、比較後は85ef83…で、09f5261d以前との速度比較ではない。外部負荷・CPU周波数は固定していない。

## 採用しなかった診断と残件

IFTでpoint候補を追加する案、イベント探索の分割軸切替、残り3種類のlens順序を足す案は、対象pilotの停止を十分に改善せず採用しなかった。値のevent boxを無条件に再収縮する案は別の局所求根拒否を起こすため採用しなかった。これらのoverlayはevidence内の実験用で、production sourceではない。

最初のpoint-disk診断はVBMの出力sourceとnative用レンズ座標の符号を混在させていた。VBM本体のTripleLightCurveを確認して符号を統一し、`point-disk-vbm-frame`へ新しく保存した。誤った`point-disk-references`は最終参照に使わず、診断履歴としてだけ保存する。

残る値25例はwide_random3、near_collinear_or_clustered22。微分27例はwide_random4、near_collinear_or_clustered23。微分case217の最初のepochは`stream_local_repair`で停止。その他は未分離root cluster、退化event、support/event budgetなどを個別に確認する必要がある。停止した値やNaNを許容誤差内として返す変更は行っていない。

次の重点は、極小imageとほぼ共線配置でのroot/event証明の表現精度、および接触点で失敗frameへ費やす仕事量。これらが残るため、「全配置で完成」「全微分が認証済み」「VBMより常に速い」とは主張しない。

### 通常APIの逐次計測結果（ms / 全curve）

| case / epoch数 | 値 前→後 | 値＋微分 前→後 | SinglePoly | MultiPoly | NoPoly |
|---|---:|---:|---:|---:|---:|
| 21 / 7 | 0.399 → 0.398 | 0.420 → 0.431 | 1.919 | 2.394 | 3.703 |
| 135 / 7 | 162.751 未収束 → 108.420 | 148.757 未収束 → 109.122 | CPU30s上限 | 2.307 | 3.983 |
| 189 / 7 | 327.958 未収束 → 59.703 | 275.559 未収束 → 59.440 | 2008.301 誤値 | 27.680 誤値 | 4.918 |
| 940 / 3 | 21.576 → 23.517 | 288.299 → 259.112 | CPU30s上限 | 65.648 | 299.902 |
| 972 / 3 | 123.165 → 123.227 | 1542.679 → 1426.615 | 11.253 | 15.938 | 50.796 |
| 1041 / 3 | 133.264 → 137.081 | 359.756 → 337.897 | 11.567 | 16.315 | 39.159 |
| 1508 / 3 | 56.492 → 56.780 | 964.788 → 872.997 | 33.434 | 46.212 | 104.890 |
| 489 / 5 | 拒否 → 拒否 | 拒否 → 140.912 | SIGSEGV | CPU30s上限 | 2.755 |

同一許容誤差であっても、VBMの誤値・CPU上限・SIGSEGVは正確な計算の速度として扱わない。今回のnative値＋微分は、成功共通のLD case940/972/1041/1508でそれぞれ約10.1%/7.5%/6.1%/9.5%短縮した。値だけのLD経路には同じ改善はなく、case940は21.58→23.52msだった。単一の逐次測定で外部周波数等は固定していない。値経路で増えた時間の内訳は未分離で、速度向上とは主張しない。

case21ではnative値0.398msに対してVBM1.919/2.394/3.703ms。ただし接触や極端配置には、VBMの正しい値が得られる場合、nativeの方が大幅に遅い例が残る。case972微分では拒否される最初のframeの仕事が支配的で、case1508微分は値のみの約15倍。要求精度を緩める方法では解決していない。CPU30s上限は各workerに同じ上限を課したもので、未完了workerの所要時間を推定の中央値に置き換えない。

全56行、全sample・返値・status・DSO hashは `final-controlled-timing.json` と `final-controlled-timing/` に保存した。beforeとVBMの不変な行は最初の逐次測定から保持し、afterの16行は最後の境界補強後のDSOで、他のcampaign・buildの終了後に再計測した。

## 再現資料

`raw-evidence.tar.gz`に全raw結果、入力、診断、script、数値source snapshotsを保存した。archiveのhashとサイズは `raw-evidence-archive.json`、全ファイルのhash一覧は `raw-evidence-manifest.json.gz`。派生DSOはcommitへ入れず、各provenanceにSHA256を記録した。今回の最終sourceは `source-snapshots/final`、速度比較前は `source-snapshots/candidate`。commitのnative sourceからビルドして、通常API workerのeditable import除外・DSO identity確認を使う。

再検証する際はarchiveをproject内へ展開し、新しい出力用evidence directoryに入力とscriptを複製する。既存rowsだけでなくstatusも上書きしない。`TRIPLE_BUILD`は使用する固定packageの絶対パスにする。各workerはこれを検査し、入力とnative DSOのhashを結果へ記録する。元の既存未追跡handoffはcommitへ取り込まず、今回の更新だけを `docs/TRIPLE_HARDENING_UPDATE_20261029.md` にまとめた。
