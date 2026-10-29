# Triple LightCurve: complete pole partitions and physical-root continuation

## 採用結果

基点 `6909059e`、branch `experiment/algebraic_boundary`。**既知の停止439・499を解消し、1880条件・8280 epochで値／全10微分付きとも全件成功。** 以前の成功条件の分類回帰0。指定rtol=1e-3/1e-4、atol=0を維持した。元の1856条件に、439/499近傍の分離・質量比・光源半径を変え、LD=0/.5/1を組み合わせた24条件を加えた。

採用DSO: `9e14a9155a432863441fa986e3f5cdec1ee81d5173ef00db09714404b8142d91`。

* 4800値比較で要求誤差超過0、最大error/requested=0.816735。既存4764比較に、439/499全10点×両経路と新規一様光源8中心点×両経路を追加した。4800件全てを今回新しく生成した参照とは扱わない。
* 独立90微分の最大scaled差2.198e-7、診断閾値1e-4以内。元の7中心70微分と新しい2中心20微分。全配置・全epochの微分誤差認証ではない。
* 169曲線・507不変性検査で失敗／超過0、最大scaled微分差2.378e-5。
* C++34件、対象Python252件成功。legacyファイルは8成功・9失敗で、旧採用版と同じ9失敗ID。全repo pytest成功とは主張しない。

完了したのは、停止反例の修復、通常の正rhoの全IFTを保つ経路、速度コストの回収、および上記の有限の検証集合である。`gradient_valid`は有限で利用可能な微分を意味し、要求した**値**rtolから任意の微分精度やfit全体の認証を導くものではない。LD高速化の未採用試作と、小さな角度微分の精度限界は後述する。

## 全成分を保持するpole partition

二重精度のglobal polar座標では、遠い極の微小像の幅が半径座標の機械幅に埋もれ、event分離が停止していた。439の明るい曲がった像は単一のfull-source Newton ballに入らず、499はsource diskがcausticを含むためpoint/hexにも逃げられない。

新しい `pole_partition.hpp` は、微小像を極ごとの局所座標で積分し、残りの像を別の完全な積分として計算する。暗い像の値も全11物理座標のIFTも実際に評価し、最後に物理座標へ戻して合算する。faint image/IFTを省略したり、倍率の上界を値の代わりに使ったりしない。

証明は元のレンズ方程式から極の特異項 `m/conj(delta)` を分離する。source disk内の全sourceについて、極を中心とするcore ball内で固定点写像が自己写像・収縮となることをoutward intervalで確認する。さらにcoreから大きなpartition regionまでのannulusにsourceの逆像が存在しないことを不等式で確認する。regionはレンズ極を含んでよいが、単一像のNewton ballとして扱わない。

この証明済みregionだけをglobal radial plannerから除外し、残りの全eventは従来のplannerで証明する。局所像の半径範囲も元の方程式からoutwardに囲む。局所積分では未分離foldを「eventなし」と誤認せず、crossing数0による空領域判定・event-free band continuationを無効にして各半径の全角度を調べる。

内部の `complete_partition_member` は全成分を合算するownerだけが使用する。通常の外部excluded-image planのgradient拒否、およびbounded component valueのgradient拒否は維持する。どれか一成分が失敗すれば全体を受理しない。合算した値誤差が元のatol+rtol*|mu|を満たすことを最後に要求する。局所chartの丸め差にも、証明された微小成分のflux上界を誤差項として加える。

一様光源で大きいレンズ質量の近傍にsourceがあり、他の極が十分遠い場合は、この完全な証明付き経路を先に試す。距離16倍の判定は仕事順のscreenであり、受理証明ではない。その他の場合は通常経路の後のrescueとして試す。失敗する全座標を巡る費用を避け、439/499で同じ値・全微分を高速に返せた。

source-disk production shortcut、微分誤差による適応制御、accepted-primal tangent replayは追加していない。異なるchartではそれぞれ値のみで制御するcoupled frontierを進め、全ての値とIFTをその場で評価する。

## rho微分の速度コストを回収

前コミットは正rhoのpoint近似をjoint hexへ変え、rho微分の0問題を修正したが、13点ごとに次数10多項式を解いて約9–11倍の費用がかかっていた。

像数が独立に証明されたhex sampleでは、保持した**全ての物理像**を元のレンズ方程式でlong double Newton継続する。負像は極とのdisplacementを保持し、丸めたglobal位置から微小像を作り直さない。残差、位置補正量、符号、像の相異性、独立像数、正負像数の必要条件を再検査し、一つでも失敗すれば元の完全な多項式solveへ戻る。rootだけを保持し、fluxや微分はcacheしない。各sampleの倍率と全IFTは新しく計算する。rho=0や明示point設定の既存契約も維持する。

## 独立参照

439/499には70桁の元レンズ方程式solveとGreen境界積分を新たに用意した。source円周上で4像の相異性・符号・元の残差<1e-52を確認し、各像の境界速度を原方程式のIFTから求めて符号付き像面積を合計した。productionのglobal polar quadratureやそのevent切断を参照計算には使わない。32/64/128点で全5 epochを保存し、64/128の一致を確認した。causticを含む499では内側境界の向きも符号付き面積に含む。

新しい20微分は2刻みの中心差分とRichardson外挿、70桁・64境界点で求めた。各摂動で原方程式を解き直す。439中心のrho微分は約851.882665、499は約-197.937964。nativeの最大scaled差は499/u0の2.198e-7だった。既存の独立source-disk参照70微分も採用DSOで再検証した。

新規24条件のうち一様光源8中心にも64/128点の独立Green参照を追加した。LDの16新規曲線全点まで独立高精度参照を生成したわけではなく、それらの正常収束を精度の全点認証と区別する。

## 速度

公開APIの全曲線、CPU0固定、ライブラリ1 thread、constructor/import除外。cold1回、warmup2回、thread CPU9回の中央値。「前」はrho微分修正済み `6909059e` のDSO `021f6216…`。外部負荷・周波数は固定していない。

|case|値 前→後 ms|全微分 前→後 ms|
|---|---:|---:|
|21|3.892→0.739|4.048→0.762|
|400|3.342→0.565|2.708→0.665|
|403|3.056→0.561|2.607→0.574|
|433|3.308→0.503|3.439→0.551|
|439|停止→1.949|停止→1.965|
|460|3.374→0.531|4.107→0.509|
|467|3.544→0.575|3.005→0.491|
|484|3.441→0.630|3.739→0.657|
|499|停止→5.684|停止→5.447|
|515|4.329→0.500|4.705→0.660|
|514|3.601→0.510|3.810→0.558|
|940|22.115→19.729|242.627→225.372|
|972|110.985→112.476|1204.206→1223.155|
|1508|50.364→51.067|758.228→767.768|

ρ依存を保った軽い領域の費用を大きく回収した。439/499の「前」は失敗なので成功速度の倍率を付けない。別に保持した、正確性修正済み・後段partitionだけの中間版では439のwarm微分375.86ms、499は4842.24msだった。早期の証明付き分割へ仕事順を変えてそれぞれ1.965ms、5.447msとなった。普遍的なspeedupやfit throughputではない。

重いLDの採用経路は維持している。940の差は新しいLD高速化として扱わず、972/1508の時間もほぼ同じ。以前確認したangular計算と失敗frontierが主な費用である。

## VBM Single / Multi / NoPolyの再検証

追加24条件と元の439/499、計26曲線を、新しいprocessで通常rtol=1e-4と厳しいrtol=1e-8でそれぞれ評価した。各VBM workerは8 CPU秒、1 thread。CPU上限、SIGSEGV、SIGABRTを区別して保存した。VBM 5.5のインストールは変更していない。

|通常精度|有限正値return|独立参照を持つreturn曲線|その要求誤差超過|
|---|---:|---:|---:|
|SinglePoly|9/26|9|9/9|
|MultiPoly|19/26|10|2/10|
|NoPoly|26/26|10|0/10|

独立参照を持つ元の2曲線は全5点、新規一様光源8曲線は中心点のみで判定した。参照のないLD曲線を誤値・正値のどちらとも断定しない。厳しいMP/NPが両方returnして全点一致する新たな参照は得られず、今回そこからnative精度の認証を増やしていない。既存の参照gate・VBM誤値除外は維持する。

`VBM_AUDIT.json` に全raw・独立誤差・厳しい二手法の除外理由をまとめた。ここでのVBM CPUはfreshな1曲線評価であり、上のnative warm中央値との同条件の速度比として扱わない。正常returnや正の倍率だけではVBMを正解にしない。

## LD候補を棄却した理由と微分の限界

同じ物理座標のまま、32 refinementで長引くPatterson streamをcoupled積分へ切り替える案も全1856条件で実行した。成功分類の回帰0で、252対象テストも通った。単独epochでは972が約1秒の停止から約36msの収束、1508が約233msから約19msとなった。

しかし940中央epochのalpha/ang、1070のalpha/angに微分差が出た。より厳しいnative値rtol=1e-9/1e-10と複数刻みの中心差分を保存して調べた。例えば1070/alphaは差分が約-12.1935に近づくのに、候補は-12.2142だったため採用しない。radial/angular levelを固定で増やす試作はこの誤差を直さず、内側の値精度を厳しくする試作はrefinement budgetやangular unresolvedで収束・速度を保てなかった。これらのsource/include木、失敗値、全3712実行も保存した。productionのLD controllerは元へ戻した。

この診断では現行版の小さなLD角度微分にも有限精度が見える。940/angは現行約-0.32402に対して厳しい値差分が約-0.32581、alphaも小さい差がある。これは今回の独立90微分の対象外であり、正常なvalue/gradient returnや値rtolを任意の微分精度保証としない具体例である。停止反例の解消と、全パラメータの任意精度保証・全面的なfit認証は区別する。既存の独立LD4参照の回帰検査は採用版でも通過した。

## 再現・保存

CPU最大8・RAM16 GiBの枠を継承。native4worker、VBM/高精度参照2worker、各ライブラリ1 thread、native/VBM worker AS2 GiB。compileと検査も含め同時CPU数を8以内にした。観測したnative worker最大RSSは70,592 KiBであり、aggregate cgroup測定値とは区別する。全jobを完了まで監視し、継続中のjobは残していない。

`PROVENANCE.json` は採用版のproject依存ファイル・実際のwrapper6件・テスト・compiler flags・各DSO・VBM nativeのhashを持つ。既存dirty binding/wrapperも再現性snapshotに含むが、その元ファイルは今回の変更としてコミットしない。採用source snapshotは実際に検証した状態を保存する。未採用DSOのbit同一再ビルドを一律には主張しない。

前報の共通入力・独立参照archiveを各evidenceディレクトリへ展開する。今回のarchiveの `raw/pole-accepted-rows` / statusを前報 `triple_local_chart_followup_20261029` の同名パスへ戻せば、`accepted_invariants.py`、`summarize.py`、`vbm_audit.py` で再集計できる。native本体はsource snapshotからビルドし、wrapper snapshotとともにaccepted-packageへ置く。前報 `campaign.py --full` は `VARIANT=../triple_component_completion_20261029/accepted-package TAG=pole-accepted INCLUDE_HOLDOUT=1 WORKERS=4`、追加集合は `native_recheck.py`、対象検査は `accepted_regression.py`、独立参照は `green_reference.py` / `green_holdout.py`。

DSO/ELF、package cache、compiler scratchを除くraw・source・試作・失敗・参照をarchiveへ保存し、全memberのSHA256を検証した。`archive.json` にarchive全体のhashとサイズを記録する。
