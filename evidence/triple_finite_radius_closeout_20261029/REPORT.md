# Triple finite-radius tangents and an index-preserving duplicate

## 結果と未完了範囲

基点 `c2d0290a`、branch `experiment/algebraic_boundary`。正のrhoに対する通常の有限光源呼出しからpoint近似を外し、同じ幾何証明・元の値許容誤差の下でjoint hexの値と全IFT微分を返す。**以前0だったrho微分の反例4件を修正した。** さらに、この変更の検証で見つかった「総像数と正負像数を保つ未収束重複」を修正した。

**全体の完成ではない。** 元の439・499は値／微分とも停止したまま。LDの重い積分も改善していない。局所support/複数chartの全成分積分は未解決であり、これを完成扱いにしない。

最終DSO SHA256: `021f6216986b54fbeea6d1bdd107c8ff8fa261c27e0c7b440cf23ab35b662863`。

* 1856条件・8160 epoch、値／微分とも1854成功・2停止。旧採用版に対する成功分類の回帰0。
* 既存の参照gateを通した4764値比較、要求誤差超過0。最大error/requested=0.816735。参照を新たに全件生成したのではなく、前報の独立参照・監査済みVBM値を再利用し、新しいnative値と比較した。
* 7条件の中心における全70微分を、既存の70桁original-equation/source-disk有限差分と比較。scaled差の最大1.060e-7、閾値1e-4以内。任意の配置・全微分の認証ではない。
* 167曲線・501不変性検査で失敗／超過0、最大scaled微分差2.378e-5。
* 対象Python226件、C++33件成功。legacy point-sourceファイルは8成功・9失敗、旧版と同じ9失敗ID。全repo pytest成功とは主張しない。

## rho微分の修正

以前は、値のpoint curvature screenを満たした正のrhoでもpointの値・微分を返していた。このモデルはrhoに依存せず、有限光源のrho微分が大きくても0になる。今は、同じ20rho clearance証明の後にjoint hexを評価する。hexが拒否されたら通常のhex/積分へ進み、pointの値に有限光源微分だけを継ぎ足さない。値のみ／微分付きの両APIで同じ方針を使う。最終rescueも、通常設定でpointを再受理しない。

rho=0、mode=0、adaptive_hex_threshold=0で従来許されるpoint経路は維持する。この明示設定でrho微分が0になることもテストした。元の5rho/20rho距離gate、rtol=1e-3/1e-4、atol=0は変更していない。新しいsource-disk production経路、微分誤差による適応制御、accepted-primal tangent replay、faint image/IFTの省略は導入しない。

独立有限光源rho微分の参照は400:2500.185909、433:318.894795、460:26.988970、515:0.0023085445。旧版はすべて0だった。新しい全70微分の比較は `gradient-comparison.json` にある。

## 総数とindexを保つ重複

最初の変更 `dacb58b6…` では、新規holdout48の中心がhex拒否→積分停止となり、成功条件が1件減った。この試作は不採用として全3712実行を保存した。

調査すると、13点の**interleaved評価順**で、明るい負像の未完了Newton候補が重複し、極近傍の暗い像を置き換えた。4像・positive1/negative3という検査を両方満たす。問題の点は70桁独立計算で150.1501289997274だが、旧カーネルは224.7253238573324を返した。独立計算では4像の相異性、正負像数、残差<1e-52を検査した。13点を `point-reference-48.json` に保存した。

Newton補正量の上界 `residual*(1+|shear|)/|det J|` が既存の位置収束スケール1e-11*(1+|z|)を超えると、保持した候補からlong doubleで再研磨し、元の残差／重複検査を再適用する。これは修復を開始するscreenであり、完全性の証明でも要求像数への切り詰めでもない。像数とindexが正しい場合にも発火できるようにした。

基点コミットのpoint-source cppを再コンパイルして同じprobeへリンクし、旧反例を再現した。修正版は150.1501289997270、hex値は独立有限光源値152.5197401167771と一致する。通常の公開point APIを通した別の評価列では旧版も正解したため、旧版がすべての呼出しで誤るとは主張しない。元のLightCurveではpoint判定が先に通り、この中心のhex拒否は表面化していなかった。

C++回帰は独立13点をexpected countなし／4あり、各3周の継続評価で確認する。Pythonにはこの曲線の値・cold/warm一致も追加した。最終版の再実行ではholdout48を回収し、1856条件で旧成功の分類回帰0となった。

## 速度の代償

CPU0固定・各ライブラリ1 thread、constructor/import除外、cold1回、warmup2回、thread CPU9回の中央値。全曲線の公開API、同じ入力・許容誤差。外部負荷や周波数は固定していない。

|case|値 前→後 ms|全微分 前→後 ms|
|---|---:|---:|
|21|0.423→3.992|0.445→4.061|
|400|0.342→3.524|0.361→3.568|
|433|0.328→3.382|0.347→3.347|
|460|0.340→3.774|0.384→4.105|
|515|0.429→4.634|0.449→4.693|
|514|0.350→3.748|0.368→3.806|
|940|23.306→23.269|256.618→256.925|
|972|121.564→121.761|1393.272→1392.776|
|1508|54.482→55.076|861.070→859.311|

**軽いpoint領域は約9–11倍の費用増。** 13点の有限光源評価へ変えた正確性上の代償であり、高速化とは扱わない。重いLDはほぼ変わらない。今回VBM Single/Multi/NoPolyの計算・速度を新たに取り直してはいない。前報の同じ入力・同じ許容誤差のraw比較と、MultiPolyの誤値／NoPoly等のcrash除外を継承した。`timing-v2.json` が最終DSOのraw、`timing-attempt1.json` と通常 `timing.json` は不採用試作の記録。

## 残る439・499と局所chart試作

439の一部微小像について、eventのradial丸め床とdominant image ballの初期半径を局所座標のスケールに合わせる試作を別include木で実行した。全成分は回収できず、primary2側は引き続きregular_branch_unresolved。明るい曲がった像もfull-source contraction ballには入らない。変更はproductionへ採用しなかった。499はcausticを跨ぐsource diskであり、point/hexの距離gateを緩めて通す解決はしない。

必要な残作業は、微小な極近傍成分と曲がった／causticを跨ぐ成分を、全像・全IFTを保持して一つの誤差制御で扱う複数chartのsupport/積分設計である。現状の二重精度global polar座標の予算増だけでは解決しない。

## 保存と再現

CPU最大8・RAM16 GiBを継承。native4worker、各数値worker AS2 GiB、BLAS/OMP1。参照・compile・単独検査を含む同時実行数も8 CPU以内。全計算を監視して完了し、継続中のjobは残していない。

`PROVENANCE.json` は205 project dependencyのhash・snapshot、実際のwrapper6件、変更テスト、compiler flags、旧／試作／最終DSOとVBM nativeのhashを記録する。既存dirty binding/wrapperは再現性snapshotに含むが、元ファイルを今回の変更としてコミットしない。

前報のarchiveをそれぞれのevidenceディレクトリへ展開すれば共通入力・独立参照を回収できる。今回の `raw-evidence.tar.gz` は最初と最終の全native実行を `raw/finite-radius-rows` / `raw/finite-radius-v2-rows` に含む。再集計はそれらを前報ディレクトリの同名サブディレクトリへ戻し、`invariants.py`、`summarize.py` を実行する。campaign自体は前報 `campaign.py` に `VARIANT=../triple_finite_radius_closeout_20261029/package TAG=finite-radius-v2 INCLUDE_HOLDOUT=1 WORKERS=4` を指定する。最終DSOはsource snapshotからビルドしてpackageへ配置し、wrapper snapshotを同梱する。`regression.py`、`check_gradients.py`、`point_reference_48.py` が個別検査。

DSO・ELF・compiler scratch・package cacheはarchive対象外。生の失敗、未採用試作、再現source、完成した参照と最終結果を保持し、archive全memberのSHA256を検証する。
