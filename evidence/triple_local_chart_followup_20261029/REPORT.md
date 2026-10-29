# Triple LightCurve: critical-loop progress and duplicate-root repair

## 結果

`experiment/algebraic_boundary`、基点 `cf0a5f9db4c34fa6b783c5a12c1e71ba2fc29039`。**1856条件・8160 epochで値／微分付きとも1854条件成功、2条件停止**。同じ1856条件の旧採用DSOでは値1814・微分1813成功だったので、値40・微分41条件を回収した。既存成功の分類回帰0。以前の1800条件の停止は9→2、新しい固定seed56条件は両経路56/56（旧版23/22）。失敗も全件保存している。

参照gateを通した**4764比較・2382点・534条件で要求値誤差超過0**。最大error/requested=0.816735。独立高精度計算を追加したのは91点（修復7曲線の全35点と新規56曲線の中心点）。新規集合はseed `2026102911`、高倍率・近接極・極端な質量比の7系列、LD=0/.5/1、rtol=1e-3/1e-4、atol=0。元の1800条件は既存の接触・cusp・広い配置等をそのまま再検証した。

167曲線・501不変性検査（角度2π、epoch/t0同時移動、逆順・重複）で失敗0。最大値相対差5.833e-7、最大scaled微分差2.378e-5（閾値1e-4）。C++33件、対象Python171件成功。追加で実行したlegacy群は18成功・9失敗で、旧採用DSOでも**同じ9件**の失敗を再現した。全repo pytest成功とは扱わない。

停止が残るのは主集合 **439・499**。また、値の許容誤差から有限光源微分全体の精度を保証したものではない。特に後述のpoint近似のρ微分を区別する。

## 修正

1. **critical-loopの丸め床で親と端点の証明円盤が同じ大きさになる問題。** 端点を親に厳密内包する検査が、分割を続けても通らなくなっていた。親に余裕を持たせ、その実際の半径で収縮を改めて証明する。証明を強制せず、従来の誤差マージンを維持する。
2. **幾何証明の予算不足と座標依存。** 通常256個の新しいphase enclosureを使い、必要時に同じfrontierを65536まで継続。depth上限24、tree上限131072 node。別のprimary frameでもcausticの距離・像数だけを証明し、fluxと全IFTは元の物理座標で計算する。pointの20ρ、hexの5ρ以上という既存の距離条件は維持する。
3. **未完了Newton反復による明るい像の重複。** 初期seedからの反復が上限に達しても残差1e-7を通り、同じ明るい像の近くに別候補として残ることがあった。403のhex sampleでは4像のはずが5像となり、valid=falseの候補倍率は約1626まで膨らんだ。独立な像数または正負像数の必要条件と矛盾した場合、保持した途中点からlong doubleで再研磨し、原方程式の残差と既存の空間重複検査を再適用する。要求された像数に合わせてprefixを切る処理ではない。独立像数なしのpoint呼出しにも適用する。
4. **仕事の順序。** 不正な初期像数候補が早期経路を抑止し、結局失敗する積分を何通りも試していた。安価な別座標の幾何証明を先に試し、必要なら証明を継続してから、独立に決まった像数でjoint point/hexを評価する。curvatureの100倍budgetは証明へ費用を払うためのscreenだけで、受理許容値ではない。最終値は元のbudgetを満たす必要がある。
5. **繰り返し評価の幾何証明。** 最大16個のwinding witness ballを保持し、距離を差し引いたclearanceを再検証する。同じ中心epochへ戻るたびに数千のready nodeを再走査する費用を避ける。レンズ幾何が更新されたら全witnessを破棄する。flux・微分を保持するcacheは追加しない。
6. **正質量のgate。** 浮動小数点の `m1+m2==1` だけではprimary massの正値性は保証されない。caustic証明にもsupport plannerと同じexact-sum符号検査を加え、表現できない別primary frameは拒否する。

productionは既存point/hexまたは値で制御するcoupled積分を使う。新しいsource-disk積分経路、微分誤差による適応制御、primal分割のtangent再生、faint image/IFTの省略は導入していない。全積分のframe選択順は変更していない。

## 独立精度と微分の限界

検証専用に、70桁の元レンズ方程式Newton solveとsource-disk quadratureを実装した。各sampleで全ての像が相異なること、正負像数、原方程式の残差を検査する。新規56条件中54条件は4像、2条件は6像。GL4/8/16と角方向16/32/64を保存し、最終のGL8/16の収束を採用した。GL4は収束過程の記録で、厳密参照の値に混ぜない。残差閾値は1e-52、最終二つのmeshの差はnative許容値の1%以内。これは独立した高精度チェックであり、任意の配置の数学的誤差証明ではない。

修復7条件の中心点で全10パラメータを独立有限差分（2刻み・Richardson、70桁、GL8×32）でも検査した。hexを使う403/467/484の全30微分はscaled差1e-4以内、最大約1.1e-7。point近似を使う400/433/460/515では**ρ微分が0**となり、有限光源そのものの参照はそれぞれ約2500.186、318.895、26.989、0.00230854だった。これはpointモデルがρに依存しないためで、値のrtolを満たすことから有限光源ρ微分の精度は導けない。例えば400では倍率の相対差は約1.25e-5でもρ微分の差は大きい。`gradient_valid`を有限光源微分の精度認証とは扱わない。この近似方針は今回変更しておらず、全パラメータのfit精度認証は未解決である。

初回の独立FDでは、レンズ幾何を摂動した後も旧seedを使い続け、一部の像が同じ解に落ちる参照側の不備があった。各摂動で根を再探索し、各quadrature sampleの像の相異性と正負像数を確認したv2のみを採用した。初回の失敗・不適格参照も保存している。

## VBM Single / Multi / NoPoly

新規56条件＋修復7条件の計63条件を同じ通常rtolで比較した。表の精度判定は独立参照を持つ中心epochについてで、プロセス上限は8 CPU秒。SIGSEGV、SIGABRT、CPU上限を区別して保存した。

|VBM 5.5|有限正値を返した曲線|中心値の要求誤差超過|CPU上限|SIGSEGV|SIGABRT|
|---|---:|---:|---:|---:|---:|
|SinglePoly|29/63|16/29|32|1|1|
|MultiPoly|56/63|1/56|6|0|1|
|NoPoly|29/63|0/29|6|24|4|

**新規case32のMultiPolyは真値約89.34204に対して約1.0428e41を返した。** 新しい独立プロセスの公開Python APIで2回同じ異常値を再現した。`VBM_COUNTEREXAMPLE.json` に入力・独立値・native・再現出力をまとめた。SinglePolyにも大きい有限正値の誤値があるため、正常returnや正の倍率だけで正解としない。インストール済みVBMは変更していない。

さらにMP/NPのrtol=1e-7、報告誤差、二手法一致を監査した。新規56条件ではcurve failureで37条件、二手法不一致で42点を参照から除外した。nativeの成功条件から除外したわけではない。独立70桁の中心値は、こうした失敗がある曲線にも用意した。前報のcase514の「厳しい指定でも1.0」問題に対する未検証flat参照除外・80桁参照への置換も維持する。MP/NPの一致や内部誤差だけを真値の証明にはしない。

## 速度

同じ入力・rtol、CPU0固定、各ライブラリ1 thread、公開APIの曲線全体。constructor/importを除き、cold1回、warmup2回の後にthread CPU9回の中央値を測定。外部負荷・周波数は固定していない。VBM列は値のみ、native微分列は全10微分付き。

修復7条件の「前」は**正確性修正済み・terminal-only中間DSO `feee9f0f…`**であり、元の失敗版ではない。

|case|cold値 前→後 ms|warm値 前→後 ms|cold微分 前→後 ms|warm微分 前→後 ms|
|---|---:|---:|---:|---:|
|400|1148.0→77.10|2.115→0.341|911.4→77.06|2.102→0.360|
|403|4504.0→87.39|2.904→1.087|3285.9→87.32|2.943→1.112|
|433|982.0→46.16|1.262→0.321|749.0→46.63|1.299→0.344|
|460|1852.4→12.77|0.589→0.367|1415.3→12.73|0.590→0.384|
|467|3899.3→4.031|3890.574→2.263|3823.6→4.067|3814.499→2.309|
|484|1652.0→91.43|3.159→1.239|1181.8→92.04|3.200→1.317|
|515|4084.4→1.893|4036.428→0.428|4042.6→1.848|4032.816→0.436|

これは特定の極端配置での改善で、普遍的な倍率・fit全体の速度比ではない。下表のVBMは同じ測定手順で取り直した値経路。数値精度・停止の制約も併せて読むこと。4764比較の参照coverageは修復7曲線では全5点、21は全7点、483/514は全5点、972は3点中2点。940/1508は今回の値参照集合では0点なので速度のみの行である。

|case|native値 / 微分 ms|VBM Single / Multi / NoPoly ms|
|---|---:|---:|
|21|0.423 / 0.443|1.830 / 2.232 / 3.668|
|400|0.341 / 0.360|CPU上限 / 1.830 / SIGSEGV|
|403|1.087 / 1.112|CPU上限 / 2.435 / 7.781|
|433|0.321 / 0.344|CPU上限 / 1.912 / 7.303|
|460|0.367 / 0.384|CPU上限 / 1.582 / SIGSEGV|
|467|2.263 / 2.309|1.940 / 2.358 / CPU上限|
|484|1.239 / 1.317|CPU上限 / CPU上限 / 4.807|
|515|0.428 / 0.436|CPU上限 / 432.904（誤差超過） / SIGABRT|
|483|343.490 / 349.817|CPU上限 / 128.991 / 240.213|
|940|22.609 / 254.638|CPU上限 / 65.474 / 297.475|
|972|119.859 / 1218.966|11.197 / 15.797 / 50.641|
|1508|54.324 / 745.028|33.314 / 45.677 / 104.451|
|514|0.326 / 0.341|0.0040 / 0.0061 / 0.0042|

速度測定の反復では515/MultiPolyも大きな値誤差を返した（`VBM_WARM_COUNTEREXAMPLE.json`）。別processでも同じ最終epochの誤値（約150.546、独立値約1.33809）を再現した。この432.904msは精度条件を満たす速度比較として扱わない。上の63条件の精度表はfreshな1回の中心値であり、warm反復の正確性とは別に記録する。

通常領域との速度基準は旧採用DSO `0cb54c74…`。21/940/1508の値をA/B/B/A順でも再測定した各18sample中央値は0.410→0.422ms、23.288→23.294ms、54.310→52.856ms。1508の初回50.915→54.324msという約7%増加は再現しなかった。LD本体の一律高速化は確認していない。全結果・初回値・再測定を `timing-summary.json` とrawに残した。

## LDの内訳と残る2条件

現在のheaderにexclusive CPU scopeを入れた**単独epoch・coldの診断**も行った。scope計測の費用があり、公開APIのwarm時間を分解したものではない。

* 972中央epoch：canonical frameの積分が1.244秒でrefinement budget停止。exclusive内訳はangular 0.741秒、topology 0.191秒、endpoint 0.148秒、chart 0.104秒。別primary frame 2では準備12.55ms＋積分8.53msで収束した。
* 1508中央epoch：canonical frameは準備8.18ms＋積分273.11ms、うちangular 180.48ms、endpoint 38.28ms、chart 28.05ms。別primary frame 1は準備5.64ms＋積分2.34msで収束した。

angularはこの積分CPUの約60%／66%を占めた。したがって重いLDでは、canonical frameでの角方向の仕事と、失敗するfrontierを最後まで進める費用が残る。ただし「速い別frameで値とvalid flagが出た」ことだけでは全微分の精度は保証できない。前のframe選択試作は595/alpha、1104/rho、別の試作は321/alphaの独立反例で棄却されている。今回も全積分を速いframeへ一律切替していない。

439はsource diskの外側caustic clearanceと4像までは証明できるが、明るい曲がった像を単一のfull-source IFT円盤で覆えず、極近傍の微小像を含めたglobal polar event分離が機械幅で止まる。像ごとのlocal chart試作では一部の微小像しか回収できず、全成分の結果として採用しなかった。499はsource diskがcausticを含むためpoint/hexの距離gateも正しく拒否する。必要なのは、曲がった像・微小な極近傍像・causticを跨ぐ成分を取りこぼさず扱う局所support/積分設計であり、単なる予算増加では解消しない。

## 再現性・保存

採用DSO SHA256: `8fc87192444d742e1dfa48bad43800b03b638f09f476d1f8638020bce36a6534`。
Python3.10.19、NumPy2.2.6、mpmath1.3.0、VBM5.5、g++11.5。CPU最大8・RAM16 GiBの枠で、native4 worker、参照3 worker、各ライブラリ1 thread、Python数値worker AS上限2 GiBを使用。観測したnative worker最大RSSは70,600 KiB。証明・compile・benchmarkは同時worker数を含めて上限内で実行した。

`PROVENANCE.json` に205個のproject依存ファイル、実際に使ったfrozen Python wrapper6個、テスト、compiler flag、VBM nativeと診断C++source/probeのhashを保存した。前報の69ファイルのmanifestではPython bindingがclosureに入っていなかったため、今回はnative targetの依存ファイルから採取し直した。旧DSOはhashで保持し、旧dirty binding objectまで遡るbit同一rebuildは主張しない。現在の既存dirty bindingはsnapshotに含め、今回のコミット対象にはしない。

入力・seed・全variantのraw row・失敗・不採用prototype・log・source/test snapshot・実行scriptはarchiveに保存する。DSO、実行binary、compiler scratchを除き、全tar memberをSHA256 manifestと照合する。`archive.json` に件数・サイズ・hash・照合結果を記録する。元の証拠は削除しない。再実行時はarchiveをこのディレクトリへ展開し、前報 `../triple_event_precision_20261029/raw-evidence.tar.gz` も元のディレクトリへ展開する。source/wrapper snapshotと環境を照合してfrozen packageを再構築する。異なるDSOで既存rowを再利用するとharnessが拒否するため、新しい出力tagを使う。

再現入口は `campaign.py`（`VARIANT=accepted-package TAG=accepted INCLUDE_HOLDOUT=1 ... --full`）、`holdout.py`、`mp_reference.py` / `mp_full_reference.py` / `mp_holdout.py` / `mp_gradient.py`、`accepted_invariants.py`、`accepted_regression_tests.py`、`accepted_timing.py`。実行時はOPENBLAS/OMPを各1 threadに固定する。図やsource-disk integrationはproductionへ追加していない。

監査用の負の証拠として、分割軸変更・interval critical-cover・tree分割変更・local-component chartの不採用試作を残した。速度harnessの初回はJSON key型の起動エラーで、数値測定には含めない。30 CPU秒上限で打ち切られた中間版の速度試行は保存し、正式比較はnative60 CPU秒で完了した52実行を使う。不変性検査の初回はnative rowの到着前に起動して44曲線だけを検査したため、全rowのreadiness gateを追加して167曲線を再実行した。これらの初回出力もarchiveに含む。

保存確認：raw archiveは12819ファイル、13.65 MiB。全memberのSHA256をmanifestと照合済み。archive SHA256: `368005857f8eef42a9bc71ecb14168cce779c4061d1cb52bef98402ee6342009`。
