# 通常APIの証明screen復帰と時刻間再利用の実験

## 採用した変更

tangent/geometry.hppのdetect、known_internal_pairs、known_upper_edge_pairsへ既存ScopedNoFlushDenormalsを追加。証明処理の間だけSSEのFTZ/DAZを解除し、終了時に呼出元のMXCSRを復元する。environment_okの拒否条件、証明条件、許容誤差、積分精度は変更していない。積分のScopedFlushDenormalsは従来どおり動く。通常APIの値・Jacobian、JAX尤度+14勾配は保存した全配列/全4条件でbit一致。値だけの通常APIはほぼ同等、勾配APIを高速化した。実験ブランチに未コミットで反映。

## master参照

master b51c00c566667fc2ed1fe640543219f480642002 を読み、8e4a63e6の光源サイズによるcoarse caustic preparationと精密fallbackを確認した。master側のメッシュ数を連続包囲へ直接移すのではなく、安い処理をまず検証し、拒否時は精密経路へ戻す構造を参照した。masterのソースとhashはmaster-reference/とmaster-reference.json。参照ブランチのcheckoutや別作業の変更はしていない。

## 原因を通常APIで確認

current-profile/のcurrentは直前版の再計測、deepは勾配幾何をさらに分離、fpは演算環境ガードを計測コピーに入れた対照。LOM uniformの有限光源410時刻すべてで、幾何検出入口のFTZ/DAZ maskが非ゼロだった。screen_bandsはenvironment_okで即座に拒否し、毎回D14の細かい検出へ進んでいた。対応する3反復中央値はdetect本体約101msから約1.3ms、screen約18.8ms。これらは計測コピーの内訳であり、タイマーなし通常APIの厳密な配賦とはしない。値・スカラー尤度・14勾配の保存出力は対照実験でもbit一致。

## 最終通常API比較

KB230598 77,843時刻・18dataset、視差+Kepler LOMおよびstatic、uniform/バンド別linear LD、float64、atol=rtol=1e-3、CPU0、OMP/BLAS/MKL各1計算thread。コンパイル・warmup・I/Oを除外し、JAXは同期。変更前の検証済みnative b3b698710041d5a23ca38de1ec3e26633c4dd2209457e3ecbf4e42cdbd3d7b2fと比較。別プロセスを順次動かし5回の順序回転・反転ペア。倍率は各ペアの旧/新の中央値で、別々の秒中央値の商ではない。

| LD | 計算 | 新LOM 秒中央値 | 実時間倍率 | cycles倍率 | 新LOM/static 実時間 | 新LOM/static cycles |
|---|---|---:|---:|---:|---:|---:|
|uniform|value|0.251859|1.0028|1.0031|1.5247|1.5345|
|uniform|likelihood_reverse|0.549032|1.1938|1.1953|1.7246|1.7307|
|uniform|changing_q|0.548084|1.1944|1.2010|1.7759|1.7361|
|paper_ld|value|0.266763|1.0040|1.0040|1.5972|1.5900|
|paper_ld|likelihood_reverse|0.566925|1.1696|1.1763|1.8058|1.7808|
|paper_ld|changing_q|0.388740|1.1926|1.1901|1.7822|1.7665|

fixed-point scalar reverseはprofile-flux尤度+14物理パラメータ勾配。changing_qはqを評価ごとに変える同じ勾配評価。秒とcyclesを区別し、他のパラメータ・ハードウェアでの普遍的最速は主張しない。

## 不採用・未採用の案

- 複数根キャッシュ: 1/4/8/16/32/64個を同じ77,843幾何のs順で比較。近傍再利用は約35,627から約76,000へ増えたが、8個でCPU timeが約1.10倍、64個で約1.28倍になった。孤立した中心根計算のプローブであり、通常APIの速度比較とはしない。製品へ入れていない。
- moving包囲の幅: 0.03125–1.0倍rhoの小規模プローブ。幅だけを広げて全体が速くなる根拠は得られず、0.125*rhoを維持。全幾何へ直接queryした診断で、通常APIの呼出し順や選択と同じ配賦ではない。
- 区間を進行方向へ寄せる案: 通常APIで値約3%・勾配約22–23%のcycles改善が出たが、uniform t=10061.301151のrho微分が0.400847493から点近似の0へ変化。独立VBM有限差分0.40084751に対する既定2e-3基準を満たさず、その変更だけを戻した。区間包囲の誤証明と断定せず、近似ルート変更によるHMC向け微分精度の拒否として保存。rejected-directional-interval/にソース・native・数値・失敗ログ。
- オフライン時刻圧縮: 全参照値とJacobianを見て選ぶoracle試験ではuniformの77,841 unique時刻を831点で指定の値1e-5/Jacobian1e-3以内に表現できた。これは1固定パラメータの圧縮可能性であり、オンライン速度の実測、パラメータ領域の保証、製品採用とはしない。compression-probe.json参照。

## オンライン時刻圧縮の研究用試作

adaptive_curve_probe.pyは全結果を事前に見ず、通常LightCurveの値と14物理Jacobianを必要点だけ呼び、degree-8時間補間をsampled errorで細分化する。256点の固定tileへ詰め、尤度+勾配を計算する。profiled source/blend fluxの微分は重み付き正規方程式のenvelope theoremで消し、残差とJacobianを縮約する。時間区間全体の誤差証明とパラメータ変化時のHMC保証は未確立なので、通常APIには入れていない。値のsampled目標1e-7、Jacobian1e-5、nominal比較はabs(delta loss)<0.1、global gradient scaled difference<2e-3、value scaled difference<1e-3を事前に設定した。

| LD | 秒中央値 | native padded時刻数 | abs 尤度差 | global勾配 scaled差 | nominal比較 |
|---|---:|---:|---:|---:|---|
|uniform|0.395421|4096|0.000578063|3.38467e-05|True|
|paper_ld|0.592726|8960|0.000150567|6.61361e-05|True|

固定ベクトル1点の探索的な比較。nominal比較が通っても区間証明やHMC認証にはならない。dense manual envelope gradientと既存JAX loss/gradientとの差、全時刻の値/Jacobian差、初回compile込み時間をraw JSONに保存。性能は誤差とともに扱い、失敗した条件を速度表の成功例として扱わない。

## 最終数値検証

- API出力24保存配列とJAX尤度/勾配4条件がbit一致。
- 独立VBM円盤+適応LD円環344条件、最大正規化誤差1.99135e-05。
- 14パラメータの方向有限差分: 153時刻、4LD×3方向×3step、max best-step relative error 6.60286e-05 <2e-3。
- rho微分8条件: 最大変化点と固定難点を独立VBMで検査。最大best-step差2.44255e-05。
- q変更28条件、全77,843行の値収束・勾配有効性を確認。support_certifiedはfalseのまま。
- C++ ctestとFTZ/DAZ復元・screen有効化の回帰を通過。Python/JAX: 554 passed, 3 skipped in 494.57s (0:08:14)
- ソース/nativeのhashを固定して検証終了時の不変性を確認。HMC samplerやposterior収束の検証は行っていない。

## 保全・再現

before-source/とbaseline-native/が開始時点、this-task.patchは今回の2ファイルだけ。final-source/、final-native/、final-source.json、verified.jsonが採用結果。compare.py/worker.pyとcomparison2.jsonに全raw・counter coverage。validate.pyが独立値・rho/方向微分・q/経路・全回帰を逐次実行。profiling binariesはcurrent-profile/before-fp-packageとpackage、instrumentedソースは同ディレクトリ。計測用setupは保存した開始時点ソースを前提とする。CPU0・thread1・TMPDIRをこのevidence内に設定し、数値ジョブは逐次実行する。

## 一括ノード評価と精度を揃えた対照

通常精度のprepared試作は約0.24–0.28秒まで下がったが、局所Jacobianのsample検査に不合格。点近似/有限光源の選択変化があり、採用していない。ノード精度を1e-9にするとsample検査は通るが、通常1e-3版とのLD勾配差が一部で事前基準2e-3を超えた。失敗例を保持し、基準を緩めていない。以下の同精度dense対照は、この差を補間とコア精度変更へ分けるためのもの。dense時間にはcompileを含み、速度比較には用いない。
- uniform: 同精度dense対補間の尤度差 3.91943e-06、global勾配scaled差 2.99467e-06、nominal対照合格 True。通常精度対dense高精度のglobal勾配差 0.000403061。
- paper_ld: 同精度dense対補間の尤度差 -3.74419e-05、global勾配scaled差 2.53838e-06、nominal対照合格 True。通常精度対dense高精度のglobal勾配差 0.00210853。

全時刻の局所Jacobian差もrawに保存。通常精度adaptive試作の局所差は約0.37–0.40であり、global尤度/勾配の比較だけで安全とは判断しない。いずれの補間試作も製品採用なし。未完了の課題は時間区間全体の誤差保証、広い物理パラメータ変化、14方向微分と実samplerに対する検証、拒否時に精密経路へ戻す実装である。
