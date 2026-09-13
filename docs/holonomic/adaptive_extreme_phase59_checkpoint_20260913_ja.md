# Phase 59: quartic warm seedの停止精度をconsumerから分離

基準fa20090。既存quartic warm Aberthは内部step約1e-15まで反復する一方、呼出し側のkWarmStepTolは1e-7である。最終物理endpointは別途phi Newtonで補正され、adaptive geometry/error ledgerで判定される。研究flag HOLO_QUARTIC_WARM_SOLVE_TOL=1e-11で、warm quartic seedの内部反復だけを早く終了する試作を検証した。

これは単なるcompile-time最適化ではなく内部seed精度の変更であり、全入力で同等精度を数学的に証明したものではない。defaultは従来の0指定（内部1e-15）のまま。macroは0より大きく既存kWarmStepTol以下にstatic_assertで制限する。最大20反復、既存1e-7受理gate、実根数一致、cold再探索、root-pair branch/certificate、endpoint polish、外側value/gradient tolerancesは変更しない。D14自体には適用しない。

## Profile A/B

Phase54中点有理式を両方に入れた。subset256行で最大scaled mu差2.27e-13、status/node差0を確認し、全7216行へ広げた。全profile比較でもstatus/node差0、max |delta mu|/max(1,|mu|)=1.0632e-11。

CPU0、同HEAD/flags、baseline→candidate→candidate→baseline。先頭epochを除く5412行のV2ProfileScope付き中央値:

|run|uniform whole ms|uniform arc ms|LD whole ms|LD arc ms|
|---|---:|---:|---:|---:|
|base1|0.240800|0.067959|0.298787|0.078057|
|candidate1|0.230416|0.057365|0.287381|0.065917|
|candidate2|0.230411|0.056944|0.286108|0.066041|
|base2|0.241488|0.067757|0.300702|0.078068|

## 数値検証

- adaptive unit:1484 checks 0 failures。
- independent reference:550行、466有効、observed violation 0（残り84行は未認証）。
- analytic5Jac:220行、stop変化0。max |delta Jac|/max(1,|Jac|)=2.0222e-8。
- quality変化1行:caustic-cross、u=0.5、RelTol=1e-4の第2成分がFiniteUncertified(2)→ToleranceMet(1)。約-2.2963e-9→-2.3032e-9。悪化はないが完全parityとは報告しない。
- gradient参照は既存候補との差とunitの検証であり、全corpusの独立高精度5Jac oracle監査ではない。

## 再現

evidence/holonomic/adaptive_extreme_phase59内のbuild_trial.sh、run_trial.shでsubset、run_full.shで全profile。summarize_profile.pyで集計。validate_trial.shでunit/reference/Jacおよび候補whole binaryを作成。build_whole_baseline.sh、run_whole.shで同HEADの通常whole A/Bを実行する。raw再実行前は既存結果を保護すること。

cacheというraw名は候補を表す慣用名で、今回cacheは実装していない。速度だけで精度緩和を正当化しない。最終採用は通常wholeと元corpusのaccuracy/statusを確認してから判断する。

## 通常whole結果と採用範囲

同HEAD/flags（macro一個だけ相違）、CPU0、各3repeat、baseline全行→candidate全行の順。timed pathにはV2ProfileScopeを入れない。両tolerance×7216=14432行でcold/warm/radialの全laneが収束。status差0、nodes差0、max relative mu差1.072e-11未満。保存referenceとのtarget超過は1e-3が0、1e-4が既存3行のままで新規増加0。

|条件|warm baseline中央値 ms|warm candidate中央値 ms|paired speedup中央値|
|---|---:|---:|---:|
|1e-3 uniform|0.254605|0.243821|1.02716|
|1e-3 LD|0.304653|0.294475|1.02697|
|1e-4 uniform|0.298569|0.283341|1.03277|
|1e-4 LD|0.354891|0.340197|1.03102|

中央値同士の比とpaired ratio中央値は異なる。

両profile合算の1e-3 cold p50 0.374101→0.363827、warm 0.284525→0.274765、radial 0.108742→0.0984205。warm p90 0.567918→0.545985、p99 1.004663→0.9531465。

1e-4 warm p50 0.3276685→0.3147065、p90 0.633690→0.6074705、p99 1.269938→1.170183。cold maxは1e-3で51.673591→51.695506msと僅かに増えているので全指標非回帰とは表現しない。詳細はwhole_summary.json。

研究flag（default OFF）として保持する。production default/routerへの昇格はしない。通常wholeのspeed/statusと既存独立referenceで小幅な利益が示されたが、内部seed精度変更の普遍的保証は未証明。将来別corpusでの反例があれば既存defaultへ戻せる状態を維持する。現状の1e-3 uniform VBM勝利目標は未達。
