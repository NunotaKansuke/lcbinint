# Phase 64: D14 physical contactからquartic seedを生成

基準cb17d0a。topologyはSturm countのみなので、midpoint全根を無費用で受け渡すことはできない。一方RadialEventにはphysical contactのradius/radius_lo/fold_t_seedが存在する。これを最初のquartic seed生成へ利用する研究試作。

## 構成と契約

HOLO_ADAPTIVE_FOLD_QUARTIC_SEED、default OFF。adaptive callbackでcached sampleがなく、force_coldでなく、arc cellの場合のみ実施。cellのr_lo/r_hiとradiusが完全一致するphysical_realかつphysically_real、t seed validなeventから近い方を選ぶ。q/rhoによるroutingはしない。該当eventがない場合は元のcold path。

m=t*、v=0で既存root_pair_dRによりdm/dR,dv/dRを求め、dr=(R-radius)-radius_loから一次予測を作る。v>0なら2実根、v<0なら共役2根をseedにする。

foldで `P(t)=(t-m)^2 (a t^2+b t+c)` と書くと、ascending係数pからa=p4、b=p3+2mp4、c=p2+2mp3+3m²p4。残り2根はこのquadraticから生成する。real quadraticは積の関係を使う安定形、negative discriminantは共役pairとする。nonfinite・一致するseedは使わない。

**ここで作るのは近似seedで、root certificateではない。** 係数はbinary64 radiusで評価し、radius_loは予測drへ使うだけで、qf eventを完全な高精度factorizationへ持ち込んだ実装ではない。離れたnodeでの一次予測は不十分なことがあり得る。

QuarticWarm.deg=4、n_real=cellの既存Sturm crossing countとして既存warm Aberthへ渡す。反復上限、step受理gate、real-root数、branch、endpoint、adaptive error ledgerは維持。失敗時は従来のcold quarticへ戻る。force_cold retryではこのseedを再注入しない。接触seedだけでphysical valueを直接返さない。D14を再solveせず、packet/sample-fitや新しいquadratureも使わない。

## 初期profile

Phase59のwarm solve tol1e-11とPhase54中点有理式を双方で有効にした。subset256行でmu最大scaled差2.657e-13、convergence/node差0。cold quartic1824→366。全7216行へ拡張してconvergence/node差0、mu最大scaled差1.256e-11、cold quartic40116→5458、warm呼出し126687→161423。

CPU0、baseline→candidate→candidate→baseline、V2ProfileScope付き。先頭epochを除く5412行の初回比較中央値はuniform whole0.230025→0.221724ms、arc0.056783→0.046752、LD whole0.286552→0.277789、arc0.065532→0.056232。cold呼出し削減とwhole改善が共に見えたため通常wholeへ進めた。呼出し数差だけからseed試行の成功率を厳密には算出しない。

## 数値検証

unit1484 checks全pass。独立reference550行中466有効、observed violation0。84行は参照未認証。解析5Jac220行はPhase59比でquality/stop変化0、max |delta Jac|/max(1,|Jac|)=1.8832e-8。全corpusの高精度独立Jacobian oracle監査ではない。

## 再現

evidence/holonomic/adaptive_extreme_phase64にbuild_trial.sh、run_trial.sh、run_full.sh、summarize.pyを保存。validate_trial.shでunit/reference/Jacと候補whole binaryを生成、summarize_validation.pyでPhase59との差を集計。build_whole_baseline.shとrun_whole.shで通常wholeの同HEAD A/Bを実行。raw再生成前に既存結果を保護すること。cache名のrawはcandidateの意味。

通常wholeはcold/warm/radial-only、両tol、3repeat、D14を含む。warmは各trajectory先頭のcold epochも含む。profileとは異なりV2ProfileScopeなし。最終判断の詳細を下に追記する。

## 通常whole結果・判断

同HEAD/flagsでseed macro一個だけを変更し、CPU0、3repeat、baseline全行→candidate全行。交互実行ではないため小さい時間差には実行順の影響が残る。

|条件|warm baseline中央値 ms|warm candidate中央値 ms|paired speedup中央値|
|---|---:|---:|---:|
|1e-3 uniform|0.243608|0.233569|1.02631|
|1e-3 LD|0.294488|0.285434|1.02281|
|1e-4 uniform|0.283414|0.275006|1.02247|
|1e-4 LD|0.339474|0.332639|1.01909|

両profile合算1e-3: cold p50 0.362980→0.355891、warm 0.274407→0.266204、radial 0.098364→0.090478ms。warm p90 0.544581→0.535984、p99 0.945043→0.939497。cold p99は2.933644→2.943862msと僅かに増えたため、全指標非回帰とはしない。

1e-4 warm p50 0.314398→0.306955、p90 0.606454→0.596918、p99 1.171513→1.164980。全詳細はwhole_summary.json。

全14432行、cold/warm/radial全laneで収束し、status/node差0。最大relative mu差1.256e-11。保存referenceとの差がtargetを超える行数は1e-3が0、1e-4が既存3行のまま。新規増加なし。

default OFFの研究候補として保持する。production router/defaultへ昇格しない。D14接触情報の再利用によって既存cold quarticを大幅に減らせたことと、wholeでも小幅な利益が出たことが根拠。ただし既存gatesを通す近似seedであって全入力での新しい厳密certificateではない。1e-3 uniformでVBMに勝つ目標は未達。
