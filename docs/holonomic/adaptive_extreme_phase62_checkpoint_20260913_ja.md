# Phase 62: Fejer nodeを中央から評価する試作

基準c8020eb。topologyのquartic_topologyはSturm根数のみを返し、既存の全根座標をadaptiveへ無費用で渡せる構造ではなかった。そのため最初のcold quarticをfold端から離す目的でnode評価順を変える試作を行った。

HOLO_ADAPTIVE_CENTER_OUTでm=2^levelのnode indexをm/2,m/2-1,m/2+1,...とした。同じnode集合、cache slot、積分/estimator加算順を維持する。新しいsampleを追加しない。ただしnearest cached seedと継続の履歴が変わるので数値結果の完全一致は前提にしない。全refinement levelで適用し、既評価nodeは従来通り再利用。

## A/B

同HEAD/flags、CPU0、baseline→candidate→candidate→baseline。Phase59のwarm seed tol1e-11とPhase54中点有理式を双方で有効化。subset256行の後に全7216行RelTol=1e-3 warm-timingへ拡大した。全行convergence/node差0、max |delta mu|/max(1,|mu|)=1.1812e-11。

全7216行のうちtrajectory先頭を除く5412行のprofile中央値:

|run|uniform whole ms|uniform arc ms|LD whole ms|LD arc ms|
|---|---:|---:|---:|---:|
|base1|0.229716|0.056541|0.286995|0.065484|
|candidate1|0.227665|0.054831|0.289810|0.064159|
|candidate2|0.228479|0.054562|0.286613|0.064531|
|base2|0.230113|0.056565|0.285901|0.065517|

全7216行でrootpair successは369593→390255、quartic warmは126687→106025、quartic coldは40116で不変。中央開始でrootpairが成功しやすくなる観測はあるが、wholeの改善は小さくLDでは一貫しない。

## 追加sweep診断

HOLO_QUARTIC_WORK_PROBEによる候補一回の診断をPhase61の保存済baselineと比較した。counter比較であり、違う実行日の時間差を性能根拠にしない。steady-state各profile2706行:

- cold callsは各15037で不変。
- cold sweepsは各353459→296896、平均23.506→19.744 sweep/call。
- uniform warm calls45555→37852、sweeps236736→203552。
- LD warm calls49307→41534、sweeps261046→228780。

狙ったcold初期探索の反復削減は起きた。ただし反復数低下をそのままwhole勝利とは扱わない。今回の主要な結論は、cold seedの位置は有効な仕事量パラメータだが、単純な全level中央外向き順序では十分なwhole改善をまだ示していないこと。

## 判定・再現

今回は本線へ追加せずsourceを戻し、center_out.patchで保存。通常whole runner、cold/1e-4、unit/独立reference/5Jacの追加監査は行っていないので、数値的にproduction適合を証明した案ではない。既存最速候補・routerは維持。VBM勝利目標は未達。

基準commitへpatchをapply後、evidence/holonomic/adaptive_extreme_phase62内のbuild_trial.sh、run_trial.sh、run_full.sh、summarize.pyで再現。build_probe.sh後、p62_probeに元input_snapshot.tsvとwarm-timingを渡すとwork_probe.tsvを再生成できる。保存済rawは実行前に保護する。cache1/cache2はcandidateのファイル名でありcache方式の実装名ではない。
