# Phase 57: hardware samplingでgeometryの実行箇所を確認

基準6538f5f。Phase56の32軌道×LD有無×4epochを通常whole runnerで20repeat実行し、perfのcycles:uを499Hz、DWARF stack 8192bytesで記録。solver変更なし。コンパイルはPhase54候補flagsに-gを追加したもの。

4989 samples、lost samples 0。raw40.5MBをgzipして保存。cold/warm/radial-onlyと両toleranceを含む混合workloadなので、以下をwarm単独・uniform単独・全14432行の割合と解釈しない。sampling overheadが入るため今回のwall時間は速度比較に使わない。

## 観測

self cycle割合の主要symbol:

|symbol|%|
|---|---:|
|d14_active_presearch|18.97|
|generic double Aberth|9.80|
|同関数constprop clone|4.82|
|sincos (perf symbol名sincosf32x)|6.88|
|polish_endpoint|5.78|
|mapped_radius|5.62|
|qf multiply (__multf3)|3.24|
|qf add (__addtf3)|2.99|
|real_root_thetas_transport|2.67|
|D14Real structured evaluation|2.37|
|transport_pairs_valid|0.66|

2つのgeneric Aberthのcallgraphにはreal_root_thetas_warm→real_root_thetas_transport→arc_intervals→mapped_radiusを確認した。したがってroot-pairの失敗時に使うquartic warm補正は、次に調べる具体的な対象となる。generic Aberthには他callerもあり得るので14.62%すべてをこの一経路に帰属させる完全なcaller別集計ではない。callgraph.txtは1%未満の枝を省略し、symbols.jsonは0.5%以上のみ保存する。inline framesはself表とは重複するので足し合わせない。

一方、transport_pairs_valid自体のselfは0.66%で、ここだけを削って全体を数倍速くする見込みはない。前回までのguard/cache微修正より、quartic Aberthの反復・複素除算・再初期化の中身を優先する根拠が得られた。

D14 presearchの比率が大きいのはcoldを含む混合負荷であることに注意。以前のwarm専用stage profileの約8usと矛盾するという判断はしない。

## 制限と再現

これは新しい高速化でも精度検証でもない。元の通常whole候補をdebug symbols付きで観測した。source変更・router変更・新しい採用案はない。1e-3 uniformのVBM勝利は未達。

repository rootからevidence/holonomic/adaptive_extreme_phase57/build.sh、record.shを実行。入力はPhase56 snapshot。元binaryは/tmp/p57_wholeで、SHA256・compiler/perf/kernelをenvironment.jsonに保存。perf.data.gzだけでは別環境で同じsymbol解決ができる保証はないため、解決済みcallgraph/symbol表も保存した。raw記録時のstderrはrecord.log。

今後quarticへの変更を試す場合は、このsubsetを棄却用に使い、採用前に既存hard/referenceと全14432行wholeを確認する。
