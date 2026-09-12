# Phase23: presearch patience A/B

基準f16e453。既存HOLO_D14_ACTIVE_PATIENCE=1をA/B。
active_tol=1e-12、後段D14Real/qf/residual/completenessは維持。
休止根は全interactionに残る。既定patience=2はまだ変更しない。

旧実験ではmax_iter固定短縮がcoverageを落としたため再実装しない。
1e-10/patience1も過去rand028でparity failureがあり不採用。
今回はtolを維持し、休止の確認回数だけを1へ変更。

profile: presearch中央値 .033798→.025985 ms、whole .339027→.333153 ms。
全7216入力のstatus/node差0、最大scaled mu差2.55e-12。
既存115-case root benchmarkはpatience1/2ともparity failure0。
その比較はexpanded-Horner vs structuredであり、patience相互の
根集合照合ではないため、別途trajectoryの全根照合を実施。

14432 root setsのbijective比較で最大scaled差2.839e-14、
role/physical/classification/event数差0、completeness差0。
qf-cold totalは双方100だが12行で発生が入れ替わる。
case105でcold/warmが変動し、case149のwarm2行が新たにcoldへ。
changed_cold_calls.tsv参照。

reference550行中usable466でobserved violation0、stop差0。
解析5Jac220行の品質分類差0、最大scaled derivative差4.81e-12。
wholeのcold/warm/tail比較は進行中。採用未決。

再現: Phase20/21と同じbinaryへHOLO_D14_ACTIVE_PATIENCE=1を設定。
profileはinput_snapshot.tsv warm、wholeはinput_snapshot.tsv whole_p1.tsv 3、CPU0。
独立bench_d14_structure.cppはroot_cases.tsv 3、双方patienceを別processで設定。
root detailはbench_d14_root_work_detail.cpp input roots events 1 both。
reference/JacはPhase20 binary、patience環境変数のみ変更。

別案のメモ: 根座標・active/stable状態の完全一致による周期1/2検出。
もし完全な作業状態が反復するなら、残り周期を省き同じ終点を返せる。
まだ未実装であり性能利益は未検証。今回のpatience変更とは別に扱う。

## Whole完了・判断

全14432行、各tol7216/7216収束。status/node差0、最大relative mu差3.05e-12。
1e-3 cold p50 .3900975→.382433 ms、warm .3176995→.3117585 ms。
1e-3 cold p99 3.178741→3.206302 ms、warm 1.159774→1.089140 ms。
1e-4 cold p50 .4338965→.4250615 ms、warm .365982→.359734 ms。
1e-4 cold p99 3.214608→3.231855 ms、warm 1.396692→1.391553 ms。
VBM reference超過数は1e-3=0、1e-4=既存3で不変。

一律patience1の既定化は見送る。warmでは有望だがcoldのp99に
小さい回帰があり、qf-coldの発生行も変わる。設定変更だけの研究結果として保持。
次はcoldを完全に旧設定に保ち、certified warm seedが利用可能な時だけ
patience1にする候補を分離する。物理パラメータheuristicではない。
新設定を導入する場合も今回の全体比較をそのまま流用せず再検証する。
