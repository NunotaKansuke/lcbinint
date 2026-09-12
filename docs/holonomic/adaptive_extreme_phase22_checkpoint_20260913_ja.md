# Phase22: batched double Aberth interactions

基準71385a3。14個の差・norm・逆数を先に作り、sumは元のj順に加算。
invalid normがある場合は元のscalar loopへ戻り、最初の失敗位置と
fail-closedを保持。trace/legacy/次数14以外は旧処理。

inline版: objdumpでd14_active_presearch内にzmm/ymm/xmm vdivpdを確認。
全7216入力でmu/node/status差0。既存D14 root-scheduling unit pass。
しかしsteady profile presearch中央値 .033798→.052356 ms、
profile whole .339027→.357778 msと悪化。
SIMD命令の生成だけでは利益にならない。配列退避/読み戻し、
追加ループ、コードサイズの費用が考えられる（個別費用は未計測）。

noinline helperでコードサイズと退避を抑える再試作を比較する。
どちらもwhole勝利の証拠ではなく、negative screening用profile。

再現: Phase21 profile compile flagsにHOLO_D14_BATCH_INTERACTIONSを追加。
input_snapshot.tsv warm、CPU0。inline_candidate.patchが初回試作。

## Helper結果と撤回

helper版も7216入力でmu/status/node差0だが、presearch .0545135 ms、
profile whole .359943 msでさらに悪化。
両試作とも採用せず、radial_events.hppを基準HEADへ戻した。
helper_candidate.patchに第二案、inline_candidate.patchに第一案を保存。
パッチは各々71385a3へ独立に適用する。重ねて適用しない。
wholeの最終性能比較には進めない。profileでpresearchが約55–61%悪化し、
後段real/residualに取り返す改善が見られないため。

精度gateの変更やproduction default変更はない。
14個の相互作用のSIMD化は今回の方式では損。次は一根あたりの
Horner/interactionの仕事量ではなく、同じ全根状態で不要な
presearch反復を減らせる条件を既存失敗evidenceと照合して再検討する。
