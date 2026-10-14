# Phase25: exact presearch cycle detection (rejected)

基準e096e6a。HOLO_D14_EXACT_CYCLE試作は根のbinary64全bitと
active/stable全状態が1/2 sweep前と一致した場合だけ、整数周期を飛ばす。
残り端数のsweepは実行し、旧max_iter終了時と同じ位相を返す。
停止条件・精度・最終certificateは変更しない。trace時とdegree>14は使わない。
iterationsは旧経路のvirtual sweep数を保持、cycle_skipped_sweepsで省略数を別記。

結果: 7216入力中2行で100 sweep省略。全mu/status/node差0。
既存D14 root-scheduling unit pass。
しかしsteady presearch中央値 .033798→.035905 ms、
profile whole .339027→.3400965 msで悪化。
完全周期は稀で、全根snapshot/比較の追加費を回収できない。

試作コードは撤回し、cycle_candidate.patchに保存。
精度gateを緩めて周期と見なすような変更は行わない。
wholeの追加runは行わず、negative profileで棄却。

再現: patchをe096e6aへ適用し、Phase21 profile flagsに
-DHOLO_D14_EXACT_CYCLEを追加、input_snapshot.tsv warm、CPU0。

解釈と未検証仮説: double presearchの長い反復の大半は、厳密周期ではない。
丸め誤差floor付近の揺らぎか、まだbasin探索中かはこの計測だけで
区別できない。次にPの大きさとHorner前方誤差規模を比較すれば、
高精度へのhandoffが妥当な状態を見分けられる可能性がある。未実装。
