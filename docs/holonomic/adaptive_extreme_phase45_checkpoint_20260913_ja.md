# Phase45: tiny-v pairの局所bracket認証

基準0a5ab4f。VFloorだけで拒否されるpairに既存のquartic_local_bracket認証を適用し、根の分離を証明して追跡を継続できるか試した。

## 診断

予測guardとNewton step判定を通過、finite m、0<v<=VFloorのpairだけを対象とする。binary64 quarticの両endpoint近傍に互いに交わらない区間を作り、外向き演算で端点異符号と区間内derivative一定符号を確認。nominal t半径1e-9、隣接分離の1/8以下。既存kernelをそのまま利用。

warm steady各2706行、uniform6153 eligible中6142、LD7933中7916で認証成功。これはbinary64係数多項式の個別2根の存在/一意性であり、元の物理パラメータからの係数誤差や全quartic根のcompleteness証明ではない。新しい全根受理の根拠には使わない。

## 研究A/B

研究flag HOLO_MV_TINY_BRACKET_ACCEPTではVFloor拒否だけをこの認証で置換し、後段の集合branch/endpoint residual/phi sign/暖機thin-arc cross-checkをそのまま実行した。predictorがwild、Newton非収束、非有限、v<=0は従来通り拒否。

|steady累積/中央値|uniform baseline→candidate|LD baseline→candidate|
|---|---:|---:|
|rootpair cold戻り|43840→37869|47400→39730|
|arc ms|.068922→.0708255|.079101→.0805675|
|whole profile ms|.2468535→.248639|.3054745→.307774|

候補生成/探索の戻り回数は減るが、認証費用を含むwallでは逆に遅い。cold solveという名前だけで重いと判断してはいけない。再利用中のquartic warm補正を、区間評価で置き換える交換費用が利益を消したと解釈する。

診断probeありbaselineでは認証費用を余分に払っているため、その値と候補を比べると改善したように見える。上表はprobeなしbaselineを別runした公平な比較。rawのprofile.tsvは診断、profile_baseline.tsvはprobeなし、profile_candidate.tsvは受理試作。

## 正確性と採否

7216行でok/node数差0、mu差/(1+|mu|)最大9.69e-12。独立reference550行中usable466、observed violation0。成功した円板がそのままobservableの要求精度を保証するわけではなく、referenceで確認した範囲の結果。

速度利益がないため受理試作は撤去しpatch保存のみ。診断flag HOLO_MV_TINY_BRACKET_PROBEとカウンタは残す。通常whole 14,432行、解析5Jac、unit追加検証には進めていない。本番VFloor/受理gate/default/router変更なし。VBM勝利は未達。

## 再現

Phase44と同じcompiler flagsとHOLO_ARC_COMPACT_STORAGE。bench_adaptive_stage_profile.cppをcompile、input_snapshot.tsv warm-timing、CPU0。診断には-DHOLO_MV_TINY_BRACKET_PROBE、受理試作にはtiny_bracket_accept.patch適用後-DHOLO_MV_TINY_BRACKET_ACCEPT。baselineはどちらも指定しない。

referenceはPhase42と同じflags（march/unrollなし）に受理flag、tests/holonomic_cpp/check_adaptive_reference.cpp INPUT warm。INPUT=evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv。

raw/summary/patchはevidence/holonomic/adaptive_extreme_phase45/。単回profile、probe→candidate→baseline順。reference CPU1作業がbaseline測定の一部と並行し、共有cache/powerの影響は残る。whole速度勝利を主張しない。

## 次の判断

tiny-vの多くは個別の根として認証可能。しかし現行bracket評価をそのまま追加しても速くならない。これ以上進めるには、後段の重複認証の共有、より安い同等の局所証明、あるいはquartic補正自体の省力化が必要。単純なVFloor閾値緩和はしない。
