# Phase30: 既にreject確定のpredictorのNewton省略（保留・コード撤去）

pred_wildは後段で無条件rejectされるため、Newton前に同じcopyを破棄する試作。debugでは従来traceを維持。7216行warm profileでmu/node/status一致。baselineはPhase29 profile_baseline。whole p50 .3079395→.312267ms、arc .0769325→.0777705ms。明確な利益がなくコードを戻してpatch保存。単一run比較から悪化の原因まで断定しない。ユーザーの指示に合わせ、次はD14構造を使うbinary64評価の誤差を調べる。新しい受理gateは導入しない。

再現: Phase28 profile flagsへHOLO_MV_SKIP_REJECTED_PREDICTOR追加、bench_adaptive_stage_profile.cpp、CPU0、input_snapshot.tsv warm。
