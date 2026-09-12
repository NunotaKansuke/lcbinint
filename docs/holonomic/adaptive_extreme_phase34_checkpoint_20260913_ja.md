# Phase34: D14Real C/G/Zの共通冪評価

基準6933775。低次3/4のC/G/Zと微分を別々のHornerで評価せず、v²/v³/v⁴を共有して評価する研究flag HOLO_D14_SHARED_POWERSを追加。D14Realの演算精度は維持するが、加算順が変わるためbitwise parityではない。qfとdoubleの評価器は変更しない。既定OFF、production router/fixed-n_r defaultは不変。

## 全sweep版

単体D/D'評価は約498→433ns。7216行warm profileのsteady D14Real中央値は21.51→18.47µs。一方qf合計が1391.66→1585.38msへ増え、whole合計4260.87→4439.43msとなり不採用。case149のdb1/epoch15,23とcase105のdb1/epoch15で追加qf費用が大きい。qf_changes.tsv参照。

## 最初の2 sweepだけ使用

HOLO_D14_POWER_SWEEPS=2でmixed Aberthの最初2 sweepだけ共通冪、以後は従来Hornerに戻す。global convergence/residual/completeness gateは維持。これは必ず3 sweep以上実行する変更ではなく、既存条件で早く収束する場合もある。D14Real精度のまま評価形だけを変える。

profile: whole中央値 .307739→.305849ms、全行合計4260.87→4108.78ms。D14Real全行229.77→218.26ms、qf全行1391.66→1248.94ms。

## Non-profile whole

baseline=Phase28 whole_candidate.tsv、candidate=限定版。両tol各7216行、repeat3、CPU0、D14/topology込み。warmは各trajectoryの初回cold epochも含む。

|RelTol|lane|baseline p50 ms|candidate p50 ms|baseline p99 ms|candidate p99 ms|
|---|---|---:|---:|---:|---:|
|1e-3|cold|.387790|.384776|2.958745|3.128002|
|1e-3|warm|.297091|.293931|1.051681|1.033936|
|1e-4|cold|.431341|.427392|3.043799|3.172723|
|1e-4|warm|.344702|.340927|1.310342|1.278895|

全行収束、status/node差0、最大relative mu差4.90e-12。VBM reference超過は1e-3=0、1e-4=既存3。radial-onlyはほぼ不変。cold p99回帰があるため一律default化しない。

## Correctness

全14432根集合は最大scaled root差4.54e-14、role/physical/event/cell/completeness差0。qf-cold94→90、16行で発生有無が変わる。全row改善の意味ではない。根座標はbinary64 export比較で区間包含証明ではない。
独立reference550行のusable466でobserved violation0、mu/stop完全一致。解析5Jac220行は値/品質/stop完全一致。adaptive unit1484 checks/0 failures。

## 判断・限界

C/G/Zの固定低次数を使う演算削減は実際に軽い。ただし演算順による候補の微小変化がqfへ落ちるrowを変える。中央値改善だけで採用しない。研究flagとして保存。まだVBM勝利ではない。
whole実行中に独立root監査をCPU1、reference/Jac/unitをCPU2/3で実行した。CPU affinityは分離したが共有cache/電力影響の可能性は残る。保存baselineとの比較であり、全workloadを交互反復した統計ではない。

## 再現

Phase28 whole/profile flagsに -DHOLO_D14_SHARED_POWERS -DHOLO_D14_POWER_SWEEPS=2 を追加。全sweep版では後者を除く。
whole: evidence/holonomic/adaptive_estimator_phase10_20260913/runner_hybrid.cpp、input_snapshot.tsv、出力whole_candidate.tsv、repeat3、taskset -c 0。
profile: benchmarks/holonomic/bench_adaptive_stage_profile.cpp、同入力 warm。
root: bench_d14_root_work_detail.cpp、同入力 /tmp/p34_roots.tsv /tmp/p34_events.tsv 1 both、native residual screen/warm noise/scalar constants/shared powers/2 sweep、比較oracle phase28。
reference/Jac/unit: -O3（march/unrollなし）、warm noise/scalar constants/shared powers/2 sweep。reference_cases.tsv warm、gradientはwarm引数なし。

全raw/summaryはevidence/holonomic/adaptive_extreme_phase34。単体timingはPhase33と同じprobeのtimingモード、scalar constants/shared powers付き。係数準備はタイマ外。
