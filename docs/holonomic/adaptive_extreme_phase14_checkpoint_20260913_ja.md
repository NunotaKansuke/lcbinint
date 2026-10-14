# Phase14: topology probeの重複quartic solve（研究中）

基準2a4002e。quartic_topologyはSturm countを認証した後、
real_root_thetasを解いてcountを照合、不一致ならqf isolationしていた。
この関数では得た根座標はすべて破棄され、下流へ渡されていない。
HOLO_TOPOLOGY_STURM_COUNT_ONLYでこの二重検査を省く研究候補を追加。
Sturmのprecision ladder、root_count、square-free判定、0根時の符号判定は維持。
下流の実際のarc境界求解は変更しない。追加照合を除く変更なので、
単なる演算削減と同じ保証だと断言せず、status/reference/topologyを監査する。
既定OFF、production routerは変更なし。

warm profile全7216入力、初回1804行を除いた5412行で:
probe p50 .036820→.003219ms、topology .236389→.195416ms、
profile付きwhole .471485→.431244ms。
全行のmu差0、node/収束差0。candidateのtest_quartic_sturm pass。
profileなし3 repeatsの全14432行wholeを開始（未完了、採用未決）。

再現: Phase13のcompile flagsに-DHOLO_TOPOLOGY_STURM_COUNT_ONLYを追加。
bench_adaptive_stage_profileの第2引数にwarmを渡してprofile_warm_count_only.tsvへ。
runner_hybridへ同じinput_snapshot、whole_count_only.tsv、3を渡す。
CPU0で逐次実行。root/event/status監査と独立referenceはまだ残件。

## 全件監査・採用

全14432行3 repeatsのwhole測定完了:
1e-3 cold p50 .469304→.432379ms、warm .425370→.389366ms。
1e-4 cold .513095→.477034、warm .486406→.451486。
全laneでmu差0、status/node差0、全件converged。
1e-3 cold p99は3.220302→3.238889ms（約0.6%増加）、warmは1.239364→1.192541。
maxは約56msのまま。全分位改善とは主張しない。
VBM1e-6参照超過は1e-3で0、1e-4で既存の3件のまま。

独立angular referenceは550行中466行usable、observed violation0。
referenceが同じtopology planを使う限界を補うため、全cold/warm planを
別途exportし、cellのr_lo/r_hi/r_mid/kind/count/status、eventのradius/
kind/physical/offset/uncertaintyが392456行すべてbyte一致と確認。
SHA256とcompressed rawを保存した。これは有限corpus上の一致であり、
全パラメータにわたるSturm実装の厳密証明を追加したという意味ではない。
adaptive unit1484 checks pass、薄い像/chartを含むquartic_sturm test pass。

この証拠と、使用されない座標を求めているというコード上の依存関係から、
count-onlyを既定に採用する。HOLO_TOPOLOGY_VERIFY_UNUSED_ROOTSを定義すれば
旧二重検査へ戻せる。Sturm自体の全gate、下流のarc root certification、
D14 authority、production routerは変更しない。

今後の比較で旧baselineを生成する場合は上記VERIFY flagを使用する。
bench_topology_plan_parity.cppを同一flagsでこのflag有無の2通りcompileし、
input_snapshot.tsvを引数に渡すとtopology_*のrawを再生成できる。
reference再現はcheck_adaptive_reference.cppへ既存reference_cases.tsvを渡す。

VBM1e-3の既存保存中央値（uniform .061369、LD .314456ms）に対し、
今回warm uniform .361802、LD .418064ms。まだ全面勝利ではない。
