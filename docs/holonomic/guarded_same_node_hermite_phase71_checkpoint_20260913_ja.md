# Phase 71 checkpoint: 同一7点Hermite radial積分のshadow評価

日付: 2026-09-13
基準: `dev/holonomic` commit `9e409a3c096bab7ca2cec7c9a8d87e47c0cc1d6d`
計画: [`guarded_same_node_hermite_plan_20260913_ja.md`](guarded_same_node_hermite_plan_20260913_ja.md)

## 結論

uniform・value-onlyに限り、既存7点Fejer-II nodeの境界解からR微分を取り、同じ7点の値と傾きでH3/H7積分候補を作るshadow経路を追加した。shadowは既定OFFで、値・誤差ledger・停止・status・assuranceを変更しない。

H7は独立GL64/GL128局所参照が利用できた2790 panelのうち2746 panelでFejer7より誤差が小さく、観測誤差中央値も `7.07e-10` から `7.72e-15` へ下がった。一方で `paper_highA` の既知panelではH7にも約51.85の誤差が残り、モデルは実誤差を下回るpanelもある。shadowの追加費用に対して、既存のvalue-detail条件を守ったまま省ける追加nodeは0件だった。したがってGuardedEstimated controller A/Bは実装せず、この段階では不採用とした。

## 実装

新しい[`same_node_hermite.hpp`](../../src/lcbinint/magnification/holonomic/same_node_hermite.hpp)に固定H7積分重み、H3重み、補償和を実装した。H7は7個のFejer-II nodeの値と一階傾きから次数13以下の多項式積分を再現し、H3は中央を含む3 nodeから次数5以下を再現する。unit testで両方の多項式完全性を検査する。

uniformの

```text
F(R) = R * sum_arc(delta_theta) / (pi * rho^2)
F_R  = (sum_arc(delta_theta) + R * sum_arc(d(delta_theta)/dR)) / (pi * rho^2)
```

を使う。各境界root pair `t_± = m ± sqrt(v)` では、同じquartic endpoint/root-pair stateに既存の `root_pair_dR` を適用する。通常chartの角幅微分は

```text
d(delta_theta)/dR
  = 2 * ((1 + m^2 + v) * v_R - 4*m*v*m_R)
      / (sqrt(v) * ((1 + m^2 - v)^2 + 4*v))
```

で評価し、reciprocal chartも既存のroot-pair変換と角幅一致条件で検査する。新しいquartic/root solveは追加しない。子panelの局所座標では、`J=dR/dxi` がpanel半幅を含むことを踏まえ、

```text
g_xi = (F_R / norm) * J^2 + (F / norm) * R_xx * h^2
```

と評価する。実装時に `h` を二重に掛ける誤りを独立IFT照合で発見して修正した。fold側では`F_R`を作ってから小さい`J²`を掛けず、pairごとの`R * numerator * J² / (norm * sqrt(v) * denominator)`を`frexp/scalbn`で指数スケーリングして直接形成する。合成積が有限で固定R微分単体が非有限なら、後者の診断値だけUnavailableにし、jetは合成積側で判定する。unit testにdoubleの中間積がoverflow/underflowする値と`J²/sqrt(v)`のfold桁テストを追加した。

Hermite診断データはopt-in workspace sidecarに置いた。既定OFFの`AdaptiveSample`と`AdaptivePanel`はHermite用に拡大していない。shadowが有効な場合だけjet/panel sidecarを確保し、その容量もworkspace budgetに算入する。

panel側のモデルは比較用に

```text
E_H = max(weighted_detail_floor_fraction * D7,
          min(D7, nested_difference_safety * abs(QH7 - QH3)))
```

を記録する。これは同じnodeから作ったEstimatedモデルであり、Hermite剰余の証明ではない。shadowのpanel budget集計では既存のinner/geometry/event/roundoff項を加えたが、R-jet固有の不確かさを上界化できていない。このため集計pass数は楽観的な診断に限り、acceptance判定には使用していない。H7を返却値に使わず、Fejer7の誤差をHermiteへ転用せず、早期停止もしない。不適格時も既存node cache・7→15経路を変更しない。unit testでshadow on/offの値・node数・stop・assurance一致と、level 3からlevel 4へ進む際に15点のうち既存7点を保持して追加8点だけ評価することを確認する。

LD、5Jac、production router/default、既存のvalue/gradient tolerance契約は変更していない。

## Panel精度・停止guardの監査

`panels.tsv`はuniform参照corpusでFejer levelを3に固定し、panel単位の独立GL64/GL128積分を記録した。GL64/GL128差が所定の品質基準内だった2790行を観測比較に使った。これは独立radial積分による診断参照であって厳密誤差boundではない。

| 指標 | 観測 |
|---|---:|
| H7の観測誤差がFejer7より小さいpanel | 2746 / 2790 |
| Fejer7の絶対誤差中央値 | `7.0707e-10` |
| H7の絶対誤差中央値 | `7.7161e-15` |
| H3/H7モデルが観測H7誤差を下回った行 | 2 / 2790 |
| モデルbudget passのうち観測誤差を過小評価した行 | 0 |
| value-detail未解像panel | 14 |
| そのうちモデルbudgetだけを見てもpassするpanel | 0 |
| 現行detail条件を通り、モデルbudgetもpassしたpanel | 852 |
| そのうち現行7点解像から追加nodeを省けるpanel | 0 |

14個の未解像panelは全て微分detail decay条件も不成立だった。852個の候補は全て現行Fejer value-detailで既に解像済みで、controller上は後続8点を要求しない。

高増光反例 `paper_highA / cell=11 / depth=0` では、

```text
Q7       = 10892.263171133356
QH3      = 10934.499909774495
QH7      = 10888.926272793440
GL128    = 10837.076301144763
|QH7-ref|= 51.849971648677
E_H      = 91.147273962109
```

となった。相対許容差1e-3でもbudgetは約10.9、1e-4では約1.09なので両方rejectする。H7への補正で誤差が少し下がることは、panelを受理できる根拠にならない。過小評価2行は同じhighA caseのdepth1 panel（2 tolerance）で、モデル約31.42に対し観測H7誤差約39.71だったが、どちらもbudgetを通らなかった。

別実装のendpoint `dtheta/dR` と7 nodeごとのIFT結果を比較したscaled差は、2790 panelのp50 `7.41e-14`、p95 `4.08e-11`、p99 `7.39e-9`、max `7.71e-7`。maxを含めこれは観測照合であって、jet不確かさの上界としてledgerへ入れたものではない。

さらにunit testのdegree-14 bumpは7 node全てで値と傾きが0のまま積分値が正になる。したがって同じ7点の値・一階微分だけから厳密剰余boundを得られるとは扱わない。

以上から、計画書の必須value-detail gateを維持するとnode節約候補がなく、微分detail・jet uncertaintyを含むglobal Estimated gateを追加しても速度利益は見込めない。係数を緩めて未解像panelを通す試験はしていない。

## Whole-epoch paired比較

入力は既存trajectory snapshotのuniform・`u=0` 3608 epoch。計時用の`steady_clock`呼出しがwhole時間へ混ざらないよう、主比較はmicrotimerなしのbinaryで独立に2回実行した。各tolerance・各runでcold/warmを同一入力・同一compiler flags・CPU0 affinityで交互順に測定。表の分位は2 runをまとめた各tolerance 7216行のnearest-rank値、時間和差は同一行のpaired比をまとめて算出した。coldはepochごとにD14/topologyを新規作成。warmはprepared L2 D14 rootsを利用し、topology reuseは無効。各runの最初のwarm rowはseedなしから始まる。

CPUはIntel Xeon Gold 6530。compilerはGCC 11.5.0。再現用scriptの`-O3 -march=native -ffp-contract=fast`等は[`run_trajectory.sh`](../../evidence/holonomic/guarded_same_node_hermite_phase71/run_trajectory.sh)に固定した。

| RelTol | path | p50 ms (off → shadow) | p90 | p95 | p99 | max | 時間和のpaired差 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1e-3 | cold | 0.3439 → 0.3516 | 0.6118 → 0.6241 | 0.7137 → 0.7275 | 2.9231 → 2.9339 | 54.80 → 51.63 | +1.01% |
| 1e-3 | warm | 0.2362 → 0.2456 | 0.5079 → 0.5189 | 0.6009 → 0.6080 | 0.8953 → 0.9120 | 51.52 → 51.58 | +1.58% |
| 1e-3 | radial-only | 0.0811 → 0.0890 | 0.1445 → 0.1612 | 0.1710 → 0.1910 | 0.2540 → 0.2755 | 0.457 → 2.629 | +10.66% |
| 1e-4 | cold | 0.3707 → 0.3807 | 0.6494 → 0.6654 | 0.7666 → 0.7837 | 2.9334 → 2.9494 | 51.75 → 51.70 | +1.54% |
| 1e-4 | warm | 0.2774 → 0.2866 | 0.5525 → 0.5628 | 0.6622 → 0.6726 | 1.0158 → 1.0403 | 51.56 → 51.68 | +1.55% |
| 1e-4 | radial-only | 0.1051 → 0.1141 | 0.1917 → 0.2064 | 0.2361 → 0.2567 | 0.3536 → 0.3685 | 0.770 → 0.854 | +8.56% |

各tolerance・各modeの7216行でbaseline/shadowとも全件value converged。mu/status/unique-node mismatchesは全て0。2 runの時間和paired差は全mode・両toleranceで正。D14/topologyを含むcold/warm wholeは約1.01–1.58%遅く、radial-onlyでは約8.56–10.66%遅い。rep2のradial-onlyに1行だけ `case 119 / bin 2 / epoch 0 / RelTol=1e-3` のshadow時間2.629 ms（baseline 0.0858 ms）がある。同じrowのfull coldはbaseline→shadowで0.332→0.330 ms、status・node数も同じで、現象はradial-only timerに限られ原因未確定。p99では0.2540→0.2755 msで、単発maxを外しても遅い。maxはそのまま保存し、外れ値を除いて主張しない。radial-onlyは各rowの`classify_cells`をtimer外で実行し、`flux_adaptive_integrate`だけを計時する。各armでAdaptiveWorkspaceのcapacityはtrajectory間で再利用する。

このshadow計測はlevel-3の全対象nodeでR-jetを即時計算する診断経路であり、通常のvalue-detail判定後に必要panelだけへjetを遅延生成する将来実装の速度予測ではない。従って上のslowdownはこのeager shadow経路の実測値として扱う。一方、panel監査では現在の7点判定で未解像かつ追加8点を実際に省ける候補が0件だったため、遅延生成へ変えても本corpus上のnode節約根拠は得られていない。GuardedEstimatedを実装・計時する条件を満たさず、そのA/Bは行っていない。

R-jetとpanelモデルの内訳は別のmicrotiming binaryを使った1 runで測った。whole-epoch欄の計時値はこのrunから採らず、jet/modelの内訳だけを診断値として使う。

| RelTol | path | jet計測合計 ms / 3608 epoch | H3/H7 model ms / 3608 epoch |
|---|---|---:|---:|
| 1e-3 | cold | 33.51 | 0.845 |
| 1e-3 | warm | 32.88 | 0.812 |
| 1e-3 | radial-only | 32.30 | 0.789 |
| 1e-4 | cold | 32.83 | 0.812 |
| 1e-4 | warm | 32.78 | 0.807 |
| 1e-4 | radial-only | 32.24 | 0.795 |

jet計測には`boundary_quartic_dR`、同じroot-pairからのIFT、panel座標chain ruleを含める。model計測はH3/H7 weighted sumと候補error model。microtimingのwhole時間はtimer呼出し自体に影響されるため採用しない。shadowでは各run/tolerance/modeにつき140490 jet要求、134389成功jet、20509 panel候補を記録する。主比較は2 runなので各値は2倍となる。shadowで追加8点を省けたepochはなく、whole-epoch上の節約は0 nodeだった。

## Reference・Jacobian control・テスト

- 既存value reference checker: 550行（110 case × 5 tolerances）、reference usable 466行、observed violation 0。1e-3と1e-4は各110行中94 usable・0 violation。既存statusはConverged 496、TopologyUnresolved 50、InnerAccuracyLimited 4で、未収束statusを成功扱いにはしていない。
- 既存adaptive Jacobian finite-difference control: 120行を保存。shadow request条件がuniform value-only限定でLD/gradient経路に入らないことを確認するcontrolであり、新しいJacobian精度主張には使わない。
- `test_adaptive_radial`: 1535 checks / 0 failures。
- CTest `holonomic_adaptive_radial`, `holonomic_m7_reference`: 2/2 pass。
- `git diff --check`: pass。

## 採否・未実装

同一node R-jetとHermite kernelは、今後の数値調査用shadowとして残す。Shadow defaultはfalse。GuardedEstimated/Conservative value replacementは実装していない。根拠は、(1)既存guardを満たして新たに省けるnodeが0、(2)高増光反例でH7誤差が大きい、(3)R-jet入力誤差を含む対応ledgerを作れておらず、panelのH3/H7 modelは厳密boundではない、(4)whole-epoch shadowが費用込みで遅い、の4点。H7へのvalue置換で精度・statusがどう変わるかはwhole epochでは未検証であり、局所の精度改善を本線の精度主張へ外挿しない。

`trajectory_all_profiles_diagnostic.tsv`は初回の広いprofile診断でLD行を含み、主比較には使わない。`trajectory_attempt0_not_used.tsv`は初期harnessのrequest flag不整合を含むため未使用のまま保存した。主比較はmicrotimerなしで再生成したuniform-only raw `trajectory.tsv` と `trajectory_rep2.tsv`、別計測の`trajectory_microtimed.tsv`、panel診断`panels.tsv`を使用する。

## 再現

```bash
cmake --build build-holonomic-m7 --target test_adaptive_radial check_adaptive_reference check_adaptive_jacobian -j2
ctest --test-dir build-holonomic-m7 --output-on-failure -R 'holonomic_(adaptive_radial|m7_reference)'
./build-holonomic-m7/check_adaptive_reference evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv > evidence/holonomic/guarded_same_node_hermite_phase71/reference_cold.csv
taskset -c 0 ./build-holonomic-m7/check_adaptive_jacobian > evidence/holonomic/guarded_same_node_hermite_phase71/adaptive_jacobian_reference.csv
bash evidence/holonomic/guarded_same_node_hermite_phase71/run_panel_audit.sh
bash evidence/holonomic/guarded_same_node_hermite_phase71/run_trajectory.sh evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv evidence/holonomic/guarded_same_node_hermite_phase71/trajectory.tsv
bash evidence/holonomic/guarded_same_node_hermite_phase71/run_trajectory.sh evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv evidence/holonomic/guarded_same_node_hermite_phase71/trajectory_rep2.tsv
HERMITE_MICROTIMING=1 bash evidence/holonomic/guarded_same_node_hermite_phase71/run_trajectory.sh evidence/holonomic/v2_adaptive_best_trajectory_20260913/input_snapshot.tsv evidence/holonomic/guarded_same_node_hermite_phase71/trajectory_microtimed.tsv
python3 evidence/holonomic/guarded_same_node_hermite_phase71/summarize.py
```

Raw inputs/results and summary: [`evidence/holonomic/guarded_same_node_hermite_phase71/`](../../evidence/holonomic/guarded_same_node_hermite_phase71/). Both untimed primary passes, the separate microtiming pass, `panels.tsv`, reference CSVs, exact commands, compile flags, logs, summary JSON, `provenance.txt`, and `SHA256SUMS` are included. The evidence directory checksum manifest validates with `sha256sum -c SHA256SUMS`.
