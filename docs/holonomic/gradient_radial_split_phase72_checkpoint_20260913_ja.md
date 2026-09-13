# Phase 72: gradient radial refinement の局所h-split再評価

日付: 2026-09-13

基準: `dev/holonomic` commit `64bb1e3dd88d66be6a98eb9d77c20da6c45733e9`
対象: experimental adaptive gradient controller。production router / fixed-`n_r` API / PF6-GM経路は変更なし。

## 結論

Phase 66–69で見つかったgradient radial errorに対し、既存の `gradient_local_refinement` を現HEADで再評価した。主候補は、level 3→4までは従来どおりp-refineし、その後もgradient detailが減衰せず、radial errorが局所budgetを支配し、inner/event/geometry/roundoffが律速でないpanelだけをh-splitする設定 (`gradient_split_min_level=4`)。

これは難例の局所実験では効果があった。64-roundのcaustic例では、uniformの `a` が2,650→577 node、観測誤差が0.515%→0.0881%に、LDの `X` が2,651→586 node、0.363%→0.0501%になった。valueは両方ともbitwise不変だった。

ただし110入力×2 value toleranceのpaired whole-epoch試験では、標準4-roundの `RelTol=1e-3` において `rand035` のu=0/.5の `X` と `Y` が `ToleranceMet` から `FiniteUncertified` へ各8回遷移した（2入力×2 repeat×cold/warm、各path/tolerance groupでは4件）。1e-4ではこの遷移はなかった。最新の再現では1e-4のp50もわずかに改善したが、warm p99は悪化し、分位改善は一様でない。測定repeatは2回で、速度差の小さい箇所には測定揺らぎが残る。独立参照が全5Jac・全corpusをカバーしていない。

したがって、この候補は研究flagに留める。production/defaultへ昇格しない。gradient quality/status非回帰という採用条件を満たしていないためである。`FiniteUncertified`をToleranceMetへ上げる変更、value meshの変更、追加root solveや高精度fallbackは行っていない。

Phase 71のsame-node Hermite結論を最終判断として尊重し、本PhaseではHermiteの実装・shadow・ledger・停止条件に触れていない。

## 先行Phaseの再整理

- **Phase 66:** valueが収束してもgradientは収束しないことを3つの選択微分で再確認。causticのgradientは非単調で、64-roundでも未認証だった。独立参照はGL radial 128/256と直接angular値積分の有限差分。
- **Phase 67:** h-split時に右子未評価のまま親を捨てる不完全panel bugとgradient全成分判定の初期値bugを修正。現在のA/Bはこのtransactional child commitを含むHEADで実施。
- **Phase 68:** 一律に15点で止めて分割する方法はケース依存。LD例の悪化もあり、一律capは不採用。nested-differenceをgradientへ流用しても独立参照への誤差は改善しなかった。
- **Phase 69:** value snapshotを固定し、gradientだけでpanelを選ぶcontrollerをresearch flagとして追加。radial/detail支配のpanelだけh-splitする設計だが、標準4-round corpusで一部statusが下がるため既定OFF。

Phase 72では一律15点capを使わず、現行の局所eligibility gateにthreshold 4/5を与えて、pure p-refinementと比較した。threshold 8は4-round中にsplitへ到達しないestimator-control arm。

## 難例のpanel監査

panel harnessは各選択caseでp-only、threshold 4、threshold 5を64 gradient roundsまで実行した。value `RelTol=1e-3`、gradient `atol=1e-4, rtol=1e-2`、gradient node budget 8,192。すべてValueFirst。以下の独立FDは、各parameter perturbationでtopologyを作り直し、GL radial 128/256と直接angular値積分を使った。stepは `rho×{1e-3,3e-4,1e-4}`。これは独立積分経路による観測参照で、厳密な誤差包含ではない。

| case / selected component | p-only nodes → h-split@4 → @5 | preferred FD reference | observed absolute error: p-only → @4 → @5 |
|---|---:|---:|---:|
| `caustic-cross`, u=0, uniform `a` | 2,650 → 577 → 715 | -2.40468841994 | 0.0123731 → 0.00211952 → 0.175811 |
| `caustic-cross`, u=0.5, LD `X` | 2,651 → 586 → 725 | 77.4530945247 | 0.281118 → 0.0387759 → 0.134176 |
| `rand030`, u=0, uniform `rho` | 203 → 203 → 203 | 0.0129986580935 | 3.21e-8 → 3.21e-8 → 3.21e-8 |

caustic `a`のFDはn=128と256のsmallest-step値の差が約5.23e-6。LD `X`は同差が約1.76e-4。rand030の`rho`は約3.80e-9。3例ともh-split後も `FiniteUncertified` のままであり、観測誤差の改善を品質認証へ読み替えていない。

panel別error ledgerでは、caustic uniform `a` のradial errorがp-only 1.40318、h-split@4 0.740854、@5 1.49018。innerは0、geometryは約2e-10、eventは1.28e-5、roundoffは約7e-14で、ここではradialが支配する。LD `X`ではradialが15.3269→8.90113→11.3448。innerは0.0563から0.1629へ増えたが、radialが最大のままで、geometry/event/roundoffは小さい。threshold 4の方が観測誤差・推定radial errorの両方でよかった。

一方、rand030 `rho` はradial 5.03e-5に対しevent error 4.07e-4、geometry 1.24e-7、inner 0、roundoff 1.2e-12で、`EventLocationLimited`。局所条件はh-splitしない判断をした。3例のpanel寄与には強い相殺があり、`sum(abs(panel contribution))/abs(sum(panel contribution))` はuniform `a`で約51、LD `X`で約12、rand030 `rho`で約21,538。最終net gradientのrelative errorだけでschedulerを決める危険が具体的に見える。

監査用eligible panel数は、caustic uniform `a`でp-only level 8に2、threshold 4後のlevel 4に2、threshold 5後にlevel 3/5へ計2。LD `X`でも各arm 2 panel。rand030 `rho`は各arm 0。全panelへ一律splitした結果ではない。

`rand035` u=0/.5のXも監査したが、独立FDはn=128のsmallest stepだけ有限で、n=256を含む他のperturbationが非有限になった。これは参考値に留め、候補の精度改善を主張する根拠には使わない。

## Whole-epoch paired A/B

[`corpus.tsv`](../../evidence/holonomic/gradient_radial_split_phase72/corpus.tsv)は既存110入力を、value `RelTol=1e-3,1e-4`、cold/warm、2 repeatで同一case内のarm順をrotationしながら測定した。1 armあたり880 rows。全armがValueFirstで5Jacを有効にし、既定gradient toleranceと4-round/4096-node gradient budgetを使う。whole timeはcoldでD14+topology+adaptive integration、warmでD14 root warm-start+再classification+adaptive integrationを含む。warmは同じ入力を1回unmeasuredで準備し、Topology plan自体は再利用しない。2回目は更新済みroot stateを使う。CPU 0 affinity、同一binary/compiler flags。

paired時間和とnode総数は各path/toleranceで220対を比較。threshold 4は以下のとおり。

| value RelTol | path | paired total time | node合計 | p50 ms | p90 ms | p99 ms |
|---:|---|---:|---:|---:|---:|---:|
| 1e-3 | cold | -5.935% | -9.494% | 0.503833 → 0.496245 | 1.032806 → 0.795637 | 1.737128 → 1.514707 |
| 1e-3 | warm | -7.754% | -9.494% | 0.326725 → 0.321172 | 0.798155 → 0.548305 | 1.495238 → 1.512871 |
| 1e-4 | cold | -3.370% | -4.630% | 0.514418 → 0.513363 | 1.040233 → 0.919542 | 1.846968 → 1.690335 |
| 1e-4 | warm | -4.452% | -4.630% | 0.348166 → 0.338303 | 0.875297 → 0.744729 | 1.694594 → 1.720127 |

threshold 5は時間和が1e-3 cold/warmで-5.585%/-7.127%、1e-4で-2.897%/-4.111%。node合計はそれぞれ-8.660%/-3.933%。threshold 8はnode数0%変化で、whole時間差は約-0.10%〜+0.24%の範囲。これは主な削減がh-splitに伴うnode変更から来ていることと整合する。

ただしthreshold 4でもwarm p99は両toleranceでわずかに悪化し、1e-4ではmaxも微増した。全体の速度利益は一様ではない。p95/max、threshold 5、armごとの完全な分位は[`summary.json`](../../evidence/holonomic/gradient_radial_split_phase72/summary.json)に保存した。

### value・quality・coverage

- 全armのpaired `mu` bitwise mismatchは0。value stop mismatchとvalue convergence mismatchも0。
- 既存corpusでvalueが収束するのは各tolerance/path groupで200/220 calls（100/110 distinct input rows）。`rand007`, `rand017`, `rand024`, `rand028`, `rand032`のu=0/.5計10入力はbaselineから`TopologyUnresolved`で、全arm・両tolerance・両pathで同じ。h-splitによる新規value failureもrecoveryもない。
- threshold 4/5の1e-3では、`rand035` row 99 u=0 と row 100 u=.5で、X/Y各成分がToleranceMet→FiniteUncertified。各path/groupにrepeat込み4 transitionずつ。1e-4ではquality transitionなし。新規Invalidは0で、aggregate `gradient_stop`もbaselineから変わっていないが、成分別quality非回帰条件は満たさない。
- rand035のFDが不足しているため、status降格を「実誤差悪化」と断定しない。同時に、候補を成功とみなす根拠にもならない。ledgerが示す推定誤差増加を尊重し、FiniteUncertifiedのまま保持する。

## 独立参照・検証範囲

- 難例FD: 3成分（uniform `a`, LD `X`, uniform `rho`）でstep/order変化を監査。rand035のXは限定的で診断用途のみ。
- 既存5Jac FD control: `plan15` / `paper_highA` の2条件×u×5成分を出力。120行中88 finite、32 nonfinite。これは全5成分の有限差分経路が動くことの限定確認で、110入力全てのgradient正確性を保証しない。
- 既存value reference checker: 550行中466 usable、observed violation 0、84 unusable。value referenceの非違反をgradient精度の証明には使わない。
- adaptive unit: 1,535 checks / 0 failures。
- CTest `holonomic_quartic_sturm`, `holonomic_quartic_local_bracket`, `holonomic_adaptive_radial`: 3/3 pass。

## 採否

**early h-split@4は研究候補として残し、production/defaultへは昇格しない。** 難例でnode削減と観測改善が明瞭で、今回の2-repeat whole measurementではp50とpaired totalが改善した。しかし、標準4-round whole corpusに成分quality降格があり、p99/maxにも小さな非回帰違反が残る。threshold 5はthreshold 4よりnode節約が小さく、難例によって改善幅も小さいため、分割を遅らせる優位性は確認できなかった。

次に続けるなら、まずrand035で有限な独立FDを得られるreference積分条件を作り、X/Yの実差とledger推定差を切り分ける。その証拠が出るまではscheduler閾値やgradient toleranceを緩めない。今回、Hermite系は再開しない。

## 再現

repo rootから:

```bash
bash evidence/holonomic/gradient_radial_split_phase72/run_all.sh
```

individual commands, compiler flags, CPU/host, input hashは[`provenance.txt`](../../evidence/holonomic/gradient_radial_split_phase72/provenance.txt)。raw TSV、summary、unit/CTest logsおよびSHA256 manifestは[`evidence/holonomic/gradient_radial_split_phase72/`](../../evidence/holonomic/gradient_radial_split_phase72/)。
