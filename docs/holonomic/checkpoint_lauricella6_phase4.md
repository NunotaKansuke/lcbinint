# Phase 4 checkpoint: physical Lauricella-6 -> 5 transport

基準は `dev/holonomic @ d44be03ececbb4f8e5777759431f8571ac62ffff`。実装と
ベンチは `src/lcbinint/magnification/holonomic` と
`tests/holonomic_cpp` に隔離し、V2/V3 production routerは変更していない。

添付 `LAURICELLA6_IMPLEMENTATION_PLAN_JA.md` は今回の数式・実装方針の設計
authorityとして読んだ。独立probeは `/tmp/claude-560/.../scratchpad/` の仮説
検証として扱い、probeの結果をwhole-epochの受理根拠にはしていない。既存の
`bench_gm_coverage` と独立angular積分が、数値比較のreferenceである。

## 実装

新しい `gm_lauricella6.hpp` は、物理 half-fluxを六因子Euler積分へ変換し、
`+i` の A因子をMöbius変換で除いて、

```
Z = (F, M1, ..., M5)
beta = (-1/2, -1/2, 3/2, 1/2, 1/2)
M_i = F + (xi_i - 1) / beta_i * dF/dxi_i
```

の6複素成分を運ぶ。Taylor係数は5個のcross-ratio、20個のlogarithmic
derivative、明示的Pfaffian作用から再帰生成する。15x15 Hermite再還元、巨大な
scalar Picard--Fuchs展開、9点K-rule/Chebyshev fit packetは本線に使っていない。

境界二次式の根はimplicit polynomial Taylorで生成し、`xi_i`、
`xi_i-1`、`xi_i-xi_j` のlog derivativeも同じgeometry jetから作る。ひとつの
受理Taylor packetを複数のGauss--Chebyshev radial nodeへdense outputし、tail
gateを超えると受理stateから再中心化する。nodeごとのconnection再構築と
physical re-seedは発生しない。現行の最大再中心化深さは16で、resonant/closeの
`R_eq_a` 近傍で深さ12では落ちた区間を通せることを確認した。

physical seedだけは実際のlens arcのEuler積分から一度作る。これは初期periodの
生成であり、係数のsample-and-fitではない。seedのdouble gateに失敗したときだけ
long double geometry/seedへ昇格するprecision ladderを入れた。108ケースの今回の
runではpromotion 0、precision failure 0だった。

5パラメータ微分は同じ6状態から

```
partial_p J = (partial_p C) F + C sum_i (partial_i F)(partial_p xi_i)
```

で取り出す。endpointはimplicit-function derivative、geometryは`Dual5`で計算し、
5本の感度ODEは追加していない。ベンチでは値のwhole wall差と、内部の
`analytic_jacobian_ms`を別々に記録する。

chart認証は、有限`t`区間、arc measure、内部3点の`phi>0`、physical branchを
確認する。反射chartの`t=tan((theta-pi)/2)` の無限遠点が元の`theta=0`であることを
修正し、`chart_p4`の両側で有限chartを選べることをテストした。foldには
`F=1, M_i=1/2` と

```
K(0) = pi sqrt(S) / (rho A^(3/2) sqrt(B))
```

のregular germ helperを実装し、physical-real eventから検査している。radial
Gauss--Chebyshev nodeは開点なので、現行epochはこのhelperをnode seedとしては
まだ使っていない。`u=sqrt(v)` または `u=sqrt(|R-R_fold|)` の幅全体を運ぶfold
germ packetと、一般的な動的Möbius chart切替は未統合である。現在の近接fold対応は
Taylor区間分割とfail-closed gateである。

## Phase 3独立監査

次のコマンドで、旧7D true GMのPhase 3を独立angular/physical reference付きで
再実行した。

```
cmake --build build-holonomic-global --target bench_gm_coverage bench_gm_angular_jac -j4
./build-holonomic-global/bench_gm_coverage \
  evidence/holonomic/gm_coverage_cases.tsv 64 jac \
  > /tmp/gm_phase3_recheck_jac.txt
```

結果は、54 limb-darkening casesの全 **13,952/13,952 node** がtransportされ、
独立value gateも13,952/13,952。解析5Jacの独立physical re-seed gateは
13,691/13,952、max physical re-seed errorは`5.741e-05`だった。case-levelの
`all_true_accurate`は46/54で、`JACFAIL`の難例は削除せず記録した。代表的な
再検証結果は [gm_coverage64_jac_refined_phase3.txt](../../evidence/holonomic/gm_coverage64_jac_refined_phase3.txt)
に残している。

独立angular Jacobian probeも次で512/2048点を比較した。

```
./build-holonomic-global/bench_gm_angular_jac \
  evidence/holonomic/gm_coverage_cases.tsv 512 \
  > /tmp/gm_phase4_angular_jac512.csv
./build-holonomic-global/bench_gm_angular_jac \
  evidence/holonomic/gm_coverage_cases.tsv 2048 \
  > /tmp/gm_phase4_angular_jac2048.csv
```

既存の [gm_angular_jac512_phase3.csv](../../evidence/holonomic/gm_angular_jac512_phase3.csv)
と [gm_angular_jac2048_phase3.csv](../../evidence/holonomic/gm_angular_jac2048_phase3.csv)
に一致した。resonantのJac mismatchは`9.407e-05`、caustic-crossのvalidな最大値は
約`1.912e-03`で、これはV2との不一致をそのまま新6Dの誤差とはみなさない。
独立reference自身の収束とtopology gateを含めた難例Jacの全面的なPF6監査は未完了である。

## PF6 whole-epoch実測

```
cmake --build build-holonomic-global --target \
  test_gm_lauricella6 bench_gm_lauricella6 -j4
./build-holonomic-global/test_gm_lauricella6
./build-holonomic-global/bench_gm_lauricella6 \
  evidence/holonomic/gm_coverage_cases.tsv 64 0 512 \
  > /tmp/gm6_phase4_final2.txt
```

Release `-O3 -march=native -funroll-loops -ffp-contract=fast`、単一processの
同一入力で測った。V2側はtransport overrideを無効にし、`epoch_value` /
`epoch_jacobian`を同じ`n_r=64`で測った。`angular_n=512`は最初の12行だけの
独立whole-epoch value referenceであり、全108行の速度比較には使っていない。

```
TOTAL cases=108 uniform=54 limb=54
  cold_true=104 warm_true=104
  cold_jac_ok=52 warm_jac_ok=52
  nodes(c/w/jc/jw)=13888/13888/13888/13888
  transported(c/w/jc/jw)=13888/13888/13888/13888
  rates(c/w/jc/jw)=1/1/1/1
median value-only LD ms   PF6 cold/warm/V2 = 9.196 / 9.059 / 1.212
median value+5Jac LD ms    PF6 cold/warm/V2 = 9.350 / 9.246 / 1.304
```

cleanなLD側は52/54 casesで値・Jacともstatus OKかつ全node true transport。
`rand008` と `rand031` はV2と同じ`TOPOLOGY_UNCERTAIN`であり、ここをPF6の成功に
数えていない。試行されたPF6 LD nodeは全てtransportされたが、Phase 3の13,952
nodeとPF6の13,888 attempted nodeは、topology uncertain cellで採用されたcell
planが異なるため同一ではない。この差をcoverage改善と呼ばず、Phase 3の13,952
node監査を別referenceとして保持する。

PF6の集計は次の通り。

```
COUNTS(value) arcs=217 seeds=217 connections=8030
  fallback=0 reseed=0 promotions=0 precision_fail=0
COUNTS(jac)   arcs=217 seeds=217 connections=8030
  fallback=0 reseed=0 promotions=0 precision_fail=0
jacobian_marginal_ms(c/w)=0.231/0.187
analytic_jacobian_cost_ms(c/w)=0.174/0.174
COST_MEDIAN(value_cold) topology/D14=0.861 arc=0.013 geometry=0.001
  seed=0.170 connection=7.034 transport=0.070 analytic_jac=0 fallback=0 ms
COST_MEDIAN(value_warm) topology/D14=0.729 arc=0.013 geometry=0.001
  seed=0.168 connection=7.031 transport=0.069 analytic_jac=0 fallback=0 ms
COST_MEDIAN(jac_cold) topology/D14=0.889 arc=0.013 geometry=0.001
  seed=0.168 connection=7.000 transport=0.069 analytic_jac=0.174 fallback=0 ms
COST_MEDIAN(jac_warm) topology/D14=0.722 arc=0.013 geometry=0.001
  seed=0.168 connection=7.014 transport=0.069 analytic_jac=0.174 fallback=0 ms
```

accepted LD valueのV2との差はcold/warmとも最大`2.041e-08`、first-12独立
angular valueとの差は最大`1.716e-09`だった。plan15、resonant、closeのlocal
seedは独立angular 2048点にそれぞれ`2.7e-16`、`5.5e-16`、`2.6e-16`のrelative
差で一致し、local analytic chain ruleのangular finite-differenceとの差は最大
`1.7e-09`だった。plan15 end-to-endではvalue/Jac value差`5.442e-10`、gradient差
`2.880e-09`だった。

warmで再利用したのは前回epochのD14 full root setだけである。cell/chart/period
stateは、cycle correspondenceと非特異経路の認証がないため再利用していない。
したがって今回のwarm結果はD14-only warmであり、steady-stateのstate reuse勝利を
示さない。全体時間はPF6がV2を上回っておらず、router採用条件は未達である。

## テストと未達事項

`test_gm_lauricella6` は次を短い固定ケースで検査する。

- physical FD6 seedと独立angular integral
- equation-derived Taylor dense outputとlocal physical re-seed
- 6状態からの5方向chain-rule Jacobian
- physical fold germ
- `chart_p4`両側のchart/branch certificate
- plan15のcold whole epoch value/value+5Jac、V2 parity、fallback 0

実行結果は `gm_lauricella6 PASS failures=0`。既存isolated CTest群も再実行対象に
含める。新方式は研究用entry pointのままで、V2/V3 production routerへの変更は
行っていない。

数値の短い機械可読サマリーは
[gm_lauricella6_phase4_summary.txt](../../evidence/holonomic/gm_lauricella6_phase4_summary.txt)
にも保存した。

未達または未検証の項目は以下である。

- PF6のwhole-epoch cold/warm速度勝利。現状はV2より遅い。
- `u=sqrt(v)` のfold germ packetによる折り目近傍の全面coverage。
- topology uncertainty 2ケースを含めたPhase 3全13,952 nodeのPF6再現。
- 108ケース全体のPF6解析5Jacを独立angular/re-seed referenceで受理すること。
- period-state warm reuseと、一般的なdynamic chart切替の安全な認証。
- interval/ball arithmeticを含む厳密な全体誤差証明。
