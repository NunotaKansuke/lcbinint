# Phase 73: gradient誤差のcell別帰属

実施日: 2026-09-14  
基準: `dev/holonomic` @ `f928e3b42d25635195c7fba985cb72b811c383f3`  
判定: **対象例では、gradientの主誤差源は固定Rでの局所微分評価ではない。caustic-cross `a` は未mapのprojective fold端点、rand035 `X/Y` は主にcell 3のradial積分誤差と帰属できる。`dvK+v dK` は大きく相殺するcellがあるが、点ごとの代数恒等式はbinary64丸め程度で閉じる。production変更なし。**

## 範囲と手順

Phase72の結論を対象caseで再点検し、whole-gradient差を (1) 同一radial nodeにおける微分評価差、(2) node列と独立radial積分の差、(3) cell間・arc間の相殺、(4) LDの `dvK` と `v dK` の相殺に分けた。

ケースはuniform caustic-cross row 17, `u=0`, `a` と、rand035 row 99, `u=0`, `X/Y`、row 100, `u=0.5`, `X/Y`。独立固定R参照はbinary128のlens方程式微分、quartic実根分離、endpoint公式、直接angular微分から作り、Phase72の同じQ8 Fejer-II nodeでも評価した。radial参照は各cellを別々にGauss–Legendre積分し、angular側も直接評価した。whole FDはPhase72のcold topology再生成referenceを照合に使う。

この診断はshadow専用で、production solver、gradient ledger/tolerance、quality gate、scheduler、router/defaultを変更していない。same-node Hermiteも使っていない。Phase72のrand035 production sumはcell 11で不完全なので、valid-onlyの部分和として扱い、欠けたnodeを補完していない。

同一node監査は16,830行すべてQF評価に成功し、angular 128/256の固定R微分値とQ8加重cell和は保存したbinary64桁で一致した。これはangular積分解像度の経験的クロスチェックであり、radial誤差やproductionのcell 11 statusを保証しない。

## 主な数値結果

### caustic-cross, uniform `a`

production Q8は `-2.228480422662`、独立 whole FDは `-2.404689164237`（FDの観測spread `8.26e-8`）で、差は `+0.176208742`。同じproduction nodeでの独立QF固定R微分との差はwholeで約 `1.57e-8` に留まるため、固定R微分評価が0.176の主因ではない。

| cell | production Q8 | 独立固定R + radial GL | production − reference |
|---:|---:|---:|---:|
| 5 | 44.4945653244 | 44.5863026275 | -0.0917373031 |
| 7 | -62.2411689837 | -62.5091150453 | +0.2679460616 |
| 12 | -0.8614222799 | -0.8614222734 | +9.21e-9 |

cell 7の2 arcはどちらも約 `-31.12058449` で同符号。主な相殺はarc間ではなくcell間で、Q8 `sum(abs(cell contribution))/abs(net) = 55.63`。cell 7誤差はnet差の約152%で、cell 5が約52%を逆向きに相殺している。

cell 5と7の左端は `chart_p4` 半径で、元のcell planはそこにfold mapを付けていない（cell 5: left/right fold mapなし、cell 7: leftなし/rightあり）。QFで `R` を `p4(R)=0` に補正すると、両方とも `theta=pi` で `phi≈4.8e-33`, `phi_theta≈0`, `phi_R≈+204.93/-200.39`, `phi_thetatheta≈-13.39/+8.53` となり、通常の実接触であることを確認した。event rawではこの同じ半径に `chart_p4` と `physical_complex` が重なり、adaptive fold-map gateが使う `physical_real` eventとしては扱われていない。有限tのquartic stationary-root probeが `p4=0` のprojective root `t=infinity` を表現できないことが原因候補として強く支持される。これはsource-levelのprojective handling欠落を示す診断であり、production修正は今回の範囲外。

独立radial評価で左endpoint mapを診断時だけcell 5/7に強制すると、全cell合計はN=128で `-2.40468917646`、N=256で `-2.40468919125`。whole FDとの差はおよそ `1e-8`台。mapなしでは全cell合計がN=128/256/512/1024で `-2.14867/-2.27643/-2.34050/-2.37258` と遅くしか収束しない。mapありのN=512/1024はbinary64 `R` nodeが端点で区別できなくなる影響とみられるdriftが出るため、単調な改善とはみなさない。したがってPhase72の「fold特異性は主要因でない」という全般的な解釈は、このcaseについては修正が必要である。moving-mapの局所相殺式は成立しているが、主要な2つのprojective foldが既存mapに入っていなかった。

### rand035, `X/Y`

同じQ8 node上のproductionと独立QF固定R微分のwhole valid-only差は、row99 `X/Y`: `-4.82e-9/-1.02e-8`、row100 `X/Y`: `-2.89e-9/-6.11e-9`。角度積分128/256の出力も表示桁で一致する。したがって観測される数e-7のproduction–FD差は局所微分kernelよりradial node列/積分側にある。

| row, u, parameter | production valid-only − FD | cell 3 production − GL reference | cell 3差 / production−FD全体差 |
|---|---:|---:|---:|
| 99, 0, X | +3.1510e-7 | +3.1052e-7 | 約98.5% |
| 99, 0, Y | +6.6590e-7 | +6.5612e-7 | 約98.5% |
| 100, 0.5, X | +1.8910e-7 | +1.8631e-7 | 約98.5% |
| 100, 0.5, Y | +3.9956e-7 | +3.9367e-7 | 約98.5% |

表の割合は `cell 3のproduction−GL差 / whole production−FD差`。別に、選択したQF radial cell referenceとの差を符号付きで合計するとcell 3はnet差の約105%で、他cellの逆符号差が一部相殺する。cell 3は絶対cell誤差総和の約91.1%でもある。whole FDとQF radial referenceの差は約1e-8–4e-8なので、割合は厳密な誤差分解ではなく帰属の目安である。独立直接QF radial系列のcell 3は112x8, 128x16, 160x32でおおむね1e-9以内に安定する。cell 2の160x32は非単調な約3e-7外れ値となるためreferenceに採用せず、128x16を使った。これは参照側も形式誤差boundではなく、binary64 `R` nodeとfold endpoint近傍の分解能に制限があることを示す。

rand035のcell間相殺は小さく、`sum(abs(cell contribution))/abs(net)` はrow99で約1.052、row100で約1.033。1 arc/cellのactive cellが多く、cell 3/8ではradial integrand内部のabsolute/net比が約2.3–3.5。

### LDの `d(vK)=dvK+v dK`

row100, `u=0.5`, Q8のproduction node上で `dvK` と `v dK` を別々に集計した。各nodeの和と既存production K-Jacobianの最大差は `1.1e-14` 以下。

| parameter | cell | radial `dvK` | radial `v dK` | LD total | `(abs(dvK)+abs(vdK))/abs(total)` |
|---|---:|---:|---:|---:|---:|
| X | 3 | +12.95327 | -9.59393 | +3.35934 | 6.71 |
| X | 8 | -22.49230 | +25.83866 | +3.34636 | 14.44 |
| Y | 3 | +7.17913 | -0.08102 | +7.09811 | 1.02 |
| Y | 8 | +29.55876 | -22.48806 | +7.07070 | 7.36 |

この相殺は丸め誤差を増幅し得るので注視対象ではあるが、今回のnodewise恒等式残差は丸め程度であり、誤った微分式の証拠はない。cell 11は48/255 source node invalid、残りにも207 K-rule rejectがあるためLD分解を完了していない。独立固定R referenceのwhole-cell寄与はこのcaseでは約 `1e-17` だが、これでproductionのcell 11 statusが認証されたことにはならない。

## 判断と次に切り分ける場所

1. caustic `a` の `0.176`差は局所微分式でなくcell 5/7のradial integration。直接原因として、`chart_p4` が表す `theta=pi` の実接触がfold mapに入っていないことを確認した。projective fold eventをphysical eventとして扱える方法の検討が次に必要。
2. rand035の残差は主にcell 3 radial integration。fixed-R derivative、angular rule、dvK/vdKの代数恒等式は今回の差の主因でない。gradient向けh-splitやscheduler変更はまだ評価していない。
3. LDではcell単位の `dvK`/`v dK` cancellationが強い場合があるが、pointwise sumは整合する。次に実装を変える前に、cell 3のproduction Fejer Q8 detailと直接QF GL系列を同じradial parameterization上で比較するのが最小の追試。
4. rand035 cell 11はproduction incomplete。全体の数値誤差を議論する際に、このvalid-only値をcomplete statusの証拠として使わない。

今回の測定は対象5 gradientだけであり、全parameter/corpusへの一般化はしていない。GL系列とFD spreadはいずれも観測収束であって形式的誤差上界ではない。

## 再現

repo rootで以下を実行する。

```bash
bash evidence/holonomic/gradient_cell_attribution_phase73/run_phase73.sh
```

GCC 11.5.0、`-O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno`、CPU affinity `taskset -c 0`。入力・compile flags・実行系列はevidence内の`run_phase73.sh`と`provenance.txt`に記録する。参照mainを診断translation unitへincludeして改名するため `control reaches end of non-void function` 警告が出るが、生成binaryは正常終了しrawを生成する。

主なrawは `evidence/holonomic/gradient_cell_attribution_phase73/` 内の `cell_ranking.tsv`, `same_node_nodes.tsv`, `same_node_cells.tsv`, `same_node_arcs.tsv`, `endpoint_audit.tsv`, `caustic_unregularized_direct.tsv`, `caustic_fold_map_diagnostic.tsv`, `rand_direct.tsv`, `rand_cell{2,3,8}_direct_high.tsv`, `ld_nodes.tsv`, `ld_cells.tsv`, `summary.json`。前段の固定interval FD試作 `cell_reference.tsv` は境界/sliver収束問題があったため、主referenceにもrankingにも使っていない。

## 次工程の提案

新しい誤差推定器を先に作らず、(a) `chart_p4` root at infinityでの `P=P_t=0` projective fold seed/physical classification、(b) rand035 cell 3でradial Fejer-Q8と独立GLのdetail列、を別実験として追う。前者はcausticの0.176差に対する直接原因、後者はrand035の数e-7差に対する主要寄与と今回の証拠が示している。
