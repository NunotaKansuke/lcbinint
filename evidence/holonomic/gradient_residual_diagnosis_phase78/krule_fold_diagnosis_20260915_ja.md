# caustic-cross cell 6 におけるK-ruleと微分精度の切り分け

## 結論

K-rule自体が使えないわけではない。今回のLD caustic-crossでは、標準16点K-ruleがfoldへ近づくにつれて必要な内部解像度を失い、8/16 value gateが正しくrejectしていた。その後の64点angular rescueも同じ内部接触を十分解像できず、微分誤差を残す。

一方、同じroot pair上でK-rule次数を上げると、K値・解析微分とも独立角度積分へ収束した。cell 6全体では、同次数のangular rescueより高次K-ruleの方が短時間で同程度の積分結果を得た。したがって「K-ruleをやめる」必要はなく、**fold近傍だけK次数を局所的に上げるshadow A/Bを次に試す価値がある**。ただし16点固定への復帰やquality gate緩和の根拠にはならない。

## 対象と方法

- case: `reference_cases.tsv` のrow 17 `caustic-cross`。
- parameter: `xs=.12, ys=0, rho=.02, q=.5, a=1`, `u=1`、非barycentric。
- 対象: right endpointがprojective physical foldであるcell 6、`R=[0.84972672439230301, 0.85720378575239498]`。
- 同じradial map、同じ192×8 radial nodes、同じboundary root pairを使った。
- K-ruleは8〜2048点、analytic 5Jacを全成分評価し、表示・積分比較はX成分。angular sideも同じ5成分を計算する。
- 独立角度参照はQFのdirect angular integration。cell積分の高解像度比較は2048点angular rule。
- K-onlyとangular-onlyのcell積分タイミングにはroot/endpoint処理を含む。D14/topology生成はshared setupとして両側のcell timing外。

## rejectの直接理由と数値上の原因

baseline cell 6では1536 radial nodesがすべてK候補を試し、K成功0、reject 1536。reject理由は全件`8-vs-16 disagreement`で、`vfloor`、`TMax`、`S2 invalid`、`B invalid`ではなかった。従ってゲートは単なる保守的誤判定ではない。

近接fold nodeではreciprocal chartのpair中心がほぼ0、`v≈0.21937`で、`max |u|≈0.46836`。不安定さはchart端やroot pair degeneracyではなく、reciprocal coordinateの中央`u≈0`（projective angle `theta≈pi`）で、deflated smooth factor `S2`が非常に小さくなることに現れる。サンプルした最小`S2`はradial map `x=.75`で約`1.08e-4`、`x=.90`で`1.73e-5`、`.98`で`7.03e-7`、`.9999`で約`1.6e-7`まで下がった。

この小さい正の内部minimumにより、K integrandの`sqrt(S2)`とそのparameter derivativeが狭い内部featureを持つ。radial foldへ近づくほど16点と低次数ruleの差が増え、value gateが拒否する。`HOLO_K_GATE=t`の16/24 shadow gateは、x=-.5のような比較的smoothなnodeを救うが、x>=0のfold側sampleはrejectのままだった。Phase 78の集計では2121 attempts中1170 success、951 disagreement rejectまで増えた一方、whole derivativeの変化は約`1.2e-6`だけだった。

## 同一radial meshでのcell 6積分

符号付きcell contribution `dmu/dX`。高次angular 2048点をradial-discretized referenceとする。

| rule | cell contribution | abs diff vs angular 2048 | cell time (ms) |
|---|---:|---:|---:|
| K 16 | -174.574323 | 2.637286 | 5.40 |
| K 64 | -177.031639 | 0.179969 | 6.96 |
| K 128 | -177.166038 | 0.045571 | 8.97 |
| K 192 | -177.191338 | 0.020271 | 11.11 |
| K 256 | -177.200246 | 0.011363 | 13.36 |
| K 512 | -177.208876 | 0.002733 | 21.62 |
| K 1024 | -177.211043 | 0.000566 | 38.13 |
| K 2048 | -177.211586 | 0.0000226 | 71.18 |
| angular 64 | -177.049319 | 0.162290 | 7.65 |
| angular 128 | -177.171148 | 0.040461 | 10.59 |
| angular 256 | -177.201612 | 0.009997 | 16.47 |
| angular 512 | -177.209229 | 0.002380 | 28.67 |
| angular 1024 | -177.211133 | 0.000476 | 57.61 |
| angular 2048 | -177.211609 | reference | 99.65 |

このcellだけを同じbaseline whole value `2.14324046796`へ差し替える推定では、K128で`2.0265214`、K256で`1.9923133`、K512で`1.9836836`、K2048で`1.9809733`となる。独立FDは約`1.9807892`、高解像度radial/angular列は`1.9809508`であり、FDとの差の一部はradial/angular reference側にも残る。

高次数K-ruleは同じ次数のangular rescueより速かった。例えばK256は13.36 ms対angular256の16.47 msで積分値も近く、K512は21.62 ms対28.67 ms、K1024は38.13 ms対57.61 msだった。この比較は**cell 6の固定radial mesh上だけ**であり、whole epochの勝利を示さない。K128はangular128より似た精度で約15%短い。

node単位では、K値はvalue derivativeより先に収束する場合もあるが、fold側では双方とも低次数で不十分だった。x=.90ではK値が16→24で約`2.1e-3`変化し、Kの解析dXは16→24で約`9.7e-2`変化する。x=.99ではdXの8→16差が約24%、16→24差が約13%で、2048点まで上げても角度側direct derivativeの点wise積分も収束が遅い。x=.999〜.9999のpointwise角度参照は8192点でも収束確認が不十分なので、そこでKの絶対微分誤差を断定しない。cell積分はより安定して同次数angular列に一致した。

## 判断

1. 現行8/16 gateは緩めない。rejectは数値的に実在するquadrature不足を捉えている。
2. 今のfallbackが64点angularで終わるため、K拒否後にgradient誤差が残る。これは「Kが使えない」ことと別問題。
3. 高次K-ruleは同じcell accuracyをより少ない内部node/costで得られる候補。ただし全radial nodeを高次数にするのは無駄なので、次の候補は、同一stateのまま16→24/32→64→128…と上げるshadow routeと、value/gradient別の局所差分ledger。
4. estimatorの独立性、他parameterの5Jac精度、adaptive radial whole-epochの費用/statusは未検証。production gate/router/defaultは変更していない。

## 再現

worktree HEAD `cd1343cc2bd7fc8bfde17c5556772ed6420f14b4`、CPU core 0固定。diagnostic sourceは`direct_radial_gradient_ld_x.cpp`。

```bash
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc \
  -DHOLO_D14_NATIVE_RESIDUAL_SCREEN -DHOLO_D14_DIRECT_CONVOLUTION \
  -DHOLO_ADAPTIVE_STATIONARY_ENDPOINT -DHOLO_D14_WARM_NOISE_HANDOFF \
  -DHOLO_D14_SCALAR_CONSTANTS -DHOLO_ARC_COMPACT_STORAGE \
  -DHOLO_ADAPTIVE_SPARSE_ANCHOR -DHOLO_ADAPTIVE_SAMPLE_GROWTH \
  -DHOLO_MV_RATIONAL_MIDPOINT -DHOLO_QUARTIC_WARM_SOLVE_TOL=1e-11 \
  -DHOLO_ADAPTIVE_FOLD_QUARTIC_SEED \
  -DHOLO_ADAPTIVE_PROJECTIVE_P4_FOLD_RESEARCH \
  -march=native -funroll-loops -ffp-contract=fast -fno-math-errno \
  evidence/holonomic/gradient_residual_diagnosis_phase78/direct_radial_gradient_ld_x.cpp \
  -lquadmath -o /tmp/phase78_cell6_krule_diag

env -u HOLO_K_GATE taskset -c 0 /tmp/phase78_cell6_krule_diag \
  evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv \
  evidence/holonomic/gradient_residual_diagnosis_phase78/krule_node_audit_alljac.tsv
```

Raw outputs: `krule_node_audit_alljac.tsv.cell6_krule.tsv`, `.cell6_krule_integral.tsv`, `.cell6_integral.tsv`, `.cell6_angular.tsv`, and `.cells.tsv`.
