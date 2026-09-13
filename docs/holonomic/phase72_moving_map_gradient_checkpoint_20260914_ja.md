# Phase72: moving-map gradient integrand shadow

実施日: 2026-09-14
基準: `dev/holonomic` @ `dfe09dee62a99bbd18f3ce01246c84c6e130464b`
判定: **moving-map微分はfold端の局所特異項を消すが、今回のgradient radial積分を全体として改善しない。research shadowに留め、adaptive/productionへ採用しない。**

## 対象と変更範囲

uniform caustic-cross の `a` 微分と、rand035 の `u=0,0.5` における `X/Y` を同じ topology、同じ radial nodeで比較した。Fejér-IIのnested列は7, 15, 31, 63, 127, 255点/セル。値は置換せず、gradient ledger、tolerance、quality gate、scheduler、router/default、production solverには変更を加えていない。Hermite rule / stoppingも使用していない。

各ケースで同一のD14 topology・CellPlan・root-pair経路を使い、固定Rのanalytic derivativeを `g_old = J F_p / D`、endpoint motion込みを `g_move = (J F_p + J F_R R_p + F J_p) / D` として評価した。テストした `X/Y/a` 方向では正規化係数 `D` はparameter非依存。

物理foldの既存event radiusと `fold_t_seed` を出発点に、局所DD `P=P_t=0` correctionだけを行った。quartic係数のparameter微分から

\[
R_{*,p}=-P_p/P_R,\qquad
t_{*,p}=-(P_{tR}R_{*,p}+P_{tp})/P_{tt}
\]

を評価した。endpointから作る `R_p(x), J_p(x)` は各 `FoldRadialMap` の解析式を微分した。`F_R` はuniformでは既存root-pair/endpoint derivative、LDでは同じ16点 `vK` ruleのdirectional derivative（既存8-vs-16 gateとrescueを維持）を使い、同じ `AdaptiveSample` のquartic/root stateを再利用した。9点rule、sample-fit、Hermite、global root solveは追加していない。

固定R微分の局所確認として、同じCellPlanのcell中心で `mapped_radius` の解析 `F_p` と直接perturbed valueのcentral FDを照合した。各プラス/マイナスsampleはreliableで、`h=1e-7` の最大相対差は caustic `a`: `6.64e-8`、rand035 `X/Y`: `1.71e-7 / 7.67e-7` (`u=0`)、`1.63e-7 / 7.31e-7` (`u=0.5`) だった。これは代表点の整合確認であり、全nodeの独立証明ではない。

## fold端の相殺と境界項

ordinary foldで、そのfoldで新たに生成/消滅する局所flux成分を `F_fold(R,p) ~= A(p) sqrt(R-a(p))` とする（既存の他のarcがあれば全fluxには滑らかな背景成分も含まれる）。左endpoint map `R=a+w s^2`, `J=w s` では

\[
F_p\sim -\frac{A a_p}{2\sqrt{R-a}},\qquad
F_R R_p\sim +\frac{A a_p}{2\sqrt{R-a}},
\]

したがってそのsingular leading termについて `J F_p` と `J F_R R_p` は打ち消し、`F_fold J_p=O(s^2)` となる。滑らかな背景fluxの項は別途有限に残り得る。右endpointも `R=b-w(1-s)^2` と `R_p=b_p` を使うと同様に相殺する。rawではcaustic cell 1/12の255点detailがそれぞれ `3.33e-8 -> 1.75e-13`、`1.95e-7 -> 7.70e-13` まで落ち、fold端nodeの `g_old` が絶対値6.0–33程度なのに `g_move` は概ね `1e-9–1e-5` になる例を確認した。

一方、変換後の差は恒等的に

\[
g_{move}-g_{old}=\frac{1}{D}\,\partial_x(F R_p).
\]

各cellの境界寄与は `[F R_p]_{left}^{right}/D`。共有するphysical foldでは左右cellが同じevent sensitivityを使うため隣接境界項が相殺する。chart/soft cutは任意の内部分割として固定し `R_p=0`、外端もこの試験方向では固定（またはfluxが0）である。このため、正確な積分なら全cell合計はoldとmoveで等しい。

255点での有限則はその恒等式を丸め精度まで再現するほど収束していない。causticの合計差はQ3からQ8で `0.3134, 0.03594, 0.004427, 5.51e-4, 6.89e-5, 8.59e-6` と約1/8ずつ減少する。rand035のQ8 valid-only差は `4.88e-9, 1.03e-8, 2.93e-9, 6.19e-9`。ここでrand035の値は後述の通り全cell completeではない。したがって不一致を消すためのgate緩和やproduction変更は行わず、有限則の差を収束誤差としてrawに残した。

## 積分結果

独立referenceは各摂動でcold topologyを再生成し、direct `phi`/arc value積分からFDを作った。caustic `a` はradial GL order 128/256/512/1024と複数stepを比較。rand035はbinary128 quartic + qf Sturm isolation + direct angular physical value積分、5点central FDを使い、Xは `160 x 32` radial/subdivision・1024 angular、Yは `128 x 16`・1024 angularまで上げた。rand035の完了済みreference rawはbase `f83e30f` の計算だが、そのbaseから本checkpointの `dfe09dee` まで `src/lcbinint/magnification/holonomic` と `tests/holonomic_cpp` に差分がないことを確認した。reference source/rawともevidence内に複製し、必要なら現checkoutで再生成できる。安定幅は観測された収束差であり、形式的な区間誤差boundではない。

| 対象 | 255点/セル old | moving | 独立FD | old誤差 | moving誤差 | 完全性 |
|---|---:|---:|---:|---:|---:|---|
| caustic-cross, `u=0`, `a` | -2.228480422662 | -2.228471835752 | -2.404689164237 | 0.176208742 | 0.176217328 | complete |
| rand035, `u=0`, `X` | 17.042609893006 | 17.042609888121 | 17.042609577901 | 3.15e-7 | 3.10e-7 | incomplete |
| rand035, `u=0`, `Y` | 36.010207112840 | 36.010207102519 | 36.010206446936 | 6.66e-7 | 6.56e-7 | incomplete |
| rand035, `u=0.5`, `X` | 16.918290140177 | 16.918290137247 | 16.918289951079 | 1.89e-7 | 1.86e-7 | incomplete |
| rand035, `u=0.5`, `Y` | 35.747525536081 | 35.747525529889 | 35.747525136525 | 4.00e-7 | 3.93e-7 | incomplete |

caustic `a` の独立FDは `h=2e-6` でradial order 128/256/512/1024に対して `-2.4046831850, -2.4046884199, -2.4046890816, -2.4046891642` と収束し、近傍step/orderの最大観測差は `8.26e-8`。したがって約`0.1762`の差はreferenceの揺れではない。Q3..Q8のcaustic積分列もこのreferenceへ届かず、Q8でまだ差が半分程度ずつ残る。

rand035の表はcell 11（両端physical fold、幅 `4.17938205094e-10`、`R ~= 3.0353`）において255 node中48 nodeが `DegenerateChart` でrejectされたため、**valid-onlyの部分和**である。missing contributionを誤差boundとは見なさず、whole-domain gradientの精度比較には使わない。失敗sampleを0扱いして値を補完することもしていない。Q3だけは全cell nodeがvalidだが、その後は同じcellが不完全になる。

## 局所改善と全体不改善の切り分け

weighted interpolation-detailの全cell和に対する moving/old 比は順に caustic `1.000193`、rand035 `1.001642` (`u=0,X/Y`)、`1.001642` (`u=0.5,X/Y`) で、全体detailは同等かごく僅かに悪化した。cell-levelではrand035 cell 2のfold端が `7.87e-8 -> 8.82e-9` と改善する一方、支配cell 3（幅 `0.0690684`、片端fold）は `0.00654442 -> 0.00655523` とわずかに悪化する。causticではfold-only cell 1/12が大幅に滑らかになるが、detailを支配するcell 5（foldなし）は `1.42769` のまま、cell 7（片端fold）は `4.17025 -> 4.17132`。このcaseの遅い減衰をphysical fold以外のchart/soft-event分割が支配している。

cell寄与の符号相殺比（`sum(abs(cell contribution))/abs(sum(cell contribution))`）はQ8で caustic `71.51 -> 55.63`、rand035 u=0 `2.901 -> 1.493`、u=0.5 `3.080 -> 1.498` と変わる。ただし相殺比の改善は積分detailやFD誤差改善を意味しない。各cell寄与、nodeごとの `JF_p`, `JF_R R_p`, `FJ_p`、inner/geometry/roundoff ledger値、event sensitivityはraw TSVに保存した。radial誤差は各cellのold/moving weighted interpolation-detail norm、inner/geometry/roundoffは既存fixed-R node ledgerのabsolute Fejer-weighted寄与、event項はevent radius uncertaintyとして機械可読summaryに分けた。moving項 `F_R R_p` の新たなerror propagationはこのshadowでは認証していないため、ledger totalやgradient qualityを新設・主張していない。

## 判断

fold germの先頭特異項がmoving mapで消えるという数学的仮説は、このshadowで支持された。しかし全cellの勾配積分はcausticで速く収束せず、独立FD誤差も改善せず、rand035はthin cellのnode rejectで全domain監査を完了できない。よって今回の全体仮説（moving-map微分によりgradient radial integrandが一般に滑らかになり、少ないnodeで誤差が改善する）は**現条件では棄却**する。adaptive gradient integrand、scheduler、ledgerへ進めない。

残る原因候補はphysical foldではなく、causticの非fold/chart/soft-event cell、cell/panel間の強い相殺、rand035の極薄二fold cellである。次の研究を行う場合は、まずcaustic cell 5/7のdetailの出所と、rand035 cell 11の`DegenerateChart`を独立課題として解析する。このcheckpointでは新しい最適化を追加していない。

## 再現と検証

repo rootから一括再現:

```bash
bash evidence/holonomic/moving_map_gradient_phase72/run_all.sh
```

主な生成rawは `evidence/holonomic/moving_map_gradient_phase72/` の `events.tsv`, `all_events.tsv`, `cells.tsv`, `nodes.tsv`, `totals.tsv`, `pointwise_fd.tsv`, `reference_fd.tsv`, `summary.json`。rand035の独立reference raw/sourceは `rand035_reference/` に保存した。個別実行コマンドとcompiler flagsは同ディレクトリの `provenance.txt`、hashesは `SHA256SUMS` を参照。

確認したCTest: `holonomic_quartic_sturm`, `holonomic_quartic_local_bracket`, `holonomic_root_pair`, `holonomic_adaptive_radial`（4/4 pass）。
