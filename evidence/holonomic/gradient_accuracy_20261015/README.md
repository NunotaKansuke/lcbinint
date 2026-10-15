# 同じ値許容誤差で微分を改善する局所実験（2026-10-15）

## 方針

APIには値の絶対・相対許容誤差だけを持たせる。勾配の許容誤差を追加せず、勾配誤差による適応も行わない。前半の `joint` / `post_gradient` / `coordinate_scaled` は診断用対照であり、提案方式には含めない。

checkoutは `/rogue1_8/nunota/lcbinint/experiments/algebraic_boundary`、branchは `experiment/algebraic_boundary`、HEADは `b31db3b3be16f9c573dc81d9c87775a045a4aacc`。既存の変更を含むnative backendを専用buildからリンクした。ソースと実行ファイルのhashは `provenance.json`。新しいbranch/worktreeは作成していない。

## 観測

ノートブックの s=0.6, q=1e-4, rho=0.05, tE=10, u0=-1e-5, alpha=0.5、linear LD=0.5 を使用。7 epoch、値の atol=rtol=1e-4/1e-5/1e-6を比較した。独立VBMとnative値経路それぞれを3刻み幅で有限差分し、全データを `reference.csv` に保持した。

t=-0.7の dA/ds の参照は VBM 0.007153591–0.007153597、native 0.007155100–0.007155338。両参照が近く、刻み幅に対して安定している。

| atol=rtol | 現在の微分 | 試作 allmotion_posth | VBM FD |
|---|---:|---:|---:|
| 1e-4 | -4.5474011 | -0.00788348 | 0.00715359 |
| 1e-5 | 0.02004577 | 0.00748135 | 0.00715359 |
| 1e-6 | -0.00096602 | 0.00715641 | 0.00715359 |

1e-5で相対差は約180%→4.6%、1e-6で約114%→0.04%。同じtolで数値計算の仕組みを変える効果がある。ただし1e-4は依然大きく外れる。

t=+0.78の dA/du0 は、1e-5で -0.0591065→0.455503（VBM FD 0.431998）、1e-6で 0.433408→0.431986。dA/ds は1e-5でも約42%の差が残るが、1e-6では -0.00121215（VBM FD -0.00121283）となった。全成分・全条件の改善を保証する結果ではない。

## 有望な仕組み

既存fused実装のmaterial motionは、両端がfoldで周囲がemptyとなる一部のweighted spanだけに適用される。それ以外では、積分領域の動きによる寄与が固定Rの微分積分の中で相殺される必要があり、値用の有限メッシュでは大きな離散化誤差が残り得る。

`prepare_allmotion.py` はfold端点の半径微分を全spanに渡し、節点Rと半径Jacobianの移動も一緒に微分する。人工的な内部cutは両側で固定として扱い、full-circle区間にもRとJacobianのprefactor微分を加える。値と節点数を保ったまま、t=-0.7のdA/dsは0.02005→0.00693となった。これは厳密な誤差包含ではなく、有限差分による観測である。

`prepare_posth.py` は、値のtolだけで確定した各panelを勾配用に固定一回二分する。微分誤差や品質値は二分の判断に使わない。左右の子の評価が両方成功してから親を置換し、値は受理済みsnapshotを保持する。これを全spanのmaterial motionと組み合わせたものが `allmotion_posth`。

**7 epoch × 3 toleranceの21条件すべてで、返却値muはbaselineとbitwise一致した。** 微分誤差の認証は追加していない。t=-0.7、tol=1e-5の節点数は1705→2565。実験の時間はseed準備を除く一回計測であり、速度の結論には使わない。

## 採用できなかった試験

- 値tolの厳格化だけ：改善する成分もあるが、LDの小さい微分に誤差が残る。
- 勾配誤差で判定するjoint refinement：原因診断には有効だが、今回のAPI・方式の対象外。
- 角度nested上限31→511：一部改善するが残差が残る。
- fallback角度節点の移動だけを微分：効果は小さい。
- 固定radial次数63/127/255開始：非単調に悪化する例がある。
- 全角度を固定256点の直接phi積分に置換：高コストで、単独ではradial側の誤差を解消しない。
- weighted radial則を単純に無効化：+0.78などで大きく悪化。
- 全span motionと固定p-enrichment：改善する点もあるが、1e-4の-0.7の位置微分は悪化。固定hの方が今回の近傍では良かった。

## 限界と次の実装判断

現時点は隔離header overlayでの試作で、通常の微分API・既定動作には反映していない。粗いtolの大きな誤差、caustic接触・cusp・異なるq/rhoへの一般化、追加節点で新しいtopologyを発見した場合の扱いが残る。value convergenceとsupport certificationは別であり、この実験でsupportの証明は得ていない。

t=0.1ではVBM t0有限差分が刻み幅で変動し、q/sではVBMとnative FDも一致しない。その行は独立参照による採用判定には使わず、両系列を残した。

次の実装候補は「全spanで座標移動を整合的に微分する」ことと「値メッシュ確定後の固定一回の接線用二分」。値tolだけで動かす方針を維持したまま、独立したregular/fold/cuspの小行列で非回帰を確認してから既定経路への反映を判断する。

## 再現

既存の `build-algebraic-boundary/python-api` native librariesを使用する。`bash evidence/holonomic/gradient_accuracy_20261015/run.sh` でbaselineと候補を再生成・再計測する。`--references` を付けると高精度有限差分も再計算する。成果物・compiler一時ファイルはこのcheckout内だけに置く。

`comparison.csv` は物理パラメータへのchain rule適用後の5微分、両FD参照、刻み幅spread、節点数、時間を含む。tE/alphaは位置微分から派生するため独立FD対象は t0/u0/rho/q/s に絞った。
