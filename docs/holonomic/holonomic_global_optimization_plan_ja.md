# lcbinint：holonomic再設計とD14グローバル最適化 実装計画

**対象**：`NunotaKansuke/lcbinint` / `dev/holonomic`  
**調査したHEAD**：`420f58861fa2f67c1109ad3045f5ef80848fde68`（checkpoint 37）  
**作成日**：2026-09-09  
**用途**：リポジトリ内で動く実装エージェントへの設計・引き継ぎ資料。

## 0. 最初に読む要約

目的は二つ。

1. **本物のholonomic計算**を、毎ステップのK-ruleや9点の補間に置き換えて済ませず、方程式から係数を生成する局所Taylor輸送として実装し、現行V2を同精度で上回れるか調べる。
2. **D14の全般的な実行費用**を下げる。構造式・複素根・イベント情報を維持しながら、除算、平方根、重複構築、過剰反復、補助多項式の解法を改善する。

low-q / small-rho / 軸上専用seedを今回の中心にはしない。数値的な条件判定に応じた精度昇格・チャート切替は使うが、特定のベンチ名や物理パラメータの経験的しきい値で勝つ方法は採らない。

**推奨の依存関係**は「現行コードの再現・誤差計測 → 共通の数値基盤 → D14改善と正則化Taylor/GMを並行 → 同一精度で統合比較」。各小実験のたびに承認待ちで止まる必要はない。実装順、分担、次数、内部構造は実測に基づいて決めてよい。ただし成功条件を遅い旧baselineとの比較へすり替えない。

現行V2と旧V3は比較・退避用に残す。新しい本線は `V4/GM-series` のように区別する。名称より実際の計算内容を重視する。

---

## 1. 調査範囲と、今回確認できたこと

### 1.1 根拠の区別

- **[実装確認]**：本書末尾の固定SHAのソースをGitHub connectorで読んだ結果。
- **[導出確認]**：同梱Pythonで今回、記号恒等式または小さい独立例を確認した結果。
- **[実装提案]**：まだこのブランチに実装・性能検証していない案。
- 過去のcheckpointの時間は**過去の報告値**。本書作成環境ではリポジトリ全体をclone/buildできなかったため、新しいlcbinint全体の速度は測っていない。代数チェックは実行済み。

### 1.2 過去の会話から修正すべき前提

| 過去に混ざった理解 | 固定SHAのソースで確認した実態 |
|---|---|
| C++のfull ODEは6状態のGauss–Maninを輸送する | `ode_rhs_jac/value` は `(m,v)` の微分方程式とflux積算。半光量の右辺は `v_times_K[_jac]` またはサンプルから作った `jet_eval`。K自身のGM状態はここにない。[S4] |
| jet構築はDCTなので連立解法がない | `ode_detail::cheb_fit` は**9×9 Gauss–Jordan**。DCT置換はまだ有効な実装案。[S4] |
| `n_r=64`でも1cellあたり8〜10点しか使わない | `flux_jacobian_integrate` は非empty cellごとに `for(k=0;k<n_r;++k)`。通常経路は**各active cellで64点**。文書の8〜10という説明は、このループのノード数ではない。[S5] |
| 角度求積ゼロcounterならHGM実装済み | `angular_sweep_nodes` は旧sweep用counter。K-rule呼び出しとその16+8点は別途残り得る。counter名だけで証明しない。[S4,S8] |
| `(m,v)` RHSのたびに必ずNewtonを解いている | `root_pair_dR` 自体は係数評価と2×2 IFT。Newton補正は呼び出し経路を区別する。過去の包括的な説明だけでコストを決めない。[S6] |
| D14のfloat128・残差・根の和で全根が証明される | 現行は数値的なsanity checkであり、全根包含の証明とは別。特に根の和一つでは重複/欠落を検出し切れない。[S2] |
| `cond(C)`が大きいことだけでODE解が不安定と決まる | 接続行列の条件数、基底変換の条件数、実際の解の誤差増幅は別。全部を同一指標にしない。[提案上の訂正] |

過去の「完全ODEが成功」「Pythonで15倍」などは、このHEADで何が実行されるかを上書きする根拠にはしない。Pythonには本当のGM還元・輸送もあるが、別経路である。[S9–S11]

### 1.3 現在の配置

`dev/holonomic` は研究・実装を隔離した追加ブランチ。`tests/holonomic_cpp/CMakeLists.txt` は独立プロジェクトであり、top-level CMakeや本家 `.so` は変更しない。[S12]

旧履歴のproduction wiring / warmup接続に関する記録は、現在ブランチへの配線完了を意味しない。現在のコードと呼び出しグラフを基準にする。

---

## 2. 固定するbaselineと測定の契約

まず同じD14実装・同じ入力・同じ出力要求の下で次を区別する。

- **V0**：per-radius quartic + direct angular sweep。
- **V1**：V0 + `(m,v)` continuation。
- **V2**：V1 + 正則化 `vK` のGC2求積。現在の速度baseline。
- **V3**：checkpoint 37のRK4 / fitted-packet研究経路。凍結参照。
- **V4**：本計画の方程式由来Taylor / GM輸送。

`bench_holonomic_ode.cpp::set_variant` と実際のflagsを最初に記録する。静的にキャッシュされた環境変数を同一process内で変えて比較したつもりになることを防ぐ。[S13]

### 誤差と速度

「旧V0と一致」は回帰確認。「真値に十分近い」は別確認。`n_r=64`も `n_r=512`も自動的に真値とはしない。

値の許容誤差は `abs_tol + rel_tol*abs(mu_ref)`、勾配は**成分別**の絶対/相対許容誤差を持つ。ゼロ近傍の微分を相対値だけで評価しない。値/勾配の目標は初期候補として `1e-4, 1e-6, 1e-8` の複数水準を測り、達成不能は正直に報告する。

参照は解像度・精度を上げたV2、独立な直接積分やVBM等のvalue、安定区間の高精度有限差分/独立微分を組み合わせる。実根分類・branch切替が絡むため、complex-stepを無条件にoracleにしない。

- 全てのfull-solve比較でmultipole shortcutは無効。
- value-only / linear-LD value / value+5-Jac / JVPを分ける。
- D14をhoistしたradial単体、D14込みepoch、系列warmを別表にする。
- 108ケースを回帰セットとして保持し、ランダム・近接イベント・薄いarc・チャート跨ぎ・厳密/近似重根を追加する。
- 独立反復の分布と、同一caseのpaired ratioを保存。`ratio of medians`と`median of ratios`は別の名前にする。
- best-ofだけを主張の根拠にせず、warm/cold、cache reset、NUMA、実行CPU、flags、commit、work量を記録する。
- timeoutは打ち切り値であり、通常の完了latencyとして混ぜない。

**採用目標**：通常ケースも含め、現行V2より同じ精度で速くする。D14新旧×radial新旧の2×2比較で寄与を切り分ける。古いinverse-ray+FDに対する2倍ゲートを、V4がV2に勝った証拠に使わない。

---

## 3. D14系：既に入っているものと未回収の計算

### 3.1 既存構造式は維持する

`d14_structure.hpp` は既に全般的な厳密恒等式を使う。[S1]

\[
\widehat D=(C^2-4vg)g^2+8C(2C^2-9vg)z-432v^2z^2,
\qquad D_{14}=4096\widehat D.
\]

ここで `C=C3(v), g=G4(v), z=Z3(v)`。これを再発明することが今回の作業ではない。

現行のroot loopは概ね、balanced double予備探索 → DD polish → 必要ならquad warm → quad cold。structured化、DD、warm seedは維持する。

### 3.2 D1：複素除算と平方根の重複を削る【優先度：高】

**実装確認**：`Cplx<T>::operator/` は分母を計算したあと、実部と虚部について各々 `T` の除算を行う。DDの除算自体も複数のdouble除算と補正を行う。Aberthは各根ごとに `1/(z_i-z_j)` を13回作り、さらに `P/P'`、その後の補正でも複素除算する。[S3,S7]

改善案：

1. **複素逆数専用** `crecip(z)` と **実数scale専用** `cscale(z,a)` / `add_real(z,a)` を用意する。
2. `crecip(x+iy)` は範囲をscaleした上で一つの実逆数を共用し、実・虚部に掛ける。一般の複素除算で不要な0倍を繰り返さない。
3. Aberth補正を厳密同値な
   \[
   \delta_i=\frac{P(z_i)}{P'(z_i)-P(z_i)\sum_{j\ne i}(z_i-z_j)^{-1}}
   \]
   にする。現行の `ratio=P/P'; delta=ratio/(1-ratio*sum)` に対し、外側の複素除算を2回から1回へ減らせる。[同梱恒等式チェック済み]
4. 収束判定用 `cabs(delta)` の各根・各反復の平方根を、scaled norm-squared比較へ置換する。`tol^2`のunderflowとNaNは明示的に処理。必要な実ノルムの出力は最後だけ。
5. `P` と `P'` の共通係数評価を使う。structured evaluatorは既に同時評価、一般Horner側は別々なので、両者を混同しない。

注意：`a/b`→`a*recip(b)`は浮動小数点ではbit同値ではない。DD/quadの残差・根集合・event・最終出力で評価する。現行の逐次更新（Gauss–Seidel型Aberth）の順序は最初は保持する。

対称な `1/(z_i-z_j)=-1/(z_j-z_i)` を再利用してpair loopを半分にするには、通常は同時更新へ変える必要がある。これは別A/Bであり、単なるCSEとして黙って変更しない。

### 3.3 D2：D/D′の共通式を一本の評価グラフにする【高】

現行は `v^2`、`z^2`、`v*g'` 等を組み合わせ、多くの定数も `Cplx<T>(constant,0)` として掛けている。[S1]

次を一度ずつ計算する。

\[
A=C^2,\ B=vg,\ W=vz,\ U=A-4B,\ V=2A-9B.
\]
\[
\widehat D=Ug^2+8CVz-432W^2.
\]
\[
A'=2CC',\ B'=g+vg',\ W'=z+vz',
\]
\[
\widehat D'=(A'-4B')g^2+2Ugg'
+8\{[C'V+C(2A'-9B')]z+CVz'\}-864WW'.
\]

恒等式はチェック済み。**実際のDD演算数が減るかは、compilerの出力とbenchmarkで確認**する。数学的に同値でも、near-multipleでの相殺が悪化し得るので、旧structured evaluatorをoracleとして残す。

`4096`は `D/D'` の比から消える。iteration用は `Dhat,Dhat'`、検証用は係数規約と整合するスケールにする。

追加候補：実係数多項式の複素点評価を二実数の漸化式へ変え、C/G/Zで `2 Re(v), |v|^2` を共用する。これも全般的な代数変形だが、精度と演算数を見て採否を決める。最初から全候補を無条件に積む必要はない。

### 3.4 D3：不要な構築・allocation・過剰polishを減らす【高】

実装で確認した対象：

- `radial_events` はstructuredがONでも `p_coeffs_in_R` の5本全てを構築。現状そのfamilyを使う主な残存箇所は `chart_p4`。[S2]
- `d14_expanded_from_struct` はvector生成を重ね、`C3*C3`を再計算する。[S1]
- structured DD経路でも、使わない `descdd` を構築している。[S2]
- generic `aberth` はseedがある場合もCauchy boundを作る。[S3]
- 複数tier間で同じ係数cast、reverse、normalization、root用vectorを作る。

`D14Workspace` と小さい `array` を使い、係数、root、型別blockを一度構築して再利用する。最大次数14であっても、有効次数が落ちる場合のサイズを維持する。

現在の固定 `tol=1e-26` を単に緩めるのでなく、**下流が必要とするevent半径・root分類誤差**とroundoff boundに基づく停止を導入する。単根で `|delta|`、backward error、分離度を見て更新不要rootを止める。未解決clusterだけ反復/精度昇格する。ただし止めたrootもAberth相互作用には残し、必要なら再活性化する。[R1]

warmは現在、前epochのqf rootをdoubleへ落として予備探索に使う。その前に**保存rootから直接DDの短い補正を試す**経路を比較できる。現在の係数で検証してから採用し、失敗時に現行探索へ戻る。古いCellPlanの使い回しとは別物。[S2]

### 3.5 D4：補助 `chart_p4` の6次解法を二つの3次へ【高・新しい恒等式】

D14本体とは別に、現在 `radial_events` は `p4(R)=0` をgeneric Aberth（最大200反復）で解く。[S2]

`boundary_polynomial.hpp` の式から直接、

\[
U(R)=R^3+(a+X)R^2+(aX-1)R-a m_0,
\]
\[
p_4(R)=(\rho^2-Y^2)R^2(R+a)^2-U(R)^2.
\]

したがって `b^2=rho^2-Y^2` とおけば、

\[
\boxed{p_4(R)=-[U(R)-bR(R+a)][U(R)+bR(R+a)]}.
\]

これは**一般パラメータで記号検証済み**。軸上・low-q極限の近似ではない。実用的には：

- `|Y| > rho` かつ `a>0,R>0`：`p4<0`、正の実chart rootなし。方程式を解かず除外可能。
- `|Y| < rho`：
  \[
  R^3+(a+X\pm b)R^2+[a(X\pm b)-1]R-a m_0=0
  \]
  の二つの実3次を解く。
- `|Y|=rho`：同一3次の二乗として扱い、重複の由来を保存。

`b`計算は `(rho-|Y|)*(rho+|Y|)` と精度昇格を併用し、境界を丸め誤差で勝手に確定しない。3次は安定な実根分離/補正で処理し、素朴なCardanoの相殺を持ち込まない。出力は元の `p4` と元のレンズ境界で検証。

この変更が入れば、structured通常経路で5本のqf `p_coeffs_in_R` を作る必要も除去できる。legacy pathには残す。速度は `chart_p4` と `pcoef` を分けて測る。

### 3.6 D5：イベント分類で3次全根を毎回解かない【中】

`double_root_is_real` は各D14実候補で `P'` の3次全根を最大120反復で求め、`P` 残差の小さい根を選ぶ。[S2]

通常の単一二重根なら、`P` と `P'` のsubresultant/近似GCDから得る一次因子が接点を表す。これをcandidate generatorにして `P≈0,P'≈0`、元の `phi`、スケールしたtangency Jacobianで検証する。

多重接触・二つの二重根・chart付近では一次GCD前提を外し、現行3次全根または高精度へ戻る。全域に有効な判定付き高速経路であり、特定q向けではない。

---

## 4. D14の安全性：速度変更と一緒に明確化する

以下はソースから読める実在の監査対象。[S1–S3]

1. **NaN集約**：`d14_worst_res` は `if(res>worst)`。NaNは比較falseで無視される。全rootのfinite checkと非finite残差→失敗を明示する。Aberthの `maxstep` 集約も同様。
2. **最終失敗の伝播**：`radial_events` は `solve_d14` の返却rootを使うが、最終cold後の `worst_res` を明示的に拒否していない。成功/不確実を型として戻し `classify_cells` まで伝播する。
3. **根の和**：P=(t²−1)(t²−4)に対し、誤ったroot列(-1,-1,1,1)は全残差0かつ根の和0。よって第一Newton和は完全性証明にならない。同梱テストに入れた。
4. **共役snap後**：共役化・merge後のroot集合で再検証する。snap前の残差を最終値の保証に流用しない。
5. **相対残差**：現行の `|P(z)|/max|c|` と、`|P(z)|/(sum |c_k||z|^k)` は別。大きいroot、小さいrootで後者のscaled backward errorも使う。位置誤差・実根分類はさらに別。
6. **次数落ち/merge**：相対係数しきい値によるdegree trim、`1e-7`近接event mergeは数学的に同一rootである証明ではない。細いbandの削除と混同しない。

実装案は `D14SolveResult{roots, actual_degree, quality, residuals, clusters, counters}`。既存API wrapperは残せる。

**検証の強さを表示する**：residual sanity / numerical isolation / rigorous enclosureを区別。単純なroot-sumを強い証明と呼ばない。厳密さが必要なoffline oracleには、実装済みのexact polynomial referenceやMPSolve/FLINT等の包含機能を利用する。[R1,R2]

全root包含の実装候補はWeierstrass補正に基づく包含disk（重複clusterは個別rootでなくclusterとして囲う）。計算誤差を外向きに囲わず、形式だけdiskを計算してrigorousとは呼ばない。既存ゲートより高速化を理由に安全性を下げない。

---

## 5. holonomic再設計：何を本物の輸送と呼ぶか

新しいV4は、以下を区別する。

- `(m,v)` の方程式から境界根の局所係数を生成する：**代数的continuation**。
- K-ruleの値を9点集めて補間する：**spectral approximation**。有用だが、それだけでGauss–Manin実装とは呼ばない。
- 積分が満たす微分形式の恒等式を作り、そのODEからperiodの係数を生成する：**本計画のholonomic transport**。

現行V3の失敗は「この全てが不可能」の証明ではない。しかし「full ODEがPythonで成立済みだから移植だけ」とも扱わない。真の接続のC++構築・安定な基底・輸送誤差が新しい実装課題になる。[S4,S9–S11]

目標は**RK4をもっと何万回も回すことではなく、少数のanchorと短い係数列で必要なRへ進むこと**。

---

## 6. H1：共通幾何と代数Taylor predictorを先に整える

### 6.1 同じ半径での重複を消す

`ode_rhs_jac` は同じRで `boundary_quartic`, `_dR`, `_dp` を呼ぶ。それぞれ `t_complex_coeffs` と類似中間量を再構築する。また `root_pair_dR` と後続の `mv_param_jac` が同じEO Jacobianを使う。[S4,S6,S14]

`BoundaryBundle{P,P_R,P_p}` と `EOEvaluation{E,O,J,scale,condition}` を一度生成し、全arc・全微分方向へ渡す。value-onlyは不要な微分を計算しない。2×2 solveは複数RHSをまとめる。真のGM導入前でもV2に適用できる全般的改善。

### 6.2 9点でfitする代わりに、E=O=0から係数を生成

\[
E=P(m)+\frac v2P''(m)+\frac{v^2}{24}P^{(4)}(m),\qquad
O=P'(m)+\frac v6P'''(m).
\]

`h=R-Rc`, `x=(m,v)=x0+sum_{n>=1} x_n h^n` とおく。第n係数は

\[
\boxed{J_0 x_n=-[h^n]F(x_0+\cdots+x_{n-1}h^{n-1},R_c+h)},\quad F=(E,O).
\]

**同じ2×2 `J0` の分解を全次数で使う**。係数を決めるための9回の小刻みODE marchも、補間fitも要らない。これは近似モデルの当てはめでなく、元の代数方程式から得る切断Taylor級数。

同梱のsynthetic quarticでは、普通の二重根anchorから5次まで生成して既知の根ペアを厳密再現した。lcbinint全域の収束確認はこれから。

通常区間では N=8/12/16/24 等を候補に、tailと残差でstepを選ぶ。必要なRへHorner評価し、最後に1〜2回のNewton projectionで `E,O` を補正してもよい。補正前後の差とbranch identityを記録。失敗はstep縮小/別anchor/完全root solveへ。

複数pairの対応、全root数、未発見pairがないかは別確認。判別式の符号だけでは0対4や2イベントをまたぐ変化を区別できない。

### 6.3 チャート切替

`t=tan(theta/2)`が大きくなるところは、同じ円を `u=-1/t` で表す。

\[
\widetilde P(u)=u^4P(-1/u),\quad
\widetilde m=-\frac{m}{m^2-v},\quad
\widetilde v=\frac{v}{(m^2-v)^2}.
\]

根ペア変換は記号確認済み。`m²-v`が小さい場合はこの切替を使わない。別チャートの重なりで検証し、向き・wrap・arc IDを維持する。固定角度チャートならパラメータ微分も変換式で求める。

チャートIDは離散分岐として固定し、チャート係数/境界位置の連続依存は微分に含める。極端に長いarcを人工的に分割する場合は相対積分の端点項が必要であり、無条件に閉periodの式を流用しない。

---

## 7. H2：F0微分の平方根を、微分した後でなく座標の段階で消す

現行 `dtheta_derivs` にある式をそのまま整理する。[S4]

\[
\Delta\theta=2\operatorname{atan2}(2\sqrt v,1+m^2-v),\quad
D=(1+m^2-v)^2+4v.
\]

固定Rでのパラメータpに対し、

\[
\partial_p\Delta\theta=\frac{B_p}{\sqrt v},\quad
B_p=\frac{-8vm\,m_p+2(1+m^2+v)v_p}{D}.
\]

B_pは普通の接点で有限。**`1/sqrt(v)`を高次数多項式で無理に近似しない。**

### 7.1 通常foldの広いpatchを最初からu座標で処理

普通のfold `Rf` の内側を `R=Rf+sigma*u²`, `sigma=±1`, `u>=0` とする。

\[
v(R(u))=u^2w(u),\qquad w(0)=\sigma v_R(R_f)>0.
\]

すると、固定Rで組み立てた微分積分の密度は

\[
R\frac{B_p}{\sqrt v}\,dR
=\frac{2\sigma R B_p}{\sqrt{w(u)}}\,du.
\]

**uの逆数は消える**。`v→1e-9`まで小刻みに進んでから最後だけu置換する現行方式ではなく、fold-facing patch全体へこの座標を使う。現行 `ode_fold_tail` にu置換自体は既にあるため、「初めてu置換した」と誤記しない。[S4]

`w=v/u²`をtiny uで直接割って作らず、E/Oの局所係数や接点展開から評価する。F0値、Fhalf、微分も同じ変換の下で計算する。

別表現として `u=sqrt(v)` も使える。その場合 `dR/du=2u/v_R` なので `v_R!=0` が必要。R方向のv最小点では使わずpatchを切る。

### 7.2 foldではない小さな正のv

`v_min>0`なら実軸上の発散ではない。近い複素特異点により近似が難しいことはあるが、「普通のquadratureは原理的に不可」とは言わない。

局所的に `v≈v_min+c(R-Rm)², c>0` なら、`R-Rm=sqrt(v_min/c)*sinh z` のような座標が候補。実際のvは元の代数式で評価し、二次近似へ置換して真値を変えない。専用変換・部分区間化・次数増加を比較する。

### 7.3 微分の定義を崩さない

上の式の `m_p,v_p` は固定R微分。uを固定した微分とは別。

実装は固定参照区間へmapし、`Rf(p)`, patch端、R(u,p)、ヤコビアンを含めた同じ有限計算を微分する方式が第一候補。既存のfixed-R微分積算を移植するなら、移動端点のLeibniz項と人工patch境界での相殺を別途導出する。

普通の消失arcでは `J(Rf)=0` でも、chart/soft boundaryでは一般に0ではない。内部境界の寄与を勝手に消さない。

---

## 8. H3：真のGauss–Manin接続をC++で構築する

### 8.1 既存Pythonをoracleとして使う

\[
Q(t,R)=P(t,R)A(t)B(t,R),\quad A=1+t^2,\quad B=(R-a)^2+(R+a)^2t^2.
\]

既存 `connection.py` は、degree8かつsquare-freeの区間で

\[
-\tfrac12 t^kQ_R=C_kQ+S'_kQ-\tfrac12 S_kQ_t
\]

を正確に検証する。[S9]

ここから `I'=C_eta I`。7形式から留数を除く6形式への変換と、その変換微分もPythonにある。[S10,S11]

この**形式恒等式を検証できる接続**をC++へ実装する。サンプルをfitしただけの6本多項式を「6次元Picard–Fuchs系」と呼ばない。

### 8.2 一つのmodular inverseを再利用する

\[
U=Q_t^{-1}\bmod Q,\quad S_0=Q_R U\bmod Q,\quad S_{k+1}=tS_k\bmod Q.
\]
\[
T_k=\frac{S_kQ_t-t^kQ_R}{2Q},\quad C_k=T_k-S'_k.
\]

Python数値版では8×8のsolve後、各kでpoly product/divisionを構築する。C++では固定サイズbuffer、`S_{k+1}`の一回shift/reduce、複数RHS、余りのスケール検証を使う。同梱チェックでgeneric square-free Qの7行恒等式とshift recurrenceを確認した。

### 8.3 接続もTaylor係数としてまとめて作る

毎ODE stageで8×8を解かず、anchorで `M0` をfactorし、

\[
M(h)u(h)=e_0,\qquad
M_0u_n=-\sum_{j=1}^n M_j u_{n-j}
\]

から逆元の係数を生成する。Qのleading coefficientで必要になる逆数もseries arithmeticで扱う。全係数で同じfactorizationを使う。

さらなる高速化候補：`Q=P4*A2*B2` が互いに素な区間では、modular inverseを4+2+2の小さい問題に分け、CRTで合成する。`Q_t mod f_i=f_i' prod_{j!=i} f_j mod f_i` を使う。因子のresultantが小さい場合は旧8×8/high precisionへ。まず8×8のreuse版との速度比較で採否を決める。

### 8.4 出力・基底・チャート

半periodを `I_k=1/2 ∮ t^k dt/sqrt(Q)` と定義すれば、既存観測covector hで

\[
J_{1/2}=\frac{2}{\rho}h^TI.
\]

片道積分/閉cycleの2倍をテストで固定する。`period_reduction.py` の `H` と留数ゼロ条件を再利用する。[S10]

通常patchでは、物理出力を先頭にした可逆な基底変換 `Z=B(R,p)I` を候補にする。ただし

\[
Z'=(B_R+BC)B^{-1}Z
\]

の **B_R項を必ず入れる**。パラメータ微分もB_pを含む。`cond(C)`だけでなく `rcond(B)`、変換による相殺、実際の誤差増幅を見る。

具体的な最初の可逆基底候補：6状態のpsi基底上の出力covectorを行スケーリングし、最大の安全なpivot成分jを選ぶ。新基底の第一行をその出力行、残り5行を `e_i (i != j)` とする。これなら基底の逆変換とその微分が小さい固定サイズ演算で作れる。pivot選択はpatch内で固定し、列/行scaleを含めた `rcond(B)` と実際の輸送誤差で検証する。これは実装を始めるための具体的候補であって、これだけでconditioning改善が保証されるという主張ではない。

foldでは `J=vK` のKを使う。Kを小さいJを割って作らず、正則化x-chartの式から初期化する。`(Jを返すcovector)/v` を単に作るだけでは、基底係数が発散して相殺を悪化させ得る。fold近傍ではx-chartでのdeflation・局所級数と適切な補助状態の正規化を併用し、通常基底への接続は重なり区間で行う。補助period全てが同じ正則性を持つとは決めつけない。消失cycle以外の対数成分が必要な場合は、局所基底を分ける。

`p4=0`の座標退化は一般に曲線そのものの特異性ではない。reciprocal/Möbius chartでQと微分形式を変換し、exact reductionで基底を接続する。旧eta/psiを同じchartで無理に貫かず、**その恒等式をoracleとして残し、新しい評価座標へ移す**。

---

## 9. H4：RK4の代わりに、ODE由来係数を直接生成する

通常patchで

\[
Z'=A(R)Z,\quad A=\sum A_n h^n,\quad Z=\sum Z_n h^n
\]

なら、

\[
\boxed{Z_{n+1}=\frac1{n+1}\sum_{j=0}^n A_j Z_{n-j}}.
\]

これが新しいhot path。**初期periodを作る少数の求積以外は、K-ruleサンプルで係数をfitしない**。[R3,R4]

`F'=ell^T Z` なら、積の係数を積分してpatch fluxを得る。F0はH1/H2の幾何Taylorから計算し、不要にGMへ入れない。接続を生成・保持する費用を全て時間に含める。

パラメータ方向の感度は

\[
Z_p'=AZ_p+A_pZ
\]

を同じ係数漸化式で進めるか、series演算自体をJet化する。前者の `Z_p` は固定Rか固定参照座標かを明記し、チャート/patch境界/normalizationの微分を含める。どちらでも5回root solverを起動しない。

### 初期値

- regular anchor：元の積分で一度、必要な補助状態も含めてseed。複数periodを同じ求積ノードで同時評価する。
- ordinary fold：局所級数からseedし、小さい正則patchへ接続。新しいarcの累積F=0は基準として使えるが、内部状態を全て0にはしない。
- foldで `f(t)=f0+f2(t-m)^2+...` なら
  \[
  K=\pi f_0/\rho+\pi f_2v/(4\rho)+\pi f_4v^2/(8\rho)+\cdots.
  \]
  係数はこの表記では導関数そのものではなくTaylor係数。rho、半period規約をテストで固定する。
- soft/chart boundary：新しい像の誕生ではない。状態と積算を継続する。パッチ差分Fのローカル原点は0にしてよいが、物理寄与の累積を捨てない。

### step / degree の選択

Nとpatch幅を誤差に応じて変える。D14の複素rootは候補となる近いsingularityの情報を与えるが、`sqrt(Re v)`はcomplex R特異点までの距離ではない。`±sqrt(v_complex)`を調べ、P/A/B因子衝突とchart/gauge singularityも加える。

推定した係数減衰はheuristic。採用には独立residual、次数/幅変更比較、少数の独立直接値、必要な領域でmajorant/interval boundを組み合わせる。seedの誤差も伝播させる。線形ODEのTaylor continuationは既存技術だが、本問題での低コスト・安定性はこの実装で検証する。[R3,R4]

---

## 10. H5：F0とrho微分の精度を最終出力から管理する

現行は

\[
\mu=N/[\pi\rho^2(1-u/3)],\quad
\mu_\rho=N_\rho/D-2\mu/\rho
\]

の相殺が強いと、ODEのepoch全体をdirectでやり直す。[S5]

対策は単なる高精度の最後の引き算ではない。**積分中の誤差が増幅される**ため、出力の目標誤差から必要なF・F_pの精度を戻して決める。

候補：

- `G0=F0/rho²`, `Gh=Fh/rho²` の正規化状態とその感度を使う。
- `rho*F_rho-2F` に対応する組み合わせを、相関を保持した同一ノード/seriesで積む。
- compensated summationやDDを**少数のaccumulator**だけに使う。
- V4の局所誤差を足し、rho微分の出力誤差が不足するpatchだけ精密化する。

正規化だけで相殺が消えると約束しない。真に小さい感度なら高い相対精度は不可能/不要な場合があるので、成分別absolute toleranceを優先する。全epoch二度計算の回数を減らしたか測る。

---

## 11. packet研究を無駄にしない、ただし役割を分ける

現行のサンプルpacketをreference/比較器として残す。構築費を減らす具体的修正はある。

### 11.1 本当にDCTにする

CG nodes `theta_j=(j+1/2)pi/N`, `x_j=cos(theta_j)` なら、

\[
c_0=\frac1N\sum f_j,\quad
c_k=\frac2N\sum f_j\cos(k\theta_j)\quad(k>=1).
\]

9×9の定数変換行列をprecomputeし、全列へ適用する。一般Gauss–Jordanを毎回呼ばない。N=9ではFFTライブラリ導入は不要。Nを増やすならDCT library/固定kernelを比較する。[R5]

同梱チェックは9点・12列の既知Chebyshev多項式を約4e-15の最大絶対差で復元した。これはmicrolensingの誤差保証ではない。

### 11.2 Chebyshevを不要にTaylorへ変換しない

既存 `cheb_to_taylor` は毎回T_nのmonomial表・binomial表を作る。補間ベースのpacketを使うなら、正規化したxのChebyshev係数のままClenshaw評価し、

\[
\int_{-1}^{1}T_k(x)dx=\begin{cases}0&k\text{ odd},\\2/(1-k^2)&k\text{ even}\end{cases}
\]

で積分する。Taylor変換が必要な経路のみ事前表で行う。`1/H^n`の大きい係数を不用意に作らない。

これは**packet baselineの改善**であって、H3/H4の本物のGM導入完了とは報告しない。両者の比較こそ価値がある。

---

## 12. データ構造とファイルの案

既存C++17、header中心、isolated buildを維持する。巨大な `holonomic_ode_transport.hpp` へ全て足す必要はない。

| 候補ファイル | 責務 |
|---|---|
| `root_numeric.hpp` | checked complex inverse/division/norm、fixed workspace |
| `d14_eval.hpp` | existing structureの共通評価グラフ、coefficient buffer |
| `d14_event_quality.hpp` | residual/cluster/degree/event status |
| `boundary_bundle.hpp` | P/P_R/P_pと共通中間量 |
| `truncated_series.hpp` | fixed-capacity truncated power series、scalar/Jet lanes |
| `algebraic_series.hpp` | EO implicit Taylor、projection、chart変換 |
| `radial_chart.hpp` | R/u/reciprocal chart、fold patch、向き |
| `gm_connection.hpp` | exact-identity-compatible modular reductionと接続jet |
| `gm_transport.hpp` | period seed、basis/gauge、Taylor march、flux/sensitivity |
| `integration_error.hpp` | cell/global output error、status、refinement |

名前は提案。既存と重複する型は再利用する。

概念データ：`PreparedD14`（read-only係数/型別block）、`D14Workspace`（call-local）、`ArcBranch`（IDと向き）、`LocalSeriesBlock`（center、scale、chart、degree、state coefficients、validity/error）、`IntegrationResult`（value/grad、error、status、counters）。

**持ち越せるのは初期推定と検証済み係数情報であって、変化したパラメータで古いpanelを無条件に再利用することではない。** warmup/public wiringは今回の内部最適化と独立に扱う。

---

## 13. テストと実装の進め方

### 13.1 再現

```bash
git switch dev/holonomic
git rev-parse HEAD
cmake -S tests/holonomic_cpp -B build-holonomic-global -DCMAKE_BUILD_TYPE=Release
cmake --build build-holonomic-global -j <利用可能な並列数>
ctest --test-dir build-holonomic-global --output-on-failure
```

初回は新しいbuild directoryを使う。`/tmp/bench_cases.tsv` を暗黙前提にせず、入力形式を確認してrepoのTSVまたは生成スクリプトから明示的に作る。shared `.so` をこの計画だけでswapしない。[S12,S13]

### 13.2 D14

- finite/nonfinite、exact/repeated/near repeated、degree drop、conjugate pair、positive-real boundary、complex soft events。
- 既存構造式/新DAG/expanded oracleをqfまたはより高精度で比較。
- `p4`二三次の全正実rootと旧6次を独立oracleで比較。`|Y|`とrhoが隣接floatになるcaseを含む。
- 根の和を通る誤ったroot集合を必ずrejectできるか。
- precision tier / iteration / D-Dprime eval / complex division / active root数を記録。

### 13.3 holonomic

- E/O Taylor係数が方程式を所定次数まで満たす。
- GMの微分形式恒等式、留数ゼロ、basis変換の微分、half/full-period factor。
- chart前後のP、J、J_p一致。
- ordinary foldのF0 derivative正則化、接点からのseries seed、対数型/高次接触を混同しない。
- 新しいarcの誕生/消滅、全周、複数arc、soft boundary、root ordering。
- value-onlyとJVP/full-Jacのvalue一致および同じ数値reconstructionとの微分整合性。
- seedのangular node、通常patchのangular node、fallbackのangular node、Newton projection、GM solve回数を**実際のprimitive**で数える。
- strict regressionは既存値との一致、accuracyは独立参照。古い `5e-2` grad差のPASSだけで高精度を主張しない。

### 13.4 最後の比較

1. current V2 / old D14
2. current V2 / new D14
3. new radial / old D14
4. new radial / new D14

さらに新radialを「正則化＋algebraic Taylor＋direct K」と「その上にtrue GM」に分け、GMの限界利益を測る。補間packetも必要なら比較に残す。

最新profileからAmdahl上限を再計算する。昔の14%+18%、1.47倍を普遍的上限にしない。fallback/retry/seedを含む実測値で判断。

---

## 14. 実装上の裁量と完了条件

この計画は「毎spikeで承認を待て」という指示ではない。D14系とGM系は別commitで並行してよく、低リスク改善を先に取り込んでよい。未確認仮定を見つけたら最小再現例と修正理由を残し、合理的に進める。

ただし、次は固定する。

- **現行コードを基準**にし、過去の会話の「既に実装済み」を盲信しない。
- **low-q等の専用caseに逃げず**、全般的な式・演算・誤差制御を主題にする。
- **true GMとsample-fitを区別**する。ODE右辺でK-ruleを呼び続けた経路を「angular quadrature除去」と報告しない。
- **精度を緩めて速度勝利にしない**。高速化が確認できない研究経路は残しても標準へ昇格しない。
- **V2との同精度比較**が主。通常ケースを捨てたtailだけの勝利へ目的を変更しない。
- **根・トポロジー保証の強さを過大表示しない**。
- modified code、再現command、raw結果、未解決リスク、採用/不採用理由をrepo内に残す。

最終報告は `docs/holonomic/checkpoint_global_optimization.md` に集約する。最初に短い現状/変更予定、その後D14とGMの結果を追記すればよい。大きい設計書を毎回読み直さずに進められる索引も添える。

---

## 15. 今回実行した独立チェック

`checks/verify_structures.py`：

- `p4(R)` の二三次因数分解（一般記号）
- `dtheta/dp` の1/sqrt(v)分離
- u=sqrt(v)による測度との打ち消し
- reciprocal root-pair変換
- D14共通評価グラフの導関数
- Aberth補正の一除算形
- 第一Newton和が完全性証明ではない反例
- generic square-free QでのGM還元/shift recurrence

`checks/verify_series_and_dct.py`：

- 通常二重根をanchorにしたsynthetic quarticのimplicit Taylor（5次、厳密一致）
- CG9点→Chebyshev12列の直接DCT（最大絶対差約4e-15）

どちらもPASS。これは**新しいC++ solver全体の実行・性能・全域精度を確認した結果ではない**。

---

## 16. 参照したコード（全て固定SHA）

以下のURLの `420f5886…` が調査対象。実装者はbranchが進んでいたら差分を確認すること。

- **S1** `src/lcbinint/magnification/holonomic/d14_structure.hpp`：`d14_struct_build`, `d14_struct_eval`, `d14_expanded_from_struct`, `aberth_d14_struct`。
- **S2** `src/lcbinint/magnification/holonomic/radial_events.hpp`：`p_coeffs_in_R`, `double_root_is_real`, `d14_worst_res`, `solve_d14`, `radial_events`。
- **S3** `src/lcbinint/magnification/holonomic/poly_roots.hpp`：`Cplx`, `aberth`, `real_roots`。
- **S4** `src/lcbinint/magnification/holonomic/holonomic_ode_transport.hpp`：`cheb_fit`, `cheb_decay_rho`, `cheb_to_taylor`, `ode_rhs_jac/value`, `ode_march`, `mv_only_march`。
- **S5** `src/lcbinint/magnification/holonomic/epoch_jacobian.hpp`：`flux_jacobian_integrate`, `rho_cancel_kappa`。
- **S6** `src/lcbinint/magnification/holonomic/root_pair.hpp`：`p_derivs`, `eo_jacobian`, `root_pair_dR`。
- **S7** `src/lcbinint/magnification/holonomic/dd_real.hpp`：DD arithmetic/EFT。
- **S8** `src/lcbinint/magnification/holonomic/holonomic_transport.hpp`：`v_times_K`, GC2規則/収束判定。
- **S9** `python/lcbinint/holonomic_ref/connection.py`：`_connection_from_QQR`, `_ck_sk_exact`。
- **S10** `python/lcbinint/holonomic_ref/period_reduction.py`：`h_coeffs`, `psi_reduction_matrix`, `arc_chart`。
- **S11** `python/lcbinint/holonomic_ref/transport.py`：`psi_connection`, `transport_psi`。本当のGM/DOP853参照経路。
- **S12** `tests/holonomic_cpp/CMakeLists.txt`：独立buildの契約。
- **S13** `tests/holonomic_cpp/bench_holonomic_ode.cpp`：variant設定、timing、parity。
- **S14** `src/lcbinint/magnification/holonomic/boundary_polynomial.hpp`：P/P_R/P_p。
- **S15** `src/lcbinint/magnification/holonomic/cells.hpp`：event merge、512/3072grid再確認、CellPlan。

GitHub permalinkのprefix：
`https://github.com/NunotaKansuke/lcbinint/blob/420f58861fa2f67c1109ad3045f5ef80848fde68/`

## 17. 外部の数学的根拠（本計画の拡張部分）

- **R1** D. A. Bini, L. Robol, *Solving secular and polynomial equations: A multiprecision algorithm*, JCAM 272 (2014), 276–292. DOI: `10.1016/j.cam.2013.04.037`。root inclusion、未解決clusterだけの精度昇格、構造を保ったrootfinding。現在のlcbinintに自動的な保証を与えるものではない。
- **R2** MPSolve公式：`https://numpi.dm.unipi.it/scientific-computing-libraries/mpsolve/`。包含半径と構造化多項式の利用。常用依存として新たに入れるかは別判断。
- **R3** Marc Mezzarobba, *Rigorous Multiple-Precision Evaluation of D-Finite Functions in SageMath*, arXiv:`1607.01967`。線形ODEの数値解析的継続・正則特異点・誤差上界。
- **R4** Pierre Lairez, *Computing periods of rational integrals*, arXiv:`1404.5069`, Mathematics of Computation 85 (2016), 1719–1752。periodからPicard–Fuchsを構成する還元法。本書の小次数実装は既存Python恒等式を出発点にする。
- **R5** NIST DLMF §3.11(ii), §3.5, §18.38：`https://dlmf.nist.gov/3.11`, `https://dlmf.nist.gov/3.5`, `https://dlmf.nist.gov/18.38`。Chebyshev係数・求積・線形ODEの直交多項式展開。

外部研究は可能な構成と検証方法の根拠であり、このsolverでの倍率やcoverageを保証しない。
