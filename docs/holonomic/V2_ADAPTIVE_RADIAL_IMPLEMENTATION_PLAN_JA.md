> 実装時訂正（ユーザー指定）: 許容誤差は全出力で `max(Tol, RelTol*abs(Q))`。原計画の加算式は、以下でもこの定義へ訂正した。

# V2：fold-aware・増分型 adaptive radial quadrature 実装計画

基準コード：`NunotaKansuke/lcbinint` / `dev/holonomic @ 3f35991a1c22f11de60f26a4c88c3628e5aa5556`

対象：isolated holonomic。V2のD14/Sturm・quartic/root-pair・F0・K-rule・解析5Jacを維持し、**固定 `n_r` のradial積分制御を置き換える**。PF6/GM研究経路を再開しない。本書は実装設計であり、lcbinintへの統合・速度勝利を報告するものではない。

## 0. 決定事項

採用する構成は次の一つに絞る。

```text
D14 + Sturm の既存cell plan
    ↓
physical fold端に応じた固定の座標変換
    ↓
open Fejér-II：7 → 15 → 31 → 63 → 必要な場合だけ127/255または局所分割
    ↓
既存sampleを保持し、未評価nodeだけ追加
    ↓
積分値の差 + 補間関数の差のweighted L2 norm + node/endpoint/丸め誤差
    ↓
最終 mu / requested Jacobian に伝播した全panel誤差を合計
    ↓
不足するpanelまたはnodeだけ更新。許容誤差に届くまで継続
```

記号の混同を避ける：radial座標は `R`、root-pairは `m,v_pair`（本文のfold式のv）、D14の変数は `disc_v=R²` で、両者のvは別物。mapの0〜1座標、half-angle chart、LD係数にもそれぞれ `map_t`, `chart_t`, `ld_u` の別名を使う。

最適化対象は「受理された最終点数」だけでなく、**初期化、捨てた候補、再試行を含む実際の物理node評価回数と総wall time**である。最小点数の大域的最適性は主張しない。既知の情報を再利用しながら、要求精度に必要な仕事量へ近づける。

最重要の禁止事項：

- 64点を完走した、root検査が通った、またはK-ruleが通ったことを、radial収束と同一視しない。
- 予算超過時に固定64点へ戻して `OK` にしない。
- 固定gridをtopology authorityへ戻さない。
- 観測した次数差や係数減衰を、無条件の厳密誤差上界と呼ばない。
- 同じnodeを解像度変更のたびに解き直さない。旧求積値と新求積値を両方足さない。
- 既存の `.claude/`、共有production `.so`、無関係なrouterを変更しない。

## 1. 根拠と今回まだ確認できていないもの

### 1.1 コードから確認済み

`epoch_jacobian.hpp` の `flux_value_integrate` / `flux_jacobian_integrate` は、各active cellへ同じ `Cheb1Dyn(n_r)` を適用する。各cell端を `1e-9 * width` 内側へ動かし、nodeで `radius_value` / `radius_terms` を呼び、fluxと微分を加算している。[S1]

`EpochValue` / `EpochJacobian` は、radial truncation errorの数値や収束した解像度を返していない。`rho_cancel_kappa` による再計算は `used_ode` の場合の対策であり、通常V2の包括的な誤差保証ではない。[S1]

K-ruleは既存8/16点比較を持つが、これは内側の角度積分の検査である。外側のradial積分精度を保証しない。またV2のK-ruleは環境変数/overrideによる有効化があり、「V2」というラベルだけでは実行経路を確定できない。[S2]

Sturm版はfixed-grid上書きを除去しているが、有限精度のsign-margin検査であり、D14 eventの完全性を含む区間演算の全体証明ではない。[S3]

### 1.2 ユーザー提示の診断結果（今回の設計入力）

- loose toleranceでも固定64点のためV2の仕事量が減らない。
- 難例で `n_r=64:12182.2736 → 512:12153.8440 → 2048:12152.9736`。
- 誤差がphysical-fold端に接する一つのcellへ集中し、uniformでも起こる。
- 未収束でも `OK`。
- 保存VBM referenceと現環境の再計算も一致していない。

今回こちらから読めたremote HEADは上記 `3f35991`。提示された `docs/holonomic/v2_vbm_pure_kernel_benchmark_20260911.md` はremote取得時404だった。従って新診断の数値は**ユーザー提示の結果**として扱い、再実行済みとはしない。実装担当はローカルの診断Markdown/TSVから難例の全パラメータ、LD設定、frame写像を採取し、コミット可能なfixtureにする。数値からパラメータを推測しない。

`n_r=2048` も真値ではない。独立参照の不一致をadaptive radialだけで直せるとは仮定しない。

## 2. 「精度保証」の契約を最初に固定する

二種類を区別する。

### 通常モード：数学的根拠のある事後誤差推定

補間誤差・Cauchy–Schwarz・級数の収束モデルに基づき `estimated_abs_error` を返す。明示的なglobal tolerance制御、停滞検出、独立参照による検証を持つ。VBMの分析的error estimatorと同じく、**区間演算で全誤差を囲い込むこととは別**である。[R1,R3]

成功時の意味は「定義した事後誤差推定と数値検査が要求精度を満たした」。`assurance=Estimated` と明記する。

### 証明モード：必要条件が検証できる場合だけ `Bounded`

analytic envelope、幾何・node誤差の上界、event位置の上界、丸め誤差まで検証できた場合だけ `certified_abs_error` を返す。§6.4の具体的な上界を利用できる。単にdoubleをqfへ昇格しただけでは `Bounded` にしない。

通常モードへ巨大なvalidated numerics基盤を持ち込まない。初期実装は通常モードを完成させ、`require_bound=true` に対して証明材料がなければ `BOUND_UNAVAILABLE` を返す。未実装の証明を「保証済み」にしない。

### なぜ区別が必要か

有限個のsampleだけでは、全sampleでゼロだが積分が正の多項式を作れる。例として15点Fejér-IIの全nodeで `U_15(x)^2=0` だが、積分は約4.73626である。7点もその部分集合なので7/15差はこれを見逃す。補間差normでも無条件には防げない。**数学的根拠を持つ推定器と、全関数に対する厳密保証は同義ではない**。

## 3. APIと結果型

旧固定 `n_r` APIをそのまま残す。新しいadaptive APIは明示的な名前を付け、整数 `n_r` と浮動小数 `tol` の危険なoverloadを避ける。

```cpp
struct AdaptiveTolerance {
    double mu_atol;
    double mu_rtol;
    std::array<double, 5> grad_atol;
    std::array<double, 5> grad_rtol;
};

struct AdaptiveConfig {
    AdaptiveTolerance tol;
    bool with_jacobian;
    bool require_bound = false;
    int initial_level = 3;   // 2^3 - 1 = 7
    int split_check_level = 6; // 63点は分割/継続の判断点。成功の上限ではない
    int max_level = 8;       // 255。変更可能な仕事量設定
    size_t max_node_evals;
    size_t max_panels;
    size_t max_bytes;
};

enum class AccuracyAssurance { None, Estimated, Bounded };
enum class AdaptiveStop {
    Converged, BudgetExceeded, RoundoffLimited, InnerAccuracyLimited,
    TopologyUnresolved, EventLocationLimited, Nonfinite, BoundUnavailable
};

// 既存Statusとの対応は別途定義し、列挙子の既存数値を変更しない。
struct AdaptiveResult {
    double mu;
    std::array<double, 5> grad_mu;
    double estimated_abs_error_mu;
    std::array<double, 5> estimated_abs_error_grad;
    bool value_converged;
    std::array<bool, 5> grad_converged;
    AccuracyAssurance assurance;
    AdaptiveStop stop;
    Status numerical_status;
    AdaptiveStats stats;
};
```

導入候補：`epoch_value_adaptive`、`epoch_jacobian_adaptive`、prepared版。`AdaptiveWorkspace&` をcaller所有とし、epochごとのheap確保を減らす。

値のみ成功・一部微分未収束を表現できるようにする。要求していない微分は計算しない。`u=0` のvalue laneはFhalfを評価しない。u微分を要求する場合はu=0でもFhalfが必要になるため別flagで管理する。

`rtol` の意味は `max(atol, rtol * abs(output))` と明文化する。微分はゼロ付近になり得るため、成分ごとの物理単位に対応した `atol` を必須とする。VBMと同じ数字のTolを渡すだけで同精度とは扱わない。

## 4. foldを先に除去する座標変換

### 4.1 通常のfoldで何が起きるか

新生/消滅するarcの寄与について、通常の非退化foldでは、内側距離 `r=|R-Rf|` に対し局所的に

```text
f0(R) = A(R) sqrt(r)
fh(R) = B(R) r
fixed-R parameter derivative of f0 = C(R)/sqrt(r) + smoother terms
```

となる。A,B,Cが滑らかな領域に限る。cell全体には他のarcからの滑らかな寄与も加わり得る。

これはroot-pairの `v≈κr` と、角幅 `Δθ=O(sqrt(v))`、`Jhalf=vK` からの局所導出である。higher-order contact、cusp、別の分岐点衝突を同一モデルで扱わない。

### 4.2 使用するmap

panelの物理境界を `[a,b]`、`w=b-a`、Fejér座標を `x∈(-1,1)`、`t=(1+x)/2` とする。

| panelの端 | R(t) | dR/dx |
|---|---|---|
| 通常 | a+w t | w/2 |
| 左のみphysical fold | a+w t² | w t |
| 右のみphysical fold | b-w(1-t)² | w(1-t) |
| 両端physical fold | a+w sin²(πt/2) | (πw/4) sin(πt) |

積分するのは必ず `G(x)=g(R(x))*dR/dx`。新しいFejér weightsは `dx` に対する重みであり、旧 `Cheb1Dyn` のsqrt重みを流用/重ね掛けしない。

単純モデルでは `sqrt(r)dr` がy²型に、`dr/sqrt(r)` が定数型になる。これがadaptive点数を減らす数学的理由である。変換だけで全cellの解析性や高次接触の正則化が保証されるわけではない。

### 4.3 eventとmappingの安全条件

- `physical_real` という名前だけでなく、通常のfold条件 `P=Pt=0, Ptt≠0, PR≠0` の数値的分離、root-pairの対応を確認する。chart/soft eventはfold扱いしない。
- mappingはpanelの存続中固定。途中でmapを変える場合、元sampleのRを勝手に変えない。
- open nodeなのでexact endpointを評価しない。旧 `1e-9*width` の無条件insetを新経路へコピーしない。
- 近接点がbinary64でeventへ丸められる場合、endpointまでのoffsetを別保持し、必要な式だけDD等で評価するか `EventLocationLimited/RoundoffLimited`。clampして同じnodeを繰り返すのは禁止。
- squared mapはFejér点の端への集中をさらに強める。high levelでの距離・ULP・P_t・root separationを必ず監視する。
- 各event位置の不確かさがpanel最小距離や誤差予算に比べて大きければ、そのeventだけrefineする。D14全体の再solveを每refinementで実行しない。

### 4.4 微分を正しく扱う

既存の `radius_terms` が返すfixed-R微分を、現在パラメータで固定した積分mapへ代入し `dR/dx` を掛ける。adaptive scheduler、node選択、panel分割をADにかけない。

パラメータ依存のcell境界の移動項は、隣接panel間で共有し相殺するものと、真の像生成端で消えるものを導出して区別する。個々のcellを独立に微分して境界項を二重計上しない。

near-foldで巨大な `1/sqrt(v)` を生成してから小さいJacobianを掛ける実装は避ける。既存 `(m,v)` IFTまたは代数的に整理した積でtransformed derivativeを評価する。ただしtiny stateでも既存の信頼性flagを単に無視しない。必要精度を判定できなければ非成功にする。

### 4.5 現行node kernelへの必要な小変更

単に `R` と `dR/dx` を外側から渡すだけでは、現行 `VFloor` / tangency gate が先に微分を失敗扱いする場合がある。そのため `evaluate_mapped_radius` adapterを用意し、**積分に必要な変換後の量を有限な形で計算してから、その誤差を判定する**。

通常点では既存 `radius_value/radius_terms` をそのまま再利用。near-foldだけ、既存root-pair式の2×2 IFTから `dm/dp,dv_pair/dp` を求め、角幅/その微分とmap Jacobianを合わせて評価する。例えば左foldで `v_pair=(R-a)*vbar` が分離できれば、

```text
(dR/dx) / sqrt(v_pair) = sqrt(w) / sqrt(vbar)
```

（§4.2のleft-fold map）となり、巨大数と微小数の積を避けられる。vbarの符号・誤差・eventの不確かさを検査する。この整理は普通のfoldの局所算術であり、広いcellを高次moment/Taylor packetへ変換する研究経路ではない。

従来flagを一律クリアするのは禁止。root/topology自体が不確かな場合はadapterでも失敗する。非正則なfixed-R微分が大きいことと、変換後積分値が計算不能であることは分けて判定する。

## 5. 増分型の求積とsample保存

### 5.1 Fejér-IIを使用する

level `L`、`m=2^L`、`N=m-1` とし、

```text
x_k = cos(k*pi/m), k=1,...,m-1
```

を使う。endpointは含まない。次levelでは旧kが新2kへ一致し、新規は奇数kのみ。[R2]

```text
7 → 15：追加8、累積15
15 → 31：追加16、累積31
31 → 63：追加32、累積63
```

旧点の**重みは変わる**。旧sampleを新weightsで再加算する。旧integralに新sample寄与だけを足すのは間違いである。

weightsは一度生成し保持する。検証用の定義式は

```text
w_k = (4/m) sin(theta_k) sum_{ell odd=1}^{m-1} sin(ell*theta_k)/ell
```

で、全weightsが正、sum=2。少なくとも次数N−1までの多項式を厳密に積分する。実装時は固定tableまたは一度だけ生成する。各panelでFFT、Vandermonde solve、trig列を作り直さない。

### 5.2 IDとcache

node IDは浮動小数Rの近接比較でなく、`(cell_generation,panel_id,map_id,L,k)` のdyadic-angle正規形を使う。kが偶数なら `(L,k)→(L-1,k/2)` を繰り返してcanonicalにする。

旧点のR・計算値はbitwise再利用する。levelごとにcosやmapを計算し直して別のRにしない。対称点/中央点もcanonical tableから作る。

sampleに保存するもの：

- 物理R、必要ならeventからのoffset。
- `f0,fh`、要求した5方向微分、local error情報。
- root-pair / quartic warm seedの小さいsnapshot、chartとarc対応。
- numerical status、計算した精度、node内精度version。

保存しないもの：全K-rule nodeの巨大な配列、PF6 state、不要なJacobian列。普段は小さなPOD・arenaで済ませ、内側求積の再利用が必要なnodeだけside storageを持つ。

### 5.3 新nodeのroot初期値

各追加nodeについて、同じcertified cell/chartの最寄りの受理済みroot snapshotを使う。新規nodeをR順に処理し、その局所バッチでcontinuationを再利用する。

旧descending全列の最後のmutable warm stateから、挿入nodeへ無条件に飛ばない。失敗したpredictor/correctorは旧snapshotを汚さない。既存の残差・branch・discriminant等のgateを維持する。

### 5.4 panel分割の再利用の限界

**同じpanelのp-refinementは旧点を全て求積に再利用できる。一方、affine二分後の子Fejér nodeは親nodeに一般には一致しない。** ここを混同しない。

したがって初期はp-refinement中心。h分割はp収束が停滞したpanelだけにする。分割後も旧sampleは保持し、同一Rなら計算値も再利用、それ以外はroot anchor・補間予測・誤差診断に再利用する。予測値を未評価nodeの真のsampleとして扱わない。

全親点を子の求積nodeへ無理に含める非標準求積を、その都度連立方程式で作る案は採用しない。再利用不能な点の数と費用もstatsへ記録する。

## 6. 安い、数学的根拠のある誤差推定

### 6.1 なぜ積分値の差だけでは不十分か

`abs(Q15-Q7)` は、局所的な正負誤差の相殺で小さくなる。細いpeakを両方が見逃す場合もある。主指標は**補間関数そのものの差のnorm**にする。explicit interpolantの差を使う発想はGonnetのadaptive quadrature研究と整合する。[R3]

### 6.2 新しい物理評価なしで計算できるnorm

level Lのsampleを補間する次数N−1の多項式を `pL`、旧levelの補間を `pC` とする。既存点では両者が一致する。新規点で

```text
delta_k = G(x_k) - pC(x_k)
```

を計算する。`pC(x_k)` は前計算したbarycentric補間行列の積。旧点のdeltaはゼロにできる。

`omega(x)=sqrt(1-x²)` とすると、Gauss–Chebyshev-IIの多項式exactnessから、厳密演算では

```text
d_L² = ||pL-pC||²_L2(omega)
     = (pi/m) sum_k sin²(theta_k) * delta_k²
```

が**正確に成り立つ**。delta²の次数は2N−2で、N点GC2のexactnessの範囲内である。

さらにCauchy–Schwarzより

```text
|integral(pL-pC) dx| <= sqrt(pi) * d_L.
```

これで「積分値だけ偶然一致」の多くを検出できる。Jacobianも要求された各出力について同じ計算を行う。これは誤差推定に使う補間であり、V2の物理値をsample-fitの別solverへ置き換えるものではない。

### 6.3 真の残差との関係・通常モードの採用規則

未来の補間増分がnormで一定率q<1以下に縮み、pLが真のGへ収束するなら、telescopingから

```text
|I(G)-Q_L| <= sqrt(pi) * q/(1-q) * d_L.
```

ただし、直近2回の収縮を観測しただけでは、この未来条件を証明したことにはならない。

初期実装は、安全側の一段遅れの推定として `E_detail = sqrt(pi)*d_L` を基本にする。これは未来の増分が少なくとも半減するモデルに対応する。以下を併用する。

1. `abs(Q_L-Q_C)` とnorm指標の整合。
2. 直近二段のdetail normの減衰。7点に含まれる3点/1点subsetも使えるので、追加評価なしで初期historyを作れる。
3. smoothness/event/chart/finite・branch checks。
4. node error・丸めfloor以下の差を「無限に良い収束」と解釈しない。
5. 収縮が確認できない場合はmodel未解像としてp追加またはh分割。最大点数到達を成功にしない。

qの値をデータセットに合わせて調整して勝利を作らない。実装で採る収縮基準と停止条件を固定して公開し、その条件が経験的screenであることも記録する。少数のoff-grid sentinelを用いる場合は、異常なゼロdetail等の疑わしいpanelだけとし、結果もcacheする。sentinelがあること自体を形式的証明としない。

low-degree多項式ではfine ruleが既にexactでもcoarseとの差が残る。この方式はときに一段余分にrefineする。その代わり、tailの単一係数や積分差のみを採用するより健全な判定にする。

### 6.4 厳密上界を返せる条件

GがBernstein ellipse E_chi（chi>1）内で解析的で、そこで `abs(G)<=M` の検証済み上界がある場合、Chebyshev係数は `abs(a_k)<=2M chi^-k`。[R4]

Fejér-IIは正weights・sum=2・次数d=N−1のexactnessを持つ。次数dのChebyshev打切りを引き、積分と求積の作用素normを使うと、本設計で

```text
|I(G)-Q_N| <= 8 M chi^(-d)/(chi-1)
```

という保守的上界が導ける。座標変換のJacobianは既にGに含む。この上界計算自体は安い。難しいのはM、chi、physical branchの正当化である。

D14のcomplex root距離はchi候補・refinement順序のヒントにできるが、それだけでMや全特異点・chart・K積分の解析性を証明できない。実軸sampleのmaxをMと偽らない。verified envelopeを一度作れたpanelだけ再利用する。全面的なvalidated complex continuationは今回の初期スコープへ入れない。

### 6.5 誤差計算の費用

N<=63では新点への固定補間行列積を使い、大きいNでのみDST等を検討する。新規点32×旧点31×出力6なら約6000個の積和であり、「数十演算で常に無料」とは言わない。物理node削減との収支を測る。

root solve、K-rule、物理関数の再評価を誤差推定器に追加しない。table生成、allocation、trigをhot loopから外す。matrix-vectorは小さい固定ループにし、FFTライブラリの呼出し固定費を初期実装へ持ち込まない。

## 7. 最終mu / 5Jacでglobalに止める

各nodeのraw vectorを `(f0,fh,df0[5],dfh[5])` とする。既存のinternal→user chain rule後、

```text
D = pi*rho²*(1-u/3)
g_mu = ((1-u)*f0 + u*fh)/D
g_j  = ((1-u)*df0_j + u*dfh_j)/D - (D_j/D)*g_mu
```

を積分する。D_jはrho方向に `2D/rho`、その他の5物理パラメータ方向では0（uは別引数）。raw vectorはcacheし、誤差制御に必要な最終出力の線形結合を作る。

各panelのabs errorを非負で合計する。

```text
E_mu = sum_panels(E_radial_mu + E_inner_mu + E_geometry_mu
                  + E_event_mu + E_round_mu)

E_grad[j] = same accumulation for requested j

T_mu = max(mu_atol, mu_rtol*abs(mu))
T_grad[j] = max(grad_atol[j], grad_rtol[j]*abs(grad_mu[j]))
```

全ての要求出力でE<=T、かつnumerical statusが成功の場合だけ収束。相対誤差のpanelごとの単純平均、RSS、都合の良い誤差相殺は使わない。

rho微分の桁落ちは特に監視する。raw F0/Fhと微分の誤差から伝播する場合、少なくとも

```text
E_grad_rho <= (|1-u| E_dF0rho + |u| E_dFhrho)/|D| + 2 E_mu/rho
```

が必要。共同に計算した最終g_rhoの推定は相関を活かせるが、実際の算術誤差floorを除いてはいけない。必要な組立・和だけcompensated/DDに昇格する。全nodeを無条件qfにしない。

## 8. 内側誤差・event誤差を無視しない

### 8.1 K-ruleとangular rescue

radialの点をいくら増やしても、同じ内側バイアスは消えない。既存K-ruleの8/16差は既に計算しているので、boolだけでなく数値のlocal error estimateとして返す。[S2]

- 正のradial weightsに対し `E_inner = sum |w_k|*eta_k`。
- value用K gateの通過を、そのまま5Jac精度保証にしない。
- angular64 rescueも無誤差として扱わない。保存したsample列の低次/高次投影差等で追加の物理評価を抑えつつerror estimateを作る。必要精度に届かないarcだけ追加評価する。
- 同じRの内側求積をupgradeした場合はcache versionを更新し、そのsampleを参照するpanelのQとEを更新する。
- 最初から全nodeで厳しいinner精度を要求しない。ただし既存gateの一律緩和もしない。
- inner誤差が予算を支配するならradialのp追加を止める。inner upgradeがない実装では `InnerAccuracyLimited` を返し、見せかけのradial収束で `OK` にしない。

このタスクは新しいK-rule研究ではない。既存の計算結果からerror情報を外へ出すことを優先する。

### 8.2 geometryとevent位置

端点の誤差は、通常の分離した根なら残差/導関数に基づき評価し、near-foldではroot-pair IFTの条件を使う。厳密な上界を称する場合は導関数の非零下界等も必要。高精度化が必要なら元のパラメータとR/offsetから係数を再生成する。既に丸めたdouble係数をDDへcastしただけでは係数生成時に失った情報は復元できない。

単純な新生arcのevent位置がdeltaずれると、落ちる寄与は代表的に

```text
f0 ~ A sqrt(r):     O(|A| delta^(3/2))
fh ~ B r:          O(|B| delta²)
df0/dp ~ C/sqrt(r): O(|C| sqrt(delta))
```

で、Jacの方が厳しい。既存arcからの有限成分、人工panel境界の相殺は別途含む。上式の係数上界がないのに形式保証を名乗らない。

D14のevent mergeや `width<1e-11` skipは今回のradial適応化だけでは正当化されない。高精度要求でその影響が無視できない場合は明示的なlimitation/errorを返す。新solverがtopologyを形式的に完全保証する、と報告しない。

## 9. スケジューラ：全体を最初からやり直さない

最初に全active cellを7点で評価し、coarse subsetも利用してQ/Eを作る。その後、requested componentの許容誤差で正規化したpanel priorityを用いる。

```text
priority(panel) = max_requested_j E_panel[j] / T_global[j]
```

最大priorityのpanelだけ処理する。小数十panelなら線形探索、数百以上へ増えたらworkspace内の小さいheap等を使う。どちらも実測で選ぶ。初期化後のpanel単位のdynamic allocationは避ける。

更新は

```text
Qtotal += Qnew - Qold
Etotal += Enew - Eold
```

のみ。定期的にcompensated再集計して差分更新の丸めを抑える。毎回全radial nodeを走査し直さない。

処理選択：

1. inner誤差支配 → 対象nodeの内側精度だけ改善。
2. event/geometry誤差支配 → 対象event/rootだけrefine、または非成功。
3. radial誤差支配、detailが速く減衰 → 次のnested levelへ。
4. radial誤差が停滞 → 同じcell内部でh分割。D14/physical eventを跨がない。

63点を超える場合も、滑らかなら127点が二分より安いことがある。次levelの追加node数と直近detailの減少を使うのは**効率の予測**としてよいが、受理誤差条件は変えない。

これにより「最終muを一度出して、全体を最初から再計算」は不要。暫定muを更新しながら不足部分だけ修正する。

## 10. 実装ファイルと分割順序

追加候補（具体名は既存命名に合わせてよい）：

```text
adaptive_radial_types.hpp       Config / result / assurance / stats
nested_fejer2.hpp               nodes, positive weights, nesting IDs, tables
adaptive_radial_error.hpp       detail norm, roundoff/node error, global ledger
radial_sample_cache.hpp         arena, sample versions, root snapshots
fold_radial_map.hpp             affine / left / right / two-fold map
adaptive_radial_integrator.hpp  refinement scheduler
```

既存変更：

- `epoch_jacobian.hpp`：adaptive entry pointを追加。既存fixed版はbaselineとして残す。
- `radius_terms.hpp`：cache用の一回のgeometry評価結果と誤差情報を取り出す小さいadapter。旧kernelの不要な再評価を避ける。
- `holonomic_transport.hpp`：既に計算するK差を数値errorとして返す。旧関数wrapperのparityを保持。
- `prepared_geometry.hpp`：workspace/cache世代管理。別epochの古いflux sampleをそのまま使わない。
- `tests/holonomic_cpp/`：以下の独立テストとbenchmark。

実装順序：

**A. 数学部品。** node/weight・canonical cache・norm恒等式・fold mapをlensなしで検証。

**B. uniform value。** K/Jacに触らず、最新の高増光難例を一cell単位で解決。変換なし/ありと同精度で比較。

**C. LD value。** 内側error ledgerを加える。通常pointで精度設定に応じた点数減少を確認。

**D. analytic 5Jac。** transformed derivativeの安定評価と最終出力のerror budgeting。controllerのADはしない。

**E. 局所h-refinementとwarm。** pだけで解けない難例に限り導入。root anchorsの再利用と実評価回数を計測。

**F. accuracy/performance受理。** toleranceを渡す新APIとして判定。合格するまで既定router・共有.soは変更しない。

研究経路PF6/GMは凍結のまま。今回大きいTaylor stateやsymbolic coefficient生成を持ち込まない。

## 11. テストと受理条件

### 11.1 求積/cache単体

- 7/15/31/63/127/255でpositive weights、sum=2、degree N−1までのexactness。
- coarse nodeがfine nodeへbitwise一致。
- 累積physical eval数が7,15,31,63。15への追加8、31への追加16。
- 再要求/同一Rの重複評価ゼロ。内側upgradeは別counter。
- 異なるmu/Jac要求・パラメータ世代・chartのcache誤用を拒否。
- constant/odd/polynomial、smooth analytic、狭い内部peak、二端fold、zero derivative、canceling contributions、nonfinite、予算枯渇。
- 細いpeak・全sampleでゼロの多項式等で、推定器の限界を隠さない。Bounded modeで証明材料なしの成功を禁止。

### 11.2 physical correctness

- 最新の約1.2万倍の難例を全パラメータ付きでfixture化する。
- fold cellのF0/Fh/5Jacを個別に、R座標固定rule、fold変換、高精度独立積分で比較。
- `nr=2048` を単独oracleにせず、更なる精度/別規則との収束を確認。
- VBM referenceのversion/build、mass/frame写像、LD規約、Tol/RelTol、再計算を記録。
- 旧wide-planet thin arc、reciprocal chart、physical folds、soft events、near-axis、close/wide、tiny rhoを保持。
- 5Jacは独立高精度差分/解析参照を複数stepで収束確認。通常doubleの一回FDだけを真値にしない。
- 修正前の `OK` が誤りだったケースは、旧status parityを理由に失敗を隠さない。

全tolerance ladder（少なくともmu rtol=1e-3,1e-4,1e-5,1e-6,1e-8、対応するatol）で、実誤差とreported error、stop reasonを保存する。

### 11.3 誤差推定器の校正ではなく検証

学習用caseへ定数を合わせず、規則を先に固定してholdout/stressを分ける。

```text
effectivity = estimated_error / independently_measured_error
violation = actual_error > requested_tolerance while reported_converged
```

を出す。medianだけでなく最大underestimate・false convergence件数を重視する。referenceの不確かさがtoleranceを下回らないcaseを、合否の正解データへ混ぜない。

観測したfalse convergenceゼロは形式証明ではない。問題が出た場合、固定安全係数を闇雲に膨らませず、aliasing/endpoint/inner/topology/referenceのどれが原因か分解する。

### 11.4 速度

同じbuild/CPU・interleaved A/Bで、

```text
A fixed64 baseline
B fold-map + fixed rule（同accuracyの作業量を確認）
C adaptive without persistent cache（診断のみ）
D adaptive + cache（採用候補）
```

を比較する。Aが誤差条件を満たさないcaseではAの速度を勝利基準にしない。D14/topology込みwhole epochと、plan後のradial kernelを分ける。uniform/LD/value/5Jacを混ぜない。

cold＝cross-epoch reuseなし。warm＝コードcacheだけでなく、どのgeometry/rootを再利用したかを明記し、実際に位置が変化するtrajectoryで測る。static同一入力反復をstate warmの証拠にしない。

stats：

- unique/new/reused/upgraded node、accepted node、h分割で求積再利用されなかったsample。
- root cold/warm、precision promotion、K/角度評価数。
- panel数、level histogram、fold map別の点数。
- estimator・cache・scheduler・table準備時間と物理評価時間。
- p50/p90/p99、accuracy違反、budget/roundoff/inner/topology失敗。

管理費の目標はsmoothケースで全体の数%級、少なくとも節約した物理評価費用を下回ること。これは実測目標であり保証値ではない。狭い難例で追加点が必要なら固定64より遅くても正しい動作である。

## 12. 完了の定義

最低限、次を満たした時点で採用を判断する。

1. 最新の未収束64点の難例に対し、許容誤差へ収束、または理由付きの非成功を返す。false `OK` を残さない。
2. loose toleranceの通常caseで、物理評価数とwall timeが同精度baselineより下がる。
3. 再利用counterが「点を増やすたび全再計算」をしていないことを実証する。
4. 推定器は導出・仮定・実測effectivity・例外を明記する。
5. 内側K/geometry/event/roundoffの誤差をゼロと仮定しない。
6. `Estimated` と `Bounded` を区別し、保証していないことを保証済みと報告しない。

完了記録は `docs/holonomic/` にcheckpoint、`evidence/holonomic/` に全case/tolerance/stop/error/time/counterのTSVを置く。旧fixed APIは基準比較として保持する。変更は独立コミットで残し、採用の根拠なしにproduction defaultへ切り替えない。

## 13. 今回こちらで実行した検証

同梱 `validate_nested_quadrature.py` は**lensなしの独立検証**。lcbinintをbuildした結果ではない。

- Fejér-II 3〜255点：polynomial exactnessの最大数値差 `4.5531e-15`。
- 7→15→31→63：bitwise nested、unique eval `7,15,31,63`、追加 `7,8,16,32`。
- 補間差weighted L2のnode式とU係数式：最大差 `1.1102e-15`。
- Cauchy–Schwarzの増分boundに違反なし。
- foldモデル `sqrt(R)*(1+0.3R+0.2R²)` と `R^(-1/2)*(1+0.3R)`：変換後は15点で丸め精度。変換前の後者は63点でも絶対誤差約0.01865。
- 同じモデルで7点の真の誤差が既に小さくても、coarse/fine detailが残って15点を要求し得ることも確認した。最小点数最適性を主張しない。
- `U_15²` counterexample：Q15は約5.5e-29、真の積分約4.73626。sample-only indicatorを厳密保証と呼べないことを数値で確認。

## 14. 出典

### コード（今回取得した内容）

[S1] `src/lcbinint/magnification/holonomic/epoch_jacobian.hpp` @ `3f35991`。
`flux_value_integrate`, `flux_jacobian_integrate`, `rho_cancel_kappa`, `EpochValue`, `EpochJacobian`。

[S2] 同 `holonomic_transport.hpp` @ `3f35991`。
K-rule 8/16、環境flag、local convergence gate。

[S3] `docs/holonomic/checkpoint_v2_sturm_topology_20260911.md`、同 `quartic_sturm.hpp`, `cells.hpp` @ `3f35991`。D14/Sturmの役割と有限精度certificateの範囲。

最新のpure-kernel診断は本文§1.2の通りユーザー提示の結果であり、remote文書未取得。実装担当がローカル原本を保存して再現する。

### 外部研究（方法の背景。V2での速度を示す資料ではない）

[R1] V. Bozza, “Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control, optimal sampling and limb darkening”, MNRAS 408 (2010), 2188–2200. DOI: 10.1111/j.1365-2966.2010.17265.x. arXiv:1004.2796。

[R2] J. Waldvogel, “Fast Construction of the Fejér and Clenshaw–Curtis Quadrature Rules”, BIT 46 (2006), 195–202. DOI: 10.1007/s10543-006-0045-4。著者公開版 `people.math.ethz.ch/~waldvoge/Papers/fejer.pdf`。

[R3] P. Gonnet, “A Review of Error Estimation in Adaptive Quadrature”, ACM Computing Surveys 44(4) (2012), article 22. DOI: 10.1145/2333112.2333117. arXiv:1003.4629。特に§4のexplicit interpolant/normに基づく推定と、sample-based estimatorの限界。

[R4] N. Trefethen, “Convergence bounds for entire functions”, Chebfun example (2016),および Approximation Theory and Approximation Practice, chapter 8。Bernstein ellipse内の上界からChebyshev係数の減衰を導く部分。

§6.2のFejér-II上のweighted detail norm式、§6.4のpositive-weight quadrature bound、V2への誤差会計とAPI/制御設計は本計画の導出・設計である。文献のアルゴリズムをそのまま移植した実装とは区別する。
