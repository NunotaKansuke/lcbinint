# 像境界の停留値からR区間を直接認証する設計・実装計画

作成日: 2026-09-11  
対象: `NunotaKansuke/lcbinint` / `dev/holonomic` / isolated holonomic  
確認HEAD: `9ee01f1028b4cb959c22b27076ebe02bb41238b1`  
位置づけ: **設計と数式確認。新しいC++経路の速度・corpus coverageを実証した報告ではない。**

## 0. 決定事項

目的を「D14の根を求める」から、**像境界と同心円の交点数が一定なR区間を、間に実接触を見落としていないこととともに返す**へ変更する。

本線は、元の境界四次式を使った **stationary-value event planner** とする。

```text
実像境界 P(R,s)=0
    ↓
実接触 P(R,s)=0, P_s(R,s)=0 だけを探す
    ↓
接触のない領域を低次数Bernstein範囲で除外
接触のあり得る細い領域は停留点のtubeと局所Newtonで扱う
    ↓
接触を囲むbox + それ以外を覆うno-contact証拠
    ↓
実イベントR、一定topologyのcell、任意の積分用hint
    ↓
既存adaptive radial / kFull / vK / ValueFirst
```

`D14Real`と既存の局所`P,P_s,P_R,P_ss,P_sR`を再利用する。通常経路では、D14展開、14複素根、degree-14 PRS、全根Vieta検査を要求しない。旧solverは独立参照と既存productionとして残す。実装時に全epochで旧solverも実行して新方式の成功を確認する構造にはしない。

これは全画像の輪郭を細かくmesh化する計画でも、角度gridでinsideを推測する計画でもない。**接触がないことを確定できた広い領域を捨て、実接触の周囲だけを解く。** Bernstein表現は既存の有限次数多項式の厳密な基底変換であり、サンプル補間packetではない。

## 1. 前回から持ち越す事実と、修正する解釈

`9ee01f1`のcheckpoint [R1] によると、positive-realのcold認証は4524/14432行。9904行がChainPivotUncertain、4行がSplitSignUncertainで既存backendに戻る。係数包含構築・PRS構築の費用を先に払い、多数が二重計算になった。これは今回の設計を考える出発点であり、新方式の速度を予測するデータではない。

soft-cutを外したBとpositive-realのCは、同じ10行で非収束だった。理由はArcWidthUnresolved、ArcKindMismatch、InnerPhiNonpositive。6行は同じR・同じcellのfresh評価で成功し、4行はなお拒否された。[R1]

この結果から言えるのは「現行のnode evaluatorとpanel構築が、分割位置の変更に頑健ではない」ということまでである。**複素D14根、またはその近傍情報が物理的に不可欠だと証明されたわけではない。** `|D|`の谷から複素根を推定する案も、係数scale・他の因子に依存するので今回の正しさの根拠にしない。

前に挙げた補助三次式の二枝`F_±`も、そのままでは完全性を解決しない。`A>0`の一区間内に複数のzeroがあり得て、両端の同符号では捨てられない。また補助変数lambdaの実重根は、元の境界四次式の実角度重根と同一ではない。そこで今回の主役は**元の四次式の角度停留点**とし、補助三次式は付録の検討に留める。

## 2. 返すもの・返さないもの

新しいplannerの出力は次の三種類に分ける。

1. `PhysicalContact`: `(R,s)`の実接触を囲む領域、半径enclosure、chart、普通foldかどうか、局所seed。
2. `RegularRadialCell`: 接触のないR区間と、そこでのquartic crossing count / kEmpty / kArcs / kFull。
3. `QuadratureHint`: 積分を始める分割候補やnear-foldの局所scale。任意であり、topologyの証拠ではない。

**証明用box境界を全部radial積分のcell境界にしない。** それをするとboxを増やすほど初期7点の費用が増え、前段の高速化を相殺する。proof partitionとquadrature meshは別に保持し、event-freeな隣接R区間は証拠を保ったまま統合する。

実接触の個数を7個や14個に固定しない。同じRに異なる角度の二接触が起きる場合もある。`Rが近い`だけで削除・mergeしない。固定容量workspaceには明示budgetとoverflow statusを用意する。

## 3. 数学的な対象

### 3.1 元の式

`boundary_polynomial.hpp` [R2] の表現を変えない。

```text
t = tan(theta/2)
A = 1+t^2
B = (R-a)^2 + (R+a)^2 t^2
n0 = -(X+iY) R^2
n1 = R(R^2-1+a(X+iY))
n2 = a(m0-R^2)
T = n0+n1+n2 + 2i(n2-n0)t + (n1-n0-n2)t^2
P = rho^2 R^2 A B - |T|^2
```

固定parameterについてPは`degree_R <= 6, degree_t <= 4`。実数の係数配列は最大`7 x 5 = 35`個である。複素数はこの導出の記法だけに使い、実装では`Re(T)^2+Im(T)^2`を実数で構築する。

`R>0, a>0, 0<m0<1, rho>0`を通常対象とする。入力モデルは現行と同じbinary64 `PrimaryFrame`をexact dyadic値としたもの。元のユーザーparameterからframeまでの丸めが消えたとは主張しない。

### 3.2 二つの有限chart

実射影円周を次で覆う。

- T chart: `t in [-1,1]`。
- U chart: `u=-1/t in [-1,1]`。

U chartの多項式は、**係数の反転から直接**

`Q(R,u)=p4-p3*u+p2*u^2-p1*u^3+p0*u^4`

を作る。`t=-1/u`を先に作って掛け戻す実装は禁止。`u=0`での無限大と桁落ちを避ける。

各chartで`G=(P,P_s)`または`G=(Q,Q_u)`を解く。`s`はそのchartの有限座標。`dtheta/ds != 0`なので、境界上では`P_s=0`が物理的な接触条件と同値になる。

chartの端`±1`は重複する。境界上の接触を落とさないよう、局所認証には必要時に少し拡張した有限chartを使う。所有chartは認証後の規則で決める。二重検出を消すには、座標変換したenclosureと局所一意性の一致を確認する。半径近接だけのdedupは禁止。

### 3.3 なぜこれでtopologyが一定か

あるR区間において実射影円周上に`P=P_s=0`がなければ、全ての境界実根は単純で、陰関数定理により連続して動く。円周はcompactなので、根はchartの無限遠へ逃げて消えるのではなく別chartへ移る。従って境界交点数は一定である。

この命題を使うには、**円周全域・R区間全域を覆うno-contact判定**が必要。中点と両端の3回のSturm countが一致しただけでは成立しない。

レンズ位置の分母zeroも確認する。式から厳密に

`P(0,s)=-a^2*m0^2*(1+s^2)^2`

`P(a,0)=-a^2*(1-m0)^2`

となり、通常対象ではどちらも負である。従って多項式化でこの2点を実境界rootとして数えない。`m0=0,1`や`a=0`は別の入力退化であり、この議論をそのまま使わない。

### 3.4 有限な探索範囲

既存のr_max式を再利用するが、数値上界の根拠を明示する。`W=hypot(X,Y)+rho`とすると`R>a`ではレンズ写像に対し

`|zeta(z)| >= R - m0/R - (1-m0)/(R-a) >= R-1/(R-a)`。

従って`R>a`かつ`(R-W)(R-a)>1`なら像がない。正の解

`R0=(a+W+sqrt((a-W)^2+4))/2`

より外向きの`Rmax`を取り、上記不等式を外向き演算で確認する。`+1e-12`だけを一般的な包含証明にはしない。探索domainは`[0,Rmax] x RP^1`。R=0および上端の扱いもcoverから除外しない。

## 4. 主判定器：局所Bernstein表現

box `B=I_R x J_s` を正規化した`(x,y) in [0,1]^2`上で、

`P = sum_{i=0..6,j=0..4} b_ij B_i^6(x) B_j^4(y)`

とする。各`b_ij`には外向き誤差包含を持たせる。Bernstein基底は非負で総和1なので、

`range(P,B) subset [min lower(b_ij), max upper(b_ij)]`

である。全係数が同符号なら、そのboxには境界自体がない。これはサンプリング検査ではない。

同様に`s`微分の係数は

`4*(b_i,j+1-b_i,j)/width(J_s)`。

微分係数の範囲にzeroがなければ、そのboxには接触がない。境界は通っていてよい。

通常の安い排除条件は次の二つだけでよい。

```text
0 not in hull(P)   -> NoBoundary
0 not in hull(P_s) -> NoTangency
```

rangeは必要なら定数の線形preconditionerを掛けた`G=(P,P_s)`にも適用できる。ただしこの追加費用は、両方の安い判定が失敗したboxだけに払う。

### 4.1 PRSとの違い

未知の小さいpivotで何度も除算する長い剰余列はない。subdivisionはde Casteljauの凸結合であり、既存係数包含を保持する。分割自体が初期係数の不確かさを消すわけではなく、微分とbasis変換には丸め・condition問題が残る。**intervalなら自動的に速い、Bernsteinならdependencyが消える、とはしない。** [S1,S2]

`double+radius -> D14Real+radius -> local qf+radius`の階層は、必要なbox/係数生成に限定する。全domainのqf係数を最初から先払いしない。

### 4.2 実装上の重要事項

- `T`の小さい実・虚多項式と正項から直接生成する。generic `PolyFamilyR`やD14の展開を挟まない。
- 分割時は親のBernstein tensorを再利用する。全boxで物理式を一から構築しない。
- 繰り返し分割のroundoff幅が支配したboxだけ、元のframe入力から**局所座標で**再構築する。粗いdouble係数のcastでは精度は戻らない。
- 幅の縮小とともに`1/width`が増える微分の評価には、normalized-coordinate係数を使う。NewtonのJacobianにも同じ座標scaleを使う。
- 正の2冪による方程式の定数scaleはよい。R依存scaleを導入した場合の微分項を捨てない。
- exact zeroと不確かなzeroを区別する。分割上限に達したboxをNoTangencyにしない。
- 丸めモード・FTZ/DAZ・FMAの契約は既存包含primitiveを監査して再利用する。新しい汎用多倍長classは作らない。

## 5. 核心：角度停留点のtubeで、二次元の探索を一次元化する

任意のboxを二次元で最後まで細分化するのは本線ではない。P_sは角度sについて高々三次なので、通常領域では停留点の枝をまとめて認証する。

### 5.1 tubeの成立条件

`B=I_R x [s_lo,s_hi]`で次を確認する。

1. `0 not in range(P_ss,B)`。
2. 全R in I_Rに対して`P_s(R,s_lo)<0<P_s(R,s_hi)`、または一括して逆向き。

端点の条件はR方向のBernstein範囲で調べる。中点での符号だけでは足りない。

これにより、**全R in I_Rに一意の停留点`s(R)`がこのtube内にある**。parametric interval Newton

`s_new = J_s intersect (s0 - P_s(I_R,s0)/P_ss(B))`

でtubeを縮めてよい。ここでは存在は端点符号、一意性はP_ssの定符号が保証する。

### 5.2 停留値の一次元問題

`F(R)=P(R,s(R))`と置くと、

`F'(R)=P_R(R,s(R))`

`F''(R)=P_RR - P_Rs^2/P_ss`

となる。第一式に`s'(R)`の計算はいらない。第二式は必要時の曲率診断とsplit選択にだけ使い、全boxで先払いしない。

tube内で`P_R`の符号も確定すればFは単調。端点の停留値を、そこでのP_sの一根bracketから包含付きで評価する。

- F両端が同符号 → tubeに接触なし。
- F両端が異符号 → **tubeにちょうど1接触**。
- 不明符号・P_Rの範囲がzeroを含む → 下記の局所contract/splitへ。

これで、枝上の根を全体相互作用なしのsafeguarded Newtonで仕上げられる。候補stepは`R <- R-F/P_R`。stepだけを成功判定にしない。

同じ枝が区間内で2回zeroを横切る場合はP_Rの定符号条件を満たさないため、誤ってemptyとして捨てない。停留点の数・枝が変わる場所でも、Pがzeroから離れていればcheap NoBoundaryで除外できる。**停留点三次式の判別式をまた全域で解く必要はない。**

### 5.3 候補生成は証拠ではない

R中点で既存の三次根solverからP_sのreal root seedを出すのはよい。そのseedでtubeやNewton boxを作る。ただし、seed以外の角度領域もBernsteinで覆って除外する。複素根のimaginary thresholdから「停留点はこれだけ」と決めない。

## 6. tubeが成立しない場所だけ、2x2 interval Newton / Krawczyk

`G=(P,P_s)`、

`J_G = [[P_R,P_s],[P_sR,P_ss]]`。

普通foldではP_s=0、`P_R != 0, P_ss != 0`なので、このJacobianは正則。現行の局所fold Newton [R2] をそのまま候補生成に使える。

normalized box B、中心x0、point inverseに近い非特異行列Yについて、

`K(B)=x0-Y*G(x0)+(I-Y*[J_G](B))*(B-x0)`

を外向き演算で作る。

- `K(B) intersect B`がempty → 接触なし。
- `K(B) subset interior(B)`かつ選んだmatrix normで`||I-Y*[J_G](B)||<1` → 存在・一意性を受理。
- その他 → 相交部分でcontractできるときはcontract、できなければsplit。

上の二条件を併記した十分条件で実装し、point Newtonの収束だけで一意性を主張しない。[S3]

新しい汎用二次元solverを導入する必要はない。2x2、Pのbidegree(6,4)、二つのcompact chartに専用化した短いkernelにする。boxの数を制限して失敗原因を記録する。

box境界上のrootは、neighborを含む小さいenlarged boxで包含を取り直す。認証済みevent boxに完全に含まれる探索boxは、その一意性証拠を参照して終了できる。**event候補の近くというだけで周囲の未確認boxを削除しない。**

## 7. cold: 完全性をcoverで確認する

初期domainを二つのchartで覆い、次のpartitionを構築する。

```text
queue := two compact-chart domains
for each box:
    cheap NoBoundary / NoTangency
    else try stationary tube (existence for every R, then F monotonicity)
    else try local G Newton / Krawczyk
    else split the uncertain coordinate and reuse Bernstein coefficients
finish only when all boxes are:
    excluded, or covered by a unique-contact certificate,
    or resolved by an explicit exact-symmetry/degeneracy handler
```

分割方向はnormalized derivative rangeやtubeの失敗理由から選ぶ。固定の二次元gridを最後まで走査しない。探索途中に見つかった接触を中心にしたR slabを使って周辺を大きく切ってよいが、両側の未被覆領域を必ずqueueに残す。

完全性の根拠は「N本見つけた」ではなく、**探索domainを覆い、残りにG=0がないことを確認した**こと。旧D14のroot countを毎回要求しない。余分な非実角度接触、非実Rの根、lambdaだけの重根を求める必要もない。

budgetを越えたら`AtlasIncomplete`。通常V2への互換fallbackは独立オプションとしてよいが、時間・件数・provenanceを隠さない。新経路単体のcoverageに数えない。caseごとのq/rho heuristicで最初から旧solverに振り分けることを今回の解決策にしない。

## 8. warm: 根のseedだけでなく、除外済み領域も再利用する

`PreparedEpochGeometry`と独立に、またはその追加メンバとしてcaller-owned `RadialEventAtlasCache`を持つ。保存物は実接触のbox/seed、no-contactのpartitionとmargin、入力generation。旧全14複素根を要求しない。

次epochでは:

1. 現在の35係数相当を構築する。
2. 旧eventの`(R,s)`から、現在のGに対して少数の2x2 Newton stepを試す。
3. 現在の式でevent boxの存在・一意性を再確認する。
4. 旧no-contact領域のmarginを現在の係数で再確認する。
5. 証拠が保てなかったbox、移動event周辺、Rmaxの増分domainだけ再分割する。
6. 全domainが再び覆われたら現在のplanとしてpublishする。

旧certificateの再検査を省略しない。新しい接触の生成/消滅、別々のseedの同一rootへの収束は未被覆領域または失効証拠として残る。

### 8.1 cheap margin reuse

旧boxで`P_old >= margin > 0`が確認済みで、現在の係数差を同じboxへrestrictionして

`sup_B |P_now-P_old| < margin`

を確認できればP_nowも正である。P_sによる除外も同様。これは古いtopologyを無条件に流用するL1ではなく、**現在の多項式に対する除外証明を更新する**こと。

係数restrictionは線形。差の範囲はnew/old係数生成の誤差も含める。差が小さいという期待から包含半径を減らさない。全てのboxへ大きい共通global誤差を適用するより、必要なboxだけ局所restrictionする。

matrixを通すなら固定小行列を事前生成する。deep treeを毎epoch全再構築するより、失効したbranchだけ更新する。全leafの点評価だけで0.3msを超えるならその費用も失敗として報告する。

### 8.2 event predictorの式

以前の普通foldでparameter方向を使う場合は、

`dR = -(P_p * dp)/P_R`

`ds = -(P_sR*dR + P_sp*dp)/P_ss`

で予測できる。5方向ADを予測のためだけに新たに先払いせず、最初は**現在のGを旧seedで評価するNewton**から始める。補正途中もhi/loを維持し、double basin presearchへ落とし直さない。

cacheの正しさは現在の入力と証拠で決める。訪問順・thread順・運動量によって異なる近似値を受理しない。原点が同じでも別trajectoryに跨ぐstale stateはgenerationで区別する。

## 9. 軸上・同時接触・高次退化

- **exact Y=0**ではPはsについてeven。`P=p0+p2*s^2+p4*s^4`、`P_s=2s(p2+2p4*s^2)`。s=0の枝と、`s^2=-p2/(2p4)>0`の枝を直接扱える。p4を割れない場合は別chart/局所boxへ。小さいYをzeroにしない。
- 正負sの接触が同じRになる場合は、exact symmetryを証拠として一つのradial eventに**二つのcontact**を関連付ける。crossing jumpは±4の場合がある。rootを一つ落とした扱いにしない。
- chart seamは§3.2の所有規則と重複認証を使う。`p4=0`の単根crossingは通常のchart eventであって、P=P_s=0とは区別する。
- `P_R=0`または`P_ss=0`の真の高次接触には、普通foldの一意性証明・平方根germを使わない。`SingularContact`を明示する。Pが非zeroなら単なるstationary-point退化なので除外してよい。
- exact特殊例を後から対応する場合は、小さい局所deflationを個別導出する。今回のordinary-fold経路が全singularityで必ず終わるとは主張しない。[S4]

## 10. topologyから積分へ渡す境界を一貫させる

前回のsoft-cut除去診断では、D14が返した半径とadaptiveが補正した半径が異なり、診断時にもcell IDの取り違えを避ける必要があった。[R1]

新plannerではeventを**一つの所有オブジェクト**にする。

```cpp
struct ContactAnchor {             // proposed API
    EventId id;
    Chart chart;
    D14Real R_center, s_center;
    OutwardInterval R_range, s_range;
    LocalFoldData local;
    Generation generation;
};
struct RadialAbscissa {
    EventId anchor;
    D14Real directed_offset;       // eventからの距離を先にdoubleへ潰さない
    Side side;
};
```

cell、fold map、root-pair初期値、inner error ledgerが同じanchorを参照する。eventを追加refineしたらversionを更新し、そのeventに依存するsampleを保持可能か明示判定する。古い幾何の値を新しい重みだけで流用しない。

event boxのR-rangeを分離したあと、その間の内部Rで既存quartic Sturmを実行する。ここで初めてkEmpty/kArcs/kFullを付与する。NoTangencyは「empty」を意味しない。

chart_p4、R=a、R=sqrt(m0)、axis L_rootなど既存のrepresentation cutは初回実装では残す。物理eventの証拠と混同しない。旧merge/skip toleranceで別のcontactを消す経路を通さない。

同じR-rangeへ投影する複数contactはgroupとして扱う。近いが異なるeventを分離できない場合、thin bandを無視するのではなく`EventClusterUnresolved`にする。

## 11. soft cutに頼らないためのnear-fold評価

単にcomplex cutを消すだけでは、前回の10行を再発させる。新plannerに不要な複素情報を求め直すより、**補正済み境界・root-pair・sample座標を一致させる**ことを実装範囲に含める。

### 11.1 端点を別々に作らない

普通foldの近くでは、angle chartの二根を

`s_± = M ± sqrt(w)`

とする。wはradialの`v=R^2`とは別の量である。四次係数p0..p4に対して

```text
E = p0+p1*M+p2*(M^2+w)+p3*(M^3+3*M*w)
    +p4*(M^4+6*M^2*w+w^2)
O = p1+2*p2*M+p3*(3*M^2+w)+4*p4*M*(M^2+w)
```

が共にzeroなら二根になる。foldのw=0では

`J_(M,w)(E,O) = [[0,P_ss/2],[P_ss,P_sss/6]]`

で、detは`-P_ss^2/2`。**普通foldなら端点各々のIFTが特異でも、この系は正則。** 既存root_pairの実装を使い、anchorとsmall offsetを保持した局所Pから解く。

初期値は`w ≈ (-2*P_R/P_ss)*(R-Rf)`を使えるが、これはpredictorであって受理条件ではない。最終E/O残差、wの非負性、期待crossing数との整合を確認する。

arc幅は、有限chartで向きを確認した上で

`delta_theta = 2*atan2(2*sqrt(w), 1+M^2-w)`

から計算する。近い二つのatanを引かない。wに誤差包含があるなら、それを幅の不確かさへ伝播する。以前の`width > de+dl`を無条件に解除するのではなく、**同じ保証対象に対する、相殺の少ない評価と誤差model**へ置き換える。

LDは既存vKをこの同じM,wで評価する。ValueFirstのgrad品質分離は維持する。値が有限というだけでbranch誤りを未認証grad扱いにしない。

### 11.2 積分hintと証拠を分ける

plannerが検証に使ったR slab端や、near-contact tubeの局所scaleを少数の初期panel hintとして渡してよい。ただし全leafをhintにしない。hintを消しても、node evaluatorは正しい値か理由付き未収束を返す。

Fがzeroでなく小さい場所について、`|F/F'|`や`sqrt(2|F/F''|)`は、必要条件が成立する場合の局所変動scaleの推定に使える。**複素rootの存在・距離の証明、topologyの証明、誤差上界とは呼ばない。** 既存adaptiveのrefine/split判定が精度を管理する。

eventの位置が不足している場合は同じcontactだけrefineする。nodeを何万点追加して解決しようとしない。refineできなければEventLocationLimited。sampleをeventへ丸める、epsilonだけ外へずらす、tiny intervalを無言で0にする変更は禁止。

uncertain radial stripの寄与を上界で扱う必要がある場合、`0<=intensity<=1`に基づくannulus面積上界を使えるが、tiny rhoでは大きくなる。これを安い近道だと仮定せず、通常はevent enclosureを局所的に縮める。

## 12. 新APIとファイル配置案

既存production APIを置き換えず、次の内部research APIを追加する。

```cpp
RadialAtlasResult build_radial_event_atlas(
    const PrimaryFrame&, const RadialAtlasConfig&,
    RadialAtlasWorkspace&, RadialEventAtlasCache*);
TopologyResult classify_cells_from_atlas(
    const PrimaryFrame&, const RadialAtlasResult&);
```

候補ファイル:

- `boundary_bernstein.hpp`: 小さいreal coefficient生成、tensor範囲、restriction、derivative。
- `radial_stationary_atlas.hpp`: tube、2x2認証、coverage管理、cold/warm。
- `radial_contact.hpp`: EventId、anchor/enclosure、局所fold受け渡し。
- 既存`adaptive_epoch.hpp` / `prepared_geometry.hpp`: 明示opt-in adapter。

再利用: `D14Real`, 既存のoutward primitive、`local_fold_quantities`, `boundary_quartic(_dp)`, `quartic_sturm`, `root_pair`, `vK`, nested Fejer workspace。

`AtlasStatus`: Complete / AtlasIncomplete / SingularContact / EventClusterUnresolved / ArithmeticUncertain / RepresentationLimited / BudgetExceeded。

`Complete`は対象frameとdomainに対する実接触coverの完了を意味する。最終muのBounded保証へ自動昇格しない。旧solver fallbackの結果には別provenanceを付ける。

## 13. 費用設計と採用条件

狙って減らすもの:

- 全14複素根の探索・仕上げ・相互作用。
- 全根certificateとcomplex-real投影用root精度。
- PRSのpivot・global係数intervalの連鎖的膨張。
- warmで済んだ「接触のない領域」の探索やり直し。

代わりに払うもの:

- 小さいbivariate coefficientの生成。
- no-contact cover / stationary tube / 局所2x2認証。
- eventとcellの整合を保つ局所算術。

**この代替費用が小さいとは未検証。** 特にthin sourceでbox数が増える、near-contact tubeが作れない、warm margin検査が全域で失効する場合は遅くなり得る。Newton候補だけの時間を成果にしない。

workspaceは連続メモリ・容量再利用とする。nodeごとのheap、毎boxのstd::function、全boxのqf、係数の再構築を避ける。proof treeにはsmall integer IDを使う。旧geometryの深い不要treeは現在のmarginで上位nodeへcoalesceする。

採用は同一条件のfull-cold/full-warm全費用で判断する。radial-onlyは副作用の切り分け用。medianを別々に引き算して排他的stage時間としない。root数、box数、tier数、fallback件数を併記する。多数caseでold D14も実行している結果を「D14不要」と報告しない。

## 14. 実装順序

**A. 多項式と局所証拠kernel。** 二compact chart、35係数生成、Bernstein hull/derivative、tubeとKrawczyk。通常foldとexact-axis対称接触を区別する。付属exact確認コードを読む。

**B. cold cover。** D14なしでdomainをexcluded/unique contactへ分ける。旧D14は計時外の照合に使う。topology countの一致だけでcover完了にしない。未処理boxを明示する。

**C. warm proof更新。** event predictorと旧no-contact marginの再確認を実装する。seed補正だけ実装してwarm完了としない。必要箇所だけ再分割する。

**D. 共通event anchorとnear-fold root-pair adapter。** 旧soft-cut除去の10行を既存diagnosticの同じR/cell IDで見る。event修正前後の境界を混在させない。新hintを加えるだけで原因を隠さない。

**E. 既存adaptiveへ接続し、既存trajectoryで測定。** incumbent、atlas+near-fold adapter、必要ならhint on/offを同じ許容誤差で比較する。既存unitと既存corpusを使い、新しいHMC/NUTSや巨大な試験環境は作らない。

各段階を独立commitにする。現在のproduction router、fixed-n_r経路、PF6/GM、`.claude/`、他セッションのartifactは変更しない。未完成項目と遅い部分を記録し、性能不足を隠すcase routingは入れない。

## 15. 必要最小限の確認と報告

既存test/benchmarkへ必要な追加だけを行う。

- 係数・reciprocal・tube/Krawczykの小さい局所test。exact source方程式との差を確認。
- 既存c9/c92、close folds、axis、seam、kFull、soft-cut除去10行の再確認。真のsingular inputは明示statusでよいが、ordinary foldをsingular扱いして隠さない。
- 既存14,432行のfull-cold/full-warm/radial-only、coverage、以前からの3参照不一致と新規不一致を分離。
- `box_created/excluded/contracted`, tube成功数、event数、warm再利用/失効数、precision tier、旧D14 call数。成功した短いsubsetだけの高速化を全体としない。

全domainのproof構築は計時内。確認用の旧solver呼び出しは計時外とし、最終性能実行では不要にする。全値・微分の厳密保証、全singularity対応、採用可能速度を今回の計画だけで達成済みと主張しない。

## 16. 補助三次式案の位置づけ

既存の恒等式は有用だが、それだけでは「低次数化して解決」にならない。[R3]

`f(v,lambda)=v*lambda^3+C*lambda^2+G*lambda-4Z`

`A=C^2-3vG`, `B=2C^3-9vCG-108v^2Z`

`27*v^2*(D14/4096)=4*A^3-B^2`

A>0の停留値は

`27*v^2*f(v,lambda_±)=B ∓ 2*A^(3/2)`。

A<0を区間全体で確認できれば、v>0のその区間にはD14実根はない。これは任意の安いslab exclusionとして将来A/Bできる。しかしA=0を全域で解き、F_±の全zeroも認証すると別のglobal solverが増える。さらにlambdaは物理angleでない。

**初回実装ではこの補助global solverを作らず、P_sという本物の角度停留点を使う。** 「停留値の微分では枝の微分項が消える」という良い部分を、§5で直接引き継ぐ。

## 17. 今回こちらで確認した範囲

付属`validate_radial_stationary_atlas_plan.py`で、境界多項式のbidegree、reciprocal変換、レンズ位置での非zero、exact-axis対称性、E/O恒等式とfold Jacobian、補助三次式恒等式、Bernstein微分/半分割、単純なKrawczyk例、端点root countだけでは見落とす例の27チェックを確認した。

**未確認:** 浮動小数点包含kernel、新plannerのglobal cover実装、corpus coverage、cold/warm速度、soft hintなしの実運用精度。独立代数確認をC++成功の代用にしない。

## 18. 参照資料

### 現行repo（設計の基礎とした実装・実測）

- [R1] `docs/holonomic/d14_positive_real_checkpoint_20260912_ja.md` at `9ee01f1`: 前回のPRS固定費・認証率・10行のsame-cell診断。
  https://github.com/NunotaKansuke/lcbinint/blob/9ee01f1028b4cb959c22b27076ebe02bb41238b1/docs/holonomic/d14_positive_real_checkpoint_20260912_ja.md
- [R2] `boundary_polynomial.hpp` at the same commit: Pの定義、parameter微分、local_fold_quantities。
  https://github.com/NunotaKansuke/lcbinint/blob/9ee01f1028b4cb959c22b27076ebe02bb41238b1/src/lcbinint/magnification/holonomic/boundary_polynomial.hpp
- [R3] `d14_structure.hpp` / `d14_lifted.hpp` at the same commit: C3/G4/Z3とcubic discriminant lift。
  https://github.com/NunotaKansuke/lcbinint/blob/9ee01f1028b4cb959c22b27076ebe02bb41238b1/src/lcbinint/magnification/holonomic/d14_structure.hpp
- [R4] `adaptive_epoch.hpp` / `adaptive_radial.hpp` / `cells.hpp` / `prepared_geometry.hpp` at the same commit: event補正、mapped node、root cache、Sturm topology、ValueFirst。
  https://github.com/NunotaKansuke/lcbinint/tree/9ee01f1028b4cb959c22b27076ebe02bb41238b1/src/lcbinint/magnification/holonomic

### 外部一次資料（一般手法の背景。lcbinintでの速度の根拠ではない）

- [S1] Mantzaflaris, Mourrain, Tsigaridas, *Continued Fraction Expansion of Real Roots of Polynomial Systems*, 2009. Bernstein型subdivision・実根の局所domain削減の背景。
  https://arxiv.org/abs/0905.3993
- [S2] Farouki, Neff, *On the numerical condition of Bernstein–Bézier subdivision processes*, Mathematics of Computation, 1990. subdivisionの線形写像とconditioningの注意。
  https://research.ibm.com/publications/on-the-numerical-condition-of-bernstein-bezier-subdivision-processes
- [S3] Krawczyk, *Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken*, Computing 4, 187–201, 1969. interval Newton/Krawczykの背景。
  https://doi.org/10.1007/BF02234767
- [S4] Imbach, Moroz, Pouget, *Numeric certified algorithm for the topology of resultant and discriminant curves*, 2015. 局所特異性を含む数値認証とsubresultant deflationの背景。
  https://arxiv.org/abs/1412.3290

本文の専用組合せ・cold/warm API・費用判断は今回の設計提案であり、上記論文にこのsolverがそのまま掲載されているとは主張しない。
