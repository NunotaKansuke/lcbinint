# D14正実根専用solver：filtered Sturm / safeguarded Newton / trajectory warm計画

作成日: 2026-09-11  
対象: `NunotaKansuke/lcbinint` / `dev/holonomic` / isolated holonomic  
確認したHEAD: `73bd1e9040a0a51c9fdc56fc7a8125cf1c6f5f71`（D14Real実装は`a38378a`）  
位置づけ: **設計・実装指示。新C++ solverの速度・実運用精度を検証済みとする文書ではない。**

## 0. 採る方針

D14を「全14複素根を揃える問題」から、**物理半径の範囲にある正の実根を、個数と分離区間を確認して求める問題**へ変更する。

本線は次の構成とする。

```text
現在のPrimaryFrame
    ↓
C3/G4/Z3からD14の係数と誤差包含を作る
    ↓
正スケールPRSによるSturm chainを1 epochに1回構築
    ↓
対象正実軸の根の総数Nを確認
    ↓
cold: 根のある区間だけ分割してisolate
warm: 前epochの実根を予測・補正し、互いに交わらないbracketを作る
    ↓
現在の根の完全性を確認
    ↓
各根だけsafeguarded / interval Newtonで必要精度へ
    ↓
real D14 event + 既存のchart / representation event
    ↓
quartic Sturmでcell分類 → 既存adaptive radial
```

`D14Real`、構造評価、fold subresultant seed、局所`P=P_t=0`評価は再利用する。新しい汎用多倍長ライブラリ、GM/PF6、matrix/QZは作らない。production routerとfixed-`n_r` APIは変更しない。

**実根solverの完全性と、複素根由来soft cutを外した積分の精度は別問題として扱う。** 全複素根solverを消せば自動的に速く・正確になるとは仮定しない。

## 1. 現行実装から確認できたこと

以下は確認HEADのコードに基づく。性能の新しい推測ではない。

- `radial_events.hpp::solve_d14()`は全根を返し、`D14Real` mixed Aberthでも全根相互作用を持つ。qf残差検査と必要時の全根再計算が残る。
- 同ファイルの`radial_events()`は、正のreal D14候補を`probe_double_root()`で分類する。それとは別に、非実根の`Re(v)>0`から`R=sqrt(Re(v))`というsoft cutを作る。
- `physical_complex`という同じkindには、**real vで非実tの重根を持つevent**と、**非実vを実軸へ投影したsoft cut**が混在する。今回外す候補は後者だけ。
- `PreparedEpochGeometry`は全複素根を保存するL2 warmを使う。古いcell planの無変更再利用L1は既定OFF。
- `quartic_sturm.hpp`はdegree-4用の有限精度sign-margin検査である。これを次数14へ広げただけで、D14の厳密なroot countを保証したことにはならない。
- `cells.hpp::classify_cells()`には近接半径のmergeと狭いintervalのskipがある。adaptiveは`restore_physical_cuts()`で保持されたphysical foldを復元している。新経路はここで認証済みの別根を再び失ってはいけない。

根拠となるファイルは末尾の[Repo]に固定commit付きで記載する。

## 2. 数学的に何を求めるか

`v=R^2`、`C=C3(v), G=G4(v), Z=Z3(v)`と置く。

```text
H(v) = (C^2 - 4 v G) G^2 + 8 C (2 C^2 - 9 v G) Z - 432 v^2 Z^2
D14(v) = 4096 H(v)
```

根計算は定数4096を外した`H`でよい。既存のdiagnosticへ戻す際は規約を明示する。

求めるのは、少なくとも`0 < v < Rmax^2`にある**異なる実根すべて**。正のD14実根だからといって、すべてが通常のphysical foldとは限らない。局所quarticによる分類は残す。

D14は一般にvの偶関数ではない。根数を7本と決め打ちしない。次数は最大14、正実根数は計算結果に従う。Rの±対称性は、既に`v=R^2`へ畳み込んだ後である。

非実根、負の実根、全複素根に関するVieta検査は、この新しいroot contractの必須出力ではない。その代わり、**対象区間のroot countと重複のないroot enclosure**を完全性の根拠にする。

### 2.1 対象多項式と入力モデル

認証対象`H*`を「現在の`PrimaryFrame`のbinary64値`a,m0,X,Y,rho`を正確なdyadic数と解釈し、構造式を厳密演算した多項式」と定義する。フレーム変換の物理的定義は変更しない。

係数の丸め誤差は、この`H*`に対して包含する。丸めたdouble係数をDD/qfへcastしても、元の係数精度は戻らない。qfで一度丸めた多項式のroot countと、`H*`のroot countも同一だと無条件に仮定しない。

認証はあくまでこの入力モデルのD14についてのもの。最終flux、quartic topology、lens-map評価、既存radial error estimatorまで形式的に保証したという表示は禁止する。

### 2.2 有限区間とスケーリング

既存の`Rmax`契約を維持する。`Rmax^2`を確実に上回る正の2冪`B`を作り、探索を`0 < v < B`に限定してよい。`v=B*x`とすれば`x∈(0,1)`になる。Bの比較・二乗はoverflowを避けた指数演算／包含演算で行う。

`P(x)=2^{-e} H*(B*x)`とし、eは係数のレンジを整える整数とする。変数と全体係数のスケールは2冪なので、根の対応と符号を正確に保てる。現在のRmaxに入らない根は分類段階で除外する。Rmaxをまたぐenclosureは先に細くする。

`v=0`の根は別扱いで、正の根として数えない。微小な正根を`v>1e-12`等の固定cutoffで除外しない。小さい最高次係数を`1e-18*scale`等で消してはならない。次数低下・zero factor除去は厳密なzeroと確認できた場合だけ行う。

## 3. 数値型：既存D14Realを使い、認証用の小さい型だけ追加する

通常の根の予測・Newton補正には、既存`D14Real`を使う。認証計算には別の小型型を設ける。

```cpp
// proposed names; existing naming conventions may be followed
struct D14ScalarEnclosure {
    D14Real center;
    double radius;        // nonnegative, outward error bound in current scale
};

struct D14RealBracket {
    D14Real lo, hi;        // endpoint values are unevaluated dyadic sums
    D14Real estimate;
    int distinct_count;   // 1 for a finished isolated root
    bool simple_verified;
};

struct D14SturmWorkspace {
    // degree <=14: at most 15 nonzero polynomials, 15 coefficients each
    // no heap allocation in the ordinary coefficient/chain loops
    // precise fields depend on double/twofold/qf enclosure tiers
};
```

`D14Real`がhi/loだから認証済み、という扱いはしない。認証用add/mulはEFTで捨てた項を数え、外向きに丸めた誤差上界を返すか、検証可能なinterval primitiveで実装する。

中心a,bと誤差ea,ebについて、例として

```text
e_add = up(ea + eb + roundoff_of_center_add)
e_mul = up(|a|*eb + |b|*ea + ea*eb + roundoff_of_center_mul)
```

を使う。`|a|`自体も上から評価する。twofold積で省いた`lo*lo`、renormalizationの捨て項も含める。二つの近似値の差だけをroundoff boundと呼ばない。

binary64の四則演算のinterval版は、IEEE環境を確認し`nextafter`等による外向き拡張で実装できる。**D14Real演算はcorrectly-rounded binary64ではないので、結果に1 ULP足すだけの流用は禁止。** qf tierでも係数生成・PRS・評価すべての包含を維持する。

符号結果は`Negative / ExactZero / Positive / Uncertain`の4値とする。包含区間が0をまたぐのは`Uncertain`であり、`ExactZero`ではない。

精度階段:

```text
binary64 enclosure
    ↓ ambiguous sign / pivot only
D14Real enclosure
    ↓ ambiguous only
qf enclosure
    ↓ still uncertain / exact degeneracy
explicit unresolved result, or separately tagged incumbent backend
```

高精度chainは必要時に1 epochで1回構築し、以後再利用する。入力・chain生成で失った精度は、末端の符号だけ高精度にしても回復しないため、依存する係数から再生成する。

FTZ/DAZ、round-to-nearest、overflow/underflow、FMA、compilerのreassociationを管理する。認証primitiveを`-ffast-math`で壊さない。範囲不足と仮数精度不足を区別し、正の2冪scalingを優先する。

## 4. Sturm本体：まず正スケールのPRSを使う

基本列は`S0=P, S1=P'`、以降は負のEuclidean remainderを正の定数でscaleしたものとする。

通常のmulti-precision多項式除算を毎段行う代わりに、次のdivision-free pseudo-remainderを使える。A,Bはdegreeが確定した実係数多項式、`b=lc(B) != 0`、`sigma=sign(b)`とする。

```text
R := A
while degree(R) >= degree(B):
    a := lc(R)
    k := degree(R) - degree(B)
    R := |b|*R - sigma*a*x^k*B
    algebraically eliminated leading coefficient := ExactZero
    optionally normalize R by a POSITIVE power of two
Snext := -R
```

各消去でBの倍数を引き、Rを正の数で掛けるので、最終Rは`rem(A,B)`の正の定数倍である。したがって`Snext`は正しいSturm符号規約を保つ。通常の`prem(A,B)`を無条件に負にするだけでは、負のleading coefficientと次数差によって符号を間違える。

これは**符号安全な正スケールPRS**であり、最適なSturm–Habicht実装を新発明したという主張ではない。真のsigned-subresultantへの交換は、この列とcount parityを確かめてから、chain構築が実測の主コストだった場合だけ行う。[S2]

### 4.1 interval PRSの規約

- pivotの符号と非zeroを包含で確認してから除去する。
- 消去した最高次項は、同一係数を使った代数恒等式によってzeroと置ける。それ以外の小さい係数をzeroへ置き換えない。
- 予期しない次数低下が起き、次のleading coefficientを確定できなければ精度を上げる。
- 正の全体scalingだけを許す。各polyを勝手に「leading coefficientが正」へ直さない。
- degreeとscaleもchainの一部として保存する。
- chain全体を毎root・毎Newton stepに作り直さない。

現行quartic Sturmの`leading_margin > constant*eps`のみをdegree14の証明として使わない。まずこの小さいPRSについて包含が成立する実装を作る。通常演算を速くすることと、符号を確実にすることを混同しない。[S3]

### 4.2 countと境界点

端点がPの根でない区間では

```text
N(a,b) = V(a) - V(b)
```

で異なる実根を数える。ここでVはSturm列の符号変化数である。

APIはopen intervalを基準にし、根に一致する端点はone-sided variationで扱う:

```text
N_open(a,b) = V(a+) - V(b-)
```

Sの点cで最初にnonzeroになるk階微分の符号がsなら、右極限の符号はs、左極限は`(-1)^k*s`である。0次値が不確かなだけなのに、higher derivativeへ進んではならない。zeroを確認できない場合は精度昇格、または別のdyadic分割点を選ぶ。

分割点mが**厳密な根**ならroot atomを1個記録し、

```text
N(a,b) = N(a,m) + 1 + N(m,b)
```

とする。`P(m)`が小さいだけではatom扱いしない。中間Sturm項の厳密zeroと、P自身の根も区別する。

公開・保存するroot bracketはv座標、Sturm内部だけx座標とする。Bとgenerationを明示し、両座標の境界を混在させない。

同一epoch内のvariationを端点キーでcacheする。係数／入力generationが変わった後に旧Vを流用しない。

## 5. warmの核心：総数とsign bracketで全根確認を安くする

通常のsimple real rootが中心のwarmでは、rootごとのSturm countと全gap countを毎回行う必要はない。

現在の多項式について次を確認する。

1. 対象区間の異なる実根の総数が、SturmでNと分かっている。
2. 対象内に、互いに交わらないN個のbracketがある。
3. 各bracketの両端のPの符号を包含付きで確認し、符号が逆である。

中間値の定理で各bracketは少なくとも1根を持つ。全体にN根しかなくbracketは互いに交わらないので、**各bracketにちょうど1根、外側とgapには0根**である。

これで通常warmの完全性確認は、**global count 1回＋各bracketの安いPの符号評価**にできる。N個のcount-one区間が別の方法で既に確認できた場合も、その個数がglobal Nと等しければgap countは冗長である。

「前epochと根数が同じ」だけでは不十分。前epochの根が別の場所へ移る、別の根が生まれて別の根が消える、二つのNewtonが同じ根へ落ちる場合もある。必ず現在のPに対するbracketと非重複を確認する。

偶数重根はsign changeを作らない。その場合は通常warmのsign shortcutを使わず、Sturm count区間とmultiplicity経路で扱う。

## 6. cold経路

```text
make_current_polynomial_and_filtered_chain()
N := count_open(0,1)
if N == 0: return certified empty positive-root set
queue := [(0,1,N)]
while queue not empty:
    I,k := pop()
    if k == 0: discard I
    if k == 1: store isolated bracket; refine only as needed
    if k > 1:
        choose an interior dyadic split m
        reuse cached V at I endpoints; evaluate V(m)
        distribute root counts; handle exact root atom separately
        push only subintervals whose count is positive
check total isolated distinct roots == N
```

等間隔scanのsign changeから根数を推測しない。両端が同符号でも、2根や偶数重根が入る。log/linear scoutや既存のrepresentation radiusを分割候補として使うのはよいが、**区間を捨てる条件はcount 0等の根拠**にする。

分割はheap-freeのstack/workspaceを使う。最大深さ／count評価数に明示budgetを持たせる。上限到達で既知rootだけを完全解として返さない。

degree14では複素全根法よりSturmが必ず速いとは限らない。chain生成・精度昇格・近接根分離の費用を含めて比較する。clusterで二分割が支配した場合だけ、後段でcount保存型Newton cluster contractionを検討する。初回実装でANewDsc全体を再実装しない。[S1]

## 7. warm経路

`PreparedEpochGeometry`に全複素根cacheとは独立したpositive-root cacheを追加する。

```text
current coefficients and chainを構築し、current global Nを取得
previous real-root estimates / bracketsを現在のscaleへ写す
各rootに少数の実数Newton predictor/correctorを試す
現在のPに対するsign bracketを作る
sortして互いに交わらないことを確認
    ↓ N本揃う
§5の完全性条件で受理し、必要精度まで局所refine
    ↓不足、重複、偶数根、current count変化
既に確認した区間は保持し、未被覆部分だけSturm isolation
```

N本揃わない場合、sign bracketは「少なくとも1根」の証拠にすぎない。そのbracketもSturmで本数を確認し、複数根を含めば内部を分離する。未確認のbracketを1根として差し引き、gapだけ探索してはいけない。

予測は`v_new ≈ v_old - H_now(v_old)/H_now'(v_old)`から開始してよい。既存twofold値のlow limbを保持する。前後係数差によるpredictorは、それがさらに安いと実測できた場合だけ追加する。

bracket候補の幅はNewton step、旧enclosure幅、現在のrepresentable spacingから作り、必要なら倍化する。これは探索方針であり証明ではない。受理は常に現在のPの包含付き符号／countで行う。候補が重なったら広い方を独断で捨てず、そのunionをcountして修復する。

- previous rootsが0本でもcacheはvalidでよい。current Nを確認し、0からのpair birthを見落とさない。
- パラメータ変化ごとにchainを再構築する。古いchainやVはwarm certificateではない。
- warm失敗で既知の区間を全部捨てて0からやり直さない。
- `warm_attempted`と`warm_accepted_without_isolation`を分ける。cacheを渡しただけでwarm成功と数えない。
- L1 verbatim cell reuseは今回もOFF。根を現在の位置へ動かした新しいcellを作る。
- 4 epoch corpusは疎なepoch列である。より長い列なら必ず速くなるとは仮定しない。

## 8. 1根のrefinement：Newtonは提案、bracketが安全性を持つ

root countで1根と確認したIを持つ。中心mの通常Newtonは

```text
xN = m - P(m)/P'(m)
```

だが、`|P/P'|`だけを根誤差の証明にしない。

PとP'の包含が利用でき、`0 ∉ P'(I)`なら、平均値の定理から

```text
J = I ∩ (m - P(m)/P'(I))
```

はI内の根を保持する。十分縮めばJを採用する。これは存在を一から証明する用途ではなく、**既に1根を含むと分かった区間の収縮**である。

Jが縮まらない、P'(I)が0を含む、Newton点が区間外、または符号が未確定なら、二分割／countによる安全な進行へ戻る。計算したJが空なら、包含・入力・countの整合性を確認し、勝手に別rootへ移動しない。

通常sign bracketなら、符号評価だけで片側を落とせる。Sturmを毎iteration呼ばない。偶数根等ではcountによるrefineを使う。終了時は`v_estimate`だけでなく最終`[v_lo,v_hi]`を返す。

### 8.1 根の精度はconsumerから決める

全根に一律`1e-26`を要求しない。分離に必要な精度と、physical eventの位置精度を分ける。

```text
root completeness: 対象区間の全実根がbracket群に含まれる
root separation: 異なる根のbracketが交わらない
radius accuracy: sqrt(bracket)がconsumerのevent予算を満たす
```

`R_lo=sqrt(v_lo), R_hi=sqrt(v_hi)`を外向きに計算する。`R_hi_part + R_lo_part`というhi/lo点表現と、enclosure半径を別に保持する。`delta_v/(2R)`はcondition estimateには使えるが、v≈0での認証上界の代用にはしない。

ULP、近接eventの分離、現在のfold mapでnodeがevent uncertaintyに入らない条件は維持する。flux RelTolが緩いことを根拠にeventを粗くしすぎ、Phase 9.1のloose-tol失敗を再発させない。

局所`P=P_t=0`補正が必要なら既存関数を使う。ただし補正が認証済みR bracketの外の別eventへ飛んだ場合は採用しない。adaptiveがより狭いevent enclosureを要求したら、そのrootだけrefineする。

## 9. 重根・軸上・次数低下は最初から別契約にする

Sturmは異なる実根の個数を扱えるが、finite precisionで「小さいremainder」を厳密zeroとみなすことはできない。

特に構造式で**Z≡0が厳密に分かる**場合、

```text
H = G^2 * (C^2 - 4vG)
```

であり、D14はsquare-freeではない。これを一般square-free chainで押し切ると、正常な対称配置で毎回高精度化に失敗する。

実装規約:

- exact zeroの条件だけを使う。`abs(Y)<epsilon`で軸上多項式へ置き換えない。
- 既知factorを保持し、square-free部分とmultiplicityを扱える別経路にする。
- Gと`C^2-4vG`に共通根がある場合、近さでdedupせずgcd／同じ代数根の証拠で統合する。
- 一般の予期しないmultiple rootでzero/gcdを確認できない場合は`MultiplicityUnresolved`。precisionを無限に上げない。
- exact dyadic/rational arithmeticは、まず小さい独立reference/退化判定用とする。通常epochを常に任意精度へ送らない。
- 初回のC++実装で厳密退化を安全に完結できない場合、そのevent/problemだけ既存backendへ明示的に移し、`LegacyValidated`と記録する。`PositiveRealCertified`と偽らない。

通常のphysical foldに必要な`P_R`、`P_tt`が非退化という条件も維持する。D14重根を全部ordinary foldとしてsqrt-mapへ押し込まない。全体が恒等zeroになる等の構造退化は専用statusにし、空の根集合で成功としない。

## 10. 複素根soft cutを外す際の条件

real-only backendで残すもの:

- 対象範囲の全real D14 root。局所分類で`physical_complex`になったものも含む。
- `chart_p4`、`R_eq_a`、`R_eq_sqrt_m0`、必要な`L_root`等の既存representation event。
- ordinary foldへの既存radial map、arc/full-circle evaluator、近接physical fold保持。

外す候補:

- **非実D14 rootの実部投影だけを理由に作ったsoft cut**。

複素根は実半径のtopology変化を直接作らないが、実軸近くの複素特異点はradial integrandを急変させ得る。**real-root countは「根がない」を保証するだけで、「積分しやすい」を保証しない。**

まず既存全根solverのまま、投影soft cutだけOFFにするA/Bを実行する。このA/Bは積分側の影響を切り分けるもので、速度の採用値としては全根計算費を含む。

no-softでradial node/split/精度違反が増える場合、tolやestimatorを緩めて隠さない。初回はその状態でreal-onlyを既定へ昇格させない。必要なら、不確かなroot-free区間を数値的な分割hintとして渡す仕組みを別変更として追加する。例えばBernstein係数の符号判定が難しい区間などをhintにできるが、これは独立の精度証明ではなく、導入費込みのA/B対象である。

§4〜9のcertificateが成立しても、既存のradial resultの`AccuracyAssurance::Estimated`はそのままである。有限sampleのlevel差が小さいことを、無条件のstrict integral boundへ格上げしない。

## 11. event / cell / cacheの接続

### 11.1 root resultを全複素rootの代用品にしない

新しい型は例えば次のようにする。

```cpp
enum class D14RootBackend { AllComplex, PositiveReal };
enum class RootAssurance { Incomplete, LegacyValidated, PositiveRealCertified };
struct PositiveD14Result {
    std::array<D14RealBracket,14> roots;
    unsigned root_count;
    unsigned total_distinct_count;
    RootAssurance assurance;
    // domain scale, generation, zero/multiplicity information, counters
};
```

全14根用`D14Solve.roots`へ実根だけを詰め、`deg`を実根数へ変えて既存codeをだましてはいけない。Vieta/conjugacyは新backendに必要ないが、legacy backendでは維持する。

fallbackも含め、`backend_used`、`count_certified`、`reason`を記録する。real-onlyのつもりで毎回裏で全複素根を解いていないことをcounterで確認する。

### 11.2 callerはactive panelで使えるevent列を受け取る

既存のevent組立から「D14根の生成」と「その他event／cellの組立」を分離する。新しい`radial_events_positive()`またはsolver policyを使い、通常APIのdefaultを変更しない。

positive-root cacheはcaller-owned workspace／prepared stateに置く。少なくともanchor入力、domain scale、根enclosure、estimate、任意のfold t seedを持つ。認証済み根を次epochの初期値として使うが、旧enclosureを現在のroot enclosureとして返さない。

全複素cacheとの変換は明示的に行い、失敗した新stateで前回の受理済みstateを破壊しない。const input・thread-local scratch・epoch generationの既存方針を維持する。

### 11.3 distinct rootをmerge / skipで消さない

`cells.hpp`の旧absolute mergeと`width<1e-11`skipを、新しいreal-only認証経路へそのまま通さない。新event列からcellを組むhelperを共通化し、**異なる認証bracketは別boundary**にする。

同じ半径へ来たchartとphysical eventはprovenanceをまとめてもよいが、異なる根を近いという理由だけでまとめない。doubleに丸めると2境界が同値になる場合はhigh/low表現で進めるか、明示的に`RepresentationLimited`を返す。中の寄与を0にしない。

境界精度を考慮した確実な内部点を選び、既存quartic topologyを適用する。通常のmidpointがenclosureへ入る場合はroot bracketを追加refineする。`restore_physical_cuts()`はno-opになり得るが、完全なcell coverageの検査まで削除しない。

## 12. 実装順序と成果物

### A. 依存分離とsoft-cut A/B

同じ現行all-root結果から`AllComplexSoft` / `NoProjectedComplexSoft`を選べるようにする。型やflagはisolated経路だけ。全positive real eventとrepresentation cutは残す。

### B. real-only coldを独立に実装

`d14_real_sturm.hpp`に固定容量enclosure PRS/count、`d14_positive_roots.hpp`にisolation/refinementを置く。まず正確なcountとinterval契約を動かし、イベントへ接続する。新しい証明を作っていない部分は`certified`にしない。

### C. precision filterを最適化

double → D14Real → qfの依存付き再生成を実装する。各precisionを通常ルートで全部先払いしない。符号付きSturm列の次数・符号規約を変えず、EFTや固定配列で軽量化する。

### D. real-only warmを実装

current global count＋前根predictor＋非重複sign bracketのshortcutを優先する。不足領域だけisolateする。ordinary warmでrootごと／全gapごとの冗長Sturm queryを発生させない。

### E. end-to-end接続と判断

`PreparedEpochGeometry`、`radial_events`、cell helper、adaptive event精度へ接続する。通常unitと既存trajectory benchmarkを使い、checkpointとraw evidenceを残す。HMC/NUTSや新しい大規模試験環境は今回不要。

各段階を独立commitにし、未実装・fallback・不採用を明記して`dev/holonomic`へpushする。real-onlyの受理率やno-soft精度が不十分なら、既存solverをdefaultに残し、未達理由を数値で記録する。

## 13. 最小限の検証と性能の読み方

新しいunitはcount / enclosure契約に必要なものに絞る。

- positive root 0本、複数本、負根・非実根のみ、近接simple根、near-real complex pair、偶数／奇数重根、zero／分割点root、leading degree drop。
- warmで根数変化、根数は同じだが根の場所が変化、二つのNewtonの同一rootへの収束、不完全な旧seed。
- c9の近接physical fold、c92のnear-real quartic、kFull、既存high-Aを維持。

全根法のimaginary filterの結果をreal-root countの絶対oracleにしない。exact dyadicの小テストはPython/SymPyなどで照合する。既存全corpusのvalue・status・独立reference比較はそのまま使う。保存VBMとの既知の差は、新しいrootcountバグと区別する。

比較するbackendは同一HEAD・同一入力・同一compile設定で以下とする。

```text
A: incumbent D14Real all-roots + original soft cuts
B: incumbent D14Real all-roots + no projected complex soft cuts
C: positive-real cold/warm + no projected complex soft cuts
```

それぞれ`full-cold / full-warm / radial-only`を既存の計時境界で測る。全epochと初回除外のsteadyを分ける。複素rootの不要な仕事を除いた分だけでなく、Sturm、event判定、変更後のradial refinementまで含める。

最低限のcounter:

```text
coefficient_enclosure_ms, chain_build_ms, variation_ms
isolation_ms, root_refine_ms, event_classification_ms
real_root_count, chain_tier, count_queries, sign_queries
warm_attempts, warm_direct_complete, repaired_intervals
subdivision_nodes, largest_root_cluster
legacy_backend_calls, fallback_reason
radial_new_nodes, radial_splits, total_ms
```

速度は同一行の測定から集計する。中央値同士の差を厳密なstage費用としない。遅い上位行も含め、成功行だけで勝率を作らない。baselineを古いgeneric qf expansionへ戻し、現行D14Realより大きな改善に見せない。

採用条件は、root取りこぼし／false OKを増やさず、no-softの誤差契約を維持し、**countと全後処理込みのcold/warm full時間が再現して改善すること**。倍率の事前保証や、個々の小変更に一律15%の基準は置かない。

## 14. この文書で確認した範囲

独立コード`checks/holonomic/validate_d14_positive_real_plan.py`をexact rational arithmeticで実行し、33 polynomial cases、391 checksがPASSした。結果は`evidence/holonomic/d14_positive_real_plan_20260911/validation.json`に保存する。

確認したのは、正スケールpseudo-remainderの符号、独立Sturm列とのopen-interval count一致、重根／近接根／near-real complex pair／端点rootの会計、warm完全性条件の例、`Z=0`の因数構造である。D14具体例には対象区間内の正実根が8本、10本ある例も含むため、7本を上限にしない。

これは**浮動小数点enclosure実装、C++速度、lcbinint corpus精度、complex soft-cut省略の実証ではない**。その区別は最終checkpointにも残す。

## 15. 参考・確認元

[Repo] 確認HEAD `73bd1e9040a0a51c9fdc56fc7a8125cf1c6f5f71`:

- `src/lcbinint/magnification/holonomic/radial_events.hpp`
- `src/lcbinint/magnification/holonomic/d14_structure.hpp`, `d14_real.hpp`
- `src/lcbinint/magnification/holonomic/quartic_sturm.hpp`, `cells.hpp`
- `src/lcbinint/magnification/holonomic/prepared_geometry.hpp`
- `src/lcbinint/magnification/holonomic/adaptive_epoch.hpp`
- `docs/holonomic/d14_precision_and_warm_design_20260911_ja.md`
- `docs/holonomic/d14_precision_warm_checkpoint_20260912_ja.md`

基準tree: https://github.com/NunotaKansuke/lcbinint/tree/73bd1e9040a0a51c9fdc56fc7a8125cf1c6f5f71

[S1] Sagraloff & Mehlhorn, *Computing Real Roots of Real Polynomials*, J. Symbolic Computation 73 (2016), 46–86. https://arxiv.org/abs/1308.4088 ; DOI 10.1016/j.jsc.2015.03.004. 実根のみのisolation、Newtonと区間細分化、係数近似精度を連動させる研究背景。これを次数14での速度保証とは読まない。

[S2] CGAL, *PolynomialTraits_d::SturmHabichtSequence* / Algebraic Kernel documentation. https://www-sop.inria.fr/members/Clement.Jamin/CGAL/Td_doc_r3/Polynomial/classPolynomialTraits__d_1_1SturmHabichtSequence.html ; https://doc.cgal.org/latest/Algebraic_kernel_d/classCGAL_1_1Algebraic__kernel__d__1.html . signed subresultantと、代数的実根をpolynomial＋isolating intervalで表す既存の枠組み。CGALをproduction依存へ追加する指示ではない。

[S3] Shewchuk, *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates*. https://www.cs.cmu.edu/~quake/robust.html . 必要な符号だけ適応的な精度で確定する設計思想。D14のPRSに、そのままorientation predicateの誤差定数を流用してはいけない。

[S4] Kobel, Rouillier & Sagraloff, *Computing Real Roots of Real Polynomials … and now For Real!* https://arxiv.org/abs/1605.00410 . 実根isolationの実装ではcluster処理と管理overheadが重要であるという参考。今回の初回実装に同論文の全機能を要求しない。

本計画のdivision-free正スケールPRSの具体形、warmの総数＋sign bracketによる冗長query除去、既存V2への接続は、上記の数学と確認したコードを基にこの作業で整理した設計である。

## Codexへの短い開始指示

`dev/holonomic`を作業中の変更を保護して最新まで取り込み、この文書のA→Eを実装してください。まずsoft-cut除去の影響を分離し、次に正実根cold/warm backendを接続してください。D14Real・現行adaptiveは再利用し、根の総数と非重複enclosureで完全性を確認すること。既存の通常テストとtrajectory benchmarkだけで確認し、production routerは変更せず、checkpoint・raw・独立commitを残してpushしてください。
