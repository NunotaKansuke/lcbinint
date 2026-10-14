# D14専用精度カーネルとtrajectory warm-startの設計・実装計画

作成日: 2026-09-11  
対象: `NunotaKansuke/lcbinint` / `dev/holonomic` / isolated holonomic  
確認したコード: **`26593fa005ac0c82eb13c11dd7c7ef79dd976867`**  
文書の位置づけ: 設計。下記の最適化を実装・計時済みという報告ではない。

## 0. 結論と今回の範囲

本線は **「D14専用のtwofold数値型」＋「演算の役割ごとの精度配分」＋「前epochの根を精度を落とさず修正するwarm solver」** とする。

単に `__float128` を `double hi,lo` のクラスへ置き換えるだけでは不十分である。現行コードには既に `DD` があり、通常のD14仕上げ反復は `aberth_d14_struct<DD>` で動いている。今回減らすのは、次の三種類の仕事である。

1. 係数生成・検査・型変換で先払いしているqf演算と、一律に高精度で処理する根間相互作用。
2. warm seedの低位部を捨てたうえで実行するdouble予備探索、および全根を同じ微小stepまで回し続ける反復。
3. D14の後に行うphysical-foldの三次根探索、topology midpointで座標を捨てるためだけに行う四次根探索。

**採用する数学・精度契約を弱めず、作業量を減らす。**

D14全根集合、近接したdistinct physical folds、complex soft events、`kFull`、Sturm補修、same-R retry、nested radial sample再利用、`None / Strict / ValueFirst` は維持する。今回、radial誤差推定器やHMC/NUTSは改変しない。L1の旧cell-plan無検査再利用も有効化しない。

通常経路の目標は **qfの係数生成・検査・cast往復を不要にすること**。一方、実際に難しいclusterに対する既存qf backstopは残す。新しい型を使っただけでqfを全面禁止してはいけない。

本MDはrepoへアクセスできる実装担当に対して自己完結した作業仕様である。付属の `check_derivations.py` は数式確認用で、実装に必須の外部部品ではない。

---

## 1. 現状の確認と、まだ推測であること

### 1.1 コードから確認した事実

| 箇所 | 現状 | 今回の狙い |
|---|---|---|
| `dd_real.hpp` | twofold `DD` は実装済み。実除算はq1/q2/q3を毎回計算 | 既存DDとの比較を前提に、mixed演算・複素補正・残差積和を専用化 |
| `radial_events.hpp::solve_d14` | double予備探索の上限はcold 200 / warm 60 | basin探索と最終精度を分離。warmでは予備探索を通常不要にする |
| 同warm seed生成 | 保存したqf根をdoubleに落とし、仕上げDDへ戻す | hi/loを持ったまま新しい多項式に対するcorrectorへ渡す |
| `d14_structure.hpp` | `C3/G4/Z3` の構造評価は既にある | 構造式自体は維持。実係数・整数倍・積和を専用演算にする |
| 同Aberthループ | 14根すべてで、他13根との逆数をT精度で評価 | 遠い根の相互作用はdouble、危険な差だけtwofold以上 |
| `d14_expanded_from_struct` | qfの可変長vectorで小さな多項式積を複数回構成 | 固定容量の係数生成。warmでは不要な展開・変換を省く |
| `d14_worst_res` | DD結果をqfへ戻して全根の残差を検査 | 検査だけのqf往復を、根拠付きのtwofold検査へ置換 |
| defaultの全根検査 | 主に残差と第1Newton和のsanity check。強い候補検査は別関数 | 既存sanityを「完全証明」と呼ばない。新しい停止には全根包含検査を用意 |
| `double_root_probe` | `P_t` の三次式を解いてfoldのstationary seedを取得 | 単純な二重根ならsubresultantの一次式から取得 |
| `quartic_topology` | Sturm count取得後もAberth、不一致ならroot isolation。座標は破棄 | countだけの呼出しと、端点座標が必要な呼出しを分離 |
| `PreparedReuseConfig` | L2 warm D14はON、L1 cell reuseはOFF | L2を強化。L1を勝手にONにしない |

参照は末尾の S1–S8。特に「現行D14は全反復がqfだから遅い」という前提では設計しない。

### 1.2 古いprofileから得られる手掛かり

Phase 8の記録ではcold 108回に対してdouble予備探索21,600 sweep、warm 324回に対して19,440 sweepであり、それぞれ200/60の上限に達していた。DD sweepも平均約15回で、「ほぼ3回」というコードコメントより多い。

これは **予備探索の終了条件や精度要求が実際の算術限界と合っていない可能性** を示す。ただしPhase 8には旧grid等も含まれる。数値を最新実装の費用内訳として流用しない。今回の最初の短いprofileで、同じ現象が残るか確認する。

### 1.3 最新trajectory数値の扱い

ユーザーから提示された数値は、steady-stateのradial-only `0.1591 ms`、full-cold `0.7927 ms`、full-warm `0.6289 ms`。全14,432行でstatus一致、warmとradialの最大相対差 `8.04e-11` という報告である。

本書作成時点では、そのtrajectory `summary.json` は確認したremoteから取得できなかった。実装担当はローカルの元evidenceとHEADを記録すること。

**異なる分布の中央値を引いた `0.6289−0.1591` は、topology時間の中央値ではない。** また4点trajectoryの後半3点の中央値は、長いlight curveの漸近速度を証明しない。本書はそれらを削減量の確定根拠に使わず、各rowのexclusive timerとpaired差を使う。

---

## 2. 最終構造

```text
物理入力 → incumbentと同じPrimaryFrameの数値モデル
          ↓
D14Model<D14Real> : C3/G4/Z3、必要なら展開係数、係数誤差情報
          ↓
cold                                   warm
doubleでbasin探索                       前epochのhi/lo根・D'・分離情報
達成可能な精度で終了                    ↓
          ↓                            predictorまたは直接corrector
          └──────────────┬─────────────┘
                         ↓
D14専用corrector
  D,D'・root update : twofold中心
  根間相互作用 S   : double中心、近接差だけtwofold以上
  終了判定        : root/clusterごと。止まった根も相互作用に残す
                         ↓
現在の係数に対する全根検査・event分類
  普通の根 : 安い検査で完了
  難しいcluster : 局所recenter / 局所昇格
  なお曖昧 : 既存の検査付きbackstop
                         ↓
RadialEvent { radius_hi, radius_lo, uncertainty, stationary_seed, root_id }
                         ↓
新しいcell planをSturm countで分類
                         ↓
現行adaptive V2
```

**前epochの結果は初期値であり、今epochの証明ではない。** 係数・event・cellは現在の入力に対して更新する。

---

## 3. D14の数学表現

既存の記号を維持する。`v=R²`、`m=m0`、`x=X`、`y=Y`、`h=rho²` とし、

\[
\beta=x^2+y^2-h,\quad d=v-1,\quad e=v-m.
\]

\[
L=vd+a^2e,\qquad U=aed+xL+av\beta,
\]
\[
C=vd^2+a^2e^2+v(v+a^2)\beta+2axv(3m-1-2v),
\]
\[
G=U^2+y^2L^2-4aexC-4a^2e^2v(4x^2+y^2),
\]
\[
B_2=[am+(x-a)v]^2+v^2(y^2-h),\qquad Z=a^2(1-m)y^2(v-m)B_2.
\]

\[
\widehat D=(C^2-4vG)G^2+8C(2C^2-9vG)Z-432v^2Z^2,
\qquad D_{14}=4096\widehat D.
\]

`C,G,Z` は次数3,4,3。根探索のhot loopは `Dhat,Dhat'` を使える。4096は根とNewton比を変えないが、残差のscaleも同じ規約に統一する。検査だけ元の4096倍を使う場合の換算を明示する。

### 3.1 係数は固定容量で作る

既存構造式の係数生成を `template<class T>` にする。最初の通常型は `D14Real`。

```cpp
struct D14Blocks {
    std::array<D14Real, 4> c;
    std::array<D14Real, 5> g;
    std::array<D14Real, 4> z;
};
struct D14Polynomial {
    std::array<D14Real, 15> a;  // 明示的にascending
    std::uint8_t degree;
};
```

小さい多項式積のためのheap allocationをなくす。`C*C`、`G*G`、`Z*Z`、`v*G` などは各builder/評価内で一度だけ作る。モニックな `C` の先頭係数1、実係数の加算、2の冪によるscaleを専用化する。

「構造式だから丸め誤差ゼロ」「必ず相殺しない」とは扱わない。`beta` やGの項同士にも相殺はある。係数の中心値と必要な誤差情報を、元の入力から生成する。

### 3.2 入力精度の契約

初期版は **incumbentが採用しているbinary64 PrimaryFrameを数学モデルの入力** とする。qf版も同じ入力から再生成し、型を変えた比較で物理座標の定義まで変えない。

raw `LensParams` → primary frame変換自体の丸めを後に改善する場合は別変更にする。すでにdoubleで丸めた係数をtwofoldへcastしても、係数生成時に失った桁は回復しない。

### 3.3 scalar 14次と構造式の使い分け

coldの粗いbasin探索では、固定容量の展開係数に対するdouble Hornerを使ってよい。最終補正は原則としてtwofoldの構造評価。前回遅かった「全root・全iterationでHornerを試し、拒否されたら同じ仕事を構造評価で再計算する」構成は再採用しない。

構造評価自体をさらに安くするため、`C,C',G,G',Z,Z'` をfusedで生成する。実係数を複素数 `(a,0)` にして汎用複素乗算させず、実scale・実加算の専用関数を使う。compilerが既に消している演算は改善実績に数えない。

---

## 4. 専用数値型 `D14Real`

### 4.1 表現と担当範囲

```cpp
class D14Real {
    double hi_;
    double lo_;
public:
    // 実数値は hi_ + lo_ の未評価和。
    // コンストラクタ・算術の出口で正規化する。
    static D14Real from_double(double);
    static D14Real from_parts(double hi, double lo);
    double high() const;
    double low() const;
    double rounded() const;  // 意図したprecision lossだけに使用
    bool finite() const;
};
struct D14Complex { D14Real re, im; };
```

これは新しいIEEE浮動小数点規格ではなく、D14の小さな算術集合に限定したtwofold型である。おおむねdoubleの倍の仮数情報を持てるが、113-bit binary128と同じ丸め規則・指数範囲・全演算精度を保証するものではない。[R1–R3]

既存 `DD` を削除・全域変更しない。まず内部EFTを再利用して独立型を作り、D14 kernel専用にmixed演算と複素補正を足す。名前だけ変更したクラスを「高速化」として採用しない。

### 4.2 演算面を限定する

必要な演算は、加減乗算、実数倍、2の冪scale、正規化、比較、finite判定、複素逆数/除算、非負値の平方根に限定する。一般的な三角関数・log・pow・任意精度allocator・式木・自動昇格operatorは作らない。

用意する専用関数:

```text
add / sub / neg / mul
add_double / mul_double / scale_pow2
square / dot2 / dot3 / complex_mul
complex_mul_double / complex_scale_real
complex_div_checked / reciprocal_checked
abs_upper / abs_lower / norm2_screen
sqrt_positive_checked  // 最終event生成等で必要な時だけ
```

高精度化の選択は型のoperator内部ではなく、**係数生成、D/D'評価、相互作用S、根補正、最終検査**の境界で行う。

### 4.3 EFTと丸めのルール

TwoSum、FMAによるTwoProdなど既知のerror-free transformを基礎にする。double-word算術の演算ごとの誤差解析はR1/R2を参照する。既存DDの演算をそのまま使用する段階では、その演算に対応する上界だけを使用する。[R1–R3]

- `lo` は表現の一部であり、真値に対するerror boundではない。
- FastTwoSumの前提を、使っているアルゴリズム全体で満たす。未正規化入力を勝手に許さない。
- hiもloもfinite検査する。汎用 `qfinite_(const T&) -> true` に依存しない。
- negative sqrt / 0除算 / NaNを0に置き換えない。
- overflow/underflowの危険はscaleまたは明示的な失敗で処理する。
- rounding modeはround-to-nearestを前提にする。EFT区間のFTZ/DAZは無効化し、scope終了時に戻す。
- `-ffast-math`、再結合、意図しないFMA contractionでEFTを壊さない。実装targetのcompiler設定を明示し、必要なら `-ffp-contract=off` と明示 `std::fma` を用いる。pragmaが効いたと思い込まない。

### 4.4 除算を目的精度に応じて終える

既存DDの実除算はq1/q2/q3を常に作る。D14ではまず、scaled doubleの複素商 `q0` を作り、

\[
r=a-bq_0
\]

をtwofoldの差分積和で評価する。その残差を使い、必要なら補正 `q1 ≈ r/b` を加える。

候補商qについて、`b`の絶対値の下界が正で、残差rの評価誤差をErとして囲めるなら、

\[
\left|\frac{a}{b}-q\right|\le\frac{|\widehat r|+E_r}{\operatorname{lower}|b|}
\]

が使える。a,b自体の不確かさがある場合はその項も含める。

この上界が根更新の演算誤差予算に入れば、さらに商の低位桁を作らない。入らないときだけ次の補正を追加する。q3を無条件に削る変更ではない。

`dot2/dot3` は残差を作るための短い積和に使う。通常のDD multiply→addで毎回正規化する構成との比較対象であり、**一度の最終正規化だけで何項でも正確になる**とは仮定しない。

### 4.5 指数範囲と仮数精度を分離する

D14Realはbinary128の広い指数範囲を代替しない。まずモデル単位のpower-of-two scalingで通常レンジへ入れる。特にWeierstrass分母の積は、必要に応じて `(mantissa, exp2)` で保持する。

`double + float` や任意の80-bit型を「ビット数が少ないから速い」として導入しない。CPUの実命令数・conversion・alignmentまで含めないと利点は判断できない。

---

## 5. 最も有望な精度配分: 根間相互作用だけを安くする

### 5.1 現在の費用

次数14のAberthは各rootについて13個の

\[
S_i=\sum_{j\ne i}\frac{1}{v_i-v_j}
\]

を計算する。現行の全root更新は1 sweepあたり182個の根間逆数をT精度で処理する。

ここを「D,D'と同じ精度でなければいけない」とはしない。高精度残差と遠距離相互作用は役割が異なる。

### 5.2 数学的な根拠

\[
g=D/D',\qquad w=\frac{g}{1-gS}.
\]

g,Sの摂動をδg,δSとすると、厳密演算で

\[
\widetilde w-w=
\frac{\delta g+g(g+\delta g)\delta S}
{(1-gS)[1-(g+\delta g)(S+\delta S)]}.
\]

したがって、**根に近くgが小さい局面ではSの誤差の影響はg²で抑えられる**。一方、`1−gS` が小さい局面や近接rootではその議論だけで安い演算を許可してはいけない。

候補値g,Sに対する誤差をEg,ESとし、

\[
H=|1-gS|,\quad dH=|S|E_g+(|g|+E_g)E_S
\]

とする。H>dHなら、

\[
E_w\le
\frac{E_g+|g|(|g|+E_g)E_S}{H(H-dH)}+E_{w,\mathrm{round}}.
\]

Hの計算も外向きに囲む。上式はg,Sを中心としたexact perturbation boundであり、実装時はgの除算誤差、Sの加算誤差、分母形成、最終商の丸めを落とさない。式の恒等性は付属コードで確認済みだが、C++の包含実装が完成したわけではない。

### 5.3 実装

- 根座標とD/D'はD14Realで保持する。
- `v_i-v_j` はまずhi/loの差として作る。差をdouble化する際の情報損失を見積もる。
- 安全に分離している相互作用はscaled double reciprocal、補償和でSに加える。
- 差の相対誤差が大きいpair、underflow/overflowリスク、Sによる補正誤差が予算を超えるrootだけtwofold reciprocalへ上げる。
- cheap Sが不十分なら、そのrootのSだけ再評価する。D,D'や全rootを最初からやり直さない。
- 許可するEwは、現在の必要補正量またはroot精度目標の小さな割合。0.05等は予算配分係数であって、実測からaccuracyを捏造する閾値ではない。

より直接的に `b=D'−DS`、`w=D/b` を使う実装も可能。そこでは

\[
E_b\le E_{D'}+|S|E_D+|D|E_S+E_DE_S+E_{b,round}
\]

と商の残差上界を使う。gを別に形成するためだけに高精度除算を増やさない。前述のg式は、Sに高精度が不要になる理由とgateの設計根拠である。

### 5.4 pair逆数の共有は別変更

Jacobi型の同時更新なら、snapshotに対して `1/(vi−vj)` と `1/(vj−vi)` を符号反転で共有でき、相互作用の組は91個になる。

しかし現行はin-place更新である。**そのまま逆数を共有すると別時点の座標を混ぜるバグになる。** 初期実装は現行更新順のままmixed precision化する。Jacobi化・SIMDは独立A/Bとして後段に置く。逆数半減がsweep数増加を補えるかで採否を決める。

---

## 6. 必要以上に反復しないための停止設計

### 6.1 三つの終了条件を分離する

1. **basin探索の終了**: 高精度correctorへ渡す初期値が得られたか。
2. **root補正の終了**: 現在必要な座標精度へ到達したか。
3. **全根集合の受理**: 取りこぼし・重複・分類曖昧性が残っていないか。

double予備探索で `1e-15` の絶対stepを全根へ要求し続けたり、すべてのrootを一律 `1e-26` へ磨くことを必須にしない。

### 6.2 stagnation-aware presearch

double予備探索では、相対step、scaled residual、最近の改善率、根間距離を安く記録する。丸め誤差床で改善が止まったら、反復上限まで空回りせずtwofold correctorへ渡す。

ただしstagnationは**最終成功**ではない。誤ったbasin、重複候補、未分離clusterの可能性を保持し、後段で検査する。単にcold200→20へ定数変更して終わらせない。

### 6.3 per-root active mask

十分なrootはactive maskから外してよいが、root集合とSの項からは外さない。これは係数deflationではない。active rootが停止済みrootへ近づいたらclusterを再分類し、必要なら停止済みrootも再開する。

root中心に保持した有限精度値、補正量、評価誤差、包含半径は別項目。Newton correction `|D/D'|` だけを真のroot誤差上界と呼ばない。

### 6.4 root精度とvalue toleranceを同一視しない

`mu`のRelTolが1e-3でも、D14 rootを相対1e-3で止めてよいとは限らない。非常に近い2つのphysical foldを混ぜればtiny arcを消す。

必要精度は少なくとも次を満たすよう決める。

- finite root/clusterの個数と分離。
- 実軸・正の実軸・積分範囲との位置関係。
- 隣接eventの順序と別物であること。
- physical endpointのlocal `P=P_t=0` 補正へ安全に渡せること。
- downstream event uncertaintyの会計。

まず既存の最終精度条件を保ったまま反復と算術を軽くする。rootの最終精度を役割ごとに緩めるのは、後述のroot enclosureとevent契約が接続できてからにする。

---

## 7. warm-startの強化

### 7.1 保持するstate

```cpp
struct D14RootState {
    D14Complex v;                // hi/loをそのまま保持
    D14Complex derivative;       // 現在のanchor入力・anchor位置でのD'
    double enclosure_radius;
    double nearest_separation;
    std::uint16_t cluster_id;
    std::uint8_t arithmetic_tier;
    bool derivative_valid;
};
struct D14TrajectoryState {
    PrimaryFrame anchor;
    D14Blocks blocks;
    std::array<D14RootState,14> roots;
    std::uint8_t degree;
    bool valid;
    // 必要なら現在のaccepted event metadata。旧cellを無検査再使用しない。
};
```

derivativeがroot update前の座標で計算されたものなら、その点を明記するか再評価する。古いD'を「新rootでの厳密値」として扱わない。

### 7.2 基本warmルート

最初は複雑なpredictorを入れず、**前epochのD14Real rootsから、現在のDhatに対するtwofold correctorへ直接入る**。

double予備探索を省いてよいかは、現在の残差・separation・degree・finiteを確認するprecheckで決める。これは既存 `HOLO_D14_SKIP_WARM_PRESEARCH` の単なるON化ではなく、旧実装で捨てていたlow limbを保った経路である。

通常の数回の補正で受理できなければcluster単位に切り替える。degree変化、巨大drift、root同士の衝突などでは既存のcold初期化を許す。必要なcold fallbackを禁止して正しさを失わない。

### 7.3 追加候補: 係数差に基づくpredictor

old/newでDhatをF0,F1とし、前root ziから

\[
v_i^{pred}=v_i-\frac{F_1(v_i)}{F'_0(v_i)}
=v_i-\frac{F_0(v_i)+\Delta F(v_i)}{F'_0(v_i)}
\]

と予測できる。`F0(vi)=0` と置いてはいけない。前回の残差とその誤差を保持する。

`Delta F` を大きな二つの多項式値の差で作らず、構造ブロックの差で計算する。

\[
\Delta(AB)=\Delta A\,B_0+A_1\Delta B,
\quad \Delta(A^2)=(A_1+A_0)\Delta A.
\]

`F=C²−4vG`、`B=2C²−9vG` としたとき、固定vで

\[
\begin{aligned}
\Delta\widehat D={}&\Delta F\,G_0^2+F_1(G_1+G_0)\Delta G\\
&+8[\Delta C\,B_0Z_0+C_1\Delta B\,Z_0+C_1B_1\Delta Z]\\
&-432v^2(Z_1+Z_0)\Delta Z.
\end{aligned}
\]

これは同一式の厳密な差分表示で、parameter微分5方向を追加する必要はない。式は付属コードで記号確認済み。計算費がcorrector 1回の節約より大きければ採用しない。

前のroot separationに対してpredicted movementが大きい場合や `F0'` が不確かな場合はpredictorを使わない。このratioは初期値選択のguardであり、今epochのtopology certificateではない。

### 7.4 cluster-local座標

近接したroot群だけ、

\[
v=v_c+\delta
\]

としてcenterとoffsetを別に保持する。`C(v_c+delta),G(v_c+delta),Z(v_c+delta)` の低次数係数を正確な多項式shiftで作れば、O(1)のroot差から小量を差し引く回数を減らせる。

これは最大3/4次ブロックの**厳密な基底変更**であり、9点補間packetやGM/PF6再導入ではない。全rootで毎epoch eagerにshiftせず、未分離clusterに限定する。

real↔complex pairの遷移に注意する。全rootを実数軸へprojectすると、実係数Newtonは実軸を抜けられず新しいcomplex pairを見つけられない。旧real pairの分離が崩れたclusterはcomplex候補を許し、必要なら局所restartまたは旧cold solverへ渡す。

### 7.5 cacheの境界

workspaceのcapacity再利用、同一epochのevent再利用、別epochのD14 warm seedを区別する。前epochのflux、誤差判定、cell endpointをそのまま再使用しない。stateはcaller所有・trajectory単位・thread単位。benchmarkは独立なtrajectory間でresetする。

---

## 8. qf検査を減らすための全根検査

### 8.1 現状より弱い検査へ置換しない

小さい残差と第1Newton和だけでは、全rootの取りこぼしがない数学的証明にはならない。例として二つの根の取り違えが和で相殺される場合がある。

第一段階は既存qf検査を保持し、新型・warm・mixed Sの効果を分けて測る。次の包含検査が動作してから、qf検査を通常経路から外す。

### 8.2 安い全根包含の候補: Weierstrass / Gershgorin

現在のdegree nの多項式をp、先頭係数をan、distinct candidateをziとする。

\[
w_i=\frac{p(z_i)}{a_n\prod_{j\ne i}(z_i-z_j)}.
\]

すると

\[
\frac{p(z)}{a_n}=\prod_i(z-z_i)
\left(1+\sum_i\frac{w_i}{z-z_i}\right).
\]

これは `diag(zi)−w 1ᵀ` の特性多項式に一致する。したがってGershgorin円板

\[
\mathcal D_i=B(z_i-w_i,(n-1)|w_i|)
\]

の合併は全n根を含む。互いに分離した連結成分にk枚の円板があれば、重複度込みでk根を含む。[R4,R5]

**14×14固有値問題を数値的に解くわけではない。** 必要なのは14個の残差と小さな分母積、円板の分離検査だけである。

wの中心をwhat、絶対誤差をEwとして囲めた場合、中心 `zi−what`、半径

\[
(n-1)|\widehat w_i|+nE_{w_i}
\]

へ広げる。中心をdoubleへ丸めた分も半径へ加える。pの係数生成誤差とanの誤差もEwへ含める。`D/D'` の推定値だけでこの円板の代わりにしてはいけない。

### 8.3 実装上の制限

- 検査は候補が十分改善した段階か最終段で行う。毎iteration全係数再構成・全円板計算を重ねない。
- `zi-zj=0` や分母積のscale問題は、root重複・未分離clusterとして扱う。
- 残差p(zi)は構造式を使ってよい。展開係数をqfで毎回作ることを前提にしない。
- nは常に14とは限らない。構造退化で次数が下がる場合は実degreeを使用する。
- doubleword中心＋外向きerror boundが完成していなければ `Enclosed` と返さない。その段階では旧検査を維持する。
- cluster円板のroot数が分かっても、physical-realとcomplex-softの分類はまだ別。clusterを1rootへmergeしない。

### 8.4 実根・complex根の区別

実係数多項式で、実軸対称な分離円板にちょうど1rootがあれば、そのrootは実数である。近実軸candidateには、中心を実軸へ移し、その移動量だけradiusを広げてから分離を確認する方法を使える。

逆に円板が実軸から離れていればcomplex rootである。実軸と交わる複数root clusterは、imaginary partの小ささだけでrealへsnapせず、局所精度を上げるか既存の処理へ戻す。

正負・Rmax²との比較も円板/区間の位置で判断する。最終半径は `sqrt(v)` をD14Realのまま評価してhi/loを保持する。

### 8.5 「保証」の適用範囲

この検査の根拠は、指定した入力モデルの多項式についての全根包含である。レンズモデル全体の実装・radial integration・勾配・入力物理量の測定誤差まで保証するものではない。

通常のnode誤差推定は引き続き `Estimated`。D14でroot enclosureが得られても、最終muを自動的に `Bounded` にしない。

---

## 9. 追加の代数的削減: foldの三次探索を一次式にする

D14の正実rootでのphysical-fold分類に使う、境界quarticを

\[
P(t)=a_4t^4+a_3t^3+a_2t^2+a_1t+a_0
\]

とする。この節だけ簡記して `a=a4,b=a3,c=a2,d=a1,e=a0` を使う。レンズseparationのaとは別。

次を定義する。

\[
A=8ac-3b^2,\quad B=12ad-2bc,\quad C=16ae-bd,
\]
\[
U=2cA^2-4aAC-3bAB+4aB^2,
\quad V=dA^2-3bAC+4aBC.
\]

厳密に

\[
16aP-(4at+b)P'=At^2+Bt+C=Q_2(t),
\]
\[
A^2P'-(4aAt+3bA-4aB)Q_2=Ut+V.
\]

したがってPとP'の共通rootは、U≠0なら

\[
\boxed{t_*=-V/U}
\]

で得られる。**毎foldでP'の三次式をAberthで全根探索する代わりに、小さい積和でstationary seedを作れる候補**である。

### 9.1 guardと使用方法

- これはordinaryな単一double rootに対するchart。Uが不確かなほど小さい、triply repeated、二組のdouble roots、leading coefficient退化では使用しない。
- 現在のRは近似eventなので、式の値をそのまま認証済みtとして返さない。local `P=P_t=0` correctorと残差/分離チェックへ渡す。
- large |t| は `t=-V/U` を巨大数にせず、reciprocal `u=U/V` を使う。最初は同次pair `(-V,U)` として持つことも可能。
- 係数は正規化し、A/B/C/U/Vは必要な箇所のみD14Realで評価する。numerator/denominatorの相殺をerror screenで確認する。
- 二つのdouble complex rootsを持つ実quarticではU,Vが消える退化があり得る。これを「real rootがない」や「full circle」と即決しない。
- 不成立の場合だけ現在のderivative-cubic probeを使用する。これは稀な表現退化の保護であって、毎回二重払いする前段ではない。

このpseudo-remainderの二つの恒等式と、double-rootへの代入は付属コードで厳密に確認した。ただしC++速度と実corpus受理率は未測定である。

---

## 10. topology側の重複を減らす

### 10.1 count-onlyとcoordinateの分離

現行 `quartic_topology()` はSturm実根数が得られてもAberthを実行し、不一致ならqf isolationで座標を得てから捨てる。

cell分類のために必要なのは、現在のquarticに対する信頼できるroot countと、root数0のときのinside符号である。次を分離する。

```text
quartic_topology_count_only
    quartic係数 → 現行Sturm判定
    count=0 → 現行の符号・分母有効性確認 → full/empty
    count=2/4 → arcs
    ambiguity → fail/既存精度ladder

quartic_arc_coordinates
    endpoint座標が必要なnodeでのみroot solve/isolation
```

これはSturmを省く変更ではない。**座標を使わない場所から座標探索を外す**変更である。

現在のSturmは丸め済み係数に対する有限精度certificateという制限を持つ。その保証レベルを勝手に強く表示しない。曖昧なcertificateをroot数確定として使わない。

### 10.2 event処理

c9で直したdistinct physical foldsの保持を維持する。root proximityだけのabsolute mergeは復活させない。complex roots由来のsoft cutも、adaptiveがあるから不要とは扱わない。

event ID、fold seed、radius hi/lo、uncertaintyをD14からadaptiveまで一度だけ生成・更新する。文字列→enum化や配列reserveは最後の小最適化であり、本線より先に大改修しない。

---

## 11. 以前の構造候補の扱い

### 11.1 Horner hybrid

Phase 8で負けた方式をそのまま復活させない。既存 `d14_hybrid_solve` ではrootごとに全rootのdouble vectorを作り直し、cheap判定の後でDD評価へ戻る構造もある。精度を切り替える思想自体と、現在の実装費用は区別する。

今回の最初のmixed precisionは、**D,D'を安くしようとせず、Sを安くする**。分岐の位置が異なる。

### 11.2 三次判別式lift / matrix QZ

\[
f(v,\lambda)=v\lambda^3+C\lambda^2+G\lambda-4Z,
\qquad \operatorname{Disc}_{\lambda} f=\widehat D
\]

という同値表示は維持する。ただし前回のlifted Newtonでは局所収束しても全根検査に失敗した例がある。独立Newtonで全根Aberthを一括置換しない。

matrix/QZ seedは主にcold向けで、warm seedが既にある今回の主標的と合わない。外部QZ依存の追加は今回の必須実装にしない。

今後試すなら、全root identityとcluster管理を保持したまま、困難clusterのcorrectorとしてliftを使う別A/Bに限定する。

### 11.3 optional threefold

D14Real二肢で不足し、qf時間がまだ支配的なときだけ、固定三肢の `D14Wide` を追加候補にする。二肢から三肢への昇格はroot/cluster単位、係数の追加精度は元入力から再生成する。

三肢はbinary128より情報量が多くても、対象演算とCPUによって速い可能性はある。ただしそれは計測事項。**最初から三肢を全rootへ使うこと、全てのoperatorに動的precision tagを持たせることはしない。**

---

## 12. データ構造と既存APIへの接続

新しい内部ヘッダの例:

```text
src/lcbinint/magnification/holonomic/
    d14_real.hpp                 // 専用twofold・mixed/complex演算
    d14_model.hpp                // 固定容量構造builder/evaluator
    d14_precision_solver.hpp     // cold/warm、active roots、mixed S
    d14_root_certificate.hpp     // 全根包含・分類用の小検査
    d14_fold_seed.hpp            // quartic subresultant seed
```

既存ファイルへの接続:

```text
radial_events.hpp     : D14Resultからevent生成、新kernel切替
prepared_geometry.hpp: caller-owned D14TrajectoryStateを保持
cells.hpp            : count-only分類への内部接続
adaptive_epoch.hpp   : event型bridgeだけ。integratorは原則変更しない
v2_profile.hpp       : 最小限の追加counter
```

`std::vector<Cplx<__float128>>` との互換bridgeを通常ループに挟み続けない。初期比較では残してよいが、qf往復削減phaseでは、内部root stateとevent生成をD14Real対応にする。legacy root exportはlegacy consumerまたはdiagnosticが要求したときだけ行う。

旧fixed API、production router、PF6/GM、未追跡 `.claude/`、別セッションのbenchmark成果物を変更しない。新しいkernelはisolated opt-inで開始し、採否を独立commitにする。

---

## 13. 実装順序

### Step A — 短いprofileと比較点の固定

既存trajectory runnerを使用し、D14/topologyを含むtimer内訳を次のexclusive項目へ分ける。

```text
frame/model build
coefficient expansion / conversion
cold or warm presearch
D/D' evaluation in corrector
S interaction + root update
root-set validation
event extraction + stationary probe
Sturm cell classification
prepared-state copying / other
```

演算ごとのclock取得はしない。細かい内訳が必要な部分は別profile passにし、採用速度はunprofiled wallを使う。

### Step B — D14Realとfixed-capacity kernel

既存DDを比較相手として、専用mixed演算、fixed arrays、実係数scale、複素残差商を実装。最初はqf係数/最終検査を維持し、新しいroot arithmeticの差だけ切り分ける。

### Step C — warm精度維持と予備探索短縮

hi/lo保存、warm direct corrector、stagnation-aware cold presearch、active maskを導入する。最終受理の検査はまだ弱めない。predictorはdirect correctorとの独立A/Bで追加。

### Step D — mixed-precision S

正しいroot座標差を保持したまま、far interactionsをdouble化。Ew gateが通らないrootだけSを昇格。in-place更新順は維持する。

### Step E — qfの通常固定費を外す

D14Realによる元入力からの係数生成、係数誤差の伝播、必要時のqf再生成、全根検査を接続。通常caseでqf build/check/castが0になる状態を狙う。

包含検査に必要な演算上界が未実装なら、このStepのqf検査除去は未完として明記する。先にB–Dの改善を止める必要はない。

### Step F — structural event seedとcount-only topology

subresultant seedとcount-only分類をそれぞれ独立に入れる。c9/c92、full-circle、reciprocal chart処理を保つ。

### Step G — profile次第の追加

cluster recenter、三肢、Jacobi pair-sharing、SIMDを検討する。全部作ることを必須にしない。今のhot spotを削れない追加機構は入れない。

---

## 14. 最小限の確認と性能の採否

HMC/NUTS、新しい推論環境、大規模な別corpusは今回不要。既存のunit・D14比較・adaptive benchmarkを使用する。

新設部品の基本確認は必要である。EFTの相殺/極端scale、正しいascending/descending評価、近接root、実↔complex遷移、subresultant退化、全根数の誤判定防止を、小さい既知解fixtureで確認する。これは型とroot solverの通常unitである。

性能は次を分けて報告する。

- full-cold / full-warm / radial-only。同じ入力・同じtolerance・同じprecision contract。
- trajectory全体と初回を除く部分。初回を時間外へ出してwarm勝利にしない。
- uniform / LD、各RelTol。混合集計だけで結論を出さない。
- qf係数生成・検査・局所補正・全根cold fallbackを別counterにする。
- root更新回数、presearch sweep、double/twofold reciprocal件数、active root数、cluster昇格数。
- value/statusの既存比較。失敗を除いた速度だけを勝利としない。

最終目標はfull-warmの大幅削減だが、何倍になるかは未測定である。小変更ごとに一律15%を要求しない。削減量とコード複雑化の釣り合いで採否を決める。

特に次を成功条件にしてはいけない。

```text
クラスを作ったので成功
qf呼出しを減らしたので成功（検査まで失っている）
DDに全置換したので成功（既存DDと同じ費用）
短いNewtonで止まったので成功（全root未確認）
muのTolが緩いのでD14 rootも粗くした
warmだから旧event位置を信頼した
```

---

## 15. 実装担当へ渡すプロンプト

```text
添付D14_PRECISION_AND_WARM_DESIGN_JA.mdを仕様として、
lcbinint/dev/holonomicの最新HEADでisolated D14最適化を実装してください。
開始時にHEADとローカルtrajectory evidenceを記録し、他者変更は保持してください。

主対象はD14Real専用twofold型、役割別mixed precision、
hi/loを失わないwarm corrector、予備探索の空回り削減、
qf係数/検査/castの通常固定費削減です。
既存DDとの差を明確にし、単なる型の改名・全体DD化にしないこと。

文書Step A→Fを独立変更として進めてください。
Step Gはprofileで必要性が見えたものだけ。
全根・event分離・Sturm・full-circle・value/gradient契約は維持。
qfの既存backstopは必要時だけ残し、曖昧な結果を成功にしないこと。
root enclosure実装が未完成ならqf検査は外さず、その限界を報告してください。

HMC/NUTSや新規大規模テストは不要。既存unitと既存trajectory benchmarkで、
full-cold/full-warm/radial-onlyを計測し、precision tierと仕事量を記録してください。
production router、PF6/GM、.claude、他セッション成果物には触れないこと。
checkpoint・raw evidence・独立commitを残してdev/holonomicへpushし、
実装した項目、未実装項目、実際の削減箇所と速度を分けて報告してください。
```

---

## 16. 出典・由来

### repo一次資料（確認したrefを固定）

- S1 `radial_events.hpp`: `solve_d14`、warm予備探索、qf検査、event生成。  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/src/lcbinint/magnification/holonomic/radial_events.hpp
- S2 `dd_real.hpp`: 既存twofoldとq1/q2/q3除算。  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/src/lcbinint/magnification/holonomic/dd_real.hpp
- S3 `d14_structure.hpp`: C3/G4/Z3、展開、構造Aberth。  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/src/lcbinint/magnification/holonomic/d14_structure.hpp
- S4 `d14_hybrid.hpp` / `d14_lifted.hpp`: 前回候補の実装。  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/src/lcbinint/magnification/holonomic/d14_hybrid.hpp  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/src/lcbinint/magnification/holonomic/d14_lifted.hpp
- S5 `docs/holonomic/v2_phase8_checkpoint.md`: 過去の上限到達profileと未採用理由。旧gridを含む歴史的記録であり、最新速度ではない。  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/docs/holonomic/v2_phase8_checkpoint.md
- S6 `prepared_geometry.hpp`: L1 OFF / L2 ON、全根warm state。  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/src/lcbinint/magnification/holonomic/prepared_geometry.hpp
- S7 `radius_terms.hpp::quartic_topology`: count取得後の座標計算と破棄。  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/src/lcbinint/magnification/holonomic/radius_terms.hpp
- S8 `adaptive_full_circle_coverage_20260911.md`: c9のdistinct folds、c92、full-circle修正。  
  https://github.com/NunotaKansuke/lcbinint/blob/26593fa005ac0c82eb13c11dd7c7ef79dd976867/docs/holonomic/adaptive_full_circle_coverage_20260911.md

### 外部一次資料（算術・検査の背景。速度を本repoへ外挿しない）

- R1 Joldes, Muller, Popescu (2017), *Tight and Rigorous Error Bounds for Basic Building Blocks of Double-Word Arithmetic*, ACM TOMS 44(2), Article 15. DOI: 10.1145/3121432. 著者publication list: https://homepages.laas.fr/mmjoldes/publicationList.html
- R2 Muller, Rideau (2022), *Formalization of Double-Word Arithmetic, and Comments on “Tight and Rigorous Error Bounds for Basic Building Blocks of Double-Word Arithmetic”*, ACM TOMS 48(1), Article 9. DOI: 10.1145/3484514. 著者記録: https://cv.hal.science/jean-michel-muller
- R3 Ogita, Rump, Oishi (2005), *Accurate Sum and Dot Product*, SIAM J. Sci. Comput. 26(6), 1955–1988. DOI: 10.1137/030601818. https://tore.tuhh.de/entities/publication/f14b3dcf-d52f-49aa-ab40-8c7808ee2b8d
- R4 Bini, Fiorentino (2000), *Design, analysis, and implementation of a multiprecision polynomial rootfinder*, Numerical Algorithms 23, 127–173. 開発者による紹介・引用情報: https://numpi.dm.unipi.it/scientific-computing-libraries/mpsolve/
- R5 Bini, Robol (2014), *Solving secular and polynomial equations: A multiprecision algorithm*, J. Comput. Appl. Math. 272, 276–292. DOI: 10.1016/j.cam.2013.04.037. https://doi.org/10.1016/j.cam.2013.04.037
- R6 Shewchuk (1997), *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates*. 著者公開ページ: https://www.cs.cmu.edu/~quake/robust.html
- R7 GCC Internals, *Routines for floating point emulation*. https://gcc.gnu.org/onlinedocs/gccint/Soft-float-library-routines.html

R7はqf算術がcompiler/runtimeのemulation helperへ落ちる場合の背景である。`__float128`の四則演算と `libquadmath` の数学関数を混同しない。実際の対象CPU/compilerで `__multf3 / __divtf3` 等がどこから呼ばれるかをprofile/assemblyで確認する。

### 本書で導出・確認した部分

mixed-S補正摂動恒等式、warm構造ブロック差分式、quartic pseudo-remainder seed式を記号的に確認した。Weierstrassの特性多項式恒等式はdegree14の有理数例でも確認した。これは数式の確認であり、実装の丸め誤差包含・実際のroot取りこぼし防止・性能の検証を代替しない。

## Appendix A. 専用mixed演算の具体例

汎用DD同士の演算へscalarを毎回変換しないため、少なくとも次の二つを直接持つ。

### A.1 twofold + double

入力 `a=ah+al`、binary64 bに対して:

```text
(s,e) = TwoSum(ah,b)
c     = RN(e+al)
(h,l) = TwoSum(s,c)
```

overflow/underflow等のEFT前提が満たされる場合、唯一の捨てた丸めは `c=RN(e+al)` のものなので、表現値に対する絶対演算誤差は `u/(1-u)*abs(c)` で抑えられる。subnormalが関与する場合は絶対誤差項を追加するかscaleする。ここで `u=2^-53` はunit roundoffであり、C++の `epsilon=2^-52` と混同しない。

### A.2 twofold × double

```text
(p,e) = TwoProd(ah,b)   // e=FMA(ah,b,-p)
c     = FMA(al,b,e)
(h,l) = TwoSum(p,c)
```

同じ前提の下で、丸めはFMAのcに集約され、その絶対演算誤差は `u/(1-u)*abs(c)` で抑えられる。これはfull twofold乗算ではないが、D14の整数倍・実scale・一部Horner処理では頻出する。

実装の通常戻り値は `(h,l)` だけでよい。演算誤差を必要とするbuilder/validator用にはboundも取得できるcompile-time variantを用意する。全operatorで常時error propagationしてhot loopを重くしない。

TwoSumをFastTwoSumへ置換する場合は条件を証明してから行う。full twofold同士の加減乗除は、R1/R2に対応する既知アルゴリズムを基準にし、上の簡易式を無条件に一般化しない。
