# 同一7点の式由来R微分を使う、guard付きHermite radial積分：実装計画

日付: 2026-09-13  
対象: `NunotaKansuke/lcbinint`, `dev/holonomic`  
確認した基準commit: `fe3fca4a3f650b2dd49edb57bb43d50a368ebfd3`（Phase 70）  
位置づけ: 実装方針。新しいC++経路の速度・精度・安全な早期停止は未検証。本書の追加案を既存実測と混同しない。

## 0. 決定する方向

**新しい半径で根を解く前に、既存7点で得た境界解からR方向微分を取り出し、区間積分を高次化する。信頼できないpanelは、同じcacheから従来の7→15→31…へ進む。**

ただし、次の三つは分ける。

1. `QH7` が高精度な近似であること。
2. 返す `QH7` に対する誤差推定が適切であること。
3. その推定で追加nodeを省いて終了してよいこと。

Phase 70で示されたのは主に1で、2・3は未完成。「smoothらしい」指標を数学的certificateとは呼ばない。現行APIの `AccuracyAssurance::Estimated` と `Bounded` の区別を維持する。

初回実装は **uniform (`u=0`)・value-only (`GradientPolicy::None`)・level 3の7点panel** に限定する。LD、5Jac、D14/event探索、K-ruleの点数、production routerを同時に変更しない。uniformの `kFull` は既存の解析的に簡単な経路を優先し、Hermiteを使うためだけに追加処理を払わない。

## 1. 出発点：確認済みの結果と、その限界

Phase 70の `summary.json` による結果:

| 項目 | 実測 |
|---|---:|
| 対象panel | 247 |
| 参照可能なpanel | 238 |
| QH7がQ7より良い | 231/238 |
| QH7がQ15より良い | 18/238 |
| Q7の絶対誤差中央値 | 1.3285198e-5 |
| QH7の絶対誤差中央値 | 2.8107901e-9 |
| Q15の絶対誤差中央値 | 4.4479420e-12 |
| 診断上のR微分追加費/base比の中央値 | 約4.96% |

最後の時間比はcoldの `reference_arcs` と短いnode timerによる値で、warm本線の追加費用、cache、重み適用、controllerは含まない。**7点Hermiteが15点より高精度、あるいはwholeで高速だと判明したわけではない。目標は15点と常に同じ精度ではなく、要求tolに十分なpanelで次の8点を省くこと。**

必須反例 `paper_highA / panel12`（両側fold map）:

- Q7 ≈ 10892.263171
- QH7 ≈ 10888.926277
- Q15 ≈ 10867.732610
- GL128 ≈ 10837.076301（GL64との差 ≈ 3.5e-5）
- Q7の観測誤差 ≈ 55.19、`abs(QH7-Q7)` ≈ 3.34
- **QH7自身にも約51.85の誤差が残る。補正差を小さくできたことだけでは不十分。**

現参照は同じtopology/境界式を使う独立radial quadratureであり、独立した全physical-topology oracleではない。9個の参照不能panelを成功に数えない。Phase 70での過小推定件数はQ7誤差との比較なので、QH7を返す今回の候補ではQH7誤差との比較を改めて行う。

## 2. 数学的契約と、今回行わないこと

### 2.1 補正の根拠

局所座標 `xi ∈ [-1,1]` の7個のFejér点で `g_i=g(xi_i)` と `d_i=dg/dxi(xi_i)` があれば、次数13以下のHermite補間H7が一意に定まる。

`QH7 = integral[-1,1] H7(xi) dxi`

`g ∈ C^14` なら剰余は

`g(xi)-H7(xi) = g^(14)(eta_xi)/14! * product_i(xi-xi_i)^2`。

これは高次化の数学的根拠になる。しかし、全区間の14階微分上界を7個の値・1階微分だけから得ることはできない。`A*product_i(xi-xi_i)^2` は全nodeで値も微分も0で、積分は非零になる。

したがって、`abs(QH7-Q7)`、Hermite同士の差、係数tail、node上のconditioningのいずれも、それだけで真の積分誤差上界にはならない。**高増光例が1個だけ通らないから安全係数を大きくする、という設計にはしない。**

### 2.2 現行controllerとの整合

基準コードでは `HybridEmbedded`、weighted-detail floor、detail decay、inner/geometry/event/roundoff ledgerを使い、`require_bound=true` は `BoundUnavailable` を返す。現行自体が全入力への厳密包含保証ではなくEstimated契約である。

本計画には二つの明示的な段階を設ける。

- **Shadow / Conservative**: 現行受理を緩めず、補正の実装と費用を確認する。
- **GuardedEstimated（research-only）**: Hermite用誤差モデルと棄却guardを検証し、7点での早期停止を試す。保証水準はEstimatedのまま。実測passをBoundedへ昇格しない。

厳密上界が必要なら別途、方程式に基づく区間全体の剰余上界が必要。本フェーズで全panelの14階jet、区間atlas、複素領域coverを新設しない。安価にその上界が得られると実証されない限り `require_bound` の挙動は変更しない。

## 3. R微分は境界解から作り、パラメータ5Jacと混同しない

以下は本書で具体化する式。微分はレンズ・ソースパラメータを固定したR微分。

### 3.1 uniformのobservable

arc幅の和を `W(R)` とし、未正規化のobservableを

`F(R)=R W(R)`, `F_R=W+R W_R`

とする。正則なendpointでは

`t_R=-P_R/P_t`, `theta_R=2*t_R/(1+t*t)`。

既存の最終endpoint/branch/chartと同じ解を使用する。probeの `reference_arcs`、cold root solve、FDを実行時の微分取得へ移植しない。角度の2π unwrap、向き、reciprocal chartは既存実装を再利用する。

### 3.2 foldで大きなendpoint微分を引かない

fold pair `t_- = m-sqrt(v)`, `t_+ = m+sqrt(v)` では、枝を整合させたarc幅を

`Wpair = 2*atan2(2*sqrt(v), 1+m*m-v)`

として扱える。そのR微分は

`Wpair_R = 2*((1+m*m+v)*v_R - 4*m*v*m_R)`
`          / (sqrt(v)*((1+m*m-v)^2+4*v))`。

この式は同一pairの幅についての式で、wrap arcや複数arcの向き・加算は既存規則に合わせる。`m_R,v_R` は既存 `root_pair_dR` 等の式由来IFT計算から取得する。代数的な同値性と浮動小数点での安定性は別なので、極端に小さいvの扱いを無条件に安全としない。

最終的に必要なのは `J^2*Wpair_R`。fold付近で `Wpair_R` 単独を巨大化させてから小さい `J^2` を掛けず、`J^2/sqrt(v)` を含む有限な積へまとめる。既存のhi/lo event radiusと同じ基準で距離を作る。v=0の端点極限を使う場合は別途式を導出する。通常のopen Fejér nodeでは端点を評価しない。

R微分だけが不可靠ならHermiteを使わず既存積分へ進む。値のnodeが正常なのに、任意追加した微分の失敗でepoch全体をNonfiniteにしない。逆に値自身の既存rejectを隠さない。

### 3.3 子panelの座標倍率を必ず含める

cell共通mapの座標をx、panel局所座標をxiとし、

`x = (xl+xr)/2 + h*xi`, `h=(xr-xl)/2`, `R=R(x)`。

すると

`J = dR/dxi = h*R_x`, `Jprime = d²R/dxi² = h²*R_xx`。

uniformの正規化N=`pi*rho²`に対して

`g = F*J/N`, `g_xi = (F_R*J² + F*Jprime)/N`。

`AdaptiveSample.value[0]` は既にmap/正規化を含むので、Jを二重に掛けない。x微分とxi微分を混ぜない。

`t=(1+x)/2`, `L=b-a` の既存fold mapでは:

| map | R_x | R_xx |
|---|---|---|
| foldなし | L/2 | 0 |
| 左fold | L*t | L/2 |
| 右fold | L*(1-t) | -L/2 |
| 両側fold | pi*L/4*sin(pi*t) | pi²*L/8*cos(pi*t) |

mapとその微分は同じradius/radius_loをownerにする。二重実装による端点ずれを避ける。

## 4. 固定重みkernel：onlineで補間行列を解かない

点順は `xi_i=cos(i*pi/8), i=1..7`（正から負）。

`QH7 = sum_i A_i*g_i + B_i*d_i`。

初期値として使える重み（実装時には80桁以上で再生成してbinary64へ丸め、保存する）:

| i | A_i | B_i |
|---|---:|---:|
|1|0.1135986494090330123|0.004903180563873449870|
|2|0.2719058719058719059|0.01549371501900593643|
|3|0.3943378585274749243|0.01528302956622608756|
|4|0.4403152403152403152|0|
|5|0.3943378585274749243|-0.01528302956622608756|
|6|0.2719058719058719059|-0.01549371501900593643|
|7|0.1135986494090330123|-0.004903180563873449870|

本書作成時に90桁演算で0〜13次momentを確認した。これはquadrature式の確認であり、物理integrandの精度試験ではない。中心nodeと中心の微分重みは厳密に0、対称性は明示的に設定する。

同じ7点に含まれる3点（i=2,4,6）からQH3も作れる。次数5以下でexact、A=`(8/15,14/15,8/15)`、B=`(sqrt(2)/30,0,-sqrt(2)/30)`。

補正の構造は、値だけの補間L7とそのnode微分行列D7を用いて

`QH7-Q7 = sum_i B_i*(d_i - L7'(xi_i))`

とも書ける。これは実装照合・診断に使えるが、onlineの主計算は14項の固定dot productでよい。毎panelでVandermonde逆行列、monomial係数展開、高次数GLを実行しない。必要な小さな補間行列・derivative行列は事前生成する。

和は既存の補償和を使う。誤差ledgerは返す式の重みに対応させ、入力誤差について少なくとも

`Edata,H = sum |A_i|*e(g_i) + sum |B_i|*e(d_i)`

を用いる。元のFejér重みで集計した誤差を、そのままHermiteの誤差として表示しない。微分誤差を0と置かない。境界/rootの不確かさと算術丸めが微分にどう伝播するかを確認し、評価できないpanelはHermite不適格とする。Estimatedな入力誤差の伝播はEstimatedのままである。

## 5. 補正値と停止判定：具体的な第一候補

### 5.1 Shadowを最初に接続する

本線と同じcached nodeからQH7、QH3、微分整合性を計算するが、返す値、node配置、停止、statusは変更しない。現在のcontrollerで既に7点終了するpanel、7→15へ進むpanel、さらに進むpanelを分ける。

同じ入力を微分あり/なしで測り、microtimer比ではなく費用総額を見る。既存7点で終了するpanelをさらに高精度化してもnode削減はないので、原則としてそこへ微分取得費を掛けない。

### 5.2 Conservativeモードは契約の接続確認用

現行のQ7と誤差E7からQH7へ値を替えるなら、元の誤差をそのまま流用しない。三角不等式による転送は

`Etransfer = E7 + abs(QH7-Q7)`。

E7が上界ならこの式も上界、E7が推定なら推定の転送にとどまる。これは元より小さい誤差を作らないため、**このモード単独で早期停止が増えるとは期待しない**。早期停止を変えず、選択値とledgerの整合を確認するためのA/Bである。

### 5.3 GuardedEstimatedの第一候補

現行7点の値detailをD7、既存のdecay flagを `value_resolved` とする。初回に比較するHermite側モデルは、現行HybridEmbeddedの構造に合わせた一つに絞る。

`E_rad,H = max(alpha*D7, min(D7, s*abs(QH7-QH3)))`

開始時の `alpha=1/8`, `s=2` は現行設定と揃える。これは **H3→H7の改善が以後も続くとする収束モデルで、Hermite remainderの証明ではない**。H3との差が粗い近似の誤差を拾い、過度に保守的になる可能性もある。Q7との補正差C7は診断に残すが、それだけをQH7の誤差にしない。

さらに同じnodeの `d_i=g_xi` に、既存と同形の3→7および1→3のweighted interpolation-detailを作り、微分detailの減衰を棄却guardにする。第一候補は値と同じ `detail7_d <= 0.5*detail3_d`、または微分の数値不確かさfloor以下という条件。座標は必ずxi、値detailと微分detailを無単位化せず直接加算しない。

新モデルの誤差は `E_rad,H + Hermite用data/roundoff ledger + event ledger` として集計する。event誤差を補正差で減らさない。既存のevent/geometry/inner制限を回避して7点受理する経路を作らない。

この第一候補が高増光反例を弾けるか、通常panelで実際に小さいEを作れるかは未確認である。まず保存データと追加のshadow dataで調べる。**反例を見逃す場合、そのままcontrollerへ組み込まない。係数を多数sweepして既知238区間に合わせ込まない。**

### 5.4 早期停止を試してよい範囲

少なくとも以下を満たすpanelだけがHermite候補となる。

- uniform/value-only、level3、完全に計算済みの7点を持つ。
- 現在のevent/cell topology、chart、arc対応、値nodeの既存検査が成立する。
- R微分が有限で、選んだchart/正則化に対して信頼できる。不確かさを算入済み。
- 現行の値detail decayを満たす。未解像のpanelをHermiteだけでresolvedへ上げない。
- 微分detail guardを通る。guardは棄却のための条件であり証明と呼ばない。
- 新式で集計したglobal error ledgerが `E <= max(atol, rtol*abs(mu))` を満たす。

global toleranceの判定対象は実際に返す混合集計値とその誤差とする。panelごとに全体tolを丸ごと割り当てない。全panelの誤差を非負で合算し、相殺で隠さない。

候補が増光率やq/rhoの固定しきい値で選ばれる実装にはしない。区間・式・node情報に基づく。安いguardでも反例が残るなら、より高価な証明機構を即追加するのではなく、この早期停止案を未達として報告する。

## 6. 反例対策を「smoothという名前」で済ませない

`paper_highA/panel12` を、式評価・R微分・map・モデルのどこで危険と認識できるか追う。両側fold mapを使った、実root数が一定、7点のP_tが非零、という事実だけでは区間内の高次構造を除外できない。

既存D14の複素根やconditioningを使う場合も、役割は追加の棄却hintに限る。未認証の根位置から「最寄り特異点までの距離が保証された」と言わない。D14根を新しく解き直さない。複素特異点の完全な位置・map後距離・分枝の正則性まで証明する処理は本フェーズ外。

同じ7点のモデルが反例を識別できないなら、既存の次levelへ進むことが正常動作である。7点への固執はしない。Q15もこの高増光panelでは十分でないため、「反例は必ず15点で終える」という仕様にもせず、従来controllerに31点以降を任せる。

## 7. cacheとincremental controllerの接続

### 7.1 微分の取得は必要なpanelだけ

まず、既存 `AdaptiveSample` のquartic/root-pair stateが**最終的なendpoint解**を再構成できるか確認する。可能なら7点評価後、controllerが7→15へ進めようとする時だけ、そのcached stateからR微分をlazyにmaterializeする。

最終endpoint情報がcacheに不足する場合、既存callback内で小さいendpoint/pair summaryを保存する方式と、初期7点に限りR微分を同時計算する方式を比較する。どちらも新規root solveを許可しない。不足を `reference_arcs` の再呼出しで埋めない。通常の7点終了panelへ常時払う費用・sampleサイズ増加を計測する。

まずは研究用sidecarを使い、OFF時に全sampleを巨大化させない。所有するworkspaceの `max_bytes` へsidecarも算入する。sampleの再配置で無効になるpointerを保持せず、sample IDとgenerationで関連付ける。

### 7.2 7→15の再利用

Hermite不適格、guard reject、または新誤差が予算を満たさない場合は、既存のp-refinementへ戻る。7点を捨てない。15点ruleでは既存7点が偶数indexに一致するので、新規の8点だけ値/rootを評価する。

15点以上では初回実装は元のFejér値・元の誤差推定を使う。H7は診断に残せるが、Q15にH7の補正を二重加算しない。Hermite失敗でepoch/D14/topologyをcoldからやり直さない。

### 7.3 h-splitはp-refinementと区別する

二分した子のFejér点は親点と一般には入れ子ではない。親7点を全部子quadratureへ再利用できる、と約束しない。既存どおりroot anchorとして再利用し、座標が厳密に一致し意味も同じ場合だけ値を再利用する。

gとg_xiはpanelのmap Jacobianに依存する。同じRでも子のhが違えば古いmapped値/微分をそのまま流用できない。未mapのF/F_R、またはmap ID・xl/xr・generationを持つmapped jetとして所有し、再写像を明示する。

Phase67で修正した「左右子が両方成功してから親を置換する」transactional splitを維持する。予算切れ、微分失敗、candidate取消でも、完成した親の値と全ledgerを保持する。

### 7.4 推奨する小さい追加状態

名前は現地の構造に合わせてよい。概念的には:

- policy: Off / Shadow / Conservative / GuardedEstimated（新規はすべてopt-in）
- radial-jet status: NotComputed / ReliableEstimated / Unavailable
- panel candidate: Q7, QH3, QH7, selected rule, selected ledger, reject reason
- counters: jet attempts/reuse/failure、7点早期受理、Hermite試行後に15点へ進んだ数、追加node数

「Hermite不適格」は値sampleの `reliable=false` とは別の状態にする。元の停止reasonを雑に書き換えない。

## 8. LD・5Jacは次段階に分離

今回のR微分は `d/dR` で、既存のパラメータ5Jacを代入してはいけない。5Jacのintegrandへ同じ方法を使うなら `d/dR(dF/dp)` が必要で、fold map/eventのパラメータ依存も整合させる必要がある。

初回はStrict/ValueFirstではHermite経路を使わず、現行と同じ値・微分・snapshot契約を保つ。高速な補正値だけを差し替えて、古いmeshの微分をその補正値の厳密な微分と表示しない。

LD拡張も、K-rule内のR微分、node/rootの再利用、mixed sensitivity費用を独立に測ってから着手する。16点K-ruleは維持。uniformで利益が出る前にLD/5Jacの実装を広げない。

## 9. 実装順序と打ち切り条件

### A. cheap kernelとR-jet

固定H3/H7重み、mapの二階微分、既存解を使うR-jetを小さい部品として追加する。near-foldの有限な積、reciprocal chart、hi/lo、子panel倍率を確認する。production defaultsと停止判定は触らない。

### B. Shadowと費用・反例監査

Phase70データに加え、現在のfast/warm node経路でshadow計測する。QH7の真の観測誤差、H3/H7モデル、値/微分detail、各ledger、反例のreject理由を記録する。既存baselineが15点以上へ進むpanelのうち何割が本当に7点Hermiteで予算内かを、参照を使ってofflineに評価する。oracleは診断ラベルでありruntime routingに入れない。

**ここで「省ける8点の費用」が微分・guard費用を上回る対象が少ないなら、controllerの複雑化へ進まない。**

### C. Conservative接続、次にGuardedEstimated研究A/B

まず値とledgerの対応、cache、取消、予算終了を接続する。その後、Bで反例を見逃さず一定の利益が認められた場合だけ、別フラグでGuardedEstimatedを有効にする。

既存のWeightedDetail/HybridEmbeddedを残し、同じHEAD・compiler・CPU・入力で比較する。基準のsafety=2を同時に1へ変更しない。既存最良構成の他のmacroも両armで揃える。

### D. checkpointと採否

uniform value-onlyで追加費用込みのwhole利益が再現し、値/status非回帰が確認できて初めて、production候補として報告する。自動default化しない。失敗時は有用なR-jet/診断のみ残せるが、未達を成功扱いしない。

## 10. コストの評価軸

次の8点を実際に不要にできる確率をp、Hermite/guardの試行費をCH、追加8点とそのcontroller費をC8とすると、試行一回あたりの単純な損益は `p*C8 - CH`。さらにcandidate管理、cacheの容量増加、拒否後のledger復元・計測されるすべての費用を差し引く。

この式は後段refinementの連鎖まで含まないため、最終判断はwhole実測とする。区間誤差が何桁良くなったか、root solve数だけの削減率、成功panelだけのtimeは採用根拠にしない。

測るのは既存のfull-cold / full-warm / radial-only。cold/warmではD14、setup、event/cell、微分、guard、追加node、fallbackを全部含む。warmはtrajectory先頭coldを含む集計とsteady-stateを区別する。

p50/p90/p99/maxに加え、paired差、全行時間和/平均、Hermite成功・失敗別費用を出す。希少な利益をp50だけで捨てず、希少な大損をmedianで隠さない。実行順を交替し、短いtimerのノイズを高速化と解釈しない。

## 11. 検証は既存資産に小さい回帰を追加する

大規模な新しいテスト基盤やHMC試験は要求しない。

- 小unit: 0〜13次のHermite exactness、map/子panelのchain rule、同じroot stateでのR微分、7→15の新規8点のみ評価、微分不適格時の既存復帰、split中断時の親保持。
- 必須反例: `paper_highA/panel12` と「7点で値・微分とも見えないbump」。後者でsame-node estimatorを厳密boundと誤表示しないことを確認する。安いguardが全てのbumpを検知できるというテスト仕様にはしない。
- 既存reference / trajectory: uniformの両tolを中心に、LDとgradient modeはOFF-path非回帰を確認する。参照不能行・既存参照不一致を明示し、新規違反と混同しない。

個別panelの精度はQH7と参照の差で測り、wholeは実際に返すmuとglobal errorで確認する。観測誤差の過小推定率、誤った早期受理、その最大程度を保存する。Q15も自動的な真値にしない。

phase70の238区間だけで採否を決めない。guardの選択に使ったケースと、それ以外の既存corpusでの結果を区別する。full corpusも全入力への証明ではない。

## 12. 実装対象・成果物・変更管理

既存接続箇所:

- `src/lcbinint/magnification/holonomic/adaptive_radial.hpp`: policy、sample/panel所有、estimate、refine、gather。
- `src/lcbinint/magnification/holonomic/adaptive_epoch.hpp`: 同じendpoint/root-pairを使うR-jet取得と正規化。
- `src/lcbinint/magnification/holonomic/nested_fejer2.hpp`: 点の順序と入れ子indexの参照。既存ruleは書き換えない。
- 既存のendpoint_dR / root_pair_dR / fold-map部品: 再利用し、実装位置は現地で確認する。
- 新しい小header案: `same_node_hermite.hpp`。重みと小kernelだけを分離する。

checkpoint、raw、再現command、compile flags、HEADを保存する。独立commitで`dev/holonomic`へpushする。作業中の無関係変更と`.claude/`、既存rawを変更・削除・stageしない。中間結果も無断で消さない。

採否報告は「同一7点の精度改善」「停止モデルの信頼性」「cache再利用」「追加費用込みwhole速度」「未検証のLD/5Jac」を分ける。microbenchや参照可能panelだけから全体の速度/保証を主張しない。

## 13. 資料と、本書の追加提案の区別

確認したrepo資料（すべて基準commitの内容）:

- [Phase70 checkpoint](derivative_quadrature_phase70_checkpoint_20260913_ja.md)
- [Phase70 summary](../../evidence/holonomic/derivative_quadrature_phase70/summary.json)
- [Phase70 probe](../../evidence/holonomic/derivative_quadrature_phase70/probe.cpp)
- [現行controller](../../src/lcbinint/magnification/holonomic/adaptive_radial.hpp)
- [現行epoch adapter](../../src/lcbinint/magnification/holonomic/adaptive_epoch.hpp)
- [Phase67のsplit/gradient修正](gradient_nodes_phase67_checkpoint_20260913_ja.md)

外部の一次資料:

- Bozza et al., *VBMicroLensing: three algorithms for multiple lensing with contour integration*, [§2、式(5)〜(8)](https://arxiv.org/html/2410.13660v1#S2)。境界の微分で補正を作り、二つの補正の平均と差を使う。これは同じnode情報を活用する動機であって、今回のHermite estimatorの安全性の証明ではない。
- [NIST DLMF §3.3](https://dlmf.nist.gov/3.3)、[§3.5](https://dlmf.nist.gov/3.5): 補間とquadratureの背景。

固定H3/H7 kernel、pair幅のR微分の明示式、子panelのchain rule、policy分離、第一候補のHermite誤差モデルと微分detail guard、選択的lazy jetは本書の追加設計であり、Phase70で実装・検証済みという意味ではない。

**設計の中心は「7点で必ず終える」ことではない。既存の7点から使える微分情報を引き出し、追加nodeを省けるpanelだけ省き、残りは情報を捨てず既存adaptiveへ戻すことである。**
