# True GM coverage phase 3

基準: dev/holonomic `e8820a918577db4e0e1ed1586a82006bcc6d5d8e`。
production V2/V3 router変更なし。実験範囲はisolated holonomicのみ。

## 実装前の判断

Phase 2の接続gateは生のR-Taylor係数の多項式恒等式残差を測る。
狭いセルでは係数に幅の負冪が現れ、局所的な輸送精度とは別にgateを
悪化させうる。またセル中央から全点への単一Taylor評価は解析半径を
超える可能性がある。まずこの二要因をbasisそのものの悪条件から分離する。

1. 定数s>0に対し x=(R-Rc)/s, Q_x=s Q_R として、同じGM還元を
   無次元Taylor係数で行う。gateの閾値は変更しない。これはsample fitでない。
2. 接続を再構築して小刻みにstateを進める。接続再構築はphysical re-seedと
   区別し、初期period以外の物理積分をtransportに使わない。
3. 以上だけで不足する場合、局所chart/basisの変更を検討する。
   既存6Dのq8^{-j} poleはそのままではdegree dropを解消しない。

これはcoverage原因を調べる段階であり、scaled 7Dを最終basisとする決定ではない。

218 arc中央のprobeではDD Order8の受理がraw 39→scaled 180、
scaled point 183、scaled quad 187。残りは正規化後も残差が大きい。
次の実装前判断: mod Qを介さず
`Q C_k + Q S'_k - Q' S_k/2 = -t^k Q_x/2`
を15×15の係数比較で直接解く。Cは次数6、Sは次数7。
行列は定数行列でequilibrateしTaylor係数の漸化式で解く。
先頭係数の逆冪を避ける狙いであり、追加マイクロ最適化ではない。
残差はequilibrateした恒等式のcomponentwise backward error。
旧gateの未重み付き係数残差とは測度が異なるため、physical referenceで
forward errorを必ず別途確認する。接続の受理だけをcoverage改善とは呼ばない。

## 結果と採否

**coverageは改善したが、速度の勝利条件は未達。production routerへは採用しない。**

- `gm_direct_connection.hpp`: 15×15直接Hermite還元。Taylor係数を恒等式から
  生成するtrue GMであり、K-ruleやsample-fit packetは使わない。
- `gm_adaptive_transport.hpp`: 無次元Taylorを途中で再中心化するDD輸送。
  各方向で受理したstateを次のノードへ引き継ぐ。embedded Order8/6の全state
  相対スケール付き差を1e-12で判定し、失敗時は刻みを半減する。接続/刻み予算を
  超えたら非OK。node physical re-seedで成功を装わない。
- `gm_sensitivity.hpp`: DDの5方向forward dual、Pの暗黙根微分、endpoint-deflated
  physical seed微分、observable covector微分。同じGM stateの解析感度を運ぶ。
  有限差分はテストの独立oracleだけ。seedとhもDDで構成して7D observableの
  cancellationを抑える。数学的basisそのものは7D etaを保持している。
- `gm_coverage_epoch.hpp`: F0のarc geometry/IFT、F_halfのtrue GM、mu/5Jacまで
  接続した研究用entry point。毎cell/arc中央に一つの初期seedを置く。
  全セルを通じて一つのseedという意味ではない。全円セルには未対応で非OK。

元の108ケース（u=0が54、u=0.5が54）、n_r=64では、limb-darkening側の
**13,952/13,952 arcノードが再seedなしで輸送**された。value接続は25,664/25,664成功、
解析5Jac側は36,345/36,345成功。元の54 LDケースで全ノード輸送は54/54、
そのうちtopologyを含めstatus OKは52/54。全108ケースのstatus OKは
GM valueとV2は104/108。解析Jacobianも局所数値gateでは104/108だったが、
以下のforward-error監査で全面受理できず、最終APIではLDの解析Jacobianを
一律GRADIENT_UNRELIABLEとして返す（local_statusは診断用に保持）。元々非OKのケースを
OKに昇格していない。トポロジーを含めた成功とnode transport完了を区別する。

独立angular参照とのvalue node gateは `abs(a-b)/(1+abs(reference)) < 1e-9`。
512点参照との差が1e-11以上なら2048/4096点で参照自身の収束を確認する。
最終value probeでは全13,952ノードが合格、最大差は `7.901e-11`。
途中で見つかった2つの約2e-9/9e-9の差は512点angular参照の収束不足だった。
2048点physical re-seedもGMと一致する。収束確認前の記録も残してある。

`chart_p4`の原chartのdegree dropを、反射chartの同じstateのまま跨ぐテストを
2イベントで追加した。イベント半径そのものと反対側の両方で独立angular参照に
一致する。原chartのexact-p4におけるarc solverは参照を返せないため、独立lens
積分側も反射座標でendpointを求める。これはGM connectionを参照に使うことではない。
一般的な動的Möbius chart切替は未実装。fold近傍は現状、刻みの縮小で扱っており、
sqrt(v)変換によるODEそのものの正則化や直接flux-priority 6D化は未実装である。

## cold / warm（同じ入力、n_r=64）

同一process内で各GM epoch直後に同じ入力のV2を測定。coldは毎回topology/D14/
physical seedを作り直す。warmは前epochのD14全根だけをwarm seedとして再利用し、
classify_cellsの認証を毎回通す。V2も既存のL2 warm-D14を使い、L1 topology再利用はOFF。
trajectoryは2点で、次点は `xs += rho*1e-3`, `ys += rho*3e-4`。
`warm_init`はwarm集計から除外する。長期steady-stateでの勝利を示す実験ではない。

以下はLD側54ケースの中央値ms。cold欄は元のepoch=0、warm欄は変位後のepoch=1。

| 出力 | cold GM / V2 | warm GM / V2 |
|---|---:|---:|
| value (u=0.5) | 140.342 / 1.169 | 140.072 / 1.117 |
| value + analytic5Jac (u=0.5) | 444.222 / 1.088 | 443.851 / 1.035 |

| GM段階（ms中央値） | cold value | warm value | cold analytic5Jac | warm analytic5Jac |
|---|---:|---:|---:|---:|
| topology/D14 | 0.882 | 0.704 | 0.871 | 0.704 |
| physical seed | 0.204 | 0.203 | 0.873 | 0.878 |
| connection construction | 136.465 | 136.407 | 428.170 | 428.110 |
| transport recurrence | 2.727 | 2.730 | 15.130 | 15.142 |
| arc geometry | 0.188 | 0.328 | 0.192 | 0.340 |
| fallback/re-seed | 0 | 0 | 0 | 0 |

各段階の中央値は同一ケースとは限らず、合計がwhole中央値になるとは限らない。
whole wallにはobservable再構成等も含む。Jacobian marginalは同一case/epochをpairして
計算し、cold/warmとも約305.6 ms、倍率中央値は3.19/3.18。
全case/変位後を含む値・5Jac・成功率の集計は `gm_phase3_summary.txt`、
各epochのraw値は `gm_cold_warm_value_phase3.csv` / `gm_cold_warm_jac_local_gate_phase3.csv`。
u=0のuniform valueはGM F_half輸送を行わないので、true GMの勝利条件に数えない。

hostはrogue1、Intel Xeon Gold 6530、Release `-O3 -march=native`、単一thread kernel。
共有hostでCPU affinity/clockは固定していない。一部の診断を並行実行したため数%の
速度差を論じる測定ではないが、100倍以上の差による不採用判断には十分である。
旧Phase2との速度比は入力/測定範囲が異なるので採用判断に使わない。

現行D14の別計測は `gm_d14_current_phase3.txt`（108+既存ref7ケース、best-of-10）。
構造化solve中央値0.598 ms、radial_events全体0.625 ms。
旧 `bench_d14_split` はlegacy内部処理を別計測し、現行構造化DD経路の費用へ
足し合わせられないことも確認した（`gm_d14_split_phase3.txt`）。

## 精度・微分の独立検証

muのV2との差は、変位後を含め最大約1.91e-8。解析JacobianのV2との差は
最大1.91e-3だが、これを即座にGMの誤差とは解釈しない。
独立の元のlens式の `dphi/(2 sqrt(phi))` 積分を512/2048点で比較した。
resonantでは独立参照自身がV2から9.407e-5ずれ、512/2048の一致は2.64e-11。
このケースのwhole GM解析5Jacは独立2048点参照と最大3.446e-11で一致し、テスト化した。

一方、独立angular微分の512/2048収束gate（1e-7）を満たしたのは41/54。
狭いarcでdouble phiのendpoint cancellationが出るため、全54ケースのJacobianが
独立angular参照により科学的受理されたとは主張しない。未収束や非正phiは
参照失敗として記録する。physical re-seedによるnode解析感度の別検証も残す。

## 次の表現に対する判断

直接Hermite還元によって「接続が作れない」と「セル幅を一回で飛べない」を
分離でき、ほぼ全セルのcoverageは得られた。ただし反射chart＋7D stateは
物理observable以外の成分も厳密に運ぶため、多数の再中心化が必要になる。
6Dの旧q8^{-j}射影を重ねても、その構造は解消しない。
この15×15還元を追加の細かな最適化だけでV2まで縮める方針は採らない。
次の研究対象は、直接physical fluxを含む低次元のregularized GM/PF stateであり、
今回の全セル・解析感度・独立参照ベンチをその判定器として使える。

warmでphysical seed/stateを再利用するには、前の中心半径からの移動だけでなく
parameter方向のGM輸送とarcの対応認証が必要になる。一次感度だけで以前のseedを
線形外挿して再利用することは誤差保証がないので採用していない。
現在seed費用はvalueで約0.2 ms、接続費用は約136 msであり、seedを完全に省いても
勝利条件に届かない。cold限定のD14削減をsteady-stateのGM高速化とは混同しない。

## 再現

入力を `evidence/holonomic/gm_coverage_cases.tsv` に固定して保存した。
過去TSVのt_jac列は入力reader互換のため保持し、今回の時間集計には使用しない。

```sh
cmake -S tests/holonomic_cpp -B build-holonomic-global \
  -DCMAKE_BUILD_TYPE=Release -DLCBININT_BUILD_HOLONOMIC_M7_PY=OFF
cmake --build build-holonomic-global -j4
ctest --test-dir build-holonomic-global --output-on-failure
HOLO_D14_LEGACY_COMPLEX=1 HOLO_CHART_P4_LEGACY=1 \
  ctest --test-dir build-holonomic-global --output-on-failure
./build-holonomic-global/bench_gm_connection_coverage evidence/holonomic/gm_coverage_cases.tsv
./build-holonomic-global/bench_gm_coverage evidence/holonomic/gm_coverage_cases.tsv 64
./build-holonomic-global/bench_gm_coverage evidence/holonomic/gm_coverage_cases.tsv 64 jac
./build-holonomic-global/bench_gm_cold_warm evidence/holonomic/gm_coverage_cases.tsv 64 2 0
./build-holonomic-global/bench_gm_cold_warm evidence/holonomic/gm_coverage_cases.tsv 64 2 1
./build-holonomic-global/bench_gm_angular_jac evidence/holonomic/gm_coverage_cases.tsv 512
./build-holonomic-global/bench_gm_angular_jac evidence/holonomic/gm_coverage_cases.tsv 2048
./build-holonomic-global/bench_d14_structure evidence/holonomic/gm_coverage_cases.tsv 10
python3 tests/holonomic_cpp/summarize_gm_phase3.py
```

通常isolated ctestは10/10 PASS（15.07秒）、legacyも10/10 PASS（15.46秒）。production経路は変更していない。


## 解析感度の全ノード監査と最終fail-closed判定

`gm_coverage64_jac_reference_phase3.txt` の初回監査では13,952 node中267で
512点physical re-seed感度との差が1e-7を超えた。参照を2048/4096点まで増やす
`gm_coverage64_jac_refined_phase3.txt` で再検証した。very-closeの最大0.224の差は
参照不足であり7.19e-11へ解消したが、small-qケースに感度のforward errorが残る。
従って、接続構成の成功率やembedded全state差だけで解析Jacobianの精度を保証できない。

最終 `gm_coverage_epoch<GmDDDual5>` は、u!=0で局所gateがOKの場合も
`status=GRADIENT_UNRELIABLE` を返す。パラメータ別の例外リストは作らない。
参考用数値とlocal_statusは保持し、value laneはこの感度用制限を受けない。
このため、**value coverage改善は成立、解析Jacobianの全面受理と速度目標は未達**。

`gm_cold_warm_jac_local_gate_phase3.csv` はこの最終status降格を追加する前の
局所gateのみの実測であり、そこにあるstatus OKを最終APIの受理率と解釈してはいけない。
計算本体は同じだが、現在のAPIのLD解析Jacobian status OKは0/54である。
再現コマンドのJacobianベンチを現行コードで走らせると、この非OKが反映される。
通常/legacy isolated ctestはいずれも10/10 PASS。全ノード監査は性能・精度の
研究上の判定であり、単体テストのPASSと科学的受理を区別する。

最終refined感度監査は、13,952 node中13,691が1e-7 gate内、261 nodeで失敗、
最大差5.741e-5。失敗は6ケース（rand003/017/018/022/032/035）に残った。
未精緻化参照による偽の失敗と、収束した参照に対する感度輸送の失敗を分離した。

最終APIのfail-closedを確認した2ケースsmokeは `gm_jac_fail_closed_phase3.csv`。
LD行のstatus=8（GRADIENT_UNRELIABLE）が記録され、uniform行はstatus=0を維持する。
