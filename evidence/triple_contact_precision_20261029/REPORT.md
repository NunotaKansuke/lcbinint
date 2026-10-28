# Triple contact resolution hardening

Base: `ed7e859d` / `experiment/algebraic_boundary`, host rogue1. この前に、それまでのtriple修正・回帰テスト・検証記録をed7e859dへ明示pathでコミットした。別作業のdirty変更を保持。push・worktree作成はしていない。

## 修正

接触点の `panel_resolution_limit` を独立C++ probeで再現し、二つの原因を分離した。

1. 評価側はcertified radial bandの境界に丸められたRをnextafterで内部へ移してcacheへ保存するが、7点の境界補間を組む側は移動前のRでcacheを引いていた。補間曲線が作れず、誤差がinfとなって無意味な二分を続ける。mapped / mapped_transportも同じcanonical radiusを使い、位置とmaterial motionの参照先を揃えた。case949はこれだけで回復した。
2. 実際に異なるradial nodeが同じbinary64 Rへ潰れる段階では、補間の分解能検査を無限に繰り返しても直らない。7点latticeに実際のR重複がある場合に限り、完全な環帯面積からscalar flux errorを保守的に評価する。

線形LDの未規格化強度は0以上1以下なので、任意のcellの真のweighted areaは `0 <= A <= U = pi*(b-a)*(a+b)`。推定値Fに対する誤差は `max(abs(F),abs(U-F))` 以下。Uは浮動小数点の丸めを上側へ余裕付けする。全panelを囲うUをcellごとに使うので過大評価になり得る。通常の規格化と値の誤差controllerへこの上界を渡す。要求rtol/atolを変更しない。R重複がなければ従来通りinfとしてrefineする。

既に評価した物理packet・微分を全て積分に残す。image componentや微分項を省略せず、support/root-count/endpoint proofを維持する。微分誤差でmeshを選ぶcontrollerは追加していない。この値の上界を微分の精度証明とは説明しない。

## 全件の結果

同じ1528条件、6520 epoch、rtol=1e-3/1e-4、atol=0、LD=0/.5/1を再計算。3056 native engine-curvesの入力hashとDSO hashを全件照合。

| 結果 | ed7e859d | 今回 |
|---|---:|---:|
| 値 成功 | 1460 | 1460 |
| 値＋微分 成功 | 1441 | 1448 |
| 以前成功→今回失敗 | — | 0 |

回復: 503, 940, 949, 1041, 1070, 1178, 1508。残りは値68 exception、微分73 exception＋7 invalid_gradient。従来の曖昧なVBM参照の除外基準を保持し、646条件・2286 source points・4547比較で要求誤差超過0。最大誤差/request比0.816735。元の全VBM 3方式のraw、追加厳密参照とLD overrideは前二回の検証物を再利用した。

周期・時刻原点・逆順＋重複epochの130軌道/390検査は全てstatus=0、gradient_valid=true。最大value relative差8.60e-10、最大gradient scaled差4.62e-9。

## 独立LD微分検証とVBM参照の問題

通常VBM LDの微小中央差分は多数のworkerでSIGSEGVとなり、成功参照として使わない。30 parameter-workersのstatusと途中値を全て保存した。case940の最初の差分刻みは接触幅に対して大きすぎたので停止し、その結果をgradient940-large-step.jsonへ分離した。

独立チェックはVBMの**一様disk**の値を別のGauss-Legendre積分でLDへ変換し、パラメータ差分を取った。式は `[(1-c)*M(R)+c*integral(sin(theta)^3*M(R*sin(theta)), theta=0..pi/2)]/(1-c/3)`。MultiPoly/NoPolyを独立に使い、積分次数・hの両方を変更した。

- case940 t0: 32/64/128/256点で安定。256点、h=1e-5/5e-6でMultiPolyの差分は−1135.736169/−1135.772885、Richardson外挿−1135.785123。native−1135.787054。NoPolyも同じ値に接近する。NoPolyは予算内で256点の小さい方のhまで完了しなかったため、完了している点とstatusを区別する。
- case1041 q2: 通常のVBM LD差分は約−3249.280で一見安定するが、独立LD積分の極小hでは積分離散化誤差が差分で増幅する。hを物理的に小さい範囲で広げ、128→256→512→1024点を比較した。
- 1024点、h=2.4085875e-7/1.2042938e-7: MultiPoly−3195.658202/−3194.511840、NoPoly−3195.657730/−3194.510925。外挿はそれぞれ−3194.129719/−3194.128656。native−3194.196222。512点からの外挿変化は約0.46であり、参照誤差をゼロとはしない。nativeと3e-4相対の診断範囲で整合する。通常VBM LDの偽の安定性と区別できた。

上記から独立参照付きの微分回帰を2件、収束・値の回帰を4件追加した。全配置・全10微分方向の精度を認定してはいない。

## 速度と残る費用

CPU0、1 thread、独立processの逐次測定、2 warmup＋15回のthread CPU中央値。import/constructorを除く。build・全件検証・参照計算と重ねていない。外部負荷・CPU周波数は固定していない。

| case (3 epoch) | 値 前→後 ms | 微分付き 前→後 ms |
|---|---:|---:|
| 940 | 9.682→9.663 | 241.365 (invalid)→186.553 |
| 949 | 39.398→39.342 | 例外→48.063 |
| 1041 | 69.889→69.980 | 例外→288.596 |
| 1508 | 22.343→22.357 | 例外→644.832 |

通常8軌道の前後比較もtiming-summary.jsonに保存した。値の追加費用は小さい。失敗していた旧微分の時間を同等の成功計算として比べない。case1508の微分は値の約29倍で、LD接触微分の費用削減は未達。今回の変更を高速化の一般的保証とはしない。内部stageの詳細profileは新たに取っていない。

## テスト・再現・残件

- 既存Python対象51 pass、追加6 pass (合計57)。C++全targetを再buildしてctest 30/30 pass。
- 追加テストのうち最初の5件は旧DSOで全てfail、新版でpass。後から追加したq2検査は新版でpass。
- DSO `f23387064f75ea2778f1c0df67fab4b2b7b8984ce3d3047b7a7cc61880f30989`、source/test hashesはfinal-provenance.json。
- native campaignは3 workers、CPU5..7、各AS 2 GiB、BLAS/OMP各1。独立参照は最大4 workers、CPU1..4。buildは-j2。既定上限8 CPU/RAM16 GiBを維持。
- 検証・独立参照・build・速度測定は終了。附属sessionを永続backgroundと説明していない。長いcoordinatorはnohup setsidで起動しPPID1とworkersを確認した。
- 失敗probe、環帯上界だけの不十分な案、canonical-only、最終resolution版の結果をそれぞれbaseline/annulus/canonical/resolutionへ分離。初期の案では949以外は未解決だった。rawを成功結果で上書きしていない。
- 再現はrun.py (既存入力/DSO hashを照合して再利用)。現在のbuild既定はcandidate-package。archiveには入力、raw、参照、スクリプト、source patch、ログとhashを保存し、DSO自体はローカルfreezeに残す。
- 未解決: 値の根分離・degenerate radial event、接触のendpoint/membership precision修復、LD微分費用、全方向の微分精度。前回の接触22条件のうち6条件が回復し16条件は残る。今回、前回棄却した広いquad endpoint/membership変更は採用していない。
