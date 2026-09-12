# Phase19: stationary endpoint reuse

基準bc2d6a5。HOLO_ADAPTIVE_STATIONARY_ENDPOINT研究flagで、
Newton補正後のthetaが元と同じbinary64値（zero符号も一致）のときだけ
既に計算済みのphi/dphiを返す。final_phiを要求するadaptive呼出しだけに限定。
step threshold、derivative reliability、nonfinite判定を緩めない。
同じ入力の純粋関数を再評価しても値は変わらないので、
これは近似・error estimator変更ではなく同一評価の再利用。
production defaultは変更しない。

既存adaptive testは1484 checks/0 failures。ValueFirst解析5Jacの220行は
同じcompile flagsのPhase17 baselineとbyte一致。
correctness runはCPU1、whole測定はCPU0で分離。
wholeはPhase18と同じ3repeat・入力・compile flagsに
-DHOLO_ADAPTIVE_STATIONARY_ENDPOINTだけを追加。
whole結果確認中。

## Whole検証完了

14,432行でmu/status/node差0、各tol7216/7216収束。
1e-3 warm p50 .340798→.339915 ms、radial .121538→.120266 ms。
1e-4 warm p50 .389658→.388800 ms、radial .157601→.155963 ms。
radial-onlyではp90/p95/p99も短縮。whole p99は1e-3で
1.146416→1.150877 ms、1e-4で1.430954→1.422879 msと上下する。
VBM reference超過数は1e-3=0、1e-4=既存3のまま。

結論: 同じ引数の再評価を省く効果はradial側で約1%だが、whole効果は小さい。
研究flagとして保持し、whole勝利/production採用は宣言しない。

次の調査箇所: radial_eventsのadaptive metadata生成における
d14_struct_eval(d14s,Cplx<qf>(vq,0),D,Dp)。実数引数に対して
複素数qf演算を大量に行っている。式・精度・順序を保った実数専用化を試す。
