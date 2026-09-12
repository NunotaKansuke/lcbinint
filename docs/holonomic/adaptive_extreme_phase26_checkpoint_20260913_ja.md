# Phase26: double noise-scale audit and early handoff

基準78aaf9d。8 sweepごとに全14根のbalanced expanded polynomial残差を調査。
proxy = 8*degree*eps64*sum |c_k| |z|^(degree-k)。
これは実数係数complex Hornerのroundoffの診断目安であり、
係数生成誤差を含む厳密上界・root certificateではない。

HOLO_D14_NOISE_AUDITは停止を変更しない。7216行中6486行で
全根がproxy以下になる観測があり、4796行は最初の観測点8 sweepだった。
観測は8 sweep刻みで、真の到達時刻ではない。初期終了で観測しない行もある。
79,536 checks、mu/status/node差0。

HOLO_D14_NOISE_HANDOFF試作では全根がproxy以下なら、その候補を
既存D14Real/qfへ引き渡す。solveの受理gateは変更しない。
初期profileではpresearch p50 .033798→.007991 ms、
D14Real .022718→.022799 ms、profile whole .339027→.3099005 ms。
7216行全converged、status/node差0、最大scaled mu差1.90e-12。

レビューでproxyのoverflow/underflowを明示拒否するguardを追加した。
radius/mass/scale/ratioが有限、scale>0の場合のみ引き渡せる。
上記profileはguard追加前の試作。最終whole/root/referenceはguard追加後の
binaryを使い、初期profileだけで採用しない。

再現: Phase21 profile flagsへHOLO_D14_NOISE_AUDITまたは
HOLO_D14_NOISE_HANDOFFを追加。wholeもPhase20flagsにHANDOFFを追加、
input_snapshot.tsv、repeat3、CPU0。root detailは1 both。
最終root parity/reference/whole検証中、production defaultは未変更。

## 最終guard後の検証

whole14432行、各tol7216/7216収束、status/node差0。最大relative mu差3.06e-12。
1e-3 cold p50 .3900975→.3592025 ms、warm .3176995→.288560 ms。
1e-4 cold p50 .4338965→.4016525 ms、warm .365982→.3371055 ms。
1e-3 cold p99 3.178741→3.321099 ms、warm 1.159774→1.025429 ms。
1e-4 cold p99 3.214608→3.369957 ms、warm 1.396692→1.346569 ms。
radial-onlyはほぼ不変。cold p99回帰があるため一律既定化はしない。

全14432根集合で最大scaled root差3.090e-14、role/physical/event数差0。
qf-coldは100→98回、発生行は14行で変化。総数低下は全row改善の意味ではない。
115-case expanded/structured比較もparity failure0、root差1.194e-9。
この115-caseのworst residualは1.099e-13であり、残差だけを完全性証拠にしない。

reference550行はbaselineとmu一致、usable466でobserved violation0。
解析5Jac220行の値・品質分類も一致。既存adaptive unit1484 checks/0 failures。
VBM reference超過数は1e-3=0、1e-4=既存3のまま。

1e-3 LD whole warm中央値は.3121045 ms、uniform .2615305 ms。
LDは保存済みVBM中央値付近まで来たが、僅差と母集団集計の問題があり
明確なVBM勝利とはしない。uniformはまだ大きく届いていない。

HOLO_D14_NOISE_HANDOFFは研究flagとして保持。既定・production routerは不変。
次はcoldを旧探索に保つwarm-only handoffを別binaryで検証する。
noise_ratioは最後の有効な観測時点の全根最大ratioであり、
全sweepを通じた最大ではない。noise_valid=0は有効観測なし。
