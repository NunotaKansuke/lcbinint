# Phase24: warm-only single-confirmation presearch

基準6e033a5。HOLO_D14_WARM_SINGLE_CONFIRM研究flagを追加。
有限なpresearch_seedがある場合だけpatience1、coldは既存設定を使う。
既定patience2や物理parameter routingは変更しない。
D14Real/qf/residual/completenessはそのまま。

Phase23の一律patience1でcold p99が少し悪化したため分離した。
warm rootsは前epochの結果に依存するので、Phase23のwarm結果の流用では
この候補の根集合を保証できない。別binaryで全trajectoryを再実行する。

reference550行中usable466でobserved violation0、stop差0。
解析5Jac220行で品質分類差0、最大scaled derivative差4.81e-12。
correctnessはCPU1、wholeはCPU0。
wholeはPhase20と同じflagsにHOLO_D14_WARM_SINGLE_CONFIRMを追加、
input_snapshot.tsv、repeat3。根集合比較も別途実行。

## 結果

全14432行、各tol7216/7216収束、status/node差0。
cold/radialのmu差0、warmの最大relative mu差3.73e-12。
1e-3 warm p50 .3176995→.3144635 ms、p99 1.159774→1.097712 ms。
1e-4 warm p50 .365982→.362699 ms、p99 1.396692→1.391284 ms。
cold p50 1e-3 .3900975→.3899765 ms、1e-4 .4338965→.432678 ms。
1e-3 cold p99は3.178741→3.185391 msと微小増加で、
coldの演算上の改善は主張しない。

全14432 root setsで最大scaled root差2.770e-14。
role/physical/event数/completeness差0。qf-cold totalは100→100、
発生が変わる行は4件。総数不変を各row不変と混同しない。
既存adaptive unitは1484 checks/0 failures。
VBM reference超過数は1e-3=0、1e-4=既存3のまま。

結論: warm-onlyは一律変更より適切な研究候補として保持。
既定設定・fixed-n_r API・production routerは変更しない。
whole短縮は約1%なので、これだけをVBMへの勝利とはしない。
次は候補seedを変えず、厳密な周期状態を省く方法を検討する。
