# Phase15: K-rule平方根の整理（研究中）

基準19dac47。HOLO_K_SINGLE_SQRTは既定OFF。
sqrt(S2)/(A sqrt(A) sqrt(B))をsqrt(S2/(AB))/Aへ変更する候補。
ABとS2/(AB)がnormal finiteの場合だけ使用、指数範囲端は従来式を維持。
積分node、8/16gate、rescue、微分kernelは変更しない。
数式は同じでも丸め演算は違うので、parity/referenceの実測が必要。

warm profile（初回を除く5412行）のLD部分:
K p50 .012376→.009299ms、physics .198133→.196382ms。
whole .463827→.463476msの差は変動と区別できず、採用の根拠としない。
uniformは変化なし。mu相対差最大5.88e-16、node/status差0。
既存test_holonomic_transportはcandidate flag付きでpass。
これはvalue、analytic JacobianのFD参照、epoch parityを含む。
rawとtest出力をevidenceへ保存。

次にprofileなしwholeの3 repeatsを測る。未採用。
再現: Phase14と同じcompile/run手順へ-DHOLO_K_SINGLE_SQRTを追加。
profileはbench_adaptive_stage_profile.cppへinput_snapshot.tsvとwarm。
wholeはPhase10 runner_hybrid.cppへinput_snapshot.tsv、whole_single_sqrt.tsv、3。
CPU0で各timingを逐次実行。

過去のsoft-cut全除去は10行のnear-fold node非収束を起こしているため、
単にcut数を減らす案は再実装しない。特にwarmでは薄いarcのcold crosscheckを
必要としており、これも根拠なく除去しない。

## 全件whole（3 repeats）

1e-3 cold p50 .432379→.429757ms、warm .389366→.387340ms。
radial .123010→.121370ms。1e-4 cold .477034→.474107、
warm .451486→.448218、radial .159425→.157596ms。
全14432行×3laneでnode/status不一致0、全件converged。
mu相対差最大5.88e-16、VBM1e-6超過は1e-3で0、1e-4で既存3件。
cold/warm p99の微小増加もwhole_summary.jsonに保存。
全体の短縮は1%未満だが、積分側で両tolとも改善が残った。
独立referenceを追加確認してから採否を確定する。

集計:
`python benchmarks/holonomic/summarize_adaptive_reciprocal.py --baseline ../adaptive_extreme_phase14/whole_count_only.tsv --candidate ../adaptive_extreme_phase15/whole_single_sqrt.tsv --output ../adaptive_extreme_phase15/whole_summary.json`

次に検討する候補（未実装）はwarm thin-arcのcold全根照合を、
各tracked rootのdisjoint bracket + P符号/derivative interval証明で
置き換える局所認証。証明できない場合は現在のcold照合をそのまま実施し、
単なる残差閾値緩和やcold照合の無条件削除にはしない。
まず認証費用と成功率をprofileしてからwholeを測る必要がある。

## 採用

独立reference550行中466usableでobserved violation0、stop分布も
Converged496/TopologyUnresolved50/InnerAccuracyLimited4のまま。
未usable行を検証成功に数えない。両tolのwhole測定と既存K-rule value/
Jacobianテストを踏まえて採用する。旧式へのスイッチは
HOLO_K_DISABLE_SINGLE_SQRT。production routerは変更なし。
改善は小さく、VBMへの全面勝利は依然未達。
