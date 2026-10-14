# Phase16: warm薄arcの局所root bracket認証（研究中）

基準d7abe34、HOLO_WARM_LOCAL_BRACKETは既定OFF。
目的は既存のcold quartic再照合を除去することではなく、局所的に証明できる
rootだけを安く認証し、曖昧な場合は元のcold照合を行うこと。

各tracked t-rootを公称半径1e-9、隣root間隔の1/8以下の区間で囲む。
丸め後の実際のoffsetが2e-9未満であることを別途要求する。
端点のP符号が反対かつ区間全体のP'が非零なら単根が一つ存在する。
区間が互いにdisjointであることも要求する。
|d(2atan t)/dt|<=2から角度誤差は4e-9未満となりincumbent1e-7より十分小さい。
0/2piのwrapをまたぐ区間はこの線形角度距離では扱わずcoldへ戻す。

区間演算はbinary64各演算結果のnextafterによる外向き評価。
subnormal/overflow、非round-to-nearest、fast-mathでは採用しない。
FTZ/DAZに耐えるよう0付近のoutward幅はmin-normalまで広げ、非零積の
0へのunderflowや非exact cancellationによる0は認証不能とする。
これはcold solverと同じbinary64 quarticの局所根の証明であり、
物理parameterからの係数構築誤差を新たに証明するものではない。
既存D14/cell authority、rootpair branch/residual/gap、endpoint gateは維持。

初期profile（wrap拒否を追加する前）では114014 attempts/113913 successes。
warm初回を除く5412行でwhole p50 .427510→.389079ms、
physics .163481→.126878、arc .102819→.077560。
全7216入力のmu差0、node/convergence差0。
wrap拒否追加後のwholeは別途測定する。初期profileを最終版の結果と混同しない。

unit:40000 point interval add/mulをqfと比較し包含を確認。
overflow/underflow拒否、単根受理、重根/duplicate/誤位置/非実根拒否、
丸めモード拒否、角度wrap拒否をテスト。

再現:Phase15と同じcompile flagsに-DHOLO_WARM_LOCAL_BRACKETを追加。
profileはbench_adaptive_stage_profile.cpp(input_snapshot.tsv,warm)。
wholeはrunner_hybrid.cpp(input_snapshot.tsv,whole_bracket.tsv,3)。
CPU0で逐次実行。採用にはwhole/value/status/reference監査が必要。

## ビット分類修正と最終検証

FTZ/DAZ環境ではFP比較のx==0がsubnormal inputをzero扱いし得るため、
regular判定をuint64ビット分類へ変更。最終版はwhole_bracket_bits.tsv。
whole_bracket.tsvはこの修正前（wrap拒否は追加済み）の参考結果で、
採用版の証拠として使わない。

最終whole p50(ms):
- 1e-3 cold .429757→.430808、warm .387340→.357182。
- 1e-4 cold .474107→.475503、warm .448218→.406858。
- warm p99は1e-3 1.195356→1.175892、1e-4 1.483183→1.439117。
- cold maxの56.28→57.14msなど微小tail変動もrawに残す。
全14432行×3laneでmu差0、node/status差0、value全件converged。
VBM1e-6参照超過は1e-3で0、1e-4で既存3行のまま。

通常のreference checkerはcold planのためこの変更を通らない。
optional warm引数を追加し、同じ物理値の初回prepared epochでseedを作り、
2回目のwarm topologyで実際にwarm認証を通す参照比較を追加。
550行中466usable、old/newともobserved violation0、stop/mu差0。
残る84行は独立参照での認証成功には数えない。
実trajectoryの異なるepoch間の値一致は上の全件benchで別途確認。

warm ValueFirst、微分追加budget0の110cases×2tolでも、value、5つの
analytic gradient、gradient qualityをexportして220行byte一致。
有限未認証を認証済みに変更したという意味ではなく、旧分類を維持している。

unitにqf polynomial/derivative interval照合を追加した際、全1000区間を
必ず受理するというテストの仮定が1区間で失敗した。零係数のoutward幅を
min-normalに広げるため、次の乗算で保守的なunderflow拒否が起こる。
認証不能を許す契約に合わせて受理区間だけ包含を検証し、900以上の受理も
要求した。結果999/1000受理、受理した全区間の包含pass。
その他40000演算照合、FTZ/DAZのsubnormal拒否、wrap拒否、重根拒否もpass。

採用:HOLO_WARM_DISABLE_LOCAL_BRACKETで旧cold照合のみへ戻せる。
production routerは変更しない。局所認証不能なら従来cold照合へ進む。
VBM全面勝利は未達。

再現補足:
- bench_warm_gradient_parity.cppを旧/新flagでcompileしreference_cases.tsvを入力。
- check_adaptive_reference.cppはreference_cases.tsv warmで実行。
- check_d14系のqf residual/completeness gateは一切変更していない。
