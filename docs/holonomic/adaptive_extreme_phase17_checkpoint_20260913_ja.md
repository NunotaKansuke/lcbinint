# Phase17: 根ごとのnative residual上界（研究中）

基準1c8a178。既存の1e-13残差gateとcompleteness/step gateは変えず、
通過を確認できるrootだけqf Hornerを省く候補。
HOLO_D14_NATIVE_RESIDUAL_SCREENは既定OFF。
HOLO_D14_NATIVE_RESIDUAL_AUDITでは必ずqfも計算し、上界の違反を数える。

## 誤差モデルの条件と導出

binary80（64-bit significand）のround-to-nearestでのみ使用。
x87 precision-controlも64-bitか確認。fast-math、別精度/丸めモード、
入力の非有限/範囲外は使用しない。非零入力の絶対値を[1e-200,1e200]に
限定し、次数<=14の中間演算とerror radiusがbinary80のnormal範囲に
収まることを前提にする。物理パラメータによるroutingではない。
qf squared normのrange端を避け、非零norm上界が[1e-2400,1e2400]外なら拒否。

u=eps80/2とする。cast誤差はeps80*|cast値|以下で囲う。
複素Hornerの実部は積→差→係数加算、虚部は積→和なので、局所丸め誤差の
L1 boundはgamma3(u)*(X*B+|c|)、gamma3(u)=3u/(1-3u)<2eps80。
qf側の丸めも含め、32eps80*(X*B+|c|)で保守的に囲う。
前段誤差Eはroot変換誤差を含むXtrueで伝播し、B*deltaXとdeltaCを加える。
Xtrueにはqf側の誤差伝播分も上回る余裕を持たせる。

非負のbound計算plus/timesは演算後に1+16eps80を掛ける。
二つのround-to-nearestの下向き誤差を含めても
(1-u)^2*(1+16eps80)>1なのでnormal範囲では安全側になる。
分母はcast(scale)*(1-16eps80)で下げる。正の1e-300を省くことも上界側。
L1 normでEuclidean normを上から囲い、qfのnorm/除算丸めの余裕も加える。
これは任意epsilon multiplierでgateを緩める案ではなく、上記演算モデルと
限定された指数範囲に基づく上界である。条件外は既存qf評価を行う。

## 実測

全7216入力warm audit、7330 residual呼び出し。
全14根が同時に上界<=1e-13になるのは368回（約5%）しかなく、
全根まとめて省く案は追加費用を回収できないため不採用。
root-wiseでは102620根中72186根（約70%）が資格を持ち、
qfとの上界比較違反は0だった。native評価中央値は約3us。

根ごとに資格を持つものだけ省くscreen profile:
residual p50 .026379→.010985ms、warm profiled whole .389079→.368329ms。
値差0、node/convergence差0。
これはprofile付きの参考値。baseline profileはPhase16初期版なので、
最終採用は最新baseとのprofileなしwholeで判断する。

screenはD14Realがconvergedの場合だけ使い、未収束経路には追加費用を課さない。
返値を通常のqf残差と混同しないためD14Solve::worst_res_is_upper_boundを追加。
後でqf再計算した場合はfalseへ戻す。
既存event-contract研究の実残差capture時はscreenを使わない。

unitでは2800 qf評価との上界比較、mixed screen、precision-control/丸め拒否、
指数範囲拒否を確認。全根の完全性を残差だけで証明するものではなく、
duplicate rootsのテストもあくまでresidual gateのテストである。

## 再現

Phase16と同じcompile/runで-DHOLO_D14_NATIVE_RESIDUAL_AUDITまたは
-DHOLO_D14_NATIVE_RESIDUAL_SCREENを指定する。
profileはbench_adaptive_stage_profile.cpp(input_snapshot.tsv,warm)。
wholeはPhase10 runner_hybrid.cpp(input_snapshot.tsv,whole_screen.tsv,3)。
CPU0で逐次測定。以下の追加検証を参照。研究flagのまま、採用未決。

profile_root_auditは各rootのqfをもう一度計算して監査するため、
residual_msからnative_msを引いて通常qf費用と見なしてはいけない。
profile_auditは初期の全根比較のみ。screenでは監査用二重計算は行わない。

## 全epoch / root-set追加検証

`whole_screen.tsv` の14,432行が完了。Phase16 `whole_bracket_bits.tsv` 比:

|RelTol|lane|旧p50 ms|screen p50 ms|旧p99 ms|screen p99 ms|
|---|---|---:|---:|---:|---:|
|1e-3|cold|0.430809|0.414842|3.214882|3.192342|
|1e-3|warm|0.357182|0.341652|1.175892|1.138409|
|1e-4|cold|0.475504|0.458476|3.247944|3.234626|
|1e-4|warm|0.406858|0.390142|1.439117|1.425398|

全laneでmu差0、status/node差0、各tol 7216/7216収束。
VBM reference超過は1e-3=0、1e-4=既存3のまま。
warm 1e-3のmaxは55.8555→58.8629msで、tail改善は示していない。
radial-only p50は0.121722→0.121088ms（1e-3）。
別時点のbaselineとの逐次比較なので実行時変動は含む。

`roots.tsv.gz` / `events.tsv.gz` をPhase13の同じ最終root pathと照合:
14,432集合、binary64 export上の根差0、role/physical/classification/event数差0。
qf coldは両者100回。これはinterval根座標一致の証明ではない。

単体テストを各根の上界を直接照合する形に強化し、2800比較pass。
CMake/CTest native residualテストpass。
reference候補は550行中466 usableでobserved violation0。
過去Phase16の出力とは微小差があるため、同じコンパイル設定で
screenなしbaselineを再生成し、flagだけの差として確認する。
同様に220行の解析5Jacもmatched buildで確認中。

## Matched buildの検証完了

SCREENの有無だけを変えた `-O3 -DNDEBUG -std=gnu++17
-fext-numeric-literals -Isrc` のbuildで、reference 550行が完全一致。
usable 466行でobserved violation0。残り84行のreference精度は認証しない。
ValueFirst解析5Jacの220行も値・gradient・qualityを含めbyte一致。
過去Phase16出力との微小差をこの最適化の影響とする根拠はない。

今回の結論: root-wise residual screenはwhole p50で約4%の改善候補。
全根一括screenは不採用。まだ研究flagとして保持し、production routerは変更しない。
VBMへの全面勝利には達しておらず、特にuniform側の固定費が大きい。
数学的境界は正常演算範囲・rounding mode・binary80精度の条件付きであり、
有限corpusのpassだけを全入力域の形式証明とは主張しない。
