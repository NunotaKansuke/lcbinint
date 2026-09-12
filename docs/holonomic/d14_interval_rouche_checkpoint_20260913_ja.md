# D14 event contract: interval Rouché checkpoint (2026-09-13)

## 結論

D14のqf-warm候補を「全14根のglobal stepが停止したか」ではなく、14個の互いに素な円板がそれぞれ根を1個含むことによって認証するresearch経路を、外向き丸め付きbinary128区間演算へ置き換えた。14,432行のmatched whole-epoch A/Bでは、cold 24行、warm 22行をqf-cold restartなしで受理し、topology、event数、status、value convergenceの差は0だった。受理行のD14/classify中央値はcold `45.179 -> 6.090 ms`、warm `51.981 -> 5.980 ms` となった。

一方、全母集団ではwarm topology p50が約2.1%増え、whole-epochのp50〜p99は測定揺らぎ程度、maxは約1〜2%悪化した。失敗候補にも区間係数と最大14回のTaylor shiftを払うためである。このためproduction default/routerは変更せず、compile-time research flagとruntime opt-inのまま残した。数学的な受理条件はproduction候補になったが、現在の実装コストはまだproduction採用条件を満たさない。

## 実装したcertificate

binary64 `PrimaryFrame`からD14係数を直接、外向き丸め区間として構築する。候補中心 `c_i` で

\[
D(c_i+w)=a_0+a_1w+\cdots+a_{14}w^{14}
\]

へTaylor shiftし、半径 `r` について

\[
\underline{|a_1|}r > 16\left(\overline{|a_0|}+
\sum_{k=2}^{14}\overline{|a_k|}r^k\right)
\]

を外向きboundだけで判定する。さらに `3r` が他中心までの距離下界より小さいことを要求する。全14円板が成功すれば、Rouchéの定理と次数14から各円板に根が1個あり、全根を捕捉したことが分かる。係数生成、複素区間演算、Taylor shift、絶対値、べき和まで丸め誤差を含む。実行環境がround-to-nearest、非FTZ/DAZ、非fast-mathでない場合はfail closedする。

点binary128の従来判定は棄却専用prefilterへ下げた。prefilterのpassだけでは受理しない。正式certificate後の2回のstructured Newtonも、更新点が認証円板内に残ることを要求する。既知反例case 49（残差 `4.63e-28` でもpositive rootを欠く候補）は引き続き棄却し、qf-coldへ戻る。

Taylor shiftはPascal/Horner型の固定次数kernelへ変更し、明示的な中心べき生成より区間複素乗算を減らした。通常経路にはコードも費用も入らず、`HOLO_D14_EVENT_CONTRACT_RESEARCH`でcompileし、`HOLO_D14_EVENT_CONTRACT_ACCEPT=1`を指定した場合だけ動く。

## matched whole-epoch A/B

同じbinary、入力、compiler option、CPU affinityを使用し、runtime flagだけを変えた。1,804 trajectory、2 tolerance、linear/log profileを合わせた14,432行、各3 repeat中央値である。value-only adaptive、coldは毎epoch D14/topologyを再生成、warmはL2 root seed、radial-onlyはtopologyをtimer外で準備した。

| target | lane | p50 baseline→interval ms | p90 | p99 | max |
|---|---|---:|---:|---:|---:|
| 1e-3 | cold | 0.546→0.544 | 0.874→0.870 | 3.454→3.438 | 56.261→57.107 |
| 1e-3 | warm | 0.491→0.490 | 0.811→0.810 | 1.486→1.475 | 56.213→57.163 |
| 1e-3 | radial | 0.149→0.149 | 0.299→0.299 | 0.502→0.501 | 0.910→0.917 |
| 1e-4 | cold | 0.591→0.589 | 0.976→0.975 | 3.503→3.506 | 56.449→57.164 |
| 1e-4 | warm | 0.557→0.556 | 0.942→0.940 | 2.219→2.222 | 56.461→57.165 |
| 1e-4 | radial | 0.189→0.189 | 0.403→0.406 | 0.849→0.855 | 2.237→2.259 |

qf-coldはcold `58 -> 34`、warm `44 -> 22`、qf-cold sweepはcold `12498 -> 5882`、warm `12912 -> 6412`。受理46行のclassifyは約85〜89%短縮した。棄却行の追加費用はcold中央値0.425 ms、warm中央値0.979 msで、希少tailの削減を全母集団の明確なwhole勝利へ変えられていない。

全14,432行でtopology status/cell/event数、cold/warm/radialのvalue status/stop/convergenceは一致した。μ差最大はcold `6.682e-12`、warm `6.633e-12`。reference比較は1e-3でviolation 0、1e-4で既存と同じ3で、新規violationはない。受理された全根集合とqf-cold oracleの最大位置差は `1.16e-11` だが、円板がroot identity/completenessを保証し、下流event/value parityも通っている。今回の契約はroot位置をqf-coldと同値にする契約ではない。

## 残存clusterの分類

単根円板が全14個揃わないcold/warm候補を、未分離根と最近傍候補の連結成分で分類した。重複profileを除くと33 clusterで、size 2/3/4/5/6がそれぞれ16/4/9/1/3。physical-real rootを含むものが12、physical positive-realを含まないsoft-onlyが15だった。

dominant `a_m w^m` による単純なm-root円板certificateも調べ、9/33 clusterで根数を認証できた。しかし9件すべてで円板が実軸を横切った。したがって「円板内にm根」は証明できても、positive-real event数やphysical/soft分類は保証できない。この9件を受理経路へ使うのは不適切であり、全33 clusterを既存qf-coldへfail closedした。cluster certificateのproduction統合は不採用とした。

## 検証と再現

isolated CTestは19/19 pass。専用testは受理既知例でqf-coldを回避しながらevent/value parityを確認し、case 49で必ずcertificateを棄却する。raw、machine-readable summary、再現command、hashは `evidence/holonomic/d14_event_contract_interval_20260913/` に保存した。

次に削るべき箇所はcertificate不成立を早期に予測する低コストbound、または複数中心で共有できるshift/bound計算である。判定marginの緩和や実軸を跨ぐcluster円板の受理では速度を作らない。失敗費用を十分削れない限り、この経路はresearch-onlyのままが妥当である。
