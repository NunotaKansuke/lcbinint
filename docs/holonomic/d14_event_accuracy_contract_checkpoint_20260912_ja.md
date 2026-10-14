# D14 event-accuracy contract 調査 checkpoint

基準は `dev/holonomic` の `c5582b12678797da9d33af7ce90343989a5a0743`。production の D14 受理条件、qf-cold backstop、router、fixed-`n_r` 経路は変更していない。追加コードは `HOLO_D14_EVENT_CONTRACT_RESEARCH` を定義した専用 benchmark だけで有効になる。

## 結論

全14根に同じ global-step 精度を要求しなくてもよい可能性は実測で確認できた。ただし、現時点で qf-warm の小残差だけを event-level contract として採用するのは安全ではない。

非収束 qf-warm 候補は cold 58行、warm 44行（profileを別入力として数え、各 tolerance で同じ集合）あった。scalar certificate、root correction/separation による曖昧性、候補 event と局所 `P=P_t=0` 補正の分類・移動量を組み合わせた研究 gate は、cold 42/58、warm 40/44を通した。この82行では qf-cold oracle に対する positive-real event数、physical/soft分類、complex soft-cut数、cell数、topology status、adaptive stop/value coverage の差が0で、`RelTol=1e-3,1e-4` の value contract violationも0だった。

ただしこの gate は corpus 上の識別器であり、正実根を全て拾った証明ではない。既存の positive-real Sturm certificate まで要求すると通るのは cold 8行、warm 2行だけだった。この10行は event/cell/value差0で、certificateは中央値約1.1--1.2 ms、局所event構築は約0.1 msだった。該当する現行qf-coldは保存済みprofileで約11--17 msなので、限定的な研究経路としては十分な利益が見込める。一方、残る72行は `ChainPivotUncertain` が主であり、全体のqf-cold tailを置換する段階には達していない。

## 反例と必要な certificate

既知の case 6/49/152/203 を含む7,216入力を cold/warm、`RelTol=1e-3,1e-4` で監査した。D14Real の有限・小残差候補だけを使うと、未収束候補で event mismatch と value非収束が再現した。

特に case 49 / `d_bin=1` / epoch 0 は、qf-warm residual が `4.63e-28` でも positive-real candidate が6から5へ減り、physical eventも6から4へ減った。adaptiveは `EventLocationLimited` で停止し、値差はlinearで約3.24だった。これは residual を event completeness の代用にできない直接反例である。

全positive-real candidateを局所 `P=P_t=0` Newtonへ入れるだけの案も不十分だった。case 49では別の接触へ約`1e-4`移動して局所方程式を解く例がある。局所収束には、元candidateとの対応を保証する移動量bound、root separation、正実根総数certificateが必要である。今回の研究 gate はこの大移動と near-real ambiguity を拒否したが、この経験的結果だけでproduction受理条件にはしない。

## complex soft-cut

qf-cold oracle rootからcomplex D14 soft-cutだけを除いた独立armも測った。7,216入力中7,076入力でcell境界数が変わり、`RelTol=1e-3` で5行、`1e-4`で6行のvalue convergenceを失った。収束した値には要求誤差超過を観測しなかったが、status coverageが落ちるためsoft-cut全削除は不採用である。

一方、研究 gateを通ったqf-warm候補ではcomplex soft radiusのoracle差は最大`1.27e-10`で、soft event数、adaptive status/value差は0だった。complex rootsはphysical eventと同じglobal-step条件までは不要そうだが、soft-cut位置のcorrection boundと順序を個別に認証して保持する必要がある。

## 実装した診断

`D14Solve` はresearch macro時だけ、D14Real候補とqf-warm候補をqf-coldで上書きする前に保存する。専用harnessは同じ候補からD14 eventを再構築し、最終oracle eventへ残りのrepresentation eventを合わせて `classify_event_cells()` と `flux_adaptive_integrate()` を実行する。

各行にはcandidate convergence/scalar certificate、曖昧root数、positive/physical/real-soft/complex-soft数、event radius差、cell/status、adaptive stop/value、局所 `P=P_t` の成功数・最大移動、positive-root certificateの成否と時間を保存した。oracleを計算してから比較する診断であり、benchmark結果がproduction計算へ流入することはない。

## 次の最小候補

最も強い次案は、qf-warm非収束時だけ次の順で限定的に試すこと。

1. scalar/root ambiguity screen。
2. positive-real Sturm count/isolationが成功した場合だけ続行。
3. 各physical eventを、そのisolation bracketに拘束した `P=P_t=0` 補正で作る。
4. complex rootsはsoft-cut用の位置boundと順序だけ確認する。
5. どれか曖昧なら既存qf-coldへ戻る。

現在の `positive_d14_roots()` は82候補中10行しか最終gateまで通せず、失敗時にも約0.30 msを払う。まず `ChainPivotUncertain` の原因をtail subsetで調べ、既存qf-warm候補を使った局所interval repairで証明を継続できるかを確認すべきである。これを解かず経験的82/102一致だけでqf-coldを省く変更は行わない。

## 検証

Release isolated buildは全target成功。CTestは18/18 pass。raw、summary、mismatch rows、再現コマンドは [`evidence/holonomic/d14_event_contract_20260912/`](../../evidence/holonomic/d14_event_contract_20260912/) に保存した。

