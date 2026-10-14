# D14 warm-event Phase 6 checkpoint（2026-09-13）

## 結論

前epoch rootを少数補正してevent listを直接作るD14省略案を、production受理へ入れる前のdiagnosticとして実装した。case 0ではdouble presearch 60/200 sweep後の候補も全行でevent個数がoracleと一致せず、rootwise structured qf Newtonを1/2/4回追加しても改善しなかった。小残差だけではnear-real rootのphysical/complex分類を保証できない既知問題を再確認した。

一方、通常D14Real経路で未使用だったDD structured coefficient blockを毎epoch構築するzero workを削除した。同一binaryの旧eager切替との14,432-row matched A/Bで値・statusが完全一致し、whole中央値を約0.8〜1.3%短縮したため採用する。

## 採用した変更

`solve_d14()`はstructured coefficient blockを実際に入るprecision kernelについてだけ構築する。通常のD14Real経路では`D14StructC<D14Real>`だけを生成し、`HOLO_D14_REAL=0`のDD経路では従来どおり`D14StructC<DD>`を生成する。root solve、residual、completeness、event分類のgateは変更していない。

| RelTol | lane | eager p50 | lazy p50 | 差 | p99差 |
|---:|---|---:|---:|---:|---:|
|1e-3|cold|0.5584|0.5532 ms|-0.93%|-0.28%|
|1e-3|warm|0.5036|0.4996 ms|-0.79%|-1.91%|
|1e-4|cold|0.6047|0.5970 ms|-1.27%|-0.59%|
|1e-4|warm|0.5718|0.5652 ms|-1.14%|-0.72%|

各target/lane 7,216/7,216 converged、status mismatch 0、mu差0。cold/warmのpaired median短縮は約2.9〜3.7 us。radial-onlyも約1%動いたため、数us未満の差にはrun順やCPU状態の影響が残るが、cold/warmの四条件とp90/p99が同方向で、削除対象が明確な未使用計算なので採用した。

## 棄却したwarm event候補

研究用event-contract captureへbinary64 presearch root setを追加し、そこからevent listとadaptive値をoracleと比較できるようにした。case 0、両lane、両Tol、全variantでpresearch候補のevent count mismatchは80/80。1/2/4回の独立qf Newton後も80/80である。最大mu差は数百に達し、production受理には全く使えない。

原因はpresearchの役割がbasin locatorであり、near-real complex pairをphysical root toleranceで分類できるforward accuracyを持たないこと。residualが約1e-15の候補でも分類が異なる。独立Newtonはcluster内のroot identityを保持する保証がなく、追加回数とともにadaptive convergenceが悪化する行もある。

この結果から、D14省略trajectoryは「前rootを数回Newtonしてeventを作る」だけでは成立しない。次はrootを経由せず、前epochのphysical contact enclosureを現在の`P=P_t=0`へ局所Krawczyk更新し、event間の旧no-contact証拠を係数差boundでまとめて再認証する必要がある。soft-cut省略で落ちる11行を必須回帰集合とする。全面stationary atlasの再構築は約42 msだったため使わず、通らない局所区間だけを分割する。

production router、fixed-n_r API、PF6/GM経路は変更していない。presearch captureとNewton比較は`HOLO_D14_EVENT_CONTRACT_RESEARCH`でcompileしたdiagnosticだけに存在し、通常buildには入らない。
