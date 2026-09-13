# Phase 69: 微分専用の局所refinement controller

基準4cfbdc8。experimental adaptive pathに実装。production router/fixed-n_r、D14、K-rule、value safety=2は変更なし。

## 設計

値が収束したらprimal snapshotとerror ledgerを固定し、以後はgradient errorだけで優先区間と停止を決める。gradient meshの一時的なvalue誤差へ引き戻されるschedulerを修正した。

新しい `AdaptiveConfig::gradient_local_refinement=true` では、微分成分ごとに

`E_radial = max(weighted interpolation detail, gradient_difference_safety * |Q_N-Q_coarse|)`

を計算する。既存sampleだけを使い、従来の保守的なdetail推定を小さくしない。inner/geometry/event/roundoff errorは別々に残し、global ledgerは区間誤差を絶対値で合計する。これは数値的推定であり厳密包含保証ではない。

各成分の独立 `max(grad_atol, grad_rtol*abs(grad))` で正規化した区間誤差が最大のpanelを選ぶ。通常はnested p-refinementを続けるが、微分detailの減衰が不十分で、そのpanelのradial errorだけでもglobal budgetを超え、level>=gradient_split_min_level（標準4=15点）ならh-splitする。全panelを15点上限にはしない。

左右子panelは両方の評価成功後に入れ替える既存transactionを維持。中断で片側積分を落とさない。Invalid/FiniteUncertified、inner/event accuracy limitを維持。

## 難例A/B

3選択微分×local無効/最小level4/5×gradient rtol1e-2/1e-3×追加round8/32/64=54trial。value rtol1e-3、gradient atol1e-4、独立node budget4096。warm-up後の同一パラメータwarmで比較、trajectory速度試験ではない。

64round・grad rtol1e-2で、Phase66独立direct-angular/GL value有限差分参照に対し:

| 微分 | 従来 nodes / 相対誤差 | local level4 nodes / 相対誤差 |
|---|---:|---:|
| caustic-cross uniform a |2650 / 0.515%|577 / 0.0881%|
| caustic-cross LD X |2651 / 0.363%|586 / 0.0501%|
| rand030 uniform rho |203 / 約2.47e-6|203 / 同じ|

全て選択微分はFiniteUncertified。実測誤差が小さくても未認証をOKへ変更しない。短budget・別levelでは非単調性が残る。点数削減をwall-time速度改善と混同しない。全5Jacの独立精度認証は未実施。

## 既存corpus

110入力×2value tolerance、各local有無で計440評価。標準の追加4round/4096nodes、gradient標準tol、全5成分を出力。

primal muは全220pairでbitwise一致、value coverageは両方200/220（既存20行未収束）、新規Invalid componentは0。
rand035のu=0/.5、value rtol1e-3のX/Y計4成分はToleranceMet→FiniteUncertified。新推定が大きくなることと4round予算による未認証であり、精度改善の証明ではない。この分類変化と小規模独立参照しかないことからlocal controllerのdefaultはfalseに保つ。利用する実験では明示的に有効化する。

## 検証

- adaptive unit: 1506 checks / 0 failures。追加の鋭いrational peakの微分積分をatan解析解と比較。Strict/ValueFirstで契約達成、局所split、primal保持、sample再利用を検証。
- relevant CTest: adaptive_radial / quartic_sturm pass。
- value reference checkerも再生成。件数・observed violationはsummary.json参照。

raw、probe、集計、CTestログは `evidence/holonomic/gradient_controller_phase69/`。

再現（repo root）:

```sh
bash evidence/holonomic/gradient_controller_phase69/run.sh
bash evidence/holonomic/gradient_controller_phase69/run_corpus.sh
cmake --build /tmp/lcbinint-holonomic-adaptive-build --target test_adaptive_radial check_adaptive_reference -j2
ctest --test-dir /tmp/lcbinint-holonomic-adaptive-build -R 'holonomic_(adaptive_radial|quartic_sturm)$' --output-on-failure
taskset -c 0 /tmp/lcbinint-holonomic-adaptive-build/check_adaptive_reference evidence/holonomic/adaptive_radial_20260911/reference_cases.tsv > evidence/holonomic/gradient_controller_phase69/reference.csv
python3 evidence/holonomic/gradient_controller_phase69/summarize.py
```

## 残課題

実装したのは微分専用の区間誤差推定と、減衰を見た局所分割。全難例の微分収束を解決したとは言えない。LD inner derivativeの独立node auditとevent-error支配例の検証を次に行う必要がある。今回の良い64round結果を、標準4roundで得られると説明しない。新controllerの広域採用には、より広い独立5Jac参照と追加費用の実測が必要。

利用例:

```cpp
cfg.gradient_policy = GradientPolicy::ValueFirst;
cfg.gradient_local_refinement = true;
// grad_atol / grad_rtol はvalueと独立に利用目的に合わせて指定。
// 追加予算は既定4roundのまま。難例A/Bだけ64へ増やした。
```

raw.tsvの歴史的列名max_levelは、このPhaseでは0=local無効、4/5=gradient_split_min_levelを意味する（value max_levelは8固定）。reference最終結果は550行、466usable、observed violation 0。
