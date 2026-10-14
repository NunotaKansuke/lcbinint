# D14 positive-real Sturm / Newton 実装 checkpoint

対象は `dev/holonomic`、仕様は `d14_positive_real_sturm_newton_plan_20260911_ja.md`。
基準 `0dec591`（数値baselineは既存D14Real mixed all-root solver）。変更はisolated holonomicのみ。

## 実装と契約

- A: `D14EventPolicyScope` に AllComplexSoft / NoProjectedComplexSoft / PositiveReal を追加。adaptive metadataを要求するisolated callerだけが切り替わる。通常APIの既定は従来どおり。
- B/C: 固定容量degree-14 PRS。double interval → D14Real center + outward radius → qf intervalの順に、元のbinary64 PrimaryFrameをexact dyadicと解釈した構造式から再生成する。係数生成、正スケールpseudo-remainder、評価の誤差を包含する。未確定符号をzeroにしない。FTZ/DAZ、非nearest rounding、fast-math環境は拒否する。
- D: current global root count、前回estimateからのD14Real Newton predictor、現在の多項式での非重複sign bracketを使う。N本揃えばgap countは不要。不足時は重なる候補のunionと未被覆gapをcountして修復する。cacheは全複素根cacheと別に保持する。古いchainは使わない。
- E: 正実根のenclosureからeventのqf radius boundsとhi/loを生成する。別の認証根を旧絶対thresholdでmerge/skipしない。cell midpointが根enclosureの外にあることを確認する。adaptiveの局所fold correctionが元の認証区間を離れれば拒否する。
- root assuranceは Incomplete / LegacyValidated / PositiveRealCertified を区別。後者が意味するのは対象H*の正実根count/enclosureのみで、最終fluxの厳密積分誤差保証ではない。adaptiveのEstimated契約は維持する。
- PositiveRealで認証不能なら既存全複素根backendに明示的に戻る。soft-cutはその場合も除去したままなので、Bとの比較でsolverの影響を分けられる。fallbackを正実根認証成功とは数えない。

## 意図的に未達として残す項目

- exact axis / Z≡0のsquare-free factor・gcd認証は未実装。MultiplicityUnresolvedからLegacyValidatedへ移る。微小Yをzero扱いしない。
- PRSのdependencyによる包含幅増大を解決する新しいsubresultant表現は実装していない。qfでもpivotの符号が不明なら終了する。
- 現在のrefinement targetはrootの相対幅2^-49。consumerから特定rootだけ追加refineするAPIは未実装。Rmaxとenclosureが交差する場合やbinary64で境界を表せない場合はRepresentationLimitedとして既存backendへ戻す。
- qfのv_lo/v_hiがenclosureのauthority。D14Real lo/hiは表示・予測用の近似であり、それ単独を外向きendpointとして扱わない。
- complex soft-cutを数値的hintに置き換える新しいpanel設計は加えていない。今回のA/Bで除去が安全かを先に判断する。

## 再現

コンパイルと測定:

```sh
c++ -O3 -std=gnu++17 -Isrc -march=native -ffp-contract=fast \
  benchmarks/holonomic/bench_d14_positive_trajectory.cpp -o /tmp/d14-positive-final -lquadmath
for mode in all-soft no-soft positive; do
  D14_EVENT_POLICY="$mode" taskset -c 0-7 /tmp/d14-positive-final \
    evidence/holonomic/d14_precision_warm_20260912/trajectory/input_snapshot.tsv \
    "evidence/holonomic/d14_positive_real_20260912/${mode}-final.tsv" 3
done
python checks/holonomic/summarize_d14_positive.py evidence/holonomic/d14_positive_real_20260912
```

同一実行ファイル・同一入力、3回のmedian反復。value-only / GradientPolicy::None、atol=1e-16、RelTol=1e-3/1e-4、uniform/linear。
1804 trajectory ×4 epoch ×2 target = 14432 rows/backend。VBMの保存RelTol=1e-6 referenceを再利用する。
座標のmappingは既存検証済みrunnerと同じ `LensParams{time,y,rho,1/q,s,true}`。LensParamsの準備はtimer外。
full-coldは毎epoch D14/topologyを生成。full-warmはL1 OFF、前epochのroot seedだけ利用してtopologyを再生成。radial-onlyはtopologyをtimer外で準備する。
全epochと初回除外steadyを別集計。失敗行も時間分布には含め、成功率を別に示す。
variation_msはisolation/refinement内の部分計測であり、stage sumへ二重加算しない。
prebuiltのroot stageはtimer外の同じcold topology準備時の診断。warmのbackend countersは実際のwarm topology結果から取得する。

## 結果と採用判断

**既定への採用は見送る。** 投影soft-cut除去に10行の収束回帰があり、positive-realの包含PRSは通常ケースの固定費を増やした。新しいsolverだから既存solverより速いとはならなかった。

全14432行、単位ms。失敗行も含む。

|backend|cold p50 / p90 / p99|warm p50 / p90 / p99|radial p50 / p99|value収束|
|---|---:|---:|---:|---:|
|all-soft|0.632 / 9.601 / 17.550|0.551 / 0.988 / 13.406|0.166 / 0.699|14432/14432|
|no-soft|0.637 / 9.712 / 17.610|0.552 / 1.057 / 13.378|0.167 / 1.301|14422/14432|
|positive|0.986 / 1.648 / 17.629|0.879 / 1.423 / 11.302|0.169 / 1.308|14422/14432|

初回epochを除外したsteady warm（10824行）:

|backend|p50|p90|p99|
|---|---:|---:|---:|
|all-soft|0.520|0.881|1.397|
|no-soft|0.520|0.936|1.864|
|positive|0.824|1.356|2.411|

positive-realでcold p90が改善した一方、cold/warm中央値とsteady warm tailは悪化した。全epoch warm p99の改善をsteady-stateの改善とは読まない。

### root側とsoft-cut側の分離

- all-soft: 14432/14432収束。
- no-soft: 14422/14432。positive: 同じ14422/14432。B→Cによる追加の収束回帰は0。
- 正実根cold認証: 4524/14432 (31.35%)。9904行はChainPivotUncertain、4行はSplitSignUncertainから既存backendへ戻る。これは「認証済み」には含めない。
- warm: positive cacheを試した3304行中1792行がglobal count + sign bracketsで直接完結した。counterのwarm_attemptsはprecision tierごとにも増えるため、epoch数は`>0`で集計する。
- 既存複素cacheが利用可能なfallbackではそれを既存backendへ渡す。runnerの旧warm_l2/l3はpositive cache有無を数えるため、legacy fallback内部のwarm成功率の根拠にしない。
- 保存VBM RelTol=1e-6 referenceとの要求誤差比較には、baselineから存在する3行のobserved violationが残る。全laneでA→B、B→Cの**新規violationは0**。既存3件を今回解決したとは主張しない。
- timing sanity PASS。非有限・負のstage、wholeを超える個別stageを機械検査した。stage差の分布はsummary.json。単なる中央値差を排他的な費用として加算しない。

### 固定費

positiveのcold診断medianは、係数包含構築0.099 ms、chain構築0.143 ms。この約0.24 msを先に払い、約69%は既存backendをさらに実行する。
isolation/refineの全行medianが0なのは、その前のchainでfallbackした行が過半数だからであり、無料という意味ではない。refine p90は0.655 ms。内訳の全分布をraw/summaryに残した。

現行の包含PRSをさらに細かく最適化して勝利を演出することはせず、ここでこの実装の採用判断を止める。次に検討するなら、係数interval間の依存によるPRS膨張と、soft-cut省略時のnode条件を別課題として扱う必要がある。

### 開発中に修正した誤り

初期prototypeは独立に正規化されたSturm S1をNewtonのP'として使い、スケールが一致していなかった。最終版はS0から導関数を再生成する。局所テストのopen-endpoint比較も、`-1 - (-1) == 0`で不明符号が誤ってPASSしないよう修正した。最終A/B/Cは両方の修正後の同一binaryによる。

## 検証・raw artifact

`checks/holonomic/run_d14_positive_validation.sh`でCTest、C++ enclosureのexact dyadic監査、soft-cut reject診断、既存exact-plan検査を再現する。CPUを測定と取り合わないよう、validationは全timing終了後に実施する。
- 最終CTest: 16/16 PASS（adaptive / prepared / root-pair / D14を含む）。
- exact dyadic監査: 17 PrimaryFrame、3 precision tier合計765係数包含を独立式で確認。6問題で認証成功した38 root enclosureについて、SymPy exact rational root countが全区間で1、global countも一致。認証できなかった11問題を成功数に含めない。
- 既存exact-plan: 33多項式 / 391 checks PASS。
- source/kernel変更後の同一binary SHA256、入力SHA256、compiler/CPUはenvironment.txt。

### soft-cut回帰の具体例

case_id=0 / configuration_id=0 / linear / d_bin=0 / epoch_index=7、RelTol=1e-3。
all-softは107 nodesでConverged。B/Cは158 nodes、level=8 / node_slot=1 / R=0.13608325925322462でArcWidthUnresolved。
D14が返すeventはR=0.13608325900448376で、adaptiveの局所補正後はR=0.13608325925605302。失敗nodeは補正後foldから約2.83e-12内側にある。同じD14正実根とtopology=OKでも、soft-cutを外したmeshがこのnodeに到達する。
具体的な拒否は既存 `width > de+dl`（endpoint位置誤差・lens-map cancellationの推定）の不成立で、D14根数不明による拒否ではない。gateを緩める修正はしていない。
全10行を、**元のrestored cell・同じR**でfresh quartic/root stateと比較した結果:

|epoch / profile|両tolの行数|最初のreject|same-R same-cell cold|
|---|---:|---|---|
|7 / linear, uniform|4|ArcWidthUnresolved|成功|
|15 / linear, uniform|4|ArcKindMismatch|同じ拒否（既存local cold retryでも拒否）|
|23 / linear|2|InnerPhiNonpositive|成功|

したがって、6行はseedを外すと局所評価が成功するが、残る4行をwarm seedだけの問題とは説明できない。局所quarticのcell整合性の追加監査が残る。今回、そのための新fallbackやgate緩和は追加していない。
診断の途中でRだけからcellを再選択すると、event補正後のRが隣の未補正cellへ入って見えることが分かった。最終diagnosticは必ず失敗時のcell IDを使う。Rだけ一致させた隣接cell評価を成功の証拠にはしない。

`soft-cut-diagnostic.txt`に全10行×3方式のイベント列・局所補正値・rejectを保存した。全回帰行は`*-regressions.tsv.gz`に保存。


rawとsummary: `evidence/holonomic/d14_positive_real_20260912/`。
`*-final.tsv`（保存時gzip化）だけが同一binaryの正式A/B/C。初期`all-soft.tsv`は開発途中の予備測定であり、正式比較には使用しない。

## Commit構成

- `c3507f8`: A、soft-cut policyと比較runner。
- `f97c436`: Bのenclosure PRS基盤。
- `f0b895a`: B/C/Dのfiltered isolation・refinement・warm repairとroot契約テスト。
- `41ba6ef`: E、positive cache・event・adaptive接続。正式benchmarkのkernel revision。
- `f33b6c5`以降: exact audit、same-cell regression診断、checkpoint/evidence。

kernelのdefaultは一切PositiveRealに変更していない。通常のV2を引き続き使う。既存の作業中figure変更・未追跡docs/evidence・`.claude/`はこのcommit群の対象外。
