# Stationary atlas 実装・検証 checkpoint

対象は `dev/holonomic`、取り込み後の基準HEADは `7f59e1a`。仕様は
`radial_topology_stationary_atlas_plan_20260911_ja.md`。作業前のtracked差分を
`/tmp/lcbinint-before-atlas.patch` に退避し、無関係な図・文書・未追跡`.claude/`は変更・stageしていない。
サブエージェントは使用していない。

## 判断

**研究用の明示opt-in実装。productionへの採用条件は満たしていない。**
全14,432行で既存V2のfull-cold / full-warm中央値は0.633 / 0.553 ms、atlasは42.085 / 42.104 ms。
value収束も14,432行からcold 6,612 / warm 6,760行へ減るため、置換しない。
境界の実接触とno-contact coverを直接扱えるが、現実のthin-source corpusでcoverの構築が高価。
「D14固定費をradial kernelの半分以下にする」目標は未達。
新しい数学的表現一般が不可能という結論ではなく、今回のtensor subdivision / stationary tube実装の測定結果である。

## A–Eで実装したもの

A. `boundary_bernstein.hpp`はexact binary64 PrimaryFrameを対象に、元の実式から(6,4) tensorを生成する。
T/Uの有限chart、係数反転、外向きBernstein変換、微分とde Casteljau分割、P/Ps排除、
Pssと両端Psが認証されたstationary tube、PR単調性＋両端停留値、normalized Krawczykを実装。
Krawczykの存在・一意性はstrict inclusionとnorm<1の両方を要求する。
Newtonは候補生成だけで、別の包含計算による認証を要求する。

B. `radial_stationary_atlas.hpp`が `[0,Rmax] × RP¹` の二chartをcoverする。
Rmaxでは外向き演算で `(R-W)(R-a)>1` を確認。
全leafが排除または一意接触に覆われる場合だけCompleteとする。
局所Krawczykによる縮約では、捨てるstripもno-contact leafとして保存する。
精度はdouble interval→D14Real ball→必要なboxのqf interval。
証明用leafを全部積分panelにすることはない。

C. warmはeventの一意性を現在の式で検証し直し、旧除外boxも現在の式で再認証する。
安いwhole-boxのP/Ps範囲で通らないものだけtensorを再構築・細分化する。
一部seedだけ残る場合もevent IDが重複しない。失敗したatlasで既存の完全cacheを上書きしない。
これは証拠更新であって、古いevent数・中点のtopology一致を信じる再利用ではない。

D. `radial_atlas_topology.hpp`が実接触のenclosureから既存cell classifierへ渡す。
旧merge toleranceで近いcontactを消さず、投影が分離できないものは未解決とする。
exact Y=0のreflectionは、座標変換したuniquenessの再認証を通る場合だけ同時接触として扱う。
chart_p4、R=a、sqrt(m0)、axis Lのrepresentation cutを保持。
`atlas_anchor`のeventはadaptive setupで二度補正しない。
R_hi/R_loとmap offsetを保持するopt-inを追加した。
near-foldの二交点・正のbounded arcでは、同じanchorからE/Oの(M,w)を解き、
interval Krawczykを通したpairで安定な角幅と既存16/8 GC2 vKを評価する。
対応外では既存node evaluatorを使うが、Rの丸めがfoldの側を変え得るときは拒否する。
新しい重いfallbackやphysical re-seedは追加していない。

E. 既存adaptiveへ明示接続したtrajectory runnerを追加。
旧D14は計時外の照合のみで、atlasのtimed pathに旧D14 fallbackはない。
Completeは実接触coverの完了であり、mu収束や全微分の厳密保証とは区別する。
production router、fixed-n_rの既定動作、PF6/GM研究solverは変更していない。
共通quartic primitiveに任意reciprocal引数と高階局所微分を追加したが、既定引数の意味は維持する。

## 実験途中の失敗と変更理由

- 強縮約後のBernstein係数差からPssを作ると有効桁を失った。
  仕上げをD14Real局所Newton候補＋元の式のphysical-coordinate Krawczykに分離。
  通常例の6接触coverが成立し、初期試作の約144 msから約9.6 msに減った。
  それでも既存D14より遅い。
- separable Bernstein変換、cheap signの先行判定で不要なqf証明演算を削減。
- box上限を2048から20000へ上げた難例は完全coverになったが、約200 msかかる。
  上限増加を性能改善としては採用しない。
- 最初からD14Realでtensorを生成すると難例の認証率が上がる一方、cold約130–150 ms、
  warmは約200–340 msへ増加した例がある。全体の既定はdouble first。
- Krawczyk縮約では除外stripを保持。box数だけでなく全wall timeで比較する。
- warmのleaf単位再構築は現在も高価で、深いleafが蓄積する。これは最適化未達の主因の一つ。
- 監査でwarm部分生存時のevent ID重複、および異なる接触が同一binary64 cell端へ丸まる場合のmergeの穴を発見。
  どちらも修正・テストを追加し、途中計測を中止して修正後binaryで再計測した。
  後者は精度を捨てて統合せず、現行CellPlanで表現できないとしてfail closedする。

## 検証と測定

### 全14,432行の結果

全費用を含むms。wholeの列は未収束試行も含む。失敗が半数を超えるatlasのradial-onlyは、拒否時間の中央値を使わず、収束行だけを下段に示す。

|方式|lane|value収束|p50 ms|p90 ms|p99 ms|max ms|
|---|---|---:|---:|---:|---:|---:|
|incumbent|full-cold|14432/14432|0.633|9.649|17.547|30.020|
|incumbent|full-warm|14432/14432|0.553|0.988|13.338|29.962|
|atlas|full-cold|6612/14432|42.085|61.886|114.812|182.338|
|atlas|full-warm|6760/14432|42.104|62.606|115.736|187.941|
|incumbent|radial-only（収束行）|14432/14432|0.165|0.342|0.697|2.261|
|atlas|radial-only（収束行）|6612/14432|3.549|8.379|23.982|101.079|

atlas収束行だけのfull-cold / full-warm中央値も33.562 / 34.652 ms。
同一の収束行で比較した `incumbent_ms/atlas_ms` の中央値はcold 0.01748、warm 0.01411。
成功subsetに限っても明確に遅い。

epoch 0を除いたsteady-stateの中央値はincumbent 0.524 ms、atlas 42.151 ms。
warmでこの費用差が消えることもない。

「固定費 < 既存radial-only時間の半分」を満たすatlas収束行はcold/warmとも0。
自分自身の遅くなったadaptive部分を分母にするとcold 16行、warm 3行が条件を満たすが、
これは目標の達成とは数えない。

### LD / tolerance別（全試行中央値）

|profile|RelTol|V2 cold / warm ms|atlas cold / warm ms|atlas cold / warm 収束行|
|---|---:|---:|---:|---:|
|uniform|0.001|0.584 / 0.489|41.210 / 41.268|1653 / 1690（各3608行）|
|uniform|0.0001|0.621 / 0.551|41.726 / 41.725|1653 / 1690（各3608行）|
|linear|0.001|0.638 / 0.559|42.315 / 42.488|1653 / 1690（各3608行）|
|linear|0.0001|0.693 / 0.627|43.005 / 43.020|1653 / 1690（各3608行）|

### 認証・参照・timing integrity

- cold: Complete 6,632、BudgetExceeded 7,156、ArithmeticUncertain 644。
- warm: Complete 6,780、BudgetExceeded 7,008、ArithmeticUncertain 644。
- Complete後のcell化で20行ずつが未解決。Completeとvalue収束は区別する。
- 旧D14 fallbackはtimed atlas pathで0。未完了をV2へ回してcoverageを作っていない。
- incumbentの独立VBM参照超過はcold/warm/radialとも既知の3行。atlasの収束行は0、新規も0。
  atlasの未収束行で誤差が検証できたという意味ではない。
- CTest 17/17、adaptive 1,482 checks、exact plan 27 checksがpass。
- 親・分割後tensorの3,360係数を3 tierでexact rational監査してpass。
- 6 laneすべてでstage非負・有限、各stage≤whole、stage sum≤wholeを機械チェックしpass。
  `summary.json` にrowごとの未分類差の分布を保存。

### 費用の内訳

atlas coldの全試行median: topology 40.520 ms、standalone coefficient 4.878 ms、
proof loop 35.731 ms、contact refine 0.398 ms。これらの別々のmedianを足してwholeとはしない。
各rowで計算したtopology未割当差のmedianはcold 0.099 ms、warm 0.105 ms。
proof loopにはtier昇格の再構築・subdivisionも入るため、35.731 msを純粋なKrawczyk算術だけの費用とは呼ばない。

収束行でのpair専用kernel利用率はcold 33.7%、warm 30.0%（node evaluation数を分母）。
局所qf pair/GC2が通常のV2 evaluatorより重く、radial-onlyも速くなっていない。
今のuniqueness boxで選ぶ方針では、元の10行の拒否回避に必要な点より広く高精度評価を行う。
これも未最適化事項として残す。

### 16 trajectory / 128行でのA/B

|構成|cold収束|warm収束|cold p50 ms|warm p50 ms|cold box median|
|---|---:|---:|---:|---:|---:|
|default|46/128|50/128|46.875|47.613|2049|
|no_contract|46/128|46/128|32.889|35.900|1857|
|no_tubes|46/128|44/128|45.276|49.975|2049|
|no_range_probe|46/128|50/128|46.630|47.001|2049|
|initial_dd|54/128|58/128|131.710|135.433|2049|
|boxes8192|112/128|112/128|83.367|100.937|2832|

入力はcorpus順に等間隔で選んだ16 trajectory。選択入力・flag・rawは`ab/`と`ab_trajectories.tsv`。
すべて参照超過0（収束行のみ）。no_contractはこのsubsetで速いがwarm coverageを失い、
初期DDや予算増加はcoverageと引き換えに高価。全corpusの勝者とは呼ばず、productionにはどれも採用しない。

### near-foldと難例

旧soft-cut除去で拒否された10行は、同じR・同じrestored cell IDで新pair評価が通る。
共通anchorでepochを積分しても10/10収束、VBM参照相対誤差は最大1.38e-5。
ただしこの診断はbox budget=20,000であり、通常の2,048 budgetでの成功を装っていない。
難例c9/c92/exact-axisの未解決、near-axis/wide/close/fullのcover結果と未処理boxは`diagnostics.txt`。

### 旧physical/softラベルとの差

Completeのうち88行でatlasの接触数と旧physical_real event数が異なる。
22個のgeometryへ重複を除いて再調査した。

- 5 geometry（Tol/profileを展開すると20行）はR-rangeが重なる接触の同定に失敗し、
  cell化を拒否する。既存unionの証明がchart方向に依存する例もあり、seam同定は未完成。
- 残る17 geometryでは、合計20接触が旧`physical_complex`と同じbinary64半径に対応する。
  atlasの接触は全てU chart。旧D14の半径は存在しており、「D14根を失った」とは解釈しない。
  旧経路ではこれらもsoft-cutとして残るので、今回の値比較が直ちに破綻するわけではない。
- 旧t-chart probeの正規化残差は1.49e-6–6.58e-3で、既存1e-6 gateを通らない。
  同じdouble quarticの係数を反転したsubresultant candidateは17/20接触でvalid。
  これは近theta=piのt-chart評価が悪条件になることと整合する。
  **candidateの成功をそのまま新しいtopology certificateとして採用していない。**

rawは`contact_count_differences.tsv`、`contact_label_audit.txt`、`contact_label_summary.json`。
全面的なatlas置換より、既存D14のphysical probeへ有限chartを導入する狭い研究候補が見えた。
今回そのproduction変更はしていない。

### artifacts

- `evidence/holonomic/radial_stationary_atlas_20260912/summary.json`: final全corpus集計
- `trajectory/`: final raw、manifest、shard log（大きいTSVはgzip）
- `ab/`: configuration A/B rawとsummary
- `diagnostics.txt`: 旧10行・難例・same-R/same-cell診断
- `enclosures.tsv`, `enclosure_audit.json`, `exact_plan.json`, `ctest.txt`: 検証
- `environment.json`: source/binary/input provenance、compiler、CPU
- `pilot_summary.json`とpilot TSV: 開発途中の失敗案。最終binaryの速度根拠とは別。


## 未達・制限

- warmは旧uniqueness boxからseedを作って再認証する。前epoch中心からの専用predictor、
  coefficient-deltaと旧marginによる安い一括認証、上位nodeへのproof-tree coalescingは未実装。
  現在はcurrent whole-box range / tensorの再認証で正しさを保つが、速度上のCは未達。
- near-fold pair専用経路はvalueの二交点bounded arcが対象。4交点、complement arc、
  同pairからのanalytic 5Jacは専用化していない。
  None / Strict / ValueFirstとInvalid / FiniteUncertifiedの既存契約は変更していない。
- 全singularityのdeflation、一般の同時接触群の完全分離は未実装。
  不明なものをemptyやordinaryと呼ばず、未解決statusにする。
  ArithmeticUncertainは証拠が足りないという意味で、物理的な特異点と断定するstatusではない。
- 16/8則やadaptive estimatorを新しい厳密積分上界へ昇格したわけではない。
  exact係数監査、接触の局所証明、全domain cover、最終値の独立参照比較は別の検査である。
- 既存D14より速くするという採用条件は未達。これはproduction代替完成の報告ではない。

### 分割座標の監査

局所Krawczyk縮約後のbox端は任意のqf値になる。この後の物理中点をqfへ丸めると、
元のboxの厳密な1/2位置とは限らない。de Casteljauだけを常に1/2で実行してよいとはしない。
EFTで1/2位置が厳密に表現できる場合だけ安い半分割を使い、その他は
`(rounded_mid - lo)/(hi - lo)` を外向きに包含して分割する。
元のtensorと物理boxの不一致を避ける修正であり、許容誤差を広げたものではない。
exact rational監査は親boxに加え、この分割後のboxも含む3,360係数へ拡張した。

### 再現コマンド

```bash
# repository root; no concurrent compiler/CTest during timing
c++ -O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno \
  -std=gnu++17 -Isrc benchmarks/holonomic/bench_radial_atlas_trajectory.cpp \
  -lquadmath -o /tmp/bench-atlas-final
python checks/holonomic/run_radial_atlas_trajectory.py \
  /tmp/bench-atlas-final \
  evidence/holonomic/d14_precision_warm_20260912/trajectory/input_snapshot.tsv \
  /tmp/atlas-trajectory-repro --cpus 4,5,6,7 --reps 1
python checks/holonomic/summarize_radial_atlas.py \
  /tmp/atlas-trajectory-repro /tmp/atlas-trajectory-repro/summary.json
bash checks/holonomic/run_radial_atlas_validation.sh /tmp/atlas-validation-repro
```

`run_radial_atlas_trajectory.py`はshard完了markerと入力/binary SHAで再開を管理する。
rawの先頭コメントには旧harnessの共通文言が残るが、方式はファイル名・manifest・各rowのatlas statusで区別する。
atlasのtimed pathでD14を実行しているという意味ではない。
全1,804 trajectory（各4 epoch）×2 Tol = 14,432行。
各shardを独立の1 CPUへ固定し、各laneの未計時warm-up trajectoryを除外する。
今回は巨大な速度差を判定する研究benchmarkなので、timed repetitionは1回。
小さい数%の速度差を主張する測定ではない。

`LensParams{time,y,rho,1/q,s,true}` は既存の検証済み変換を使用し、構築は計時外。
RelTol=1e-3/1e-4、AbsTol=1e-16、gradient policy=None、uniform/linearを同じ入力で比較。
VBMの保存済みRelTol=1e-6参照を再利用し、VBM再計算は行わない。

- full-cold: 全domain cover・topology・adaptiveと一時object破棄まで毎epoch計時。
- full-warm: 前epochのeventとproofを再認証する全費用を含む。
  trajectoryのepoch 0はcold。steady-stateは位置1–3を別集計する。
- radial-only: 各方式のtopology/anchorを計時外で用意した同じadaptive本文。
  未完了atlasを渡した短い拒否時間を、速いradial kernelとは解釈しない。
- 既存経路ではfull-warmはL2 D14 root warm-start、L1 cell plan reuseはOFF。
- 同じraw rowのwholeから排他的stageを引き、残差をunclassifiedとして記録する。
  stageの別々のmedianを引き算して残差とはしない。
- atlasのproof_msはrange/tube/Krawczykとsubdivision、および不確かなboxのtier昇格時の係数再構築を含む。
  refine_msにはwarm seedの一意性再認証も含む。
  dd_boxes/qf_boxesはtensorのtier数であり、局所contact/pairのqf演算回数そのものではない。
  failed attemptの末尾scope時間とcache/管理費もwholeに含め、未割当分はunclassifiedへ残す。

A/B flagは `ATLAS_MODE=1` に加え、`ATLAS_NO_CONTRACT=1`、
`ATLAS_NO_TUBES=1`、`ATLAS_NO_RANGE_PROBE=1`、`ATLAS_INITIAL_DD=1`、
`ATLAS_MAX_BOXES=8192`。物理parameterによる選別はしていない。

```bash
ATLAS_MODE=1 ATLAS_NO_TUBES=1 taskset -c 4 /tmp/bench-atlas-final \
  evidence/holonomic/radial_stationary_atlas_20260912/ab_trajectories.tsv \
  /tmp/atlas-no-tubes.tsv 1
```

### ラベル診断の再現

```bash
c++ -O3 -march=native -funroll-loops -ffp-contract=fast -fno-math-errno \
  -std=gnu++17 -Isrc checks/holonomic/audit_atlas_contact_labels.cpp \
  -lquadmath -o /tmp/audit-atlas-labels
/tmp/audit-atlas-labels \
  evidence/holonomic/radial_stationary_atlas_20260912/contact_label_inputs.tsv
python checks/holonomic/run_radial_atlas_ab.py /tmp/bench-atlas-final \
  evidence/holonomic/radial_stationary_atlas_20260912/ab_trajectories.tsv \
  /tmp/atlas-ab-repro
```

### commitの区分

A: `14c3426`、B: `2386bfc`、C: `839903e`、D: `5537b17`。
追加guardは`f8f8cbb`、物理分割の包含修正は`b4d2991`。
最終計測のkernelは`b4d2991`で、Eとしてrunner・checkpoint・evidenceを別commitに保存する。
