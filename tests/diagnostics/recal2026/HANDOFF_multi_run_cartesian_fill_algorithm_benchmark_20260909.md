# 引き継ぎ資料：multi-run Cartesian fill のアルゴリズムとベンチマーク

作成日: 2026-09-09  
対象ブランチ: `multi-run-cartesian-fill`  
HEAD: `08c9e399ba026d4b8ffde0a9f706238612f6b3e6` (`Harden multi-run component refinement`)  
remote: `origin/multi-run-cartesian-fill` は同じコミットを指す。  
対象ホスト: `rogue1`

この資料は、次の作業者が「なぜこの実装にしたか」「何をベンチしたか」「どこまで検証済みか」を再調査せずに再開するためのメモである。アルゴリズムの説明は現在の作業ツリーを基準にする。現在の作業ツリーには、HEAD 後の未コミット変更もある。

## 1. 現在の状態

### 実装の状態

- `3af6985` で決定論的な multi-run Cartesian fill を追加した。
- `08c9e39` で coarse component の fine-grid refinement を multi-seed 化した。
- Jacobian parity を新しい run fill のトポロジー境界として使わないようにした。
- 新しい fill は通常の native Cartesian 経路から呼ばれる。
- ただし、現在のソースでは `LCBININT_CARTESIAN_FILL` 未設定または `run` のときに新経路を選ぶ。つまり現状は新経路が既定であり、以前の会話にあった「まだ default にしない」という方針とは一致していない。merge 前にこの点を必ず確認する。

### 未コミット状態

この引き継ぎを書いた時点で、関連する未コミット変更は次の通り。

```text
 M python/bind_lc.cpp
 M python/lcbinint/__init__.py
 M src/lcbinint/magnification/cartesian_run_fill.hpp
 M src/lcbinint/magnification/finite_source_magnifier.cpp
 M src/lcbinint/magnification/finite_source_magnifier.hpp
 M tests/regression/test_binary_cusp_component.py
 M tests/regression/test_component_refinement.py
 M tests/regression/test_vbm_consistency.py
 M tests/unit/test_core.cpp
```

`.claude/` と `interactive-lensing-explorer/` は、この作業のアルゴリズム差分とは別の未追跡変更である。削除・reset・checkout で巻き戻さないこと。

## 2. 設計思想

旧経路の問題は、像の形を推測しながら flood fill を修理していたことである。claim/owner と Jacobian parity を「ここから先へは進まない」という境界に兼用していたため、seed の順番や局所形状によって、次の問題が起こり得た。

- 同じ image component の中を parity の違いで分断する。
- 1 行に複数の interval がある形を、banana や horseshoe のような個別形状として修理する。
- 先に処理した seed が、後から来る seed の探索範囲を決めてしまう。
- duplicate coverage の accounting と topology discovery が混ざる。

新経路では責務を次のように分離する。

```text
seed completeness       : component certificate / caustic / boundary probes
topology discovery      : inside predicate + shared lattice + 8-neighbor run fill
duplicate accounting    : run/component の union と共有セル registry
integration             : run 端の boundary correction と limb-darkening correction
resolution improvement  : component-local coarse -> fine refinement
error decision          : diagnostics と fail-closed の convergence model
```

したがって「shape-specific heuristic を topology discovery から外した」というのが正確であり、solver 全体からすべての heuristic を消したわけではない。

## 3. 新しい native アルゴリズム

主な実装箇所は次の通り。

- `src/lcbinint/magnification/cartesian_run_fill.hpp`
- `src/lcbinint/magnification/finite_source_magnifier.cpp`
- `src/lcbinint/magnification/finite_source_magnifier.hpp`
- `python/bind_lc.cpp` の `_evaluate_preplanned_xy`

### 3.1 seed の準備

1. 点光源 image position を共通の Cartesian lattice に snap する。
2. 同じ lattice cell に落ちる seed は sort/unique する。
3. production の新経路では seed の順番を canonicalize する。
4. source center の point images だけに依存せず、finite source が caustic をまたぐ場合に必要な component certificate、caustic probe、source-boundary probe を保持する。

有限光源の端だけが caustic の内側に入る場合、source center に対応する image とは別の image component が生まれ得る。そのため、run fill に変えたからといって「source center の seed だけで全像を発見できる」という保証にはならない。seed completeness の仕事は引き続き certificate/probe 側にある。

### 3.2 maximal horizontal run fill

`fill_cartesian_runs` は一セル単位の visited set ではなく、各 row の maximal inside interval を保存する。

```text
row iy:       [lo ---------------- hi]
row iy + 1:   candidate [lo - 1 -------- hi + 1]
row iy - 1:   candidate [lo - 1 -------- hi + 1]
```

- 1 個の seed から左右へ伸ばし、`[lo, hi]` の maximal run を作る。
- 各 run から上下の row に `[lo-1, hi+1]` を frontier として渡す。これで lattice 上の 8-neighbor connectivity を表す。
- seed run は frontier 消化より先に全て登録する。後から出てくる certified seed が、先に出た seed の traversal によって抑制されないようにする。
- 同じ row の隣接/重複 interval、または上下 row の 8-neighbor 接触は union-find で component を統合する。
- claim/owner は topology の壁ではない。既に積分済みのセルを二重加算しないための accounting と、component の統合に使う。
- ordinary path では trace event を保持せず、diagnostic trace のみ最大イベント数を持つ。
- 記憶量は概ね run 数に比例し、無限に広い empty plane の visited-cell set にはならない。run/evaluation budget を超えた場合は fail-closed する。

この設計により、1 行に複数 run がある component、pinched/horseshoe、斜め接触、duplicate/permuted seed を形ごとの特別処理なしに同じ規則で扱う。

### 3.3 面積と境界補正

各 run の内部セルの contribution を足し、左右端の outside/inside state から Bennett-style の boundary correction を計算する。linear limb darkening の boundary strip correction も run の左右端ごとに残っている。

従って新経路は単なる整数セル数えではない。topology の発見と、run boundary の sub-cell 面積補正を別々に行っている。

### 3.4 component-local refinement

粗格子で component を得た後、component ごとに必要なら fine grid を実行する。

- coarse component の全 run から代表 seed を 1 個ずつ選ぶ。
- その seed を refinement factor 倍して fine lattice に lift する。
- fine grid では coarse の 8-connected component が複数 component に分裂することがあるため、最初の run だけを seed にする方法は使わない。
- fine result が finite/positive で、coarse の別 component に侵入しない (`refined_run_footprint_is_private`) 場合だけ coarse contribution を置き換える。
- fine fill が失敗した場合や footprint が private でない場合は coarse result を保持する。

この multi-seed 化と coarse-to-fine split の回帰は `tests/unit/test_core.cpp` と `tests/regression/test_component_refinement.py` にある。

## 4. Jacobian parity の扱い

`lattice_snapped_seeds(..., preserve_jacobian_side)` は legacy と新経路を分けている。

- legacy walker: 必要なら seed を Jacobian sign ごとの側に残す。
- new run fill: `preserve_jacobian_side=false`。parity sign で fill を分断しない。
- `diagnostics.parity_free_topology=true` を設定する。
- fold 近傍 seed の個数 (`fold_seed_count`) は diagnostic として残る。

「parity を topology に使わない」と「error model から fold 情報をすべて消す」は別である。現在の convergence/error estimator には、`max_jump_cells`、`rows_with_multiple_runs`、high-magnification の fail-closed floor など、まだ数値的な heuristic が残る。ここを将来整理する場合も、topology 判定と error diagnostic の責務は分けること。

## 5. legacy rollback と diagnostic selector

canonical production run では diagnostic 用 environment variable を設定しない。

| variable | 動作 |
|---|---|
| unset または `LCBININT_CARTESIAN_FILL=run` | deterministic multi-run fill |
| `LCBININT_CARTESIAN_FILL=legacy` | 旧 walker |
| `LCBININT_DIAGNOSTIC_UNSORTED_SEEDS` | seed order 依存性を再現するため legacy を選択 |
| `LCBININT_DIAGNOSTIC_DISABLE_COMPONENT_REFINEMENT` | component refinement を無効化 |
| `LCBININT_DIAGNOSTIC_DISABLE_BENNETT_LIMB` | boundary limb correction の diagnostic 比較 |
| `LCBININT_DIAGNOSTIC_SEED_ORDER=...` | fold-first/fold-last/Jacobian/reverse の順序診断 |
| `LCBININT_AREA_DIAGNOSTICS` | area diagnostics を追加 |

変更を legacy と比較するときは、同じ build、同じ input、同じ thread 数で一方ずつ実行すること。結果を混ぜた平均や、legacy の値を自動的に正解値とみなすことはしない。

## 6. JAX 経路の扱い

このブランチの今回の multi-run diff は native `finite_source_magnifier` 経路に対するもので、現在の `git diff` に JAX ファイルの変更はない。

現行 JAX binary Cartesian 経路は概ね次の別実装である。

```text
python/lcbinint_jax/api.py
  -> python/lcbinint_jax/cpp_backend.py
  -> python/bind_jax_ir.cpp
  -> discover_cartesian_support + fixed_support_kernel
```

JAX 側は tile queue、inside probe、fixed-support integration を使い、native の `fill_cartesian_runs` を直接呼んでいない。従って「native の新 run fill を入れたので JAX も同じアルゴリズムになった」とは、この資料の時点では言えない。JAX の現在の seed sort/unique、probe discovery、tile capacity は既存経路として別途存在するが、今回の pure-kernel benchmark には含めていない。

JAX も同じ topology 規則に揃えることが要件なら、次の作業を独立に行う。

1. JAX の tile discovery と native run fill のどちらを共通 core にするか決める。
2. binary/triple、forward/Jacobian、固定 shape/capacity の制約を確認する。
3. JAX の `tests/jax_ir` と native regression に同じ split/multiple-component corpus を追加する。
4. native pure-kernel benchmark とは別に、JAX の compile/warm/call/gradient を分けて測る。

## 7. C++/Python テストの現状

### 通過したもの

```bash
ctest --test-dir build --output-on-failure
```

結果: `unit_core` 1/1 passed、0.11 秒。

### まだ green ではないもの

次を実行した結果は `99 passed, 13 failed` だった。

```bash
python3 -m pytest -q \
  tests/regression/test_component_refinement.py \
  tests/regression/test_binary_cusp_component.py \
  tests/regression/test_vbm_consistency.py
```

失敗は次のグループである。

- `test_component_refinement.py`: triple cap と wide-equal-mass/planetary の refinement 3 件。
- `test_binary_cusp_component.py`: segment-interior cap 2 件、two-coefficient limb boundary 1 件、uncalibrated tolerance 1 件。
- `test_vbm_consistency.py`: polar/high-magnification route、auto error estimate、auto Nbin/refinement level、explicit grid の Cartesian/Polar、absolute-only tolerance status の計 6 件。

これらは「新 run fill の algorithm correctness が 13 件すべて壊れている」という意味ではない。route selection、現在の default、refinement 値、既存 API の期待値が混在しているため、merge 前に failure ごとの原因を分類する必要がある。ただし、現時点で Python 回帰を全面 green と報告してはいけない。

## 8. controlled pure-kernel benchmark

### 8.0 provenance warning

この節の数値は、2026-08-13 の archived paper run と同じ数値ではない。
archived run の protocol は旧 `master` 系の実装で取得され、linear LD の勝率は
`54.4%` (`1e-3`) と `59.4%` (`1e-4`) だった。一方、今回の 2026-09-06
再測定は branch `multi-run-cartesian-fill` の HEAD `08c9e39` に加えて、
作業ツリーの未コミット native changes を含む build で実行されている。

特に現在の `finite_source_magnifier.cpp` では `LCBININT_CARTESIAN_FILL` が
未設定でも deterministic multi-run fill を選ぶ。これは旧 archived run の
legacy walker とは別の実装である。multi-run 関連の commit は 2026-09-05
(`3af6985`, `08c9e39`) で、archived run のファイル時刻は 2026-08-13、
現在の extension の build artifact は 2026-09-06 である。

なお、results JSON は extension の絶対パスだけを保存し、build hash や git
commit を保存していない。そのため archived JSON と現在 JSON が同じ
`.so` パスを指していても、同じ binary だとは言えない。今回の 256-case
coverage report は q--rho coverage の作業結果として保存してあるが、旧
`54.4%/59.4%` paper table の置き換えには使わない。旧実装で q--rho だけを
拡張するなら、旧 commit/build を固定して別の extension timing を取り直す
必要がある。

### 8.1 protocol

元の benchmark protocol は次の別 worktree にある。

```text
/home/nunota/.codex/worktrees/8d2b/lcbinint/tests/diagnostics/recal2026/BENCHMARK_controlled_pure_kernel_20260813.md
```

この protocol は現在の checkout に tracked file として存在しない。再現性を保つには、将来この protocol と speed/accuracy map script を本ブランチへコピーして commit するか、上記 worktree を保持すること。

ベンチの測定対象は production LightCurve 全体ではなく、cache-warm な pure finite-source kernel である。

- lcbinint: preplanned direct-XY (`_evaluate_preplanned_xy`) の forced Cartesian/Polar inverse-ray。
- VBM: direct `BinaryMag`/`BinaryMagDark`。`BinaryLightCurve` や point-source shortcut は使わない。
- lcbinint の Nbin 選択: VBM の値を使わず、3 個の増加する grid 値の self-convergence。
- `route-filter=all`。production dispatcher の勝率ではない。
- source profile: uniform、linear limb darkening `c=0.5`。
- targets: `1e-3`, `1e-4`。
- reference epoch: block の `(0, 7, 15, 23)`。
- repeats: 5。
- maximum source bins: 400。
- point/search timeout: 30 秒、job timeout: 0。
- thread budget: `OMP_NUM_THREADS=1`。

速度比は次で定義する。

```text
R = t_VBM / t_lcbinint
R > 1 なら lcbinint が速い
```

### 8.2 corpus

run directory:

```text
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906
```

生成条件:

- 160 independent binary configurations。
- `s`, `q`, `rho` を独立 log-uniform に抽出。
- ranges: `s=[0.2,4]`, `q=[1e-4,1]`, `rho=[3e-5,1]`。
- measured `d/rho` bin: `[0,0.4)`, `[0.4,0.8)`, `[0.8,1.2)`, `[1.2,1.6)`, `[1.6,2]`。
- 1 configuration あたり 5 source positions、合計 800 positions。
- uniform/linear の 2 profiles、合計 1,600 profile rows。
- targets 2 個を含め、3,200 jobs、各 job 4 epochs。
- corpus seed `20260812`、generation worker `1`。
- reference floor の最大値 `6.988571248269248e-4`、p99 `1.1426340046751195`。

generator の再実行例:

```bash
python3 tests/diagnostics/recal2026/generate_controlled_pure_kernel.py \
  --output /tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/corpus \
  --cases 160 --seed 20260812 --workers 1 \
  --profiles uniform linear
```

### 8.3 実行分割

20 parts を serial に実行した。`d02,d06,d10,d14,d18` は各 `d/rho` bin の代表 factor `0.2,0.6,1.0,1.4,1.8`、`c0,c1,c2,c3` は case id 範囲 `[0,40)`, `[40,80)`, `[80,120)`, `[120,160)` である。

各 part は概ね次の条件だった。

```bash
OMP_NUM_THREADS=1 python3 \
  tests/diagnostics/recal2026/bench_grid_vs_vbm_pure_kernel.py \
  --input /tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/corpus \
  --output /tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/parts/d02_c0 \
  --case-count 160 --case-id-min 0 --case-id-max 40 \
  --factors 0.2 --profiles uniform linear \
  --route-filter all --seed 20260809 --repeats 5 \
  --point-timeout 30 --search-point-timeout 30 --job-timeout 0 \
  --max-source-bins 400 --targets 1e-3 1e-4
```

factor と case range だけを各 part で変更する。全 20 個の `results.json` は生成済みで、campaign log は次にある。

```text
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/campaign.log
```

merge は次で再実行できる。

```bash
python3 tests/diagnostics/recal2026/report_controlled_pure_kernel.py \
  --corpus /tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/corpus \
  --parts /tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/parts/*/results.json \
  --output /tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/merged
```

### 8.4 実行環境と provenance

- build extension: `/rogue1_8/nunota/lcbinint/build/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so`
- extension SHA256: `3c5c3da4463801ca8bf1ef88b1c74f3240454bdaebf41a4db6837e038be8ee12`
- Python `3.10.19`
- NumPy `2.2.6`
- VBMicrolensing `5.5`
- CPU: Intel Xeon Gold 6530、2 sockets、32 physical cores/socket、64 logical CPUs
- RAM: 約 251 GiB
- reported part elapsed の合計: 約 10.074 時間

注意: 実行時には `OMP_NUM_THREADS=1` を設定したが、`taskset` などによる OS-level CPU affinity は明示していない。既存 report の「one physical CPU に pinned」という文言は、この rerun については検証済み事実として扱わないこと。実行時 host load も完全 idle ではなかった。

### 8.5 結果（初回 2026-09-06 rerun）

> この表は初回 rerun の記録として残す。2026-09-09 に同じ branch、同じ
> build、同じ corpus、同じ protocol で全件を取り直した最終確認値は §8.9
> を authoritative な結果として扱う。

| profile | target | measured epochs | lcbinint wins | VBM wins | unresolved | median R | p10--p90 R | median Nbin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 3200 | 707 | 2493 | 0 | 0.574 | 0.049--1.346 | 25 |
| uniform | `1e-4` | 3199 | 19 | 3180 | 1 | 0.157 | 0.011--0.464 | 75 |
| linear (`c=0.5`) | `1e-3` | 3200 | 2278 | 922 | 0 | 2.520 | 0.358--13.191 | 25 |
| linear (`c=0.5`) | `1e-4` | 3200 | 2025 | 1175 | 0 | 1.840 | 0.144--10.278 | 75 |

candidate self-search の status は次の通り。timeout/unresolved candidate を raw result から削除していない。

| profile | target | self-converged | self-unresolved | self-timeout |
|---|---:|---:|---:|---:|
| uniform | `1e-3` | 6357 | 0 | 43 |
| uniform | `1e-4` | 5115 | 1153 | 132 |
| linear | `1e-3` | 6356 | 0 | 44 |
| linear | `1e-4` | 5398 | 871 | 131 |

### 8.6 benchmark の解釈

初回 rerun の数値に基づく記述であり、最終確認の丸め値と細部が異なる場合は
§8.9 を優先する。

この結果から安全に言えるのは、今回の pure-kernel 条件では次の傾向が出た、というところまでである。

- uniform source では VBM が優勢。特に `1e-4` では lcbinint の median `R` は `0.157`。
- linear limb darkening では lcbinint が優勢。`1e-3` の median `R` は `2.520`、`1e-4` は `1.840`。
- これは end-to-end LightCurve の speed law ではない。dispatcher、point/hexadecapole、warm-up setup、Python overhead を含まない。
- `vbm_mismatch` は独立停止規則間の disagreement diagnostic であり、どちらが正しいかの裁定ではない。
- `1e-4` の uniform には最終的な unresolved epoch が 1 件残っている。これを勝敗の測定値として数えていない。
- したがって、source profile・target・geometry によって勝者が変わり、単一の「常に速い」結論や即時 dispatcher heuristic にはできない。

### 8.7 生成済み artifact

merged result/report:

```text
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/merged/results.json
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/merged/summary.json
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/merged/REPORT_controlled_pure_kernel.md
```

3 列 2 行の speed/accuracy map は uniform/linear × `1e-3`/`1e-4` の 4 枚を生成済み。各図は上段が runtime ratio `R`、下段が cross-engine relative difference で、q-rho、A_fs-rho、A_fs-q の 3 列である。

```text
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/figures/controlled_pure_kernel_speed_accuracy_map_uniform_1e-3.png
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/figures/controlled_pure_kernel_speed_accuracy_map_uniform_1e-3.pdf
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/figures/controlled_pure_kernel_speed_accuracy_map_uniform_1e-4.png
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/figures/controlled_pure_kernel_speed_accuracy_map_uniform_1e-4.pdf
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/figures/controlled_pure_kernel_speed_accuracy_map_linear_1e-3.png
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/figures/controlled_pure_kernel_speed_accuracy_map_linear_1e-3.pdf
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/figures/controlled_pure_kernel_speed_accuracy_map_linear_1e-4.png
/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/figures/controlled_pure_kernel_speed_accuracy_map_linear_1e-4.pdf
```

PNG は目視確認済みで、軸・凡例・パネルの重なり・clip は見つかっていない。plot script はこの checkout ではなく、benchmark protocol と同じ `/home/nunota/.codex/worktrees/8d2b/lcbinint/tests/diagnostics/recal2026/plot_controlled_pure_kernel_speed_accuracy_map.py` を使った。図を長期保存するなら script も本ブランチへ移して commit すること。

`/tmp` の raw artifact は ephemeral であり、現時点では git 管理下にない。論文・再現用の正式 artifact にする場合は、容量と repository policy を確認して保存先へコピーし、build hash と commit hash を併記する。

### 8.8 q--rho coverage extension（2026-09-09）

元の 160 configuration corpus では、protocol の有効範囲
`q=[1e-4,1]`、`rho=[3e-5,1]` を 12 x 12 の log bin に分けたとき、
144 bin 中 93 bin しか占有されず、51 bin が空いていた。864 configuration
を追加する案は時間が大きいため採用せず、論文で扱いやすい合計 256
configuration にした。

- 既存の 160 configuration に、case id `160--255` の 96 configuration を追加した。
- 追加 96 のうち 51 は、既存 corpus で空だった q--rho bin を 1 configuration
  ずつ埋める coverage-targeted sample とした。残り 45 は extension 内の
  remainder sample である。
- coverage extension は `s=1` 固定にした。これは q--rho coverage を増やす目的に
  限定し、極端な `s` と `rho` の組み合わせによる長時間 job を増やさないためである。
- 追加分の timing は targeted 51 configuration のみ、`d/rho=1.4` と `1.8`
  の 2 strata で実行した。profile 2 種、target 2 種、reference epoch 4 点なので、
  extension の実測は `51 x 2 x 2 x 2 x 4 = 1,632` epoch 相当、job row 408
  （各 job は 4 epoch）である。`d/rho=0.2, 0.6, 1.0` と remainder 45 は
  extension timing の対象外である。
- extension の timing は、元 benchmark と同じ `OMP_NUM_THREADS=1`、repeats 5、
  `Nbin<=400`、point/search timeout 30 秒、job timeout なしで実行し、対象 408
  job に unresolved はなかった。
- この最終 extension 6 part の reported elapsed 合計は約 `1,212` 秒（約 20.2 分）
  だった。これは timing part の合計であり、corpus generation と merge の時間は
  含まない。864 configuration の full run は開始していない。
- 追加 extension は iid log-uniform sample ではない。従って、256 configuration
  を結合した overall speed summary は q--rho coverage の診断用であり、iid の
  canonical performance law や dispatcher の勝率として解釈しない。iid 性能の主張は
  元の 160 configuration report に限定する。

q--rho map の binning は protocol の物理範囲をそのまま使うように固定した。
具体的には `q=np.geomspace(1e-4,1,13)`、`rho=np.geomspace(3e-5,1,13)` とした。
従来の plot helper の decade-floor は観測最小値より下に edge を広げ、rho の
protocol 下限 `3e-5` 未満の帯を作ることがあるため、最終図では
`plot_controlled_pure_kernel_protocol_edges.py` を wrapper として使用した。
`min-count=8` の coverage map では、4 条件すべてで q--rho の speed/error
cell が `144/144` 有効となり、q--rho パネルの空白はなくなった。A_fs--rho、
A_fs--q の灰色領域は q--rho の未充填ではなく、そこに十分な物理サンプルがない
ことを示す別の projection なので、q--rho の充填判定とは分けて扱う。

追加 corpus、結合 report、最終図は次にある。

```text
/tmp/lcbinint_pure_kernel_coverage256_20260909/corpus
/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/REPORT_controlled_pure_kernel.md
/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/results.json
/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/summary.json
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_linear_1e-3.png
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_linear_1e-4.png
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_uniform_1e-3.png
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_uniform_1e-4.png
```

`merged_final/REPORT_controlled_pure_kernel.md` の先頭の canonical base table は
この handoff §8.9 の最終確認結果（linear `71.2%/63.5%`）を保持する。
256 configuration を結合した値は同じ report 内の coverage-extended diagnostic
table に分離してあり、q--rho map の入力を記録するための値である。

再現用の追加 script は次の通り。本 benchmark では generator で coverage
extension を作り、`append_controlled_pure_kernel.py` で base corpus と結合し、
wrapper plotter で protocol-clamped q--rho edge を指定する。

```text
tests/diagnostics/recal2026/generate_coverage_extension_pure_kernel.py
tests/diagnostics/recal2026/append_controlled_pure_kernel.py
tests/diagnostics/recal2026/plot_controlled_pure_kernel_protocol_edges.py
```

報告書の結合 timing summary は参考値として次の通りである。これは上記の通り
coverage extension を含むため、元の 160 configuration の表と直接同じ意味ではない。

| profile | target | measured epochs | lcbinint wins | VBM wins | unresolved | median R |
|---|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 3608 | 798 | 2810 | 0 | 0.566 |
| uniform | `1e-4` | 3607 | 22 | 3585 | 1 | 0.150 |
| linear (`c=0.5`) | `1e-3` | 3608 | 2452 | 1156 | 0 | 2.236 |
| linear (`c=0.5`) | `1e-4` | 3608 | 2166 | 1442 | 0 | 1.607 |

### 8.9 最終確認 rerun（2026-09-09）

「直近の handoff の強い結果を同じ条件で確定する」ため、base の 160
configuration を全 20 parts、4 profile/target 組、各 5 repeats で取り直した。
既存の結果を混ぜず、新しい output root に保存してから全 JSON を merge した。

- branch: `multi-run-cartesian-fill`
- HEAD: `08c9e399ba026d4b8ffde0a9f706238612f6b3e6`
- worktree の native changes を含む同一 build
- extension: `/rogue1_8/nunota/lcbinint/build/lcbinint/_lcbinint.cpython-310-x86_64-linux-gnu.so`
- extension SHA256: `3c5c3da4463801ca8bf1ef88b1c74f3240454bdaebf41a4db6837e038be8ee12`
- input corpus: `/tmp/lcbinint_pure_kernel_balanced_loguniform_20260906/corpus`
- benchmark seed: `20260809`、corpus seed: `20260812`
- `env -u LCBININT_CARTESIAN_FILL`、`OMP_NUM_THREADS=1`、repeats `5`
- point/search timeout `30 s`、job timeout なし、`max-source-bins=400`
- fresh output root: `/tmp/lcbinint_pure_kernel_confirm_full_20260909`
- reported part elapsed の合計: base `36,348` 秒（約 `10.10` 時間）、extension
  `1,211` 秒（約 `20.2` 分）

全 20 個の `results.json` は JSON として読め、各 part は 160 job rows、各
VBM status は `completed` だった。合計は 3,200 job rows、12,800 nominal
epoch measurements である。最終確認の canonical base summary は次の通り。

| profile | target | measured epochs | lcbinint wins | VBM wins | unresolved | win rate | median R | p10--p90 R | median Nbin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 3200 | 710 | 2490 | 0 | 22.2% | 0.574 | 0.049--1.339 | 25 |
| uniform | `1e-4` | 3199 | 21 | 3178 | 1 | 0.7% | 0.158 | 0.011--0.464 | 75 |
| linear (`c=0.5`) | `1e-3` | 3200 | 2277 | 923 | 0 | 71.2% | 2.518 | 0.358--13.122 | 25 |
| linear (`c=0.5`) | `1e-4` | 3200 | 2031 | 1169 | 0 | 63.5% | 1.846 | 0.144--10.262 | 75 |

論文用に丸めた headline は、linear LD で `71.2%` (`1e-3`) と `63.5%`
(`1e-4`)、uniform source で `22.2%` と `0.7%` である。handoff の
`71.2%/63.3%` と方向・規模は一致し、今回の同一条件の取り直しで「ボロ負け」
への反転は再現しなかった。`uniform, 1e-4` の unresolved epoch は 1 件残る
ため、その行は measured win/loss の分母から除外している。

candidate self-search の最終 status は次の通り。

| profile | target | self-converged | self-unresolved | self-timeout |
|---|---:|---:|---:|---:|
| uniform | `1e-3` | 6356 | 0 | 44 |
| uniform | `1e-4` | 5116 | 1153 | 131 |
| linear | `1e-3` | 6355 | 0 | 45 |
| linear | `1e-4` | 5395 | 871 | 134 |

q--rho extension も同じ fresh build で 6 parts（対象 51 configuration、408
job rows）を取り直した。base と合わせた coverage diagnostic は §8.8 の
更新済み table に対応し、q--rho の protocol edge
`q=np.geomspace(1e-4,1,13)`、`rho=np.geomspace(3e-5,1,13)`、`min-count=8`
で、4 条件すべてが speed/error とも `144/144` cell 有効だった。測定点は
`1e-3` が 3608 records、`1e-4` が 3607 records（後者の unresolved 1 件を除く）
である。従って、q--rho 図の空白は今回の protocol-clamped map では残っていない。

最終 report と図は次の通り。

```text
/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_base/REPORT_controlled_pure_kernel.md
/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/REPORT_controlled_pure_kernel.md
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_linear_1e-3.png
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_linear_1e-4.png
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_uniform_1e-3.png
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_uniform_1e-4.png
```

`merged_base` が論文の canonical performance table、`merged_final` と q--rho
図が coverage extension を含む diagnostic artifact である。extension を含む
結合値を 160-configuration の iid speed law として再解釈してはいけない。

### 8.10 VBM reference の `RelTol=1e-6` 再確認（2026-09-09）

q--rho 図で大きな error cell に見えた点を恣意的に除外しないため、元の
benchmark で使った同じ点を VBM reference の `RelTol=1e-6` で再計算した。
対象は、従来の error map で特に reference under-convergence が疑われた
`case_id=49,119`、linear source、`d/rho=0.6`、元の `1e-3/1e-4` の両方、
各 4 epochs（計 16 epochs）である。既存の lcbinint の grid 選択、`Nbin`、
runtime は変更せず、error map の VBM reference 値だけを置き換えた。

この再計算で、例えば `case_id=119, target=1e-4, epoch=1` の error は
旧 reference で `3.26e-2` だったものが `1.93e-6` になった。同様に
`case_id=49, target=1e-3, epoch=2` は `3.33e-2` から `2.38e-4` になった。
従って、以前の大きな error は lcbinint の失敗というより、低い tolerance の
VBM reference が未収束だったことによるものと判断した。該当点は除外せず、
元の全点を含めた図を作り直した。

参考として、同じ限定点を lcbinint 側も target `1e-6` で走らせる full harness
を background で完了させた。ただし `80` VBM epochs のうち `75` cell が
`grid_unresolved` となった（`max_source_bins=400`）。これは target `1e-6`
の lcbinint self-search がこの構成では重く、speed benchmark の canonical
結果として採用できる状態ではない。そのため、論文用の比較は元の lcbinint
runtime/grid choice を保持し、accuracy reference のみ `RelTol=1e-6` にした
targeted recheck を用いる。

再計算の manifest、targeted result、全点を含む q--rho 図は次の通り。

```text
/tmp/lcbinint_pure_kernel_precision_recheck_1e-6_20260909/manifest.json
/tmp/lcbinint_pure_kernel_precision_recheck_1e-6_20260909/targeted_reference_recheck.json
/tmp/lcbinint_pure_kernel_precision_recheck_1e-6_20260909/q_rho_linear_1e-3_vbm1e-6_reference.png
/tmp/lcbinint_pure_kernel_precision_recheck_1e-6_20260909/q_rho_linear_1e-3_vbm1e-6_reference.pdf
/tmp/lcbinint_pure_kernel_precision_recheck_1e-6_20260909/q_rho_linear_1e-4_vbm1e-6_reference.png
/tmp/lcbinint_pure_kernel_precision_recheck_1e-6_20260909/q_rho_linear_1e-4_vbm1e-6_reference.pdf
```

修正版の linear q--rho map は両 target とも `3608` records を含み、protocol
edge の `144/144` cells が有効である。修正後の global summary は次の通りで、
speed ratio 自体は元の benchmark から変更していない。

| profile | target | measured epochs | lcbinint win rate | median R | cross-code error p95 |
|---|---:|---:|---:|---:|---:|
| linear (`c=0.5`) | `1e-3` | 3608 | 67.960% | 2.236 | `9.07e-4` |
| linear (`c=0.5`) | `1e-4` | 3608 | 60.033% | 1.608 | `1.02e-4` |

raw benchmark parts は変更していない。この section の corrected parts は
`/tmp/lcbinint_pure_kernel_precision_recheck_1e-6_20260909/plot_parts_vbm_reltol_1e-6_reference`
に保存している。

### 8.11 全パターンの accuracy reference-only rerun（2026-09-09）

§8.10 の targeted recheck を全パターンへ拡張した。入力は canonical
benchmark の 26 parts / 3,608 jobs で、各 job の 4 reference epochs、計
14,432 epoch points を VBM `RelTol=1e-6` で再計算した。lcbinint の出力値、
選択された `Nbin`、lcbinint/VBM の測定時間、速度比 `R` は入力 JSON から
そのままコピーし、再測定していない。従ってこの rerun は accuracy reference
だけを変えたものであり、速度比較の workload と分母は元 benchmark と同じである。

全 3,608 jobs / 14,432 reference epochs が completed で、speed-related fields
（`chosen_grid`、`chosen_nbin`、`chosen_seconds`、`ratios_vbm_over_lcbinint`）
が元データと一致することを検証した。protocol-clamped q--rho map は全 4 条件で
`144/144 cells` 有効、最小 cell population は `8` epoch points だった。
`uniform, 1e-4` の measured points が `3607` なのは、元 benchmark に残っていた
lcbinint grid-unresolved 1 epoch を速度比較から従来通り除外したためであり、
VBM reference の再計算失敗ではない。

全点を使った最終図の summary は次の通り。`R>1` は lcbinint が速いことを表す。

| profile | target | measured epoch points | lcbinint win rate | median R | p95 cross-code error |
|---|---:|---:|---:|---:|---:|
| uniform | `1e-3` | 3608 | 22.118% | 0.566237 | `8.10e-4` |
| uniform | `1e-4` | 3607 | 0.610% | 0.149860 | `8.14e-5` |
| linear (`c=0.5`) | `1e-3` | 3608 | 67.960% | 2.23571 | `7.93e-4` |
| linear (`c=0.5`) | `1e-4` | 3608 | 60.033% | 1.60750 | `7.97e-5` |

再現用 script と provenance は次の通り。

```text
tests/diagnostics/recal2026/recheck_vbm_reference_all_pure_kernel.py
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/manifest.json
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/plot_parts_vbm_reltol_1e-6_reference
```

最終 q--rho 図は次の通り。

```text
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_uniform_1e-3_vbm1e-6_all.png
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_uniform_1e-4_vbm1e-6_all.png
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_linear_1e-3_vbm1e-6_all.png
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_linear_1e-4_vbm1e-6_all.png
```

### 8.12 `1e-4` map の低誤差 colour scale alternative（2026-09-09）

`1e-4` の下段 p95 difference は、全 cell 値が概ね
`3.5e-6--1.1e-4` に入るため、標準の error boundary（最初の境界が
`1e-4`）では低誤差側が飽和して見える。そこで data selection、speed
map、q--rho binning、minimum-count rule は変えず、下段の colour boundary
だけを

```text
1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3
```

へ移した alternative を作った。これは同じ p95 values の表示スケール変更で
あり、測定結果や速度比較の変更ではない。低誤差版の再現用 wrapper と図は次の通り。

```text
tests/diagnostics/recal2026/plot_controlled_pure_kernel_protocol_edges_low_error.py
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_uniform_1e-4_vbm1e-6_all_lowerr.png
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_linear_1e-4_vbm1e-6_all_lowerr.png
```

論文用には、この低誤差版を `1e-4` の accuracy diagnostic として使う方が、
標準 scale より cell 間の構造を読み取りやすい。標準 scale の図も比較用に残す。

### 8.13 target 間で揃えた runtime colour scale（2026-09-09）

上段の runtime map は標準 scale の高側（`>=64x`）が実際の cell median
分布に対して広すぎたため、profile ごとに `1e-3` と `1e-4` で共通の
colour boundary を使う matched-scale 版を作った。下段の error scale は
`1e-3` では標準、`1e-4` では §8.12 の低誤差 scale を使っている。

| profile | shared runtime boundaries |
|---|---|
| uniform | `0.01, 0.1, 1, 2, 3, 4, 6, 8, 16` |
| linear (`c=0.5`) | `0.03, 0.1, 1, 2, 4, 6, 8, 12, 16` |

この設定では `R=1` の class を白（`1--2x`）に戻している。linear の cell
median 最大値（約 `14`）も `12--16x` の範囲に入り、旧 `>=64x` にまとめて
潰れない。target 間で色の意味は同じなので、`1e-3` と `1e-4` の比較に使える。
再現用 wrapper と図は次の通り。

```text
tests/diagnostics/recal2026/plot_controlled_pure_kernel_protocol_edges_matched_scale.py
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_uniform_1e-3_vbm1e-6_all_matchedscale.png
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_uniform_1e-4_vbm1e-6_all_matchedscale.png
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_linear_1e-3_vbm1e-6_all_matchedscale.png
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/figures/q_rho_linear_1e-4_vbm1e-6_all_matchedscale.png
```

## 9. 次にやるべきこと

1. 上記 13 Python failure を、run fill、component refinement、route selection、既存 API expectation に分類する。
2. 現在の uncommitted native/Python/test 差分をレビューし、無関係な `.claude/` と `interactive-lensing-explorer/` を含めずに commit する。
3. native new/legacy を同一 corpus・同一 build・同一 environment で A/B 比較し、accuracy arbiter を別途用意する。
4. JAX binary Cartesian FFI が新 run fill と同じ topology 要件を満たすのかを決め、必要なら JAX 側を別実装として変更・テスト・ベンチする。
5. run cap と high-magnification/multi-run の fail-closed 範囲を測る。大きな run 数を含む stress case を残す。
6. default を本当に新経路にするかを、テストと accuracy arbitration の後に明示的に決める。現在はソース上すでに default なので、方針を維持するならその事実を commit message/README に反映する。

## 10. 重要な再開ポイント

最初に確認するコマンド:

```bash
cd /rogue1_8/nunota/lcbinint
git status --short
git branch --show-current
git rev-parse HEAD
ctest --test-dir build --output-on-failure
```

topology の主経路を読む順番:

```text
src/lcbinint/magnification/finite_source_magnifier.cpp
  inverse_ray_cartesian_core
    -> lattice_snapped_seeds
    -> fill_all_cartesian_components_multirun
      -> fill_cartesian_run_union
        -> fill_cartesian_runs
      -> lift_cartesian_component_run_seeds
```

まず「seed completeness は certificate/probe」「topology は run union」「accuracy は refinement/error model」という三つを混ぜないこと。ここを守る限り、中心 image seed だけで全 finite-source image が必ず発見できる、あるいは parity を外せば seed 問題も自動的に解決する、という誤った一般化を避けられる。
