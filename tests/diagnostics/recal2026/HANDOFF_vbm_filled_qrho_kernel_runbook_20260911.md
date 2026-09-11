# q–rho 追加サンプルを正しい bin で使い、指定 kernel だけを測る runbook

## 目的

ユーザーから渡された **現在の checkout にある実装**を、q–rho coverage extension を含む既存入力で測るための最小手順。これは実装作業の引継ぎではなく、1つの指定 kernel の測定手順である。

## 最重要ルール

- この資料や過去の結果に書かれた branch / HEAD は provenance であり、checkout 指示ではない。受け取った checkout・branch・作業ツリーをそのまま使う。
- `checkout`、`reset`、`clean`、merge、cherry-pick、別実装への差し替えをしない。未コミット変更も消さない。
- 実装を直したり、別の kernel を一緒に測ったり、ベンチマーク全体を作り直したりしない。測るのは依頼された `KERNEL_UNDER_TEST` だけで、VBM は固定比較側である。
- 現在の実装が呼べない、入力や reference がない、計時範囲が一致しない場合は、勝手に代替せず `BLOCKED` として報告する。

旧 `HANDOFF_multi_run_cartesian_fill_algorithm_benchmark_20260909.md` の「次にやること」は、この runbook の作業範囲には引き継がない。

## 「q–rho を埋めた」の意味

前回の追加サンプルは実際に作成済みである。元の 160 configuration に coverage extension 96 configuration（うち 51 は元の空 bin を狙った targeted sample、残り 45 は extension の remainder）を追加し、合計 256 configuration とした。extension は q–rho coverage 用なので `s=1` 固定であり、全体を iid performance corpus と解釈しない。

図の q–rho bin は、追加時と同じ protocol-clamped 定義に固定する。

```python
q_edges = np.geomspace(1e-4, 1, 13)   # 12 bins
rho_edges = np.geomspace(3e-5, 1, 13) # 12 bins
min_cell_population = 8
```

この定義なら、現在の input snapshot は `uniform` / `linear` とも **144/144 cell populated**（非ゼロ cell の最小 count は 8）である。snapshot の q–rho の exact pair 数が 211 でも、それは 144 bin を埋めることと矛盾しない。1 bin に複数の散在した pair が入るためである。

逆に、`q=np.logspace(-4,0,13)` と `rho=np.logspace(-4,0,11)` の 12×10 bin は別の図であり、同じ input をそこへ入れると 106/120 cell になる。この 14 個の灰色はサンプル追加の失敗ではなく、図の bin 定義を変えた結果なので、canonical q–rho 図には使わない。`rho<1e-4` の 856 行も protocol 下限 `3e-5` 以上なら有効な追加サンプルであり、捨てない。

追加後の元図と merged artifact は次に残っている。

```text
/tmp/lcbinint_pure_kernel_coverage256_20260909/corpus/manifest.json
/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/results.json
/tmp/lcbinint_pure_kernel_confirm_full_20260909/figures_protocol_edges.md
/tmp/lcbinint_pure_kernel_confirm_full_20260909/q_rho_linear_1e-3.png
```

## 固定する入力と比較側

q–rho 充填済みの canonical input は次である。

```text
evidence/holonomic/v2_vbm_pure_kernel_20260911/input_snapshot.tsv
```

header を除く 7,216 行で、`profile={uniform,linear}`、`epoch_index={0,7,15,23}` を含む。入力を間引いて図を埋めてはいけない。adaptive runner を使う場合も、同じ入力キー・同じ reference が対応していることを hash と行数で確認する。

速度比較は既存 VBM artifact を再利用する。

```text
/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/results.json
```

対象は VBM `RelTol=1e-3` / `1e-4` × uniform / linear。精度の照合だけは既存 VBM `RelTol=1e-6` reference を使う。

```text
/tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/plot_parts_vbm_reltol_1e-6_reference
```

`1e-6` に速度 target の意味を持たせない。速度側の条件と精度側の reference を混同しないこと。

## 実行前に残す provenance

`KERNEL_UNDER_TEST`、entry point、runner のパスを evidence report の冒頭に明記する。runner は現在の実装を呼ぶものに限り、古い `/tmp` の実行ファイルを再利用しない。

```bash
set -euo pipefail
EVIDENCE_DIR=evidence/holonomic/<kernel_label>_YYYYMMDD
INPUT=evidence/holonomic/v2_vbm_pure_kernel_20260911/input_snapshot.tsv
RUNNER=<runner_that_calls_the_current_kernel>
BIN=/tmp/<kernel_label>_runner

mkdir -p "$EVIDENCE_DIR"
{
  git branch --show-current
  git rev-parse HEAD
  git status --short
  git ls-files --others --exclude-standard
  sha256sum "$INPUT" "$RUNNER"
  wc -l "$INPUT"
} | tee "$EVIDENCE_DIR/provenance.txt"
git diff --binary > "$EVIDENCE_DIR/source.diff"
sha256sum "$INPUT" > "$EVIDENCE_DIR/input.sha256"
test -f /tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/results.json
test -d /tmp/lcbinint_pure_kernel_reference_all_1e-6_20260909/plot_parts_vbm_reltol_1e-6_reference
```

`<...>` は実際の値に置き換える。未追跡の kernel source を使う場合は、その source のパスと hash も provenance に追加する。`git diff` に出ないからといって、実装 identity を省略しない。

## compile / run

既存 V2 fixed-value runner の compile 条件は次を基準にする。新しい kernel の runner でも、compiler flags や timer の違いを黙って変えない。

```bash
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 \
  -I/rogue1_8/nunota/lcbinint/src -march=native -funroll-loops \
  -ffp-contract=fast -fno-math-errno \
  "$RUNNER" -o "$BIN" -lquadmath
sha256sum "$BIN" > "$EVIDENCE_DIR/binary.sha256"
```

計時境界は、比較対象と明示的に揃える。fixed V2 の既存例では
`flux_value_integrate(64,u,pf,topo)` 本体だけを 3 回 warm-up 後に計時し、`LensParams`、`PrimaryFrame`、`classify_cells`（D14/topology）、最終 blend、I/O は timer 外である。adaptive kernel を指定された場合は、対応する adaptive path の境界を同じように report に書く。実装全体を測った数字と pure-kernel の数字を混ぜない。

runner は stderr に `progress N/TOTAL` を出す。fixed runner は `TOTAL=7216`、1入力から `1e-3` と `1e-4` の2 targetを出す runner は `TOTAL=14432` とする。進捗表示は timer 外で、kernel の計算や判定を変更しない。

```bash
nohup "$BIN" "$INPUT" "$EVIDENCE_DIR/raw.tsv" \
  >"$EVIDENCE_DIR/run.log" 2>&1 &
echo $! > "$EVIDENCE_DIR/pid"
tail -f "$EVIDENCE_DIR/run.log"
```

## 集計と図の完了条件

- raw の data-row 数、重複 key、profile、epoch、status を確認する。期待行数に達しない実行は成功扱いにしない。
- q–rho 図は上記の protocol-clamped edge（12×12、期待 144/144）と `min_cell_population=8` で作る。灰色が出たら、まず bin edge が `rho=3e-5` になっているかを確認する。入力不足・key 不一致・filter のどれかを切り分け、点を恣意的に足し引きしない。
- 非 OK、non-converged、非有限値は raw と status 集計に残す。図で除外する場合も、除外規則と除外数を report に書く。
- 速度比は `R = t_VBM / t_kernel`（`R>1` は指定 kernel が速い）。速度は VBM `1e-3` / `1e-4`、誤差は VBM `1e-6` reference に対して別々に計算する。
- evidence には少なくとも `command.txt`、provenance、input hash、raw、joined table、summary、PNG/PDF、report を残す。図は実際に開いて q–rho の空白、単位、凡例、status 表示を目視確認する。

ここまで揃えば、この runbook の作業は完了。性能改善、失敗点の修正、別 kernel の A/B、branch への反映は別タスクとして依頼する。
