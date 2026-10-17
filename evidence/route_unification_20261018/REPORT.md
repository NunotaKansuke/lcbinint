# Algebraic-boundary 経路の一本化 — 2026-10-18

## 確定した実行経路

対象は `/rogue1_8/nunota/lcbinint/experiments/algebraic_boundary`、
ブランチ `experiment/algebraic_boundary`、基点 `a66ebfa2` の既存変更を保持した作業ツリー。
`40e66e51` は旧 API 統合前の seeded 値経路を調べる基準として使用した。
その後の微分安定化の保存済み修正も保持しており、ブランチ全体を過去へ戻していない。

公開入口は通常の `LightCurve(times, params)`。
`with_gradient=True` と `with_diagnostics=True` は同じ入口の指定である。

1. point / hexadecapole の共通 selector。
2. 積分が必要なら caustic-aware seed と seeded radial frontier + nested Fejér。
3. 微分なしは value-only kernel。微分ありは同じ C++ epoch controller が
   fused の値と tangent を一緒に返す。独立した値積分をもう一度行わない。
4. 微分の桁落ちが問題になる滑らかな光源では、既存の source-disk 座標による
   値と tangent の組を使う。接触時の既存 regular-chart / Gaussian replay 修正も保持。
   これは旧 algebraic-boundary / 逆レイ積分器への切替ではない。

`experimental_algebraic_boundary_integral` と旧帯積分 AD は通常バックエンドから呼ばない。
独立していた `_lcbinint_holonomic_fused` 拡張は通常ビルド・インストールから除外。
その後の最終整理で旧 Python helper、standalone binding、D14 preplanned は削除した。
現行実装は `algebraic_boundary/` と内部 `tangent/` に置き、ビルド時の overlay 生成も廃止。
Cartesian/Polar と通常 flood fill は独立した既存機能として維持している。
最終状態は [cleanup report](../backend_cleanup_20261018/REPORT.md) を参照。

対象は静的 single-source binary、uniform / linear LD。
二次 LD (`d != 0`) は対象外と明示し、旧実装へ黙って回さない。
rho=0 と通常 API の省略パラメータ既定値にも対応する。
VBM、重心、従来座標の変換を検証した。point solver / seed generator に渡る座標は既に
COM lens frame に変換済みなので、半径積分側で再びユーザー原点を適用しない。

## 診断と返り値

`value_converged` は数値積分の収束、`support_certified` は全支持領域の認定。
公開 Python が以前この二つを混同していた箇所を修正した。
`status` は支持未証明ならその状態を保持する。
`gradient_valid` は候補の利用可能性を示し、`gradient_precision_checked=false`。
未認定の有限候補は保持し、別積分器、有限差分、ゼロ、1 等で穴埋めしない。
有限候補がない実際の計算失敗は明示的なエラーとし、NaN の数値配列を返さない。
`gradient_stop` に seed 未証明などの理由を残す。

JAX CPU FFI も同じ C++ 値／ヤコビアン入口を使う。
具象入力での `with_gradient=True` は `jax` オプションや辞書順に関係なく
既に計算した fused 結果を使い、別の Jacobian 走査を行わない。
Parametersコンテナでも物理7パラメータの同じ勾配を返し、
LD係数等の未対応微分を有限差分で穴埋めしない。
通常の native 呼び出しの tracer 判定で JAX を不要に import しない。

## 検証結果

- 公開 API 回帰テスト 42 件、および C++ `unit_core` を通過。
- 既存120ケース、3960時刻をすべて実行。値3960/3960、
  微分27720/27720成分が有限。値専用／fused とも数値収束3960/3960。
- 比較可能な過去119ケースの forward 値は全点ビット一致。
- 同じ atol=reltol=1e-3 で、値専用と fused の最大相対差は
  9.25178097104059e-4。変化する quadrature chart の結果も要求値許容差内。
- `case_107`、`case_117` とその接触前後、case_033/081、および
  以前の修復ケース004/049/058/109を含めた。
- 微分の信頼性フラグは3928/3960時刻で true。
  残り32時刻の有限候補は保持され、理由は `seed_support_unproven`。
  これは全ケースの支持・微分精度認証を達成したという意味ではない。

`finite_differences.json` に同じ値エンジンを1e-7へ厳しくした差分の
ステップ走査と各支持・収束フラグを保存した。
case_081 の u0 と rho、引き継ぎ case_117 の q で旧勾配の問題を確認。
case_117 の off-contact q は fused 81.5603 に対して差分81.5519〜81.5648。
接触近傍の t0 差分にはステップ依存が残り、全微分精度の保証は主張しない。
小さい微分信号では相対差だけで正誤を断定しない。

## 実測時間

float64、atol=reltol=1e-3、1 CPU、OpenMP/BLAS 1 thread。
対象のパラメータ依存状態を持たない新規 worker で測定。
別レンズで Python とパラメータ非依存の数値則を初期化した後、
未評価の対象 light curve を1回計算する。
各5回の中央値、before/afterを交互に実行。
caustic、seed、selector、公開 API の処理を含み、微分なしは微分を計算しない。
VBM は BinaryLightCurve、同じ LD/許容差、新規インスタンス、3回の中央値。
`cold_timing.json` に全サンプルを保存。
初回積分の数値則初期化も含める別計測は `first_integral_timing.json`。

単位: ms。before は作業開始時の旧積分器を呼ぶ公開 API。

| ケース | before 値 | after 値 | before 値＋微分 | after 値＋微分 | VBM 値 |
|---|---:|---:|---:|---:|---:|
| point | 0.434 | 0.365 | 0.465 | 0.484 | 0.225 |
| crossing_uniform | 3.477 | 4.373 | 67.482 | 5.740 | 2.365 |
| crossing_ld | 3.608 | 9.266 | 72.862 | 10.799 | 5.895 |
| wide_ld | 2.198 | 2.032 | 2.187 | 2.169 | 0.510 |
| case_107 | 0.233 | 0.203 | 0.220 | 0.217 | 0.064 |
| case_117_contact | 5.314 | 2.753 | 26.095 | 8.072 | — |

pointは201点、通過・wideは101点、case107は33点、case117_contactは3点。
case117のVBMは既存の光源縁付近の参照不安定性が記録されているため、
この表では速度・真値の基準として使わず、値エンジンの差分検証を別に保存した。
値のみは旧積分器の方が速い条件もある。例えば通過LDでは旧値ルートより遅い。
旧積分器への条件分岐を復活させてこの差を隠していない。
微分追加費用1.3倍以内を全入力で達成したとも主張しない。

ノートブックの401点、linear LD .5、atol=reltol=1e-4 の1回計測では、
値37.71ms、VBM133.85ms、値＋微分73.69ms。
最大VBM相対差6.24e-5、値401/401数値収束、微分401/401利用可能。
この1回計測は上の5回中央値とは分ける。

## 再現と provenance

- `manifest.json`: HEAD、対象 source/binary SHA256、build root。
  source と build の Python wrapper は一致。
  compiler dependency に別 checkout の source は含まれない。
- `source-before/`, `before.diff`, `before.json`, `before-package/`: 既存変更を含む開始状態。
- `after.diff`: 終了時の tracked source 差分。
- `stress.py`, `audit.py`: 保存120ケースの再計算。
- `cold_benchmark.py`: target-dependent 状態を再利用しない速度比較。
- `run_tests.py`: editable install に遮られず指定 build をテスト。
- `execute_notebook.py`: plain Python の全コードセルと図を実行。
  sandbox がローカル socket を許さないため、Jupyter kernel 自体ではなく
  セルを順番に実行して出力を保存した。図は目視確認した。
- `notebook-cells.log`, `notebook.png`: 実行ログと図。

ビルドは `cmake -S . -B build-route-unified -DCMAKE_BUILD_TYPE=Release
-DCMAKE_PREFIX_PATH=/home/nunota/.miniconda3/envs/myenv` と
`cmake --build build-route-unified --target lcbinint_python test_core -j2`。
private derivative header の生成にビルド時 Python を使う。
数値計算本体は C++ で、native の通常値計算に JAX は不要。

main/feature の checkout、環境の editable install、他の作業ツリーは変更していない。
実装・検証済み、Git commit/push はこの作業では行っていない。
