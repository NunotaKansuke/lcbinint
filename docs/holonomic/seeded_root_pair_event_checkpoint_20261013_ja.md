# Seedから既存root-pair追跡でD14 eventを置き換える試作

日付: 2026-10-13

- ブランチ: `experiment/seeded-boundary-events`
- ベース: `dev/holonomic` の `f3547723b79828d9c501182746028e99f7bad116`
- 独立worktree: `experiments/seeded-boundary-events`
- 状態: 研究用の接続経路を実装・代表ケースで検証。production dispatchは変更していない。

## 実装

`seeded_boundary_events.hpp` は供給された像seedから、既存の `RootPair{m,v}`、
`root_pair_dR`、`E=O=0` の補正を再利用する。Cartesianの別contour tracerは採用しない。

1. Seedの半径で角度境界quarticを解き、隣接するroot pairを作る。
2. 内側のarcだけでなく、境界が合流する外側のgapのpairも調べる。
3. `v` のゼロを予測し、既存の局所 `P=Pt=0` 方程式を補正してeventを得る。
   通常のfoldでE/OのJacobianは正則なので、別の境界なぞりへ切り替えない。
4. 新しいeventの近傍だけに追加の開始点を置く。追跡済みのpairは状態を補正して照合し、
   同じ経路なら省略する。失敗したstepは縮小し、受理済みの状態・eventを保持する。
5. 既存の `classify_event_cells` と `flux_adaptive_integrate` に渡す。
   D14を解く呼び出しは新経路に含まれない。BenchmarkのD14は比較用のみ。

幅の二乗 `v` の誤差を `sqrt(v)` の尺度で受理すると、狭い像のradial bandを飛び越す。
新driverでは `m` と `v` の補正誤差を別々の尺度で判定し、縮小済みstep幅を次stepにも使う。
既存積分のtiny-arc gateや積分アルゴリズムのパラメータは変更していない。

逆数座標については、既存の homogeneous polynomial evaluatorを直接使う。
`t=-1/u` を再構成する式は `u=0` のprojective foldで発散するため、直接の逆数多項式へ置換した。
Adaptive event reuseも `fold_u_seed_valid` を受け取るように接続した。

## Cellとsoft cut

物理foldは粗い半径mergeをせず、積分前のcell境界として残す。
同じcontactに重なるrepresentation cutは数ULP以内で重複を避ける。

D14のcomplex-root soft cutを外すと、既存の適応積分の初期分割が変わる。
最初の接続では1条件で相対差約0.00286があり、tol=0.001を超えた。
そのため、occupied cell内に既存seedがあれば、中央に最も近いseed半径を最大1個だけ
soft cutとして再利用する。全seedの半径を分割点にする処理は入れていない。
追加のレンズ方程式求根やD14求根は不要。この変更後に同条件の差は約1.64e-6になった。

## 検証と計測

入力は保存済みの48 geometry。`nbin=256` Cartesian fixtureからgeometryを選んだが、
新経路に格子・flood fill・nbin依存の計算はない。
Seedは既存最適化経路のsource/build snapshotからexportし、primary frameに変換して保存した。
入力・source・compiler・baselineのprovenanceは `manifest.json` に記録。
48入力は少数のgeometry群を微小に摂動したものを含むので、独立な広域holdoutではない。

最終計測は1 worker、OMP/BLAS 1 thread、各入力7反復の中央値。
**供給済みseedからのevent生成とcell plan構築**を両経路で計測した。
積分、seed構築、D14との照合、正しさの検証は計測外。
`rootpair_budget*.tsv` の一部は並列の探索的検証であり、最終速度の根拠には用いない。

| 指標 | Seed root-pair経路 | D14経路 |
|---|---:|---:|
| 入力ごとの中央値の中央値 | 277.998 us | 478.751 us |
| 入力ごとの中央値のp90 | 2111.181 us | 2111.121 us |
| 48入力の時間合計の比 | 0.7500 | 1 |

中央値では約41.9%短縮、時間合計では約25.0%短縮。
p90はほぼ同じ。30/48入力で新経路が速い。難しいケースまで一律に速くなったとは言えない。

- Eventとcell topology: **48/48一致**。
- D14の `physical_complex` ラベルの一部も、前後のdegree-4 Sturm countが変化するものは実event。
  その判定を候補のeventから独立に行って照合した。case46が該当。
- 既存適応積分: `u=0,0.5`、`mu_rtol=1e-3,1e-4`、計192条件。
  D14側が収束した **178条件はすべて新経路も収束し、差は指定tol内**。
  最大の相対差/tolは 0.112994。
- **14条件はD14側が未解決**。うち4条件は両方未解決。
  比較不能な条件を成功に数えないため、strict adaptive auditのexitは1のまま。
- 固定128点の値比較は別の診断であり、高精度の真値とは扱わない。
- 既存M7 test: **10,397 checks / 0 failures**。
- Signed-vとprojective eventの単体test: `unit_tests.txt` に実測件数を記録。

## 残る制約

1. **Cold seed構築を含めた高速化は達成していない。** 保存snapshotのcold seed構築は
   中央値約1219.2 usで、caustic準備を含む。
   上の速度はこの費用を含まない。以前の約116usというaugmented-seed部分だけの計測とは境界が違う。
   このcold seed providerをそのまま新経路の前に置けば、D14単独より固定費が増える。
   全体を速くするには、geometryが毎回変化する前提でcaustic全体の構築を避けるproviderの接続が必要。
   Snapshotの費用と新計測の合算を、matchedなend-to-end実測とは呼ばない。
2. 全極値の一般的な完全性は証明していない。`coverage_certified=false` を保持する。
   `completed` は開始点queueが予算に収まったことだけを表し、各pairのstep上限やNewton探索に
   よって未発見eventがないことを意味しない。今回の一致は独立D14照合によるfixtureでの実測。
3. 広域holdout、接触・topology遷移・極端なmass ratio、seed provider込みのcold計測がまだ必要。
   Productionの標準設定へ昇格する段階ではない。

## 再現

Worktreeのrootから:

```sh
./benchmarks/holonomic/run_seeded_boundary_audit.sh
```

`rootpair_final.tsv`、`adaptive_final.tsv`、`summary.json` を生成する。
Strict adaptive auditのexit=1は上記14件のreference未解決を含む。
候補の退行は `regressions_vs_converged_reference=0` と別に集計される。
保存されたseedを使うので、元checkoutのsnapshot buildはこのauditの実行には不要。
Seedを再exportする場合のsource/buildは `export_minimal_boundary_seeds.cpp` とmanifest参照。

初期Cartesian試作と失敗結果は `retired-cartesian/` と `width_*_initial.tsv` に保存。
現行経路の採用根拠にはしていない。
