# D14置換の比較範囲の修正

2026-10-13、ユーザーの明示訂正による。

## 前提

- Causticは代表seedの配置と積分器の切り替えに使用する。
- 「D14自体はseedを置かないのでcausticが不要」という説明は、causticを外す要求ではない。
- 対象は既存のcaustic由来の代表seedから、D14のevent/cell準備を置き換える経路。
- Caustic-free point providerのv2/v3は別の研究試作として保持する。今回の本経路の代替にはしない。

`FiniteSourceMagnifier` のcausticとcritical curveは同じgeometry cacheで生成される。
`binary_critical_curve_branches_cached` も `ensure_binary_caustic_cache` を使用し、
既に同epochで準備したcausticをseed生成で再利用できる。
以前のcold seed全費用を新経路だけの追加費用とみなす説明は、この構成には適さない。

## 接続

`seeded_epoch.hpp::epoch_adaptive_seeded` を追加した。
Callerが既に用意したprimary-frameのseedを受け取り、既存root-pair追跡、cell plan、
既存適応積分へ渡す。Caustic準備の所有権とmethod selectionはcallerに残る。
新入口自身は追加のレンズ方程式求根、caustic再構築、D14求根をしない。
研究用であり、全極値の完全性を認定するAPIではない。

V3で試したendpoint補正の削除、chart選択の削減、固定restart幅変更、
seed soft cutの削減は未受理なので、従来のevent driverへ戻した。
試作ファイルと負の検証結果は `seeded_local_repair_v3/` に保持する。
Point providerの局所位相追跡は別の試作として残す。

## 最初の接続診断

`export_caustic_prepared_seeds.cpp` は毎geometryで新しいPM/FMを作り、
center imagesとcaustic距離を先に準備し、その後のseed追加費用を別に記録する。
過去epochの同配置を使い回す実験ではなく、同epoch内での共通geometryの再利用。
保存された最適化source snapshotと `LCBININT_REPRESENTATIVE_PROBES=critical-seeds` を使用。

- 従来48配置: event/cell 48/48一致。
- fold88配置: 87/88がqueue予算内。1件はseed375個で256 start上限を超えた。
- deeper88配置: 86/88がqueue予算内。2件はseed279/308個で上限を超えた。
- この3件は、実際に発見したeventの独立照合ではmissing/extraとも0。
  queue完了判定で未受理となった。境界の見落としと区別する。
- Seed追加費用の単発snapshot中央値は従来48で約147us、新holdoutで約126us。
  最終的な反復・全体A/B計測ではないので倍速の根拠にはしない。

Source/build snapshot:
`tests/diagnostics/results/recal2026/seed-construction-stage-only-20261013-v1/`。
診断: `evidence/holonomic/caustic_seeded_connection_v4/`。

次は、供給される複数のcritical seedをそのまま全て独立startにせず、
既に追跡した像/境界の情報から必要な代表startだけ処理する。
受理済みeventと追跡状態を保持し、未解決の境界だけ追加処理する。
供給seedの削減と、境界の全極値発見を別々に検証する。
比較は共通caustic準備、seed追加、event/cell、積分を分割し、同じ前処理を両経路に適用する。
