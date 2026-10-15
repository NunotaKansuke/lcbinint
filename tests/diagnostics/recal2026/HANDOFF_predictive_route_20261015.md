# 最新: light curve autoのデフォルトへ接続

`inverse_ray_grid="auto"`かつ`nbin="auto"`のbinary linear-LD積分段階で、今回の選択器を使う。点源/hexadecapole判定・接線付近の誤差下限は既存のまま。中心像をseed構築と予測で共有し、一方のadaptive積分だけを実行する。相対精度を初期fluxで判定した後にも、最終値でlight curveの誤差予算を確認する。

Cartesianの初期16から512までの倍増を可能にするため、`lcbi_default_options().max_source_bins`を400から512へ変更。明示的な上限は維持。固定グリッドと非linear LDは従来経路。詳細は `docs/design/predictive-adaptive-route.md`。

最終検証は `lightcurve-auto-validation.json`、識別は `lightcurve-default-identity.json`。以下の速度比は、この接続前のpreplanned境界の計測であり、新しいlight curve全体の速度比ではない。

---

# Cartesian / Polar 事前選択（2026-10-15）

正本: [REPORT.md](../results/recal2026/predictive-route-20261015/REPORT.md)、識別: 同ディレクトリ `final-identity.json`。

- `binary_adaptive_auto_flux`を明示的に呼ぶ試験APIを追加。既定の積分経路は変更していない。
- 中心像の幾何15特徴→深さ3・40本の木で、どちらか一方だけを事前に選ぶ。native preplanned側は既存rootを共有できる。Python入口は追加root solveあり（追加中央値4.47µs）。
- 現行カーネルで768入力を再計時。学習124イベント、調整31、評価48に分離。378評価条件の直接指定/自動選択で数値・誤差・受理状態・評価数・停止理由が一致、未達0・精度外受理0。
- 実際の選択処理込みで、Polar固定総時間/選択版総時間はUniform1e-3/1e-4が1.427/1.112、LDが1.274/1.192。総時間の比でありケース別比中央値ではない。Uniform2入力はCartesian例外のため有限ペア計時から除外したが、自動選択側はPolarで受理。
- 全14,432保存数値への選択replayでは未達21・例外0・精度外受理22。再積分ではなく、全kernel object hash一致を確認した既存監査の再利用。要求精度・停止条件は変更していない。精度認証ではない。
- 216特徴/選択チェック、Python96条件parity、無効入力6、校正範囲外2も確認済み。c=0/0.5・相対精度1e-3/1e-4・校正したs/q/rho範囲外はPolarを選び、校正内フラグfalse。
- 今回6パネル・VBM再計時なし。未commit・未push。次は別イベント群での一般化と全件での実時間比較が候補。
