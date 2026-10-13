# LD追加処理を誤差指標だけに限定する経路

2026-10-13。実験overlay内の `binary_ld_flux_estimate` を追加した。

## 外した処理

LDの増光率は既存の直接inverse-ray計算で求める。Bennett補正、輝度テーブル、光源flux規格化、通常のcomponent refinement、seed経路をそのまま使う。この入口は入れ子円盤provider、半径別のfill、境界再求根、Hermite面積状態、LD半径quadrature、Python provider callbackを呼ばない。旧入れ子円盤経路は診断用に保存する。

追加するのは同じ走査で既に得た情報からの指標だけ。

- 境界の輝度モデル残差：光源縁の輝度と最近傍2サンプルを再利用。
- 外側方向の形状指標：既存の各行のfluxから2階差分を計算。
- 数個の集計counter。誤差推定用のレンズ写像呼び出しは追加しない。

行ログ、各セル距離の大きなキャッシュ、全面的な再探索は追加しない。
走査中の隣接サンプル保持とO(行数)の集計は追加費用として残る。
thread-localのscoped modeで有効化し、例外を含めてmodeとcounterを戻す。
通常のプロファイル計測は有効化しない。

完成した増光率だけのキャッシュには誤差指標がないので、この実験入口ではそのキャッシュヒットを回避する。通常入口のキャッシュ動作は維持する。geometryを変える速度測定では、通常側も各入力を実際に計算する。

## 数値的な意味と制限

対象はCartesian、linear LD 0<c<=1、d=0。`error_indicator` は境界残差の絶対値和と外側形状指標を光源fluxで割った量。`signed_boundary_residual` と各寄与も返す。モデルが作れなかった辺は `unresolved_edges` で明示する。

これは全誤差の上限ではない。内側quadratureの残り、輝度テーブル誤差、像のcoverageを保証しない。`certified=false`、`complete_error_model=false` とし、許容誤差を受理するflagは返さない。指標が大きいという理由だけでnbinを自動増加させる経路にはまだ接続しない。

5geometry×4nbinで指標/実誤差は約2〜261。今回の標本では実誤差より大きかったが、全配置で安全とは言えない。特に実誤差の小さいケースでは過大評価する。高価な半径積分に戻さず、内側と境界の残差を安価に分離することが次の課題。

## 検証と速度

`tests/diagnostics/recal2026/cheap_ld_error_audit.py`。
同じ5geometry、c=.5、nbin64/128/256/512、1 thread。
通常のcomponent refinementは有効、追加support保証のflagは無効。

- 20条件で通常のscalar計算と増光率がbit単位で一致。
- 追加ray評価counterは全条件0。
- instrumented fill回数と再利用サンプルcounterが正であることを確認。
- scope後の通常計算も一致。
- 独立参照は一様VBM円盤面積をSciPyで積分した既存artifact。VBM LDの固定inverse-CDF精度に依存しない。
- 速度は同一build、入力と処理範囲を揃え、順番を交代した7batchの中央値。

最終v3の5入力合計は以下。

| nbin | 通常計算 [s] | 指標付き [s] | 比率 |
|---|---:|---:|---:|
|64|0.00928|0.00969|1.044|
|128|0.01513|0.01604|1.060|
|256|0.03579|0.03805|1.063|
|512|0.11260|0.11904|1.057|

約4〜6.3%の追加費用。通常側は実験buildの既存scalar経路であり、別buildのmasterとの同一性を主張しない。前のnested LDの0.42秒は別の初期解像度/実験積分方式のpilotなので、その時間比を厳密な同一アルゴリズム比較として扱わない。同じ要求1e-5に対して重い入れ子円盤構成を必要としなくなった。

artifact `cheap-ld-error-v1` は完成fluxキャッシュから値が返りcounterが0になる初回試行。INVALIDATED.mdを付け、誤差推定の証拠から除外した。v2以降は実際のfillとサンプル再利用をassertする。

production/masterは変更していない。高速化した実験入口と誤差モデルの科学的完成を区別する。

## 共通化・過大評価の改良

後続の `nbin-shared-flux-error-report.md` を参照。入口は `binary_flux_estimate` に統合し、uniformとlinear LDで共通の行・端点momentを使う。本文の数字は初期方式の記録として保持する。
