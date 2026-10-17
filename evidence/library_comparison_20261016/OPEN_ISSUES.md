# Light-curve benchmark: unresolved issues

更新日: 2026-10-17

このメモは、比較ベンチで見つかった未解決点を追跡するためのもの。以下は「既知の出力差・検証上の課題」であり、原因や正誤が確定していない項目をバグと断定しない。

測定条件と生データは [12ケース比較](corrected_value_gradient_12cases/README.md)、追加の15反復監査は [API timing audit](api_timing_audit.md) を参照。集計だけで正解を決めず、特に接触点のライブラリ間中央値は真値ではない。

## backend の範囲

**以下の7件は、すべて代数境界 backend 固有の問題ではない。** 2026-10-16保存ベンチで lcbinint の値のみ API は明示的に `finite_source_backend="algebraic_boundary"` を使った。一方、同じベンチの値＋勾配は旧 public 関数の seeded fused-gradient 経路であり、代数境界の値 API に対応する勾配測定ではない。2026-10-17以降、対応する eager static-binary の `LightCurve(..., with_gradient=True)` は同じ fused kernel を内部で使用する。これは値のみの代数境界 integrator とは別の積分経路である。

ここでいう代数境界 backend は、既存の point-source／hexadecapole fast path を含む選択経路で、標準 Cartesian／polar inverse-ray 経路は使わない。従って「backend が代数境界」と「各 epoch で境界積分を実行した」は同じ意味ではない。保存 benchmark では method 名も記録している。

| Issue | lcbinint の記録経路 | 代数境界固有か | 現時点の整理 |
|---|---|---|---|
| 1. 10.92% 値差 | 当初の数値ペアは run/raw/build がなく backend 不明。修正版保存分は値のみが代数境界、値＋勾配が旧 `holonomic_gradient()` | **不明。現行代数境界 API の矛盾とは言えない** | 保存済み同一ケースでは再現せず。出所不明の報告として保持 |
| 2. 非有限件数・失敗理由 | 当初件数の実行 provenance は不明。保存分は値のみが代数境界、勾配側は旧経路 | **件数主張は不明。失敗理由の API 契約は共通 API の課題** | 当初件数は raw で裏付けられない。現行 API には有限値と status を返す変更が入っている |
| 3. 勾配精度 | 比較集計の lcbinint 勾配は旧 `holonomic_gradient()`。Cartesian の先行プローブは `finite_source_backend="default"` の JAX 経路 | **旧比較は代数境界勾配を検証していない** | 現行代数境界 `with_gradient=True` は smooth な1点のみ確認。一般精度は未検証 |
| 4. `near_limb_contact_117` の VBM 差 | cross-library 比較。現行 lcbinint は代数境界 `LightCurve` | **ライブラリ間比較であり固有ではない** | VBM 5.5/5.6.1 とも外れ、単位丸め差にも極端に敏感。公開 API と低水準 API の変換を揃えると値は一致するため、規約取り違えではない。q>0 の参照値と VBM 内部原因は未確定 |
| 5. `stratified_09` | 現行 `LightCurve(..., with_gradient=False/True)`、既定代数境界 | **lcbinint の有限値契約は修正、中心3点の全ライブラリ参照比較は未完** | 24/27 epoch は5実装で値取得。VBM/autodiff-vbm は24点で最大相対差 `2.21e-9`、中心3点は両方未完了。中心3点の参照値は未確定 |
| 6. 勾配追加コスト | 外部 `autodiff-vbm` とブランチ既定 `lcbinint` を分けて測定 | **旧 `6.03x` は未再現・出所不明** | 対応 native calls は既存 fused kernel に切替済み。今回の401点では値＋勾配が値のみより短時間だが、別 solver なので値/status は一致せず、この難ケースは科学的な速度評価に使えない |
| 7. 最速勾配経路 | 外部 public APIs で vector full Jacobian と scalar 出力を別々に比較 | **測定形状に依存** | vector 27/30出力では MicroLUX の reverse が `2.27–2.52x` 遅いが、同じ接触例の scalar 出力では reverse/forward `0.681`。autodiff-vbm は両方ほぼ同じ。lcbinint はこの forward/reverse 比較に含まれない |

このブランチの値のみ `LightCurve` 既定は、対応する静的 single-source binary では代数境界に変更した。未対応モデルは従来 router を使い、対応 binary でも `finite_source_backend="default"` を指定すれば従来 router を明示できる。対応 eager binary の `with_gradient=True` は、別 solver の既存 fused seeded-radial kernel を内部利用するため、値のみとの magnification/status 一致は自動ではない。先行の x64 1 epoch smoke は両モード一致、今回の401 epoch stress pair は不一致で、詳細を issue #6 に記録した。triple は標準 router に残り、low-level `binary_ray_shooting` の `Options()` も引き続き受け付ける。既存の2026-10-16結果は歴史的比較として残し、新しい測定と混ぜない。

## 先に整理した5項目

### 1. lcbinint の値のみ API と値＋勾配 API が一致しないとの報告（再現せず）

- **当初報告:** `notebook_alpha180` の `t=0.084` で値のみ `71.3991769987`、値＋勾配 `63.6010937043`、相対差約 `10.92%` と記録されている。この数値対に対応する run metadata や raw 出力は見つかっていない。
- **スクリプト provenance:** 2つの raw 保存系列は同一スクリプトではない。`combined_value_gradient/metadata.json` の `script_sha256` は `5cea000d...`、修正版 `corrected_value_gradient_12cases/metadata.json` は `e48be726...`。現存する `benchmark.py` は後者と一致するが、前者の本文は保存されていないため、正確なコード差分は復元できない。親の `metadata.json` にある `57589112...` は `progress.json` 上 `ordinary_point` だけ完了しており、`notebook_alpha180` の出力はない。修正版 README には、初回比較で microlux/microjax の軌跡規約変換を見落とし、その後修正したとある。2つの raw 保存系列はどちらも値のみ・値＋勾配の両モードを記録し、lcbinint の出力は互いに一致する。したがってスクリプト hash の違いは確認できたが、それが当初報告の10.92%差を生んだ証拠はない。
- **経路・入力:** 現存する修正版 benchmark は一つの `LightCurve` と同じ `lc_params` を値 API と `holonomic_gradient()` の両方へ渡している。値 API は value router、勾配 API は seeded fused-gradient 経路を通るため、数値経路自体は異なる。`alpha=180` は lcbinint の契約ではラジアンで、修正版はその値を両 API に同じまま渡している。保存 raw の当該点では `gradient_valid` は7成分すべて trueだが、これは勾配精度の保証ではない。旧スクリプトでは同じ入力処理だったか、保存記録だけでは確認できない。
- **独立した同一ケース記録:** `evidence/holonomic/gradient_disk_20261015/notebook_alpha180_rad_ld0.5_tol0.0001_matrix.npz` でも、同じ `t=0.08400000000000005`、同じケース入力に対して値 API `mu=71.39916533927625`、値＋勾配 API `gradient_mu=71.39916533952365`。相対差は `3.46e-12`、全401点の最大相対差も `4.78e-9`（`t=-0.316`）。これは10.92%差の報告より前の別測定で、同じ API 対の整合を独立に支持する。
- **判定:** 10.92%差は、確認できた3つの同一ケース保存測定（10/15の独立診断、10/16の2系列）のいずれでも再現しない。元スクリプトが誤っていたとは確定できない。確定できるのは、報告ペアに対応する実行記録がなく、`63.6010937043` は checkout 内の当該ケース raw に存在しないこと。よって現時点では solver/API の不整合ではなく、出所不明の報告値として扱う。具体的な転記元・取り違え元は未特定。記録 metadata の value native hash `e6f7ef...` に一致する shared object は現 checkout に残っておらず、同じ native binary での再実行もできていない。
- **次の確認:** `63.6010937043` を出した元のコマンド、入力、route/diagnostic、native build を特定する。特定できるまでは10.92%差を再現済みの solver/API 不整合として扱わず、コード修正や回帰テストの根拠にしない。

### 2. lcbinint の非収束結果と API の失敗理由の表現（件数報告は未確認）

- **当初報告:** LD=0.5 の値のみ比較 `2916` 点中 `1321` 点が `np.isfinite` で非有限、また `large_source_uniform` の401点中400点が NaN とされた。これらの数値に対応する raw、run metadata、理由別の出力は見つかっていない。
- **保存 raw との照合:** `combined_value_gradient/raw/` と `corrected_value_gradient_12cases/raw/` は、それぞれ合計3317 epochを保存し、両系列とも lcbinint の値のみ・値＋勾配の全3317点が有限。両方の `large_source_uniform` も401/401点が有限。さらに inputs では `large_source_uniform` は limb=0.0 であり、修正版12ケースにおける limb=0.5 の点数は1260で、当初記載の「LD=0.5、2916点」と一致しない。従って、当初の件数と代表例は手元の保存測定から確認できず、出所不明として扱う。
- **コード上の数値挙動:** algebraic-boundary 積分は画像 support の証明が済んでも、推定積分誤差が要求予算を超えることがある。その場合 `finite_source_error_budget_exceeded` として未収束にする。これは物理量が未定義という意味ではなく、要求精度を満たした値と認定できないという意味。fail-closed 自体は精度契約を守る挙動。
- **公開 API の表現:** native の `LightCurve.__call__` は失敗 epoch があると配列全体を `RuntimeError("numerical error")` で失敗させる。default backend の `LightCurve.info()` は epoch ごとに `magnifications=NaN`、`statuses="numerical_error"`、収束フラグ、誤差推定値を返し、計算候補は `finite_source_magnifications` に残すが、内部の `finite_source_stop` 理由文字列は公開しない。一方、専用の `algebraic_boundary_info()` は `stop` と support/収束診断を返す。JAX の algebraic-boundary 値のみ FFI は通常の値 FFI、微分時の JVP は value+Jacobian FFI を呼び、どちらも失敗値を NaN にするが status は返さない。さらに深い有限光源ケースでは、値側は `binary_mag` が積分まで行う一方、Jacobian 側は `multipole_only_probe` で経路だけ選び、別の Jacobian 積分を実行する。従って微分あり／なしで実測値が違うと確認したわけではないが、primal の計算経路・失敗表現を共有していない設計上の分岐がある。
- **上記は整理前の実装:** native/JAX/algebraic-boundary ごとに通常値・診断値・勾配値の入口が分かれ、非収束時の通常値が例外または NaN になっていた。
- **テスト現状:** 既存回帰テストは収束・support の bool と scalar API の失敗を確認しているが、`finite_source_error_budget_exceeded` という停止理由の公開や、通常 `info()` への理由伝播を確認するテストは見つからない。
- **整理後の API 契約:** 値のみ・値＋勾配のどちらも、各 epoch について同じ `LightCurveInfo` 相当の結果レコードを返す。値の計算経路と受理判定は共有し、勾配を要求しても magnification と value status を変えない。未収束や support 未証明でも、利用可能な最良の有限推定値を magnification として返し、status で保証状態を区別する。NaN/Inf や値なしにはしない。通常値として使えると誤解させないよう、失敗時の推定値にも失敗 status を必ず付ける。未収束と support 未証明（要求精度／機能が実装の保証範囲外の場合を含む）を値の失敗理由の二分類とし、入力不正は計算開始前の引数エラーとして扱う。勾配だけが無効な場合は値 status を失敗にせず、勾配の有効性を別フィールドで返す。失敗した反復の候補値を捨てず、status 付きの推定値として返す。計算経路が有限候補を一つも生成できない場合も NaN/Inf や未設定値で代用せず、有限の fallback 推定値を用意し、失敗 status を残す。勾配にも NaN/Inf を使わず、有効性を別フィールドで示す。
- **現在の実装状態:** `LightCurve.__call__` は値のみ・値＋勾配で同じ `LightCurveInfo` 型を返す。対応する eager static-binary の `with_gradient=True` は既存 fused native value/Jacobian kernel をそのまま呼び、値のみの algebraic-boundary router とは別の値計算・support 判定を使う。他のモデルと traced/JAX 呼び出しは共通の値結果に JAX 勾配を付ける。status code は `0=ok`、`1=unconverged`、`2=support_unproven`。値は有限候補を優先し、なければ有限 fallback を返す。勾配が無効な場合は有限のゼロ成分と `gradient_valid=False` で示す。`info()` は同じ API の互換 alias。
- **軽量確認 (2026-10-17):** checkout `experiment/algebraic_boundary` (`40e66e510799f9d3d15b459be2b38694daad52ee`) で `cmake --build build-algebraic-boundary/python-api --target lcbinint_python -j2` が成功。実行時は conda 環境の `_lcbinint_editable` import hook が親 checkout の旧 package を優先したため、その hook だけをテストプロセス内で除外し、build dir 内の `lcbinint` package と `_lcbinint` extension を読み込ませた。`tests/regression/test_light_curve_wrapper.py` は `2 passed`。別の3 epoch smoke では値のみ／`with_gradient=True` の magnification と status が完全一致し、値と勾配は有限、3点すべて `gradient_valid=True`。1 epoch の `jax.jit(...with_gradient=True)` も `LightCurveInfo`、有限値・勾配、`status=ok` を返した。さらに既存の crossing 条件を201 epochで評価し、`ok=181`、`unconverged=3`、`support_unproven=17`、201点すべて有限値だった。
- **判定:** 当初の非有限件数や `large_source_uniform` の失敗は再現・裏付けされていない。10.92%差も再現されていない。一方、issue #6 の401点 stress pairでは、統一 API の勾配モードが既存 fused 経路を選ぶため、値のみとの magnification/status が揃わないことを確認した。有限値と status を返す基本 API は動作するが、モード切替時の値・status 一致は未達であり、勾配精度や全経路も保証していない。
- **残作業:** 全モデル種・JAX backend を含む status 伝播、勾配のみ無効なケースの有限値、JAX/algebraic-boundary の失敗ケースを含む広い回帰確認は未実施。当初の件数の元入力・呼び出し API・build・raw が見つかった場合は、別途 provenance を調べる。

### 3. lcbinint の勾配精度の検証が未完了

- **元の観測:** [保存した12ケース比較](corrected_value_gradient_12cases/README.md)では、ライブラリ間の勾配相対偏差の95 percentileが大きかった。ただし、この集計は近ゼロ成分を含み、独立な導関数基準ではない。さらに当時の lcbinint 値＋勾配 API は `holonomic_gradient()`、値のみ API は `finite_source_backend="algebraic_boundary"` の `LightCurve` で、同じ計算経路ではない。これだけで lcbinint の勾配不具合は確定しない。
- **前回の誤診 (撤回):** 2026-10-17 の先行プローブは `finite_source_backend="default"` を使い、既定 JAX backend の method 1 (Cartesian) を調べていた。これは代数境界 backend でも、保存比較の `holonomic_gradient()` でもない。固定支持・Cartesian 分類・Polar 候補についての上記説明と数値は #3 の原因を示す証拠ではないため、撤回する。
- **現在の代数境界 API の確認 (2026-10-17):** checkout `experiment/algebraic_boundary` (`40e66e510799f9d3d15b459be2b38694daad52ee`) の build package を明示的に import。実行 extension SHA256 は `3783c72ceaa436f92bfc0ebd831480c0e1955a046ff5dedb7c044d33416c2f37`。JAX x64 を有効にし、`Options(jax=True, param_type="center_of_mass", nbin="auto", finite_source_tol=1e-4, finite_source_reltol=1e-4, finite_source_backend="algebraic_boundary")`、linear LD `0.4`、`t=0.22`、`(t0,u0,tE,alpha,rho,q,s)=(0,0.15,1,0,0.05,0.28,1.35)` で `with_gradient=True` を評価。値は `4.064907149710015`、method は `algebraic_boundary`、value convergence/support/gradient_valid はすべて true。値のみ API と値＋勾配 API の値は一致した。7成分すべての AD と同じ値 API の中央差分 (`h=1e-5`) は、`t0 4.97854493909 / 4.97854509369`、`u0 12.2322048533 / 12.2322087624`、`tE 1.09527988660 / 1.09527989731`、`alpha -3.43786680859 / -3.43786724506`、`rho -6.10824800404 / -6.10824231155`、`q -6.65282849398 / -6.65282899179`、`s 9.95585449891 / 9.95585634960`。この smooth な1点では最大相対差は約 `9.4e-7`。実装は `binary_algebraic_boundary_batch_ffi` から `experimental_algebraic_boundary_jacobian` を呼び、primal は radial-band/角度積分を行う。Cartesian 積分関数はソースに別途存在するが、検索した限りこの backend からの呼び出しはない。
- **難ケースでの制限:** 保存比較の `large_source_ld05_user` から `t=-0.7` を取り、現在の統一 API・同じ algebraic-boundary backend・許容誤差 `1e-3` で確認すると値は有限 (`4.37730638857`) で support は true だが、method `algebraic_boundary` は未収束で `gradient_valid=false`。基準値と差分端点が収束条件を満たさないため、この点は勾配精度の判定には使えない。
- **判定:** 以前の Cartesian の再現は対象違い。保存比較の勾配 API も現在とは別経路であり、#3 の問題は未確認。今回の現在 API の結果は代数境界の smooth な代表ケース1点に限られ、他の topology・難ケースを保証しない。
- **残作業:** 保存比較で差の大きかったケースを特定し、現在の統一 API の値経路に対する有限差分刻み幅スイープを、regular/crossing/contact 等で実施する。集計上の相対差だけでバグと判定せず、差分の安定性、value convergence、support、gradient_valid を併記する。

### 4. 接触付近で公式 VBM が他実装から外れる（VBM の丸め感度を確認）

- **再現条件:** 現行 `near_limb_contact_117` 入力、VBM parameter convention、LD=0.5、x64。元データの最大差 epoch は `t=-1.670997233467463`（source center が limb の内側へ約 `1e-6`）。
- **許容誤差列:** 公式 VBM 5.6.1 は `Tol=RelTol=1e-3` で `9.5258195`、`1e-6` で `9.5234346`。値は収束するが他実装には近づかない。別の公式 VBM 5.5 wheel でも4段階の全出力が同じだったので、5.6.1 固有の回帰ではない。
- **照合:** 同 epoch で autodiff-vbm (vendored VBM 5.5) `7.4619659`、MicroJAX `7.4542913`、MicroLUX (`1e-6`) `7.4690958`。lcbinint の有限候補は `7.4619655` だが status は `unconverged` なので採用値ではない。MicroLUX は複数 epoch で内部 sampling capacity の警告を出しており、これ単独を正解扱いしない。
- **独立 limit 参照:** 同じ source geometry の `q=0` 単一レンズ有限光源積分を独立数値積分すると、LD=0.5 で `7.4619188`（積分誤差推定 `4.9e-9`）、一様光源で `7.6904201`。実際の `q=7.886e-7` binary そのものの解析解ではないが、他 binary 実装の値に近い。公式 VBM 5.6.1 の一様光源出力は同 epoch `11.0383412` (`tol=1e-6`) で、limb-darkening 係数だけでは説明できない。
- **丸め感度と API 経路:** 低水準 `BinaryMag2` に物理入力を渡すと `8.6462092`。q を1 ULPだけ上げると `7.7688046`、相対 `1e-12` の変更では `6.8914–9.5236` に振れる（いずれも tol=`1e-6`）。同じ入力で直接呼んだ `BinaryMagDark` は各 q で `BinaryMag2` と同じ結果なので、この変動は point-source/finite-source dispatch ではなく finite-source annulus 積分側にある。公開 `BinaryLightCurve` は `9.5234346`。その log parameter 展開と軌道座標計算を低水準 `BinaryMagDark` に再現しても `9.5234346` となり、公開 API と一致する。よって convention/API の取り違えではなく、丸め差に対する VBM 有限光源計算の不安定性が確認された。ただし各変換を個別にすると q の log/exp だけでは公開 API との差を説明せず、annulus 計算内部のどの条件が不安定化するかは未特定。
- **判定:** VBM 5.5/5.6.1 の有限光源値はこの接触点で再現性のある外れ値を示し、さらに ULP 規模の入力差にも不安定。VBM 値はこのケースの参照値に使わない。独立 q→0 積分は強い照合材料だが、実際の q>0 binary の真値ではない。原因は VBM 側に局在しており、この lcbinint ブランチから修正できない。q>0 の独立参照と VBM 内部の正確な不安定化条件は残作業。
- **証拠:** [許容誤差監査](issue4_tolerance_audit_20261017/metadata.json)、[公式 VBM 5.5 control](issue4_vbm55_control_20261017/metadata.json)、[一様／LD 切り分け](issue4_contact_ld_audit_20261017/metadata.json)、[VBM 5.6.1 一様光源 control](issue4_vbm561_uniform_20261017/metadata.json)、[q→0 単一レンズ参照](issue4_single_lens_limit_20261017/metadata.json)、[q の ULP 感度（BinaryMagDark 直接呼び出しを含む）](issue4_vbm_q_ulp_20261017/metadata.json)、[公開 API 変換の切り分け](issue4_vbm_roundtrip_components_20261017/metadata.json)。再現コードは各監査名の `.py`。

### 5. `stratified_09` の全ライブラリ比較が未完了

- **実測:** 同一の27 epoch input、tol `1e-3`、x64・CPU 1 core で、MicroLUX / MicroJAX / 現行 lcbinint `LightCurve` の値・全7成分勾配を保存。初回と定常時間を分離した。新しい lcbinint の値のみ／値＋勾配の magnification と status は27点すべて完全一致。
- **完了した実装:** MicroLUX と MicroJAX は値27/27、勾配189/189が有限。lcbinint は値27/27が有限で、値のみ／値＋勾配の値と status が全27点で完全一致。`ok` は24点、中心3点は `support_unproven`、勾配有効は168/189成分。VBM 5.6.1/autodiff-vbm も epoch ごとの隔離測定で24点ずつ完了した。全5実装の値が揃う24点では、VBM family を一つにまとめた4 family の最大相対 spread は `1.16e-5`。これは記述的な一致度で、真値の認定ではない。中心3点で従来の `mu=0` は誤解を招くため、積分値がなく support 未証明のときは有限な点光源 magnification に切り替え、`diagnostics["value_fallback_kind"]="point_source"` として返すよう修正した。この fallback は有限光源値の承認ではなく、status は `support_unproven` のまま。
- **原因と残作業:** 中心3点の積分停止理由は順に `odd physical boundary-root count`、`covered certified seed was not active`、`covered certified seed was not active`。標準 inverse-ray router は同3点で90秒以内に結果を返さず、fallback先としては使えなかった。公式 VBM 5.6.1 と autodiff-vbm は24/27点を取得し、完了点の両者は同一 VBM family のため独立照合ではないが、相対差は最大 `2.21e-9`。残る中央 index 12–14 は各々2秒／5秒の上限で未完了。MicroLUX の既定容量では中心3点に sampling-capacity 警告が出て大きく不安定だった。容量を6000に拡大すると警告は消え、MicroJAX との差は各々 `-0.0329%`、`-0.0211%`、`+0.1114%` まで縮まったが、MicroLUX の `return_info` は最終 annulus について tolerance `1e-3` に対し誤差推定和 `3.70e8–5.18e9` を報告する（この診断は limb-darkened 総和全体の誤差保証ではない）。容量拡大後の値も精度確認済みとは扱えない。中心3点の有限光源値と妥当な独立参照値は依然未解決。
- **証拠:** [3実装の現行 API 比較](issue5_stratified_09_remaining_apis_20261017/summary.json)、[更新後の27点 fallback 検証](issue5_stratified_09_fallback_verification_20261017/summary.json)、[24点の5実装比較](issue5_stratified_09_24_epoch_comparison_20261017/summary.json)、[3点の fallback 値と同一 API 結果](issue5_fallback_route_20261017/metadata_algebraic_boundary_value.json)、[標準 inverse-ray timeout](issue5_legacy_fallback_timeout_20261017/metadata.json)、[VBM/autodiff-vbm 27 epoch 分離 summary](issue5_vbm_epoch_isolation_20261017/summary.json)、[epoch ごとの出力 metadata](issue5_vbm_epoch_isolation_20261017/metadata.json)、[MicroLUX 容量6000と誤差状態](issue5_microlux_capacity_6000_info_20261017/metadata.json)、[未完了を含む実行 metadata](issue5_stratified_09_remaining_apis_20261017/metadata.json)。

## 追加した2項目

### 6. 勾配追加コスト（ブランチの重複評価は修正、旧 `6.03x` は未再現）

- **API/ケース:** `ordinary_point`（401 epoch）、`large_source_ld05_user`（401）、`near_limb_contact_117`（27）、`near_double_root`（30）の4入力を使った。内容は点源、大光源、接触、近重根の異なる形状だが、元監査資料には「なぜこの4件か」の選択基準は記録されていない。LD on/off の4条件ではなく、唯一の大光源ケースも LD `0.5` だけで、対応する uniform 条件は元の4ケース測定に入っていなかった。計測 API は外部ライブラリ `autodiff-vbm` の `BinaryLightCurve` であり、このブランチの `lcbinint.LightCurve(..., with_gradient=...)` の速度測定ではない。各ベクトル値出力に対する全7パラメータ `jax.jacfwd` を比較した。
- **追試:** 指定 integration checkout、x64・CPU 1 core、`tol=1e-3`。4ケースで値のみ／値＋勾配を15組、順序を交互にして測定し、毎回 JAX 出力を同期した。初回 compile は別記録。
- **結果:** 値＋勾配/値の定常中央値比は順に `1.019`、`1.004`、`0.999`、`0.978`。4ケースの中央値合計は値 `798.926 ms`、値＋勾配 `798.711 ms`（比 `0.9997`）。両モードの magnification は全ケースで完全一致。
- **一致させた LD 比較:** `large_source_uniform` と `large_source_ld05_user` は同じパラメータ、401 epoch。保存された時刻配列間には最大 `3.33e-16 d` の丸め差があるため、計測時は uniform 側の同一時刻配列を両条件に使った。値のみ／値＋全 Jacobian の比は uniform で `1.358`（`17.813/24.199 ms`）、linear LD `a1=0.5` で `1.003`（`244.143/244.772 ms`）。LD を入れると値のみは `13.71x`、値＋勾配は `10.12x` 遅くなり、このケースでは比が1に近づいた。
- **修正前のブランチ API:** 同じ401 epoch pair で `with_gradient=True` が値計算のあと `jax.jacfwd` を別実行していた。保存中央値は uniform `306.560 → 802.203 ms`（`2.617x`）、LD `0.5` `354.528 → 1052.785 ms`（`2.970x`）。この経路の重複評価が実装上の原因だった。
- **ブランチの修正と追試:** 対応する eager static-binary call を、既存の `BinarySeededFusedGradient.evaluate` に内部 dispatch するよう変更した。15組交互の再計測では uniform `306.483 → 229.652 ms`（比 `0.749`）、LD `0.5` `355.161 → 265.129 ms`（比 `0.747`）。両モードの値・勾配は有限で `gradient_valid=401/401`。ただし値のみは algebraic-boundary、値＋勾配は seeded radial の別 solver で、最大相対値差は uniform `1.366e-2`、LD `7.525e-3`。値のみ status は uniform `unconverged=401/401`、LD `unconverged=400/401`。fused 側は value-converged `401/401` だが support-certified `0/401` で、status は `support_unproven=401/401`。従ってこの速度比は同一 solver の微分追加コストを測っておらず、科学的な速度・精度受容の根拠ではない。
- **AutoDiff VBM で LD 後に比が1へ寄る原因:** この確認済み checkout は `autodiff-vbm` HEAD `6900260`。uniform の `native/vbm_ffi.cpp` は値のみ contour を `finite_contour_primal` で計算し、JVP のときだけ `finite_contour_magnification` の Jet 経路で5成分を伝播する。対して linear-LD の `native/vbm_limb_ffi.cpp` は、値 FFI と Jacobian FFI の両方が同じ `evaluate_limb_epoch` を呼び、結果の `Jet` から値を読むか Jacobian もコピーするかだけが異なる。つまり LD の値のみ実行でも native 内で6 lane の derivative を計算して捨てており、勾配追加時間が既に primal 値計算に含まれている。これは `1.003x` の直接の実装上の説明。uniform `1.358x` は値-only scalar contour と Jet contour の仕事量の差で説明できる。source file SHA は `_kernel.py=f8a8c9…`, `_limb.py=420879…`。
- **判定:** 修正前の lcbinint `2.617–2.970x` は重複評価の実装ミスで、対応する eager call は既存 fused kernel に切替済み。一方、現在の低比率は別 solver の比較で、値と status が揃わないため、同一値関数に対する1.2x達成とは判定しない。外部 `autodiff-vbm` の LD 比 `1.003x` は、値-only LD kernel も Jet lanes を計算しているためで、値-onlyが軽いという比較にはならない。旧 `6.03x` の provenance は未発見。
- **証拠:** [15組交互・外部 AD-VBM 経路監査](issue6_7_gradient_api_audit_20261017/metadata.json)、[外部 AD-VBM の一致 LD と scalar 出力](issue6_ld_scalar_output_audit_20261017/metadata.json)、[修正前ブランチ既定測定](issue6_lcbinint_matched_ld_20261017/metadata.json)、[修正後 fused API 測定](issue6_lcbinint_fused_api_20261017/metadata.json)、[追試スクリプト](audit_lcbinint_fused_api_20261017.py)、[AutoDiff VBM `_kernel.py`](../../../../../moasarc-ecosystem/integration/autodiff-vbm/src/autodiff_vbm/_kernel.py)、[AutoDiff VBM `_limb.py`](../../../../../moasarc-ecosystem/integration/autodiff-vbm/src/autodiff_vbm/_limb.py)、[旧 timing audit](api_timing_audit.md)。

### 7. 勾配速度の比較経路（測定した範囲を完了）

- **確認した API:** MicroJAX CPU の公開推奨経路は `jax.jacfwd`／`jax.jvp`。MicroLUX `analytic=True` は reverse mode に対応。autodiff-vbm の `BinaryLightCurve` は `custom_jvp` FFI 上で、`jax.jacfwd` と `jax.jacrev` の両方を実行できる。lcbinint の現行 `LightCurve(..., with_gradient=True)` は値と全7勾配を返す API だが、今回の forward/reverse 時間比較には含めていない。
- **出力形状:** 既存測定は scalar 出力ではなく、27 epoch／30 epoch のベクトルから `27x7`／`30x7` 全 Jacobian を作っていた。
- **scalar 追試:** 同じ `near_limb_contact_117` の同じ27 epoch入力から `t=-1.670997233467463` の1 epochだけを取り、公開関数の `mag[0]` を返す scalar 関数（出力 shape `[]`、勾配 shape `[7]`）を `jacfwd`／`jacrev` で15組測定した。autodiff-vbm は `0.865/0.869 ms`（reverse/forward `1.004`）、MicroLUX は `35.840/24.409 ms`（`0.681`、この測定では reverse が `31.9%` 速い）。scalar の MicroLUX 勾配2モード間には最大相対差 `3.26e-4` があり、これは速度比較であって精度判定ではない。実行時には MicroLUX の sampling capacity 警告（使用 `581`、上限 `478`）も出た。MicroLUX native `.so` は未追跡 artifact で、SHA-256 と build provenance の限界は metadata に記録した。
- **ベクトル結果:** `near_limb_contact_117` と `near_double_root` の full-Jacobian 測定で、autodiff-vbm の reverse/forward 時間比は `1.004`、`1.000`。MicroLUX は `2.27`、`2.52` で reverse が遅かった。MicroJAX CPU reverse は公開 API のサポート外のため未実行。official VBM は勾配 API なし。
- **数値一致の範囲:** autodiff-vbm は両ケースで Jacobian 差がほぼ機械精度。MicroLUX は `near_double_root` では相対差 `2.4e-8` 以下だが、接触ケースでは最大相対差 `2.25e-3`。この差の精度評価は、依頼どおり保留中の issue #3 に残す。
- **判定:** 前回の「forward が速い」は27/30出力の full Jacobian 測定についてのみ正しい。scalar 出力では同じ一般化はできず、今回の MicroLUX 例では reverse の方が速かった。これは一つの入力・実装での測定であり、全パラメータ領域での性能や精度保証ではない。autodiff-vbm は vector・scalar とも両モードほぼ同じ。
- **証拠:** [逆／順モード全 Jacobian と scalar 出力の測定](issue6_7_gradient_api_audit_20261017/metadata.json)、[scalar 追試と一致 LD 測定](issue6_ld_scalar_output_audit_20261017/metadata.json)、[API timing audit](api_timing_audit.md)。

## 判定の要約

- 保存済み raw は1件目の10.92%差を再現しない。差を記録した元 run の provenance は未特定。2件目の非有限件数・代表例も保存 raw と一致せず未確認だが、非収束時の fail-closed と通常APIの理由情報が粗い点はコードで確認できる。
- #4 は VBM の接触近傍での丸め不安定性まで確認したが、外部 VBM の内部原因と q>0 の独立参照は未解決。
- #5 の有限値 API 契約は修正・検証済み。中心3点は `support_unproven` の点光源 fallback を返す。MicroLUX の容量拡大で他実装に近い値は得られたが、誤差状態は精度を保証せず、有限光源値と独立参照は未解決。
- #6 の lcbinint 重複評価は既存 fused kernel の再利用で修正した。修正後の難ケースは solver と status が異なるので性能受容は保留。AutoDiff VBM では LD 値-onlyもJet derivative lanesを計算することを source で確認した。歴史的 `6.03x` は出所不明。#7 の既存 forward/reverse はベクトル全 Jacobian、追加 scalar 追試では MicroLUX の reverse が速かった。勾配精度は依頼どおり issue #3 として保留中。
- 速度値は対象ケースと実装版に限った測定結果であり、ライブラリ全般の性能保証ではない。
