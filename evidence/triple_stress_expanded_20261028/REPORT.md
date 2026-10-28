# Triple LightCurve 拡張検証（2026-10-29 JST）

`experiment/algebraic_boundary @ 37e7f6118a65`。前回の192条件から範囲を広げ、新たに1,528条件・6,520 source epochをVBM 5.5のSinglePoly / MultiPoly / NoPolyと比較した。通常native LightCurveの値・値＋10成分微分を検査した。変更は未コミット。

新たに2件を修正した。Python37テスト、修正後ヘッダを再ビルドしたC++30テストが通過。最終native全3,056 engine条件で、以前成功していた条件の新規失敗は0。成功条件は微分が1条件増えた。極端な配置に対する明示的な拒否は残る。

## 修正と微分誤差の原因

1. `LightCurve(param_type=..., options=...)` は、optionsが後ろにあると明示した座標規約や許容誤差を上書きしていた。Optionsを初期値として先に読み、明示キーワードをその後に適用する。引数順序の8テストは旧版で3失敗、修正後は全通過。これにより既存のlcbinint座標の微分テストも、実際にその座標規約を検査するようになった。
2. 一様光源の外側fold / complex fold用の変換区間が長く、光量は収束しても鋭い境界運動の積分が粗かった。通常のuniform chartの幅上限を3から1.5へ変更した。値だけを見る同一の適応制御を維持し、微分誤差の別コントローラや一律の許容誤差引下げは追加していない。局所quad昇格とsupport証明のゲートも維持。

問題の配置192、rtol=1e-3、同一epochの例。参照はVBM NoPolyのRelTol=1e-8中央差分で、3差分幅・MultiPolyとの一致を確認した。

| 微分 | 修正前 | 修正後 | 独立差分参照 |
|---|---:|---:|---:|
|u0|0.082187724|0.071711819|0.071704698|
|rho|0.064672449|0.075393088|0.075394063|
|q2|25.259457652|25.673116297|25.673137059|

この配置の全10成分は `atol=1e-4 + rtol=2e-4` の回帰基準を通過。さらに周辺24配置・96成分を、差分幅を小さくして安定性を確認した独立参照で検査し、従来から使う診断予算 `1e-3 + 3e-3*abs(FD)` の超過は修正前13→修正後0。別の9配置（一様/LD=.5/LD=1）・90成分も同じ診断基準で全通過。いずれも全配置・全成分への微分精度保証ではない。

## 拡張範囲と終了状態

- seed=2026102802。wide random 400配置、ほぼ共線/近接120配置、rho=0の48配置、24個のレンズ配置のcaustic標本から選んだ960接触付近条件。caustic標本は標的選択だけに使い、数値経路の証明には使っていない。
- sを0.01〜100、sep2を0.005〜170、質量比を1e-8〜1e4、rhoを0〜1まで拡張。実際の入力はcases.jsonが正本。線形LD=0,.5,1、rtol=1e-3/1e-4、絶対許容誤差0。
- 最大8 CPU / RAM16 GiB。各数値ライブラリ1 thread。探索workerはAS 2 GiB、CPU 20/22秒、wall 45秒で隔離。上限到達・異常終了・非有限値も記録。
- 行列はfrozen baseline packageで実行後、修正版の全1,528条件×2 native経路を再実行し、各rowにDSO hashと入力hashを保存。途中停止は保存済みrowから再開し、孤児processがないことをホスト側で確認した。RESUME.mdと各PID/logを参照。

| engine | worker判定okの条件 | 例外/上限/未収束など |
|---|---:|---|
|value|1420/1528|exception=91, nonconverged=17|
|gradient|1400/1528|exception=97, invalid_gradient=31|
|SinglePoly|1351/1528|worker_exit=149, invalid_value=28|
|MultiPoly|1515/1528|worker_exit=13|
|NoPoly|1513/1528|worker_exit=15|

VBMのokは正の有限値が返ることを示す。nativeではstatus0、微分経路ではgradient_validも必要。VBMのokを数値精度の認定としては扱わない。nativeの主な拒否はoverlapping_root_brackets、degenerate_radial_event、refinement_budget。有限候補があっても収束・supportゲートを緩めて成功扱いにはしていない。表の単位は曲線条件。

## 精度参照とVBMの未収束値

参照の最終選択は636条件。途中の診断を含め641条件・1,282 engine rowの厳しいVBM参照を保存。両方式の不一致を分離し、確認できたLD参照問題を独立求積で置き換えた結果、588条件・1894 source点の3733個の収束magnification比較で要求誤差超過0。最大相対差は0.00081673459、要求予算に対する最大比は0.8167。全6,520点に厳しい参照がある意味ではない。

参照同士が十分一致しない比較は151件、厳しい参照engineの失敗は58件。rawを保存し、nativeの正誤判定や速度優位の根拠に混ぜていない。

特に855/955/1510では、MultiPolyとNoPolyがよく一致していてもLD積分が100 annuliの上限に達し、内部therrが要求予算を超えたまま値を返した。上限300でも値が動き、1000への増加では異常終了/CPU上限も観測。インストール済みVBMは変更せず、同じ5.5 sourceを診断用に別ビルドしてnannuli/therrを記録した。

VBMのuniform disk magnification M(r)だけを使い、LDを別のGauss-Legendre求積で評価した：

`[(1-c)*M(R) + c*integral_0^(pi/2) sin(theta)^3*M(R*sin(theta)) dtheta] / (1-c/3)`。中心の点源値に依存せず、16/32/64/128点（1510は256点まで）の収束を確認。

| 条件・epoch index | 通常VBMの厳しいLD値 | 独立LD求積 | native最終微分経路の値 |
|---|---:|---:|---:|
|855 / 1|15.412044169|15.406299378|15.406299383|
|955 / 1|2.246245859|2.245733830|2.245730519|
|1510 / 2|12.662571781|12.660767525|12.660767528|

独立LD求積はVBMの根・uniform contourとコードを共有するので、完全に別の物理エンジンではない。LD積分を独立化し、nativeの別の面積積分・精度依存・微小なsource移動の連続性も併用した。

## 座標・順序・近似経路

同一物理配置のVBM↔lcbinint/center_of_mass表現を100条件で検査し、92条件で両表現を評価でき、要求誤差超過0（最大相対差2.96e-5）。8条件は少なくとも一方が明示拒否。レンズラベルの交換は80条件を試し、両表現を評価できたものでは要求誤差超過0。時刻反転/重複、時刻原点の移動、角度への2pi加算も120条件で値・微分を確認。

点源近似と有限源積分の切替ではrho微分が0↔非0となる別の現象が残る。点源式はrhoに依存しないため、その解析微分は0。今回修正した有限源積分の鋭い境界運動とは分けて記録した。gradient_validは有限性と経路収束を示し、全成分の物理的な微分誤差を認定するフラグではない。

## 性能

修正前後を同じCPU0・同じ8軌道・同じ許容誤差・1 threadで逐次測定。import/IPCを除き、2 warmup後15回のthread CPU中央値。参照計算/build/profileを止めた後に実施。外部負荷とCPU周波数は固定していない。各軌道7 epoch。

8軌道の条件別微分時間比の中央値は1.0116（約1.2%増）。問題の192は1.884→2.268 ms（約20%増）。値のみは1.720→1.734 ms。古い版を一律rtol=1e-4にした同じ軌道は2.293 ms、修正版rtol=1e-3は2.268 msで、後者のq2微分はより参照に近い。大幅な一般的高速化は主張しない。

局所probeでは問題epochの境界評価数が431→681。一律1e-4へ下げた旧版は1214。角方向の精度だけを100倍厳しくしても問題が動かず、変換区間を分けることで改善したため、支配的な原因はこの区間でのradial material transportの解像度と判断した。probe-results.json / probe-widths.jsonに切り分けを保存。

VBMとの同一8軌道のwarm比較はcontrolled-vbm-timing.jsonとtiming-comparison.csv。VBMは値のみ、native微分時間は値＋微分。Tol=0/RelTol=requestはVBM既定設定の速度とは異なる。以前の48条件・内部profileは前回evidence/triple_stress_20261028/REPORT.mdを参照。


同じ8軌道のwarm時間（ms / 7 epoch、rtol=1e-3）：

| 条件 | native値 | native値＋微分 | SinglePoly | MultiPoly | NoPoly |
|---|---:|---:|---:|---:|---:|
|192|1.734|2.268|1.207|1.573|1.748|
|24|2.456|3.468|1.472|1.892|3.384|
|54|12.300|20.863|598.422 不一致|2.254|6.640|
|222|0.851|0.955|1.278|1.751|2.277|
|16|1.885|2.878|1.109|1.586|1.676|
|80|4.406|6.829|23.419|34.023|44.418|
|116|9.465|16.203|56.328|1.910|3.676|
|200|1.874|5.716|8.108 不一致|4.083|5.757|

MP/NPは8軌道の測定出力でnativeと要求誤差内に一致。SinglePolyは54と200で要求誤差を超えたため、その速さを同精度の比較としては採用しない。表はこの8例の結果で、一般的な速度順位ではない。

## 保存物と限界

- baseline-build、api-fixed-package、chart-fixed-packageに測定バイナリを分離。provenance.json / final-provenance.json / artifact-manifest.jsonが入力・source・DSO・scriptの同一性を記録。
- rows/ は5 engineの初回raw、chart-fixed-rows/ は全native再実行、tighter-reference/ は厳しいVBM参照。成功値と失敗を一緒に残した。
- gradient192-*、gradient-matrix/、chart-neighborhood*に差分のplus/minus値、刻み幅、微分値を保存。
- tests/regression/test_triple_expanded_contract.pyに新規回帰テスト。既存の無関係なdirty変更は保存し、commit/pushはしていない。
- 旧API/旧既定backendを期待する既知9失敗の書き換えはしていない。今回のC++30/Python37通過を全repositoryのgreenとは扱わない。
- 極端配置の収束、全source点の独立参照、point shortcutを含む全微分成分の一律精度保証は残る限界。1e-6の全域収束を今回の目標に戻していない。
