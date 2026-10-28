# Triple LightCurve 大規模検証・修正・性能監査

## 結果

- `experiment/algebraic_boundary @ 37e7f6118a65` の通常native LightCurveを検査し、2種類の不具合を修正した。変更は3ヘッダと新規回帰テスト。未コミット。
- 最終192条件、4416 source epoch、値／微分の8832出力で例外0、非ゼロstatus0、無効微分0。値／微分の経路不一致は0。
- より厳しいVBM MultiPoly／NoPolyの730 source点との2920比較で要求誤差超過0。最大相対差0.000343333、要求予算に対する最大比0.343。全4416点に厳しい参照がある意味ではない。
- C++30/30、通常APIのPython回帰28/28通過。全リポジトリのテストがgreenという意味ではない。旧契約を期待する既知の9失敗の書き換えは行っていない。

## 範囲・比較条件

- 既存12軌道 × LD=0,.5,.9 × rtol=1e-3,1e-4 の72条件、各33 epoch。接触・かすめ・大光源・近接／広離角・等質量・惑星質量比を含む。
- seed=20261028の120配置、各17 epoch。s/sep2はlog-uniformで約0.22〜4.47、q/q2は1e-5〜10、rhoは1e-4〜0.40、斜め軌道、非ゼロt0、異なるtEを含む。実入力はcases.jsonに固定。
- 通常の `LightCurve(lens="triple",param_type="vbm")`、backend指定なし。static・single source・一様／線形LD。動的モデル、JAX、二光源、quadratic LDは対象外。
- VBM 5.5 のSinglePoly、MultiPoly、NoPolyを別workerで実行。Tol=0、RelTol=request、linear LD。Tol=0はVBMのpoint shortcutを無効にするため、VBM既定設定の速度とは区別する。
- 行列のworkerは各1 thread、120秒のwall timeout、AS 8 GiB。後段の参照は最大2 worker。全作業の承認上限は8 CPU / RAM16 GiB。

## 修正した問題

1. **geometry端点の局所昇格不足。** doubleのNewton反復が未解決のとき、既存の局所quadへの経路に入らず打ち切っていた。初回の9配置・53 epochで `patterson_geometry_repair` による非有限muを再現。古い候補を安く棄却し、継続・独立再探索でも未解決なら保持した候補を局所昇格する。slope、順序、非重複、内部membership等の物理検査は維持。
2. **stationaryイベントの二重登録。** `random-079` のLD点で、同じ物理根の機械精度幅の2つの箱をdoubleの一意性証明が同一視できなかった。半径差が約1 ulpの偽panelを作り `panel_resolution_limit` となっていた。重なった局所箱の合併にquadの一意性証明を追加し、証明できた場合だけ重複を除く。半径の近さだけでは除かない。値は7.977506728（未収束）から約7.945456096（収束）へ回復。独立の厳しいMP/NP参照は約7.945453177。

- 最初の昇格案は古い候補にもquadを使い、積分を含む一部条件の微分を約40〜50%遅くしたため採用しなかった。最終版は昇格を遅らせ、その追加費用を抑えた。途中試行のrawを保存しているが、最終性能表には混ぜていない。
- 値の許容誤差、微分の列、support完全性条件を緩めていない。微分誤差によるrefinementやsource-disk shortcutは追加していない。

## VBM側の不一致と判定

- 3方式の出力が揃う4331点中、SinglePolyがMP/NPから要求誤差以上に離れる点は26。これは不一致の観測であり、それだけでMP/NPを真値とはしない。
- SinglePolyのworkerが120秒でタイムアウトした条件が4件、`double free or corruption (!prev)` を伴うabortが1件。worker timeoutは17 epochの3回評価・import等を含み、単一epochの時間ではない。失敗を除外せずvbm-summary.jsonに記録。
- `random-039` の一例ではSinglePolyが約2.48e19、MP/NPが約1.45598。別例 `random-004` では要求を厳しくするとSinglePolyが3.75→21.76→279.77と離れ、nativeとMP/NPは約3.269へ寄った。SinglePolyを単独の正解基準にしない。
- MP/NPは厳しい730点の参照では最大相対差約3.78e-8。ただし同じVBMの積分・LD処理を共有する部分があり、両者の一致だけで厳密解とはしない。nativeの別積分法、精度依存、微分の差分幅も併用した。

## 微分とキャッシュの検査

- 時刻順序の反転を192条件で検査。status異常・無効微分0。値の最大相対差9.16e-07、微分の最大 `abs(a-b)/(1+abs(a))` は0.00049。微分の要求誤差証明ではない。
- 修復した9配置の各10列について、独立なVBM NoPolyの中央差分に全90比較が `abs=1e-3 + rel=3e-3*abs(FD)` の診断基準で一致。3つの差分幅を保存。MPとNPの両方で最後の2差分幅が安定したのは85/90。残り5比較では主にMP差分が揺れ、NPとnativeの一致を全精度認定とは扱わない。
- point shortcutのrho微分=0と、より厳しいhexの非ゼロrho微分を混ぜると偽の不一致になる。同一経路での確認をgradient-triage.jsonに保存した。

## 性能

同じCPU0、各engine独立プロセス、1 thread、逐次・順番回転／反転。constructor＋first、warm（2 warmups＋8回中央値）、s微小変更4回を分離。thread CPU時間でimport/IPCを除く。build・精度参照・profileを重ねていない。CPU周波数・外部負荷は固定していない。

最終48条件の条件別before/after比の中央値：値の時間削減0.5%、値＋微分0.6%。数%程度の差は環境の揺らぎを含む。大幅な一般的高速化や、全ケースでの改善は主張しない。

以下はrtol=1e-3、warm ms/65 epoch。VBMは値のみ。全条件のcold/warm/geometry更新はtiming-summary.csv。CSVのpoint_baseはmode0の診断用ablationであり、同等な有限光源計算でも正確なstage timerでもない。

| ケース | LD | native値 | native値＋微分 | SinglePoly | MultiPoly | NoPoly |
|---|---:|---:|---:|---:|---:|---:|
|miss|0.0|4.752|5.478|9.652|12.644|23.000|
|grazing|0.0|18.858|27.203|11.065|15.127|33.829|
|close|0.5|12.827|22.577|33.027|43.227|119.315|
|large|0.5|49.880|70.367|353.577|516.593|1259.342|

最終DSOをwarm後、各条件3 thread-CPU秒、199 Hzのperfで調べた排他的なスタック分類。各約600 sampleの概算であり、直接のstage timerではない。strip前の別relinkと実DSOの.text/.rodata/.eh_frameのbytes・アドレス一致を確認してsymbol解決した。

| ケース・mode | 求根/hex | caustic geometry | support/events | 積分 | その他 |
|---|---:|---:|---:|---:|---:|
|miss-value-warm|96.0%|2.3%|0.0%|0.0%|1.7%|
|miss-gradient-warm|95.5%|1.5%|0.0%|0.0%|3.0%|
|grazing-value-warm|3.2%|3.8%|67.6%|23.1%|2.3%|
|grazing-gradient-warm|0.8%|1.3%|38.7%|52.5%|6.6%|
|large-value-warm|0.7%|0.3%|13.5%|84.7%|0.8%|
|large-gradient-warm|0.8%|0.3%|12.0%|84.5%|2.3%|
|close-value-warm|16.8%|4.3%|33.3%|44.0%|1.5%|
|close-gradient-warm|11.8%|1.7%|26.5%|56.8%|3.2%|

- 非接触hexの主要費用は求根。大光源LDは積分本体が支配的。かすめる値計算ではsupport/event証明が大きい。1つのstageを全ケースのボトルネックとはしない。
- 必要なquadを一律doubleへ落としたり、独立old/fine差を弱める改善は行わない。今回の採用変更は不具合の回復と、その回復処理の不要な費用の抑制。

## 残る限界

- 追加のrtol=1e-6検査では、修復した9配置の代表点中7点で `refinement_budget` が残る。有限候補が独立差分と近くても、status=1／gradient_valid=falseを維持する。1e-6以下の全域収束・微分精度は認定していない。
- 静的native・限られた行列に対する結果であり、全配置・全接触点・全許容誤差での数学的保証ではない。有限の成功テスト数と一般的な認定を混同しない。

## 再現・保存物

- `cases.json`、`inputs/`：固定入力。`rows/`：初回の全方式raw。`fixed-rows/`：最終native raw。各最終workerにDSO hashを保存。
- `references/`、`reference-summary.json`、`independent-gradients-final.json`：厳しい参照、微分。`invariance.json`：順序反転。
- `timing-rows/`、`timing-summary.csv`：最終比較。途中で棄却した試行はfirst-fix/deferred-trial等の別出力へ保存。
- `final-profiles/`、`final-profile-summary.json`、`final-symbol-identity.json`：最終profileとsymbolの同一性。
- `final-provenance.json`、`task-only.patch`、`initial-dirty.patch`：出所と今回の差分。既存の10ファイルのdirty差分が初回と同一であることを確認。
- `baseline-build/lcbinint` と `final-package/lcbinint` に測定DSOをローカル保存。portable archiveには巨大DSO・object・生perf.dataは含めず、再buildまたはこれらのpackageの別途移動が必要。
- `validation-artifacts.tar.gz` にraw JSON・script・log・最終source snapshot・symbol化済みstackを収録。`artifact-manifest.json` にhashを保存。既存evidenceを上書きして再実行しない。

最終nativeの行列再実行：

```bash
OPENBLAS_NUM_THREADS=1 /home/nunota/.miniconda3/envs/myenv/bin/python evidence/triple_stress_20261028/fixed_matrix.py
```
完了済みrowはskipする。新しいコードでは出力先を新規にし、native hashが混ざらないようにする。
