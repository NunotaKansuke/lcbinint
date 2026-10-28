# LD frame selection: rejected prototype and gradient regression guards

## 結論

base `656b7e03`。LDの微分付き積分で、値だけの短い予備計算からprimary frameを選ぶ案を実装・全件検証したが、**採用しない**。production sourceを元に戻して再ビルドし、DSOが採用版 `a9b2ca2c3d1cff8795a0329c009719c7865259f3fcc08143460694308f7030a6` とbitwise一致することを確認した。復元版のC++32件、独立LD差分の4件を追加した対象Python100件は全て通過した。

今回の追加は診断証拠と回帰テストで、productionの座標選択や許容誤差を変更しない。値1757/1768、微分1755/1768成功、監査済み4394値比較で要求誤差超過0という採用版の結果は前報告のまま。残る停止は値11・微分13条件。全微分の精度認証とは区別する。

## 試作と検証

試作DSOは `e292a97270a486096d9376663bcab882ce5d4ce88934766429fd3592749979cc`。linear LD、rtol>=1e-4、元の完成supportに10以上のeventがある場合に値だけの予備積分を行った。予備積分は最大8 refinement、alternate frameはevent budget8192。元の仕事量192以上または未収束の場合に、値が収束し、評価点が元の半分以下のframeを候補とする。候補でも改めて通常のcoupled gradient積分を行い、support / IFT / value / gradient-valid条件を全て要求した。微分値を選択に使わず、予備積分の値・mesh・微分は再利用していない。

1768条件の両経路3536実行で成功・失敗の分類回帰0、4394監査済み値比較で要求誤差超過0。156曲線・468変換チェックと、940/972/1508の独立プロセスでの正順・逆順・重複評価も通過した。214曲線で新しいframeが選ばれた。

しかしcampaign内の成功曲線のthread-CPU合計は、微分付き44.62→63.46秒に増えた。値経路は43.32→41.78秒。別runで周波数・外部負荷を固定していないため、これは厳密な逐次速度比ではない。重い940/972/1508の予備試験では短縮したが、軽い多数の条件に予備計算を加える費用が大きい。通常APIの専用勾配streamと値積分の仕事量も同一ではない。全体に対する高速化として採用できない。

さらに、値の要求精度に収まることから全パラメータ微分の精度を推定できない。595/1104等ではframe間の微分差が大きく、独立した差分で確認した。

## 独立した微分診断

VBMの一様円盤の倍率を、LDの半径方向重みで独立にGauss-Legendre積分した。通常のVBM LD annulus driverを使わないので、そのannulus上限を正解と誤認しない。MultiPolyのuniform RelTol=1e-9または1e-10、GL 128/256/512と有限差分の刻みを比較した。NoPolyはこれらの独立参照試行でSIGSEGVしたため、以下を二手法一致の認証とは主張しない。raw uniform-disk値も保存した。

- **595 / alpha / LD=1 / rho=.03 / t=0**: 採用版は -0.0149846588211、試作は -0.146104935706。GL128/256、h=1e-4から1.25e-5の左右3点差分は約 -0.01498。GL512で完成した負側差分も -0.0149742〜-0.0149816。採用版を支持し、試作を明確に棄却する。正側GL512以降を含む全参照runは120 CPU秒上限で停止したので、そのmesh全体を完成扱いにしていない。
- **1104 / rho / LD=1 / rho=.03 / t=0**: 採用版91.3595028364、試作122.359935606。最初のh=3e-4から3.75e-5では十分な差分収束が得られず、失敗出力を保存したまま別の細かいrunを実施。GL128/256でh=4.6875e-6の左右差分は91.376〜91.414へ近づいた。直前の刻み・mesh依存を含む保守的な診断許容差0.3の回帰テストにした。任意のrho微分の高精度証明ではない。
- **940 / t0 / LD接触直前**: GL128/256/512、中央差分h=2e-6/1e-6のRichardson値は -1278.18739、-1278.17807、-1278.17919。採用版 -1278.17630 と診断相対差1e-5以内で一致する。
- **1508 / t0 / LD接触直前**: 同じ検査でRichardson値720.5067088、720.5067096、720.5067093。採用版720.5066892と一致する。

新しい4件のテストは `tests/regression/test_triple_ld_contact_gradients.py`。採用版は4/4通過し、不採用の試作ではalpha/rhoの2件が失敗した。元の96件とC++32件だけでは、この試作の問題を検出しなかった。独立差分の診断許容差は各fixtureに明記した。生産設定のvalue rtolを緩めた変更ではない。

## 残る研究上の課題

今回確認したのは、通常のvalue-only適応条件から、別の座標系の微分付き仕事量・微分精度を一律に予測する案が不十分ということ。採用版の当該2条件の微分は独立差分に支持されたため、速い座標系へ切り替える変更を入れない。任意の配置・全微分を保証したとは主張しない。

今後のframe選択には、接触近傍で微分の条件を満たす局所座標と、予備計算の重複を避ける設計が必要。微分誤差でmeshを制御する方法、寄与を捨てる方法、許容誤差を緩める方法へ置き換えていない。極端なfaint componentの証明・積分限界で停止する11/13条件も前報告のmanifestを維持する。

## 再現と状態

試作の正確なsourceは `finite_source_magnifier.candidate.cpp`、対応hashは `candidate-provenance.json`。`reconstruct_candidate.py` はbase sourceからの再構築後にhashを検証する。これらは**不採用の実験用**であり、production sourceではない。復元の確認は `accepted-state.json`、テストは `accepted-regression-100.log` と `restore-accepted-ctest.log`。試作を意図的に棄却する2失敗は `rejected-regression-4.log`。

全入力・raw結果・部分的に停止した参照・source snapshot・scriptは `raw-evidence.tar.gz`、各内容のhashは `raw-evidence-manifest.json.gz`。archive全体のhashと検証結果は `archive.json`。native DSO・static library・compiler一時ファイルは含めず、前報告の採用版source snapshotと本試作sourceを保存した。全jobは終了しており、実行中の計算を残していない。

追跡する `rejected-regression-4.log` はgitの空白検査用に行末空白だけを除いた表示用copy。変更前の生出力は上記archive内にそのまま保存し、manifestのhashで検証した。数値・判定は変更していない。
