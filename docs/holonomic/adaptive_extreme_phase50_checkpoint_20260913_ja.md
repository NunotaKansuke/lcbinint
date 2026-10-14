# Phase 50: endpoint Newton の仕事量診断

基準は d07cf69。Phase 49で一様輝度の radial-only もVBMに届かないことを確認したため、endpoint再評価の削減余地を測定した。今回の変更は診断のみで、速度改善の採用・production router変更・収束条件変更はない。

## 実装

`HOLO_ENDPOINT_WORK_PROBE` で `polish_endpoint` の実評価回数と同一引数の再訪を記録。引数一致はbinary64の値とzeroの符号で判断する。直近最大7引数を保存し、診断なしのビルドには探索処理を入れない。停止理由はstationary、small step、iteration limitに分離した。既存の計算順序・6反復・最終phi評価・reliability判定は保持。

値はV2Profile scope内の全polish_endpoint呼出しを数えたもので、adaptive sampleだけに限定したカウントではない。引数履歴は標準6反復と最終評価の7回に対応する。任意に7を超える反復を指定した場合、履歴全体の再訪率を表さない。

## 実測

RelTol=1e-3、7216 trajectory rows、warm-timing。各trajectory先頭を除くsteady-stateは各profile 2706 rows。

|profile|calls|phi評価|同一引数の再評価|6反復上限|評価/call|
|---|---:|---:|---:|---:|---:|
|uniform|437614|1397642|79481 (5.69%)|81244 (18.57%)|3.194|
|LD|484366|1556971|87701 (5.63%)|90996 (18.79%)|3.214|

同一引数のキャッシュで直接省けるphi評価は約5.7%に留まる。これはwhole時間の5.7%削減ではない。履歴探索費用も必要なので、今回cache最適化は実装しなかった。一方、約19%の上限到達には次の調査余地がある。ただし丸めによる停滞か真の未収束かは未分離で、反復上限を減らす根拠にはしない。

raw timingには引数再訪チェックの費用が含まれる。Phase 48より速い/遅いという判断には使わない。今回の証拠は仕事量の分布でありwhole-epoch勝利の証拠ではない。

## 検証と残課題

- Phase 48の同じ7216 rowsと一対一比較: mu / convergence / nodes 全て完全一致。
- adaptive unit: 1484 checks, 0 failures。
- 今回数値演算は変更していない。独立referenceと5Jacの新規再実行はしていない。
- cold側、1e-4側の仕事量診断は未実施。
- 次に上限到達時のresidual/step履歴を分離する場合も、geometry誤差ledgerと既存reliability条件を維持する必要がある。
- 全体目標の1e-3でVBMに勝つ条件は未達。現在の候補とPhase 49の比較結果は変更なし。

再現: `bash evidence/holonomic/adaptive_extreme_phase50/reproduce.sh`。
raw: `profile.tsv`、集計: `summary.json`、unit: `unit.txt`（同ディレクトリ）。CPU0、既存と同じcompiler flags。比較対象は保存済Phase 48 profileであり、交互タイミングA/Bではない。
