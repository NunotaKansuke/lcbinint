# V2 adaptive の full epoch / trajectory warm benchmark

q–rho corpusで、D14/topologyをtimer外に置いた従来のpure adaptive kernelと、production control flowに近いfull epochを分離して測定した。全結果と再現手順は [trajectory evidence](../../evidence/holonomic/v2_vbm_adaptive_trajectory_20260911/REPORT.md) に保存している。

今回の主な比較は次の3つである。

- `radial-only`: topology/D14を事前生成し、adaptive積分本体だけを計時
- `full-cold`: 各epochで `epoch_adaptive` を呼び、`classify_cells` とD14を毎回含める
- `full-warm`: trajectory内で `PreparedEpochGeometry` を転送し、L1を無効にしたL2 D14 warm seedを使う

1,804 trajectory × 4 epoch × 2 target = 14,432 rowsを、uniform/linear、`RelTol=1e-3, 1e-4`で評価した。全laneで value convergence/status は14,432/14,432。全条件を混ぜた中央値は radial-only 0.1618 ms、full-cold 0.7951 ms、full-warm 0.6682 ms。初回epochを除くsteady stateでは full-cold 0.7927 ms、full-warm 0.6289 msで、D14 warm seedによる中央値の改善を確認した。

warm mixはL3 cold build 3,608、L2 warm recompute 10,824、L1 reuse 0で、各trajectoryの最初のepochだけcoldになる期待値と一致した。accuracy、status、stage timing sanity、raw hashはevidenceの `summary.json` を正とする。

既存の論文用形式と同じ2×3図を [evidence](../../evidence/holonomic/v2_vbm_adaptive_trajectory_20260911/REPORT.md#2×3-whole-epoch-figure) に保存した。各図は旧図と同じく、上段がVBM/V2速度比、下段がVBM `RelTol=1e-6` referenceとの差で、`full-cold` と `full-warm` を別図にしている。列は `q×rho`, `A_fs×rho`, `A_fs×q`。カラーバーも旧図の境界値・色をそのまま使用し、灰色の斜線はデータ不足の射影binである。
