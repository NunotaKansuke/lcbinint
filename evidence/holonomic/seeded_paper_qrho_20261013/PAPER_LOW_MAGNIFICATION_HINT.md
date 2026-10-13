# 論文用経路の低増光性能から得られる手掛かり

2026-10-13。実装変更なし。既存結果と元ベンチマークの計時条件を照合した。

## 元の論文用ベンチマーク

参照した実行仕様は `/home/nunota/.codex/worktrees/8d2b/lcbinint/tests/diagnostics/recal2026/BENCHMARK_controlled_pure_kernel_20260813.md` と同ディレクトリの `bench_grid_vs_vbm_pure_kernel.py`。2026-09-09 の実装・結果の履歴は親checkoutの `tests/diagnostics/recal2026/HANDOFF_multi_run_cartesian_fill_results_20260916.md` に残る。

論文用計時は cache-warm の `_evaluate_preplanned_xy`。同じ入力を先に一度実行し、二回目のnative時間を測る。Cartesian/polar両候補のself-convergenceによるnbin探索と、速い候補の選択は計時外。seed cacheはsource位置・rho・s・qなどの完全一致を要求するため、同一入力の反復ではseed構築が省略される。この条件は、毎回seedを構築する今回の計時とは異なる。異なるepochに対するキャッシュ再利用を提案する根拠ではない。

元ベンチマークのVBMは uniform=`BinaryMag`、linear=`BinaryMagDark`。先の再現監査スクリプトの uniform=`BinaryMag2` と同一呼び出しではない。再現監査が低増光の保存時間を近似再現した事実だけで、元の呼び出し経路を特定してはいけない。

元の2026-09-09のraw JSONは指定された旧出力ディレクトリに現在残っていない。したがって論文用の低増光勝率や、同じ入力の旧nbin・選択methodは今回再集計していない。

## 今回の実装の同一入力による内訳

入力は本ディレクトリの `joined_v2_vbm.tsv`。`target == 1e-3`、`reference_1e-6 < 1.1` で全行を抽出。uniform=248行、linear=251行。seed構築込み、caustic構築は除外。時間は既存実測の列の中央値で、今回再計時した値ではない。

| Profile | VBM µs | seed µs | planning µs | integration µs | full µs |
|---|---:|---:|---:|---:|---:|
| uniform | 46.85 | 40.23 | 26.67 | 34.91 | 112.76 |
| linear | 50.21 | 40.23 | 27.03 | 52.97 | 133.33 |

各列の中央値は同じ行とは限らず、fullにはその他の準備・終了処理もある。段階別中央値の和をfullとしてはいけない。

同一入力ごとの `VBM time / integration time` の中央値はuniform=1.341、linear=1.080。積分段階だけならuniformの77.8%、linearの57.4%でVBM呼び出し全体より速い。一方 `VBM time / full time` の中央値は0.391、0.408。fullでの勝率は0%、11.6%。積分のみの数字をend-to-end速度として扱わない。

## 判断

低増光でも積分kernelそのものには既に競争力がある。毎回約40 µsのseed構築と約27 µsのplanningが重なることがfullの負けに大きく効く。論文用のcache-warmな勝率をそのまま転用することはできないが、積分前の固定費を省いたときの可能性を示す。

次の比較対象は、低増光でのCartesian/polarの一回評価に含まれるseed・探索・積分と、現行seeded経路の対応段階。元の論文用候補選択を後から真似て速い結果だけを選ぶのではなく、一回の評価に必要な準備とその費用を含めて比較する。現段階では全局面の勝利や特定の新しい近似の安全性を確認していない。
