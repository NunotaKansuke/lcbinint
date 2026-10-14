# Master seed + existing frontier integration

seed providerをmaster b51c00c566667fc2ed1fe640543219f480642002と一致するmain checkoutのlibraryへ変更。該当finite/point sourceにはmasterとの差分なし。既存の境界追跡・積分コードを使用。master Cartesianのpreplanned経路と同じseed cache引数（caustic distance infinity）を使用。caustic自体の構築は維持。各評価でfresh objects、3反復、1 thread。caustic構築・距離は比較時間から除外、seedを含む。

通常87条件: seed 0.10020 ms、境界準備0.12687 ms、積分0.40679 ms、全体0.63608 ms。前回削減seed版0.73399 msより約13.3%短縮。Cartesian前回同一条件0.77080 msより約17.5%短縮（別測定、微小差の断定には不向き）。nbin選択は従来のoffline最速passing gridをそのまま使う。

896評価、D14参照が解決した754評価で相対差が各許容値以内・candidate収束・support成立を確認。参照未解決142評価は精度受理対象から除外。独立真値・全域保証ではない。

build: benchmarks/holonomic/build_seeded_radial_master.sh
run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_seeded_radial_master evidence/holonomic/caustic_seeded_connection_v4/{initial,fold,deeper}_minimal_seeds.tsv OUTPUT.tsv 3 0 （各入力を個別に指定）

製品の公開経路への昇格は行っていない。実験用full benchmarkのseed引数とlink先を変更し、既存積分器へ接続した。
