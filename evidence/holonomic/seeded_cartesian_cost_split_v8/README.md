# Cartesian / seeded frontier cost split

同じ87 source/u条件、relative tolerance 1e-4、各3回の中央値を条件ごとに取得し、その合計を87で割った平均時間。geometry reuseなし。caustic構築・距離探索を除外し、seed構築は含む。Cartesian nbinはv7で参照との相対差1e-4を満たした最速のグリッドを再利用し、選択費用を含まない。

|処理|Cartesian mean ms|seeded frontier mean ms|
|---|---:|---:|
|中心像求根|0.00117|0.00150|
|seed生成|0.09978|0.23106|
|seed後の探索・積分|0.66928|0.49995|
|合計（caustic除外）|0.77080|0.73399|

新経路のseed後は境界・動径帯準備0.08836 ms、角度・動径積分0.41159 ms。Cartesianは格子seed配置・flood fill・面積積分が一体のため合算。個別ステージの中央値と合計の中央値は独立に取るため、丸め以外にも微小な残差がある。

seedの役割は共通だが、比較対象の実装は異なる。Cartesianは現行main checkoutのseed provider、新経路は保存済みdiscriminant-algebraic-minimal-seeds-20261012-v1 provider。seed数中央値は80対5。数が少ないこととseed生成費用が小さいことは同義ではない。この計測だけでは新providerの内部ボトルネックまでは特定できない。

今回の全体合計は67.059627 ms対63.857091 msで、新経路は約4.8%短い。条件ごとの全体中央値はCartesian0.453682 ms、新経路0.528137 msで、新経路の方が遅い。新経路のseed後の削減14.731725 msに対しseed生成が11.421036 ms増え、削減の約78%が相殺される。

検証: Cartesian増光率はv7の対応87条件と完全一致（最大絶対差0）。参照D14との最大相対差Cartesian9.91095e-5、新経路1.21448e-5。新経路は今回192評価、参照が解決した178評価上で失敗0。これは独立な真値検証や全域のcoverage保証ではない。

計測手法: Cartesianで実際のpreplanned経路が要求するseed cache key（caustic_distance=infinity, center hintsあり）を先に生成し、その後preplannedを計測。同一epoch内で計算結果を再利用する。後段にはキャッシュlookup/copyの小さな費用も残る。製品コードの変更なし。

再現コマンド（このcheckoutから）:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_cartesian_cost_split evidence/holonomic/seeded_inverse_comparison_v7/initial_final.tsv evidence/holonomic/seeded_cartesian_cost_split_v8/cartesian.tsv
LCBININT_REPRESENTATIVE_PROBES=critical-seeds OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_seeded_radial_full evidence/holonomic/caustic_seeded_connection_v4/initial_minimal_seeds.tsv evidence/holonomic/seeded_cartesian_cost_split_v8/candidate.tsv 3 0
```

入力選択selected.tsv、raw結果cartesian.tsv/candidate.tsv、集計summary.json、設定とhash manifest.jsonを参照。
