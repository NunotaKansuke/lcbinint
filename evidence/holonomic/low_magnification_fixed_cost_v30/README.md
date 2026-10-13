# v30: caustic空間索引によるseed固定費削減

## 採用した変更

現在の研究checkout `experiment/seeded-boundary-events`、HEAD `f3547723b79828d9c501182746028e99f7bad116` に接続した。共有の親checkoutには変更を加えていない。

seed providerはv26のprobe・求根・append・support判定を維持し、caustic構築時に既に用意されている128×128のbranch vertex索引を使って、今回のsource diskに関係し得る頂点だけを取り出す。前のsourceやepochの計算結果は再利用しない。caustic構築、索引構築の従来の費用は計時外、sourceごとの索引検索・filter・sort・seed構築は計時内。

既存scannerの頂点保持条件は `distance < rho + curvature_margin`。隣接線分の最大長をLとすれば `|v_prev - 2*v + v_next| <= 2L`。盲目的な線分横断にも、disk内の点からrho+L以内に線分端点が存在する。このためrho+2Lを丸め誤差分だけ外側へ広げた領域を問い合わせれば、従来scannerが使い得る頂点・線分を保持できる。box・距離二乗にも外向き丸めを使う。候補indexを従来のbranch/vertex順に戻してから、元のextremum refinementとprobe処理を行う。これは既存polyline探索の省略条件であり、連続した真のcausticの新しい完全性証明ではない。

候補が多いqueryは従来の全点scannerを使う。これは求根前の選択であり、計算済みの像や積分を捨てて再起動するfallbackではない。普通のscanner利用者は全点版を使う。疎な内部support descriptorでは、消費されないnearest-distance診断を作らない。`finalize_disk_support`と`resolve_certified_probes`が使うextrema/probeは維持する。

実装は `seed_indexed.cpp`。`build_seeded_image_seeds.sh` がそのprovider objectを作り、`build_seeded_radial_master.sh` と `build_seeded_paper_qrho.sh` がそれをリンクする。過去のv26 objectと既存のpaper raw結果は保存してある。

## 計測

paper corpusの低増光全499入力と、既存tolerance probeの難例・誤差tailを合わせた684入力。relative tolerance=1e-3/1e-4、absolute magnification tolerance=1e-8、各5反復。各反復でmagnifierとworkspaceを新しく作る。中心像の求根・seed構築を含み、caustic構築・object構築・I/Oを除外。OMP/BLAS/MKLは1 thread。build完了後にbefore→after、after→beforeで直列実行した。

入力ごとの5反復中央値を取った後、その中央値を集計した。低増光はVBM 1e-6参照のA<1.1。uniform=248、linear=251。単位はµs。

| profile | seed before→after | planning before→after | integration before→after | full before→after | 同一入力のfull削減率中央値 |
|---|---:|---:|---:|---:|---:|
| uniform, 1e-3 | 39.96→4.19 | 26.22→25.62 | 35.54→35.11 | 113.80→73.36 | 34.3% |
| linear, 1e-3 | 40.15→4.21 | 26.73→26.13 | 53.65→52.73 | 134.58→92.54 | 29.7% |

逆順でもseed約40.1→4.2 µs、full削減率はuniform=33.0%、linear=28.2%。planning/integrationの微小差は採用変更による改善として主張しない。段階別中央値の和とfullの中央値は一般には一致しない。同じ行から引いた未計上時間は約0.3 µsで、別の10 µs固定費が存在するという根拠はない。

保存VBM時間との比較では `R=t_VBM/t_full` の低増光中央値は約0.39/0.41から約0.61/0.61。R>1が勝ち。forward順で勝つ入力数はuniform=0→8/248、linear=29→49/251。reverse順ではuniform=0→7、linear=31→47。全局面の勝利は達成していない。保存時間はhistorical comparatorであり、今回VBMを再計時した値ではない。

## 精度と探索の検証

- 全7,216入力でseed座標をhex float出力して比較。座標、順番、個数、support flagがbyte-for-byte一致。
- 全7,216入力×2 tolerance=14,432行のbefore/afterで、mu、誤差推定、収束flag、support flag、seed個数、node数、評価数、stop、数値statusが全て一致。
- 従来版の収束行には1e-4目標を超える参照差が3行ある。今回はそれを修正しておらず、新しい超過も追加していない。既存の `coverage_certified=false` も維持する。
- derivative/integratorの実装は変更していない。

## 棄却した実験

1. 全点走査の平方根を減らすだけの案はseed時間を約42→58 µsに悪化させたため不採用。
2. 境界の初期二分探索を12→4回にすると684入力では値が一致しplanning約3割削減。しかし全corpusでは44行の値変更、16行の収束flag変更、新規参照誤差超過27行が生じたため不採用。
3. 2回、0回の案にも誤差・収束の悪化があり不採用。現在の経路は12回を維持する。

細かいbracketが現在のevent選択・band構築に必要であることを示す反例がある。これを減らすなら、回数の変更に加えてevent/band構築自体の整理が必要。残る低増光planning約26 µsが次の候補だが、回数を下げた版は接続していない。

## 再現

このexperimental checkoutから実行。scratchとbuildは全てproject内。`build_seeded_fixed_cost.sh` はbefore/採用版と棄却案を別binaryにする。

```bash
bash benchmarks/holonomic/build_seeded_fixed_cost.sh
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build-seeded-events/bench_fixed_cost_before_v30 \
  evidence/holonomic/low_magnification_fixed_cost_v30/input.tsv \
  evidence/holonomic/low_magnification_fixed_cost_v30/accepted_before.tsv 5
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build-seeded-events/bench_fixed_cost_indexed_v30 \
  evidence/holonomic/low_magnification_fixed_cost_v30/input.tsv \
  evidence/holonomic/low_magnification_fixed_cost_v30/accepted_indexed.tsv 5
python3 evidence/holonomic/low_magnification_fixed_cost_v30/summarize.py
```

`summary.json`、accepted/reverse/full TSV、seed hex比較、棄却案のraw、test logとhash manifestを本ディレクトリに保存。反復数を変えても速度集計は入力ごとの中央値を使う。fullの1反復は数値同値性の検証用であり、full-corpusの速度主張には使っていない。
