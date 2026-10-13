# v31: radial boundary planning の実根数判定を軽量化

2026-10-13。`experiment/seeded-boundary-events`、基点 HEAD f3547723b79828d9c501182746028e99f7bad116。未コミットの既存 v30 実装を比較基準にしている。集計は `summary.json`、再集計は `python3 evidence/holonomic/planning_count_v31/summarize.py`。

## 採用した変更

`quartic_fast_count.hpp` と `seeded_radial_frontier.hpp` の境界探索用実根数判定。各 radius probe で Sturm 列を組み立てる代わりに、まず二元四次形式の正定値性の十分条件、その後四次多項式の不変量 I,J,P,D の符号で 0/2/4 根を分類する。符号が丸め誤差の上界と分離できない場合だけ従来 Sturm 判定を実行する。既存の計算結果や witness はそのまま使う。

係数の long double 構築、12回の二分探索、局所 fold 求根、積分精度判定、微分経路は維持。積分器の点数削減や epoch 間キャッシュ、複数コアは使っていない。既存の全用途の `certify_quartic` は変更していない。

## 速度

同じ入力、rtol、CPU、単一スレッド。seed を毎回構築して計測に含め、caustic 構築は除外。684入力、rtol=1e-3/1e-4、各5反復。before→after と after→before の両順序で計測。各入力の5反復中央値を取り、その中央値を表に示す。低増光は reference A<1.1、uniform 248入力、linear LD 251入力。

| rtol=1e-3 | 準備 before→after [µs] | seed込み全体 before→after [µs] |
|---|---:|---:|
| uniform、順方向 |25.744→15.890|73.518→62.101|
| uniform、逆方向 |25.744→16.484|74.354→62.738|
| linear LD、順方向 |26.085→16.466|93.470→81.470|
| linear LD、逆方向 |26.139→16.721|93.449→83.071|

準備費用の集団中央値は約36–38%削減。入力ごとの比率の中央値は44–45%削減で、集団中央値の比とは異なる。全体の入力ごとの比率はuniform約15%、LD約13%削減。境界判定の約99.2%が高速経路を通る。seed・積分時間の小差は測定変動として扱う。

従来から使っていた87組の比較行列、rtol=1e-4では、各組5反復中央値の平均：

|順序|準備 [ms]|seed込み全体、caustic除外 [ms]|
|---|---:|---:|
|順方向|0.072666→0.047673|0.294220→0.267764|
|逆方向|0.073170→0.048071|0.296121→0.271010|

全局面でVBMに勝ったとはしていない。特に低増光で保存済みVBM中央値約47–50µsに対し、今回まだ約62–83µs。

## 数学的根拠と保証の範囲

係数を a,b,c,d,e とすると I=12ae−3bd+c²、J=72ace+9bcd−27ad²−27eb²−2c³、Δ=4I³−J²。Δは通常の判別式の27倍。定義の一次資料：[Cremona & Fisher (2008)](https://www.dpmms.cam.ac.uk/~taf1000/papers/quarticequiv.pdf)。Δ<0なら実根2個。Δ>0なら実根0個または4個。

P=8ac−3b²、D=64a³e−16a²c²+16ab²c−16a²bd−3b⁴。depressed monic quartic y⁴+p y²+q y+r では P=8a²p、D=16a⁴(4r−p²)。P,D<0なら y=±sqrt(−p/2) で小さい方の値が r−p²/4−|q|sqrt(−p/2)<0となり、少なくとも2実根、Δ>0なので4実根。逆に4実根ではp<0。積r<0ならD<0は自明。2正2負の根u,v,−s,−tではu+v=s+t=S、uv,st≤S²/4から (S²−uv−st)²>4uvst が distinct roots では成立しD<0。したがってΔ>0でPまたはDが正なら0実根。

正定値の十分条件は Young 不等式で奇数項と負の二次項を x⁴,y⁴ に吸収する。符号評価にはterm絶対値和と gamma_1024 の保守的な浮動小数誤差スクリーンを使用。通常範囲内での2冪係数正規化は厳密。subnormalへの係数変化や不明な符号、p4=0の射影chart境界は従来法に渡す。

この高速判定は入力double係数多項式の実根数を扱う。係数そのものの物理モデル誤差、像のcoverageや面積積分全体の厳密保証を追加したものではない。`coverage_certified=false` は維持している。fallbackの従来Sturmも数学的な完全保証と同一視しない。

## 数値検証

全7216入力×2許容誤差=14432行で比較。以前成功した行はすべて mu、積分node数、evaluation数が完全一致。新たな許容誤差超過は0。未収束は10→6、4条件が回復。既存の未収束6条件と既存の精度超過は未解決のまま。

物理入力の全判定probeを従来法と監査して、食い違いは1種類の係数、4条件のみ。従来法は2根、新方式は0根。double係数をFractionで厳密有理数に変換した丸めなしSturm列と独立binary128計算は0根を確認した。従来法のremainder計算の相殺による誤判定であり、これが4条件の回復に対応する。詳細は `physical_audit.log`、`audit_exact_counts.py`、`exact_count_audit.json`。

- 新規fast-countテスト：188407 checks、0 failures。187906 accepted cases を独立binary128 Sturmと照合。既知根数、重根、近接根、chart境界、極端な係数scale、実測反例を含む。
- seeded frontier：321 checks、0 failures。
- adaptive radial：1614 checks、0 failures。微分と資源上限を含む。
- 87組の従来行列：referenceが解決した条件にcandidate失敗なし。D14 reference未解決14条件は成功の根拠に使わない。

## 再現・資料

`benchmarks/holonomic/build_seeded_planning.sh before after audit` でbaseline snapshotと現在の採用方式を同じフラグでbuild。実行形式は `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/bench_planning_after_v31 INPUT OUTPUT REPS`。beforeも同じINPUT/REPSを用いる。計測ファイル `forward_*` / `reverse_*` が採用速度、`full_*` が全数値比較。`profile_*` はinstrumented診断で通常速度表に混ぜない。`normal*` は従来行列。

`before_include` に変更前のplanning/count headerを保存。build flagsはbuild script、ソース・入力・結果のhashは `SHA256SUMS`、環境は `provenance.json`。親checkoutの標準ライブラリとv30 indexed seed objectを使用。

不採用：Sturm列内部のcount専用簡略化は合成benchmarkで約5%遅かったため撤回 (`quartic_summary_rejected.hpp`)。doubleのみの係数生成と誤差box伝播は改善約0.3–0.5µsに対して複雑さが大きく不採用 (`double_coefficient_rejected`)。計測を残すが採用版の主張に含めない。
