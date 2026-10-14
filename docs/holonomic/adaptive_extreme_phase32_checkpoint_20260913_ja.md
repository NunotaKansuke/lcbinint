# Phase32: C/G/Z評価と最終discriminant組合せのprecision切り分け

基準3cfb0f9。Phase31と同じ202048候補点で計算のprecisionを分離した。production solver/default/gateは一切変更しない。

method 0–2はPhase31と同じ。追加:

- 3: double係数・double C/G/Z評価 → qfで最終組合せとD/D'
- 4: qf係数・qf C/G/Z評価 → doubleへ丸めて最終組合せとD/D'
- 5: doubleへ丸めた係数をqfで再評価・最終組合せ（係数丸めの診断）
- 6: 現行D14Realで係数/評価/組合せ/除算
- 7: double係数・double C/G/Z評価 → D14Realで最終組合せとD/D'

3/4/7では共通係数4096を除算前に相殺した。これは研究評価器であり、非有限やunderflowを含む全入力で既存演算と同じとは主張しない。double係数をqfへcastするmethod5は精度回復策ではなく、失われた係数情報の寄与を見る診断。

## 探索途中101024根での結果

|method|scaled correction error p99（finiteのみ）|非有限数|全14根が診断閾値内のgeometry /7216|
|---|---:|---:|---:|
|1: 全double blocks|1.92e-3|20|4614|
|3: 最終組合せqf|2.22e-10|0|5272|
|4: blocksだけqf|1.93e-3|22|4984|
|5: coefficient丸めのみ残す|1.11e-10|0|5750|
|6: 全D14Real|7.80e-19|0|7216|
|7: 最終組合せD14Real|2.22e-10|0|5272|

診断閾値は誤差<1e-12*(1+abs(v))かつ誤差<1e-6*nearest separation。これは後からreferenceを見て評価する目安であり、誤差上界・本番受理gateではない。

最終候補点ではmethod7のp99=1.45e-9、5250 geometryが閾値内。全D14Realでも最終候補点は7194/7216。最終点のbinary64 exportと極端clusterでのqf referenceの有限精度に注意し、現行qf gateの不要性は主張しない。

## 解釈

重大なtail誤差は最終discriminantの組合せの相殺が大きな原因。そこだけD14Realへ上げればqfとほぼ同等の改善が出る。一方、係数丸めを残すmethod5でもp99=1e-10級が残るため、全体double化の証拠にはならない。低次Hornerだけを改善しても最後がdoubleなら失敗する。

次はmethod7の評価費用と、足りないrootだけ現行D14Realへ渡す候補を実測する。全rootを一括で余計に評価してしまうと利益が消えるので、準備・判定・最終検査の費用を含める。必要なら局所shiftしたC/G/Zを検討するが、shift準備のqf費用を隠さない。

このフェーズは精度診断のみ。whole速度改善は未検証。独立referenceによるintegrator検証はsolverへ組み込む時に行う。qf off-root式照合はPhase31同様2.10e-28以下。

## 再現

Phase31のprepareコマンドで/tmp/p31_input.tsvを生成。probeは引数省略で従来3method、追加引数8で本フェーズ。

```
c++ -O3 -DNDEBUG -std=gnu++17 -fext-numeric-literals -Isrc -march=native benchmarks/holonomic/bench_d14_factor_accuracy.cpp -o /tmp/p32_probe -lquadmath
taskset -c 0 /tmp/p32_probe /tmp/p31_input.tsv 8 > /tmp/p32_accuracy.tsv 2> /tmp/p32_identity.txt
python benchmarks/holonomic/summarize_d14_factor_accuracy.py summary --methods 8 --input /tmp/p32_accuracy.tsv --output /tmp/p32_summary.json
```

raw1616384行をaccuracy.tsv.gzに保存。summaryはNaN/Inf件数を別集計し、finite quantileと明記。
