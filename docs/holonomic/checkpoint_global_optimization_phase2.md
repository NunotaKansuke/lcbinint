# Global optimization phase 2 checkpoint

基準は `dev/holonomic` の `7fd15113bcee5c5d5d100c781e02a3e94663d228`。
作業範囲は `src/lcbinint/magnification/holonomic`、
`tests/holonomic_cpp`、`docs/holonomic`、`evidence/holonomic` に限定した。
添付計画の原文は [holonomic_global_optimization_plan_ja.md](holonomic_global_optimization_plan_ja.md)、
実測の生データと集計は
[checkpoint_global_optimization_phase2.txt](../../evidence/holonomic/checkpoint_global_optimization_phase2.txt)
に記録した。

`chart_p4` は、`b^2 == 0.0` の場合だけ一因子、その他は二つの三次因子を解くようにした。
因子化ヘルパー内の近接根統合は削除し、既存の `radial_events` のイベント統合へ残した。
108ケース中適用可能34ケースで、根の相対差は最大 `6.548e-15`、p4残差は最大
`1.846e-16`。通常経路と `HOLO_CHART_P4_LEGACY=1` のテストはともにPASSした。
最新の50反復ベンチでは適用可能34ケースの因数分解が中央値 `0.0008 ms`、
旧generic経路が `0.7124 ms`、比率は `843.066x`だった。

GM係数生成は、固定次数・固定8次多項式に合わせて次を専用化した。

- `Q=P(1+t^2)((R-a)^2+(R+a)^2t^2)` の偶多項式積を直接展開。
- `S_{k+1}=tS_k mod Q` のシフト再利用、先頭係数逆数の再利用、LUの再利用。
- 系列積の一時オブジェクトを避ける fused accumulate/subtract と固定容量16の多項式。
- 値専用laneでは不要なdual演算・広いscratch・ゼロ係数の汎用処理を避けた。
- dual laneの演算順序は既存の有限差分parityを保つため維持した。

同じ `bench_gm_taylor` で、Order-10 `__float128` の全体は前段の約12.9 msから
最新測定で中央値 `3.8128 ms`へ下がった。内訳は係数生成 `3.6826 ms`、輸送再帰
`0.0988 ms`であり、残る固定費は汎用poly演算よりlibquadmathの高精度scalar演算が支配する。
`perf`でもOrder-10 qf接続生成のchildrenは約75%、その内部は `__multf3` 35%、
`__subtf3` 20%、`__addtf3` 17%だった。point接続は `0.0027 ms`、16段RK4は
`0.1497 ms`。double/DDの精度gateは、選択42セルでdouble Order-8が0/42、DD Order-8が
42/42だが最大誤差 `6.984e-7`、DD Order-10は38/42で最大 `6.538e-24`だった。

6D residue-freeの実験も追加した。`psi=(eta0,eta1,eta2,eta4-b1 eta3,eta5-b2 eta3,eta6-b3 eta3)`
の閉包は点で `1.337e-18`、DD Order-8の42セルで全て受理された。6D投影は `0.0152 ms`、
輸送再帰は `0.0050 ms`だが、これは7D eta接続生成後の追加費用であり、直接のflux-priority
companion basisではない。物理seedを使ったqf Order-10の短セル検証では、eta observableと
6D observableの差はそれぞれ最大約 `4.8e-15`、相互差は `7.3e-16`だった。既存研究で
有望だった `s=max(H,r_a/4)` のflux-priority scalingは、Python feasibilityの条件数
`30..900`という結果を記録したが、scalar companion ODE自体は今回のC++本線へまだ実装していない。

physical period seedを実際の境界積分から作り、`F0/F_half`、`mu`まで通すdiagnosticを追加した。
GM laneには9点K-rule/Chebyshev packetを呼び出していない。V3 packetとの比較は既存の
`bench_holonomic_ode`へ分離した。n_r=64でV2と同じ半径求積点数に揃えた108ケースの結果は、
DD Order-8で次の通りだった。

- V2 status `104/108`、GM status `52/108`。value-onlyはGM `52/54`、value+5Jacは
  `0/54`、全ノードtrue transportは `0/54`。
- GM/V2のμ相対誤差は median/p90/max `2.801e-15 / 1.280e-12 / 1.944e-08`。
  GMと毎ノードphysical re-seedとの差は最大 `5.007e-12`。
- 独立angular積分との比較は direct/reference 最大 `5.368e-09`、reference/V0最大
  `1.904e-08`。物理seed経路の誤差ではなく、V2 incumbentとの積分・境界差を分離できた。
- whole epoch中央値はGM `1.7218 ms`、V2 `1.0509 ms`で、GMはV2の `0.610x`。
  value-onlyはGM/V2 `1.0493/0.8652 ms`。GM value+5Jacは解析微分未実装のため、
  完全なGM value laneを5方向中心差分した診断で中央値 `26.4887 ms`、同じGM valueの
  `2.8076 ms`に対して `9.435x`。その勾配相対差は median/p90/max
  `4.880e-03 / 1.726e-01 / 3.874e-01`。
- 接続試行/失敗 `214/179`、transport試行/失敗 `2496/2131`。接続 `41.871 ms`、
  transport `27.334 ms`、node arc `26.883 ms`、topology/D14 `96.709 ms`、
  physical direct fallback `45.215 ms`が108ケース合計の計測値だった。

precision ladderのwhole-epoch診断でも、double8は接続 `218/218`失敗、qf10は
`214/188`失敗だった。qf10は局所的にtransport受理数を増やすが、whole epochの
value+5Jac statusを `0/54`から改善せず、n_r=16の中央値も `2.0018 ms`でV2を上回らない。
したがって、現段階ではtrue GMをV2/V3本番ルータへ切り替えない。係数生成の固定費削減、
physical seedと再seed基準の分離、6D閉包の検証までは成立したが、chart_p4近傍を含む
全セルで使えるflux-priority companion basis、または同等の正則化されたscalar GM transportが
次の必要条件である。

主な再現コマンドは以下の通り。

```text
cmake -S tests/holonomic_cpp -B build-holonomic-global \
  -DCMAKE_BUILD_TYPE=Release -DLCBININT_BUILD_HOLONOMIC_M7_PY=OFF
cmake --build build-holonomic-global -j8
ctest --test-dir build-holonomic-global --output-on-failure
HOLO_CHART_P4_LEGACY=1 ctest --test-dir build-holonomic-global --output-on-failure
./build-holonomic-global/bench_chart_p4 /tmp/bench_cases.tsv 50
./build-holonomic-global/bench_gm_taylor /tmp/bench_cases.tsv 3
./build-holonomic-global/bench_gm_e2e /tmp/bench_cases.tsv 3 108 64 0 dd8
./build-holonomic-global/bench_gm_e2e /tmp/bench_cases.tsv 1 108 64 256 dd8
./build-holonomic-global/bench_gm_e2e /tmp/bench_cases.tsv 1 108 64 0 double8
./build-holonomic-global/bench_gm_e2e /tmp/bench_cases.tsv 1 108 16 0 qf10
```
