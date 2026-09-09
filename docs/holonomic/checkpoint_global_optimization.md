# Global optimization checkpoint

基準は `dev/holonomic` の `420f58861fa2f67c1109ad3045f5ef80848fde68`。作業は
isolated holonomic buildの範囲に限定した。添付計画の原文は
[holonomic_global_optimization_plan_ja.md](holonomic_global_optimization_plan_ja.md)、
実測の全記録は
[checkpoint_global_optimization_benchmark.txt](../../evidence/holonomic/checkpoint_global_optimization_benchmark.txt)
に保存した。

実装した変更は次の三つである。

- `poly_roots.hpp` と `d14_structure.hpp` に、スケール付き複素除算、Aberth更新式
  `p/(p'-p sum)`、平方ノルム停止判定、warm seed時の不要な係数走査削減を追加した。
  `HOLO_D14_LEGACY_COMPLEX=1` で旧経路を再現できる。
- `radial_events.hpp` の `chart_p4` を、
  `p4=-(f-bR(R+a))(f+bR(R+a))` の二つの三次式へ分解した。
  `HOLO_CHART_P4_LEGACY=1` で旧六次Aberthを再現できる。
- `gm_connection.hpp` と `gm_taylor_transport.hpp` に、
  `Q_t^{-1} mod Q` のLU再利用、GM恒等式からの `S_k/C_k` 級数生成、
  `Z_{n+1}=(n+1)^{-1}\sum C_jZ_{n-j}` の真のTaylor輸送、5方向dualを追加した。
  これは既存の9点K-rule/packet経路とは独立している。

テストは、C++のisolated ctest 8/8、旧複素演算経路8/8、旧chart_p4経路8/8がPASSした。
Python `tests/holonomic` は269 passed、3 skippedであった。`test_chart_p4`では108ケースの
根集合差が最大 `6.548e-15`、p4残差が最大 `1.846e-16`。`test_gm_taylor`では、受理した
短セルのOrder-10 `__float128` Taylorと256段RK4の差が最大 `2.064e-09`、GM恒等式残差が
最大 `3.647e-09`で、反復根・次数低下もfail-closedになった。

実測上、chart_p4因数分解は適用可能34ケースで、三次式経路の中央値 `0.0008 ms`に対し
旧六次式経路は `0.7107 ms`だった。D14複素演算改良は、同一ケースのA/Bで
`radial_events`中央値約 `1.046x`、epoch中央値約 `1.066x`の小さな改善を示した。
D14 block-formの根集合・残差・tierも維持した。寄与がD14係数生成などに埋もれるため、
chart_p4のwhole epoch差は採用根拠にしていない。

GM Taylorは数学的には通過したが、品質ゲートを通った42セルで、point接続中央値
`0.0056 ms`、16段RK4 `0.3290 ms`に対して、サンプルを使わないOrder-10係数生成と輸送は
`12.9346 ms`だった。double Order-3は最大 `1.335e-03`で、V2と同精度の代替としては
棄却した。したがって、今回のTaylorは独立した検証可能な研究カーネルとして残し、
`epoch_jacobian`やwhole-epochの本番V2/V3ルータには切り替えていない。現行V2の比較値は
value-only中央値 `1.0521 ms`、value+5-Jac中央値 `1.0666 ms`（測定run、reps=10）で、
V2のV0比はそれぞれ `1.080x` と `1.230x`。V2のmu最大相対差は `2.044e-13`、statusは
V0/V1/V2とも `104/108`で変更なしである。

主な再現コマンドは以下の通り。

```text
cmake -S tests/holonomic_cpp -B build-holonomic-global \
  -DCMAKE_BUILD_TYPE=Release -DLCBININT_BUILD_HOLONOMIC_M7_PY=OFF
cmake --build build-holonomic-global -j8
ctest --test-dir build-holonomic-global --output-on-failure
taskset -c 0-7 ./build-holonomic-global/bench_chart_p4 /tmp/bench_cases.tsv 50
taskset -c 0-7 ./build-holonomic-global/bench_gm_taylor /tmp/bench_cases.tsv 10
taskset -c 0-7 ./build-holonomic-global/bench_d14_structure /tmp/bench_cases.tsv 10
taskset -c 0-7 ./build-holonomic-global/bench_holonomic_ode /tmp/bench_cases.tsv 10
PYTHONPATH=python pytest -q tests/holonomic
```

V0/V1/V2/V3のwhole epoch、value-only、微分込み、D14単体、GM本体の区別、旧経路A/B、
測定ホスト、精度・status・未解決リスクは証拠ファイルにまとめてある。
