# D14研究branch集約 checkpoint 2026-09-25

## 方針

`dev/holonomic`をD14/holonomic研究の集約branchとして扱う。masterへは
入れない。旧`claude/holonomic-solver-m1`の研究・統合線を、現行
`dev/holonomic`の後続研究（D14 root scheduling、Rouché/event certificate、
projective fold、adaptive gradient）へmergeした。

merge baseは`207c22c`。旧線は`8a8d534`、集約先の基準は`71ef8bd`。
両者は別々に進化していたため、同じheaderの競合は現行`dev/holonomic`側を
採用した。旧線からは次を保持している。

- `FiniteSourceMethod::holonomic_binary`の明示preplanned route
- rolling `PreparedEpochGeometry`を使うL2 warm-D14経路
- isolated `_lcbinint_holonomic_m7` binding
- Cartesian run-fill研究実装と関連unit変更
- 旧M0--M7のdocs/evidence/test資産

通常のfinite-source routerが自動的にD14へ切り替わる変更ではない。D14 routeは
preplanned/explicit pathであり、既存のproduction defaultは維持する。

## 検証

通常treeのRelease build（Python/JAX FFIを含む）は成功した。

```bash
GSL_ROOT=/rogue1_8/nunota/local/gsl cmake -S . \
  -B /tmp/lcbinint-holonomic-consolidated-build \
  -DCMAKE_BUILD_TYPE=Release -DLCBININT_BUILD_PYTHON=ON \
  -DLCBININT_BUILD_TESTS=ON -DLCBININT_ENABLE_OPENMP=OFF \
  -DLCBININT_ENABLE_JAX_FFI=ON
cmake --build /tmp/lcbinint-holonomic-consolidated-build -j2
ctest --test-dir /tmp/lcbinint-holonomic-consolidated-build --output-on-failure
```

結果: `unit_core` **1/1 passed**。

isolated holonomic CMakeも全targetをbuildし、次を実行した。

```bash
cmake -S tests/holonomic_cpp \
  -B /tmp/lcbinint-holonomic-consolidated-iso-build \
  -DCMAKE_BUILD_TYPE=Release -DLCBININT_BUILD_HOLONOMIC_M7_PY=OFF
cmake --build /tmp/lcbinint-holonomic-consolidated-iso-build -j2
ctest --test-dir /tmp/lcbinint-holonomic-consolidated-iso-build \
  --output-on-failure
```

結果: **22/22 passed**。fixture pathを引数から受け取る小さなtest修正も含めた。

## 境界

この集約は研究資産を一つのbranchへまとめる作業であり、masterへのproduction
昇格ではない。MOASARC/ecosystemへの接続は別repoの明示batch ABIとして移植し、
このbranch全体をecosystemへmergeしない。
