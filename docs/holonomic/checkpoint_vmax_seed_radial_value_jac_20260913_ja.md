# Vmax seed-based radial boundary value/Jac checkpoint

## 実装した経路

今回の経路で旧inverse-rayを使うのは`cached_binary_image_seeds()`によるcomplete image-component seed探索だけである。magnificationの面積grid値は使わない。

```text
certified component seeds
  -> component connection proofによるseed縮約
  -> seedからR方向へsupportをmarch
  -> image-free notchをsolidify
  -> 各adaptive R nodeでquarticをcertified Sturm分類
  -> arcsは(m,v) root pair + vK K-rule
  -> adaptive Fejer radial integration
  -> 同じstateからanalytic 5Jac
```

value-onlyは`GradientPolicy::None`、value+5Jacは`ValueFirst`で測った。production routerは変更していない。

## 実測

trajectory corpus先頭200 trajectory、各lane 1,600行、1 repeat。D14版とのmatched比較。

| lane | value converged | status OK | p50 ms | p90 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| D14 value | 1600 | 1600 | 0.692 | 1.426 | 53.701 |
| seed-warm value | 1210 | 1210 | 1.608 | 23.447 | 82.255 |
| D14 value+5Jac | 1600 | 1584 | 0.989 | 2.429 | 54.291 |
| seed-warm value+5Jac | 1207 | 1191 | 1.525 | 24.162 | 82.084 |

26回のendpoint bisectionを4回のbracket生成へ削り、最終event精度を既存DD/qf `P=P_t=0` ladderへ任せた。20 trajectory A/Bではseed-warm p50がvalue `1.46 -> 1.06 ms`、value+5Jac `1.63 -> 1.27 ms`へ改善した。それでもD14版に勝たず、200 trajectoryではtailとcoverageが悪化した。

5Jacは別実装ではなく、`mapped_radius()`の同じquartic/root-pair/K-rule stateから解析的に得ている。両laneで`ToleranceMet`となった成分だけを比較すると、D14 topologyとの差は成分別最大`3.1e-8, 1.4e-7, 1.1e-7, 9.3e-8, 8.8e-8`だった。未認証成分をparity成功へ数えていない。

## 根本原因

complete seedから得られるradial supportは「そのRに像が存在するか」のunionであり、band内部のangular topologyを与えない。同じsupport bandの内部にも、

- `kFull <-> kArcs`
- 2 crossing <-> 4 crossing
- chart/branch event

が存在する。D14はこれらをすべてpanel cutとして列挙するが、seedからsupportの両端だけを探す実装では列挙できない。

現行adaptive research pathは各nodeでquartic Sturmを行うため通常nodeの値は正しい。しかしfold eventがpanel内部に残り、nodeがevent近傍へ来ると`TopologyUnresolved`または`EventLocationLimited`でfail closedする。200 trajectoryのseed-value stop内訳はConverged 1210、TopologyUnresolved 46、EventLocationLimited 340、Nonfinite 4だった。

これはtoleranceを緩めたりangular gridを増やして直す問題ではない。physical support endpointだけでなく、seed arcのroot pairをR方向へcontinuationし、`v=0`、root-pairの再結合、full/arcs遷移をeventとして発見するplannerが必要である。

## 採否

今回の経路は要求された積分方式を実装しており、grid magnificationへの切替は行っていない。ただしcoverageと速度がD14版を下回るため採用しない。既存D14 production pathとrouterは維持する。

次段の必要条件は、各certified component seedについて「存在union」をmarchするのではなく、そのseedを含むquartic arcの`(m,v)`を両R方向へcontinuationし、最初の`v=0`を局所`P=P_t=0`で認証すること。見つけた全eventをmergeした後、現在のadaptive value/Jacobian kernelへ渡す。このevent continuationができるまでD14を安全に除去できない。

Raw evidence: `evidence/holonomic/vmax_seed_radial_phase2_20260913/`。

