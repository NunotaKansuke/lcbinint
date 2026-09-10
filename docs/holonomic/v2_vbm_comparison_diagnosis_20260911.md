# V2 と VBM 比較の診断レポート

日付: 2026-09-11
基準: `dev/holonomic` `cb45434cfd416394575944933686da0d69b5e534`
対象: isolated holonomic の診断とレポートのみ

## 判定

今回の「V2がmain側のVBMより遅く、精度も悪い」という図は、そのままの勝敗判定には使えない。速度の測定境界が違い、診断用V2 runnerのパラメータ写像も誤っていたためである。

一方、写像を直しても、現行V2のcell topology判定には独立の問題が見つかった。細いphysical image arcがboundary quarticでは検出されているのに、固定3072点の角度gridが見落とし、そのgrid結果でquartic結果を`empty`に上書きしていた。これは単なるベンチ条件の問題ではなく、外側imageを落として値を約1ずつ小さくする実装上の原因である。

根本修正はまだ採用していない。簡単な修正候補は代表例を直したが、既存M7のwide-planet parityを18件壊したため全てrevertした。従って、今回のcommitではproduction sourceとrouterを変更せず、再現条件と未解決原因を記録する。

## 比較条件の不一致

既存VBM測定は `tests/diagnostics/recal2026/bench_grid_vs_vbm_pure_kernel.py` の `pure_kernel_cache_warm_direct_xy` であり、事前計画済みのgeometryを使ったcache-warmな1 epoch kernelの時間である。D14、topology、radial cell計画、root search、F0 geometryは含まれない。

今回のV2診断runnerは `epoch_value(p, u, 64, false)` 全体を測っており、D14からcell classification、radial geometry、root-pair、F0、K-ruleまで含む。`LensParams`構築、入力parse、出力formatはtimerの外であり、LensParams初期化由来のoverheadがV2の測定を悪化させたわけではない。

したがって、以前の図のV2/VBM時間比は、VBM pure kernel対V2 whole epochの比である。V2がVBMより速い、または遅いという公平な結論にはならない。公平な比較には、同一node・同一root-pair・同一cell planを共有するLD evaluator比較と、D14込みのwhole-epoch比較を別の表にする必要がある。

## パラメータ写像の誤り

最初の診断runnerは次を使っていた。

```cpp
const LensParams p{-time, y, rho, q, s, true};
```

このVBM corpusでは、内部V2 frameへの対応は次である。

```cpp
const LensParams p{time, y, rho, 1.0 / q, s, true};
```

つまり source x の符号と、VBM raw conventionからbarycentric V2 conventionへの質量比の逆数を同時に合わせる必要があった。元runnerはこの二つを合わせていなかったため、zero outputや不正確なstatusを含み、精度図を壊していた。

代表点は次の通りである。

```text
s    = 0.2310772011499552
q    = 0.5633224083213993
rho  = 0.0001726634539771850
y    = -4.047226593265940
time = 1.1409565850100225
```

VBMを `RelTol=1e-6`, `Tol=1e-16`, `a1=0.5`, `a2=0`, `LDlinear` に設定した直接計算では、概ね次を得る。

| quantity | value |
|---|---:|
| uniform `BinaryMag` | 6.8139830013407305 |
| linear `BinaryMagDark` | 7.037894384544265 |

正しい写像を現行V2 stock pathへ渡したprobeでは、uniform約5.81186、linear約6.03582となった。この差は`n_r=16..1024`でも消えず、radial resolutionやGM/PF6 transportの差ではない。

## 見つかったV2 topology bug

現行 `src/lcbinint/magnification/holonomic/cells.hpp` は、各cellについてboundary quarticのtopologyを求め、512点gridと比較する。食い違うとM7互換の3072点gridを3つのprobe radiusで実行するが、最終的にはそのgrid結果を採用する。

代表点の外側imageでは、D14が次の2つのphysical-real eventを返している。

```text
R_lo = 4.4727647394129066
R_hi = 4.4730933909894857
R_img = 4.4729290649084206
```

このcellの幅は約`3.2865e-4`である。mid radiusのquarticは正しい2 crossingを返し、arc intervalはおよそ

```text
[5.0189217002724167, 5.0190030316414118]
```

で、幅は約`8.1e-5 rad`しかない。一方、3072点角度gridのspacingは約`2.045e-3 rad`なので、gridはarcを一度もサンプルせず`empty`を返す。結果として、quarticが検出したarcがgridのemptyで置き換えられ、outer imageの寄与が積分から消える。

同じ構造はwide-planet fixtureにも現れ、別の薄いarcが

```text
R = 2.5013971414257967
theta = [6.2830817045081035, 6.2831295188810881]
width  = 4.7814e-05 rad
```

に存在する。従って「quartic結果を常に優先する」という局所修正だけでは、legacy parityを安全に証明できない。

## 試した修正とrevert理由

一時的に、次の条件を満たす場合だけquarticのarc intervalをgrid結果より優先する候補を試した。

- 3つのprobe radiusでquartic signatureが一様
- quarticのreal root/arc intervalが非空
- 独立gridとの不一致が細いarcで説明できる

代表点はlinear value約7.03841（`n_r=64`）、`n_r=1024`で約7.03792まで改善し、VBM値に戻った。しかし既存M7の`test_holonomic_m7`は、候補適用前のHEADで`10397 checks, 0 failures`だったものが、候補適用時にはwide-planetの18件で失敗した。

このため候補は採用せず、作業treeから完全にrevertした。重いangular rescueへ無条件に逃がす修正や、threshold緩和で成功率だけを上げる修正も入れていない。必要なのは、quartic root set、符号区間、chart/branch、D14 event境界を合わせた「arcが存在すること」の独立certificateであり、certificateを作れない場合にgridの`empty`を成功扱いしない根本修正である。

## D14、cold/warmの扱い

D14を比較から除外する意図はない。既存Phase 8の本番V2 profileでは、cold/warmを分け、D14 construction/search/polishとtopologyを`epoch_value`経路の固定費として測っている。代表的なwhole-epoch baselineは次の通りである（別のPhase 8 supported set、`n_r=64`）。

| lane | p50 ms | p95 ms | status |
|---|---:|---:|---:|
| cold value | 0.880494 | 1.684426 | 104/108 |
| cold value+5Jac | 1.038950 | 1.944592 | 104/108 |
| warm value | 0.800960 | 1.474118 | 728/756 |
| warm value+5Jac | 0.893731 | 1.806424 | 728/756 |

ただし、今回のVBM corpusで正しい写像とtopology修正を確立した後のD14込みcold/warm再測定は未完了である。従ってこのreportから、現行V2のVBMに対するwhole-epoch勝敗を断定しない。

## 再現手順

既存のisolated test buildとM7 parityは次で確認できる。

```sh
cmake -S tests/holonomic_cpp -B build-holonomic-m7 -DCMAKE_BUILD_TYPE=Release
cmake --build build-holonomic-m7 -j2
ctest --test-dir build-holonomic-m7 --output-on-failure
```

VBM側のLD reference設定は、`bench_grid_vs_vbm_dark.py`の設定に合わせる。

```python
vbm.RelTol = 1e-6
vbm.Tol = 1e-16
vbm.a1 = 0.5
vbm.a2 = 0.0
vbm.SetLDprofile(vbm.LDlinear)
```

V2の一時diagnostic runnerはHEAD sourceに対して次の条件でコンパイルした。

```sh
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 \
  -I/rogue1_8/nunota/lcbinint/src -march=native -funroll-loops \
  -ffp-contract=fast -fno-math-errno /tmp/v2_correct_runner.cpp \
  -o /tmp/v2_correct_runner -lquadmath
/tmp/v2_correct_runner /tmp/v2_current_input_unique.tsv \
  /tmp/v2_correct_fixed.tsv
```

このrunnerの測定対象は`epoch_value(p,u,64,false)`で、D14/topologyを含み、`LensParams`構築とI/Oを含まない。`/tmp/v2_correct_fixed.tsv`は候補patch適用時の診断出力も含むため、production benchmarkの採用結果として扱わない。

## 検証結果と残作業

- 現在のsource差分はない。既存未追跡`.claude/`には触れていない。
- HEADのM7 parityは`10397 checks, 0 failures`。
- `test_gm_coverage` と `test_holonomic_transport` はPASS。
- V2/VBMの元の4-way図は、whole epoch対pure kernelかつ誤った写像を含むため、性能・精度の結論として無効。
- 正しい写像で確認できた差は、まずcell topologyの細いarc見落としを解決しない限り評価できない。
- V2 production router、D14既定経路、PF6/GM経路は変更していない。

次の実装では、固定gridの解像度を増やすだけではなく、boundary quarticから得た候補arcを符号判定・root completeness・event境界・chart/branch consistencyで認証し、legacy parityを全corpusで確認する必要がある。その後に初めて、D14込みcold/warm whole epochとshared-LD-kernelの二層比較をやり直す。
