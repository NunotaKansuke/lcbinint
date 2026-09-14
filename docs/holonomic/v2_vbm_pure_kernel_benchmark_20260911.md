# V2 と VBM の純有限光源 kernel 比較

> これはこの結果を記録した report であり、再実行時に branch を checkout する指示ではない。新しい実装を測る場合は、現在渡されている checkout の kernel をそのまま compile し、[指定 kernel 用 runbook](../../tests/diagnostics/recal2026/HANDOFF_vbm_filled_qrho_kernel_runbook_20260911.md) の provenance と計時条件を使う。

実行日: 2026-09-11  
V2基準: `dev/holonomic` `3f35991a1c22f11de60f26a4c88c3628e5aa5556`  
対象: `controlled_pure_kernel` の既存 1804 geometry、各4 epoch

q–rho の coverage extension（base 160 configuration + extension 96 configuration、うち51は空 bin targeted）を含む input を使っている。q–rho 図は追加時と同じ protocol-clamped edge `q=geomspace(1e-4,1,13)`、`rho=geomspace(3e-5,1,13)` の12×12 binで作り、`min_cell_population=8` では uniform/linear とも144/144 cellが埋まる。`rho<1e-4` の行は下限 `3e-5` 以上なので有効であり、図から落とさない。

今回の比較は full epoch の比較ではない。VBM側は既存の paper benchmark にある warmed direct `BinaryMag` / `BinaryMagDark` の計時をそのまま再利用し、V2側だけを同じ入力で再計算した。V2では各行について `LensParams`、`PrimaryFrame`、`classify_cells`（D14・radial event・topologyを含む）を先に構築し、1回のwarm-up後に次だけを3回計時して中央値を採用した。

```text
flux_value_integrate(n_r=64, u, pf, topo)
```

したがってV2の計時には、radial nodeごとのquartic/root-pair処理、F0/F_half、V2 K-ruleと必要なangular rescueは含む。一方、`LensParams`生成、D14/topology、`epoch_value`、最終blend、入力・出力I/Oは含まない。`n_r=64`固定なので、V2の時間はVBMの`RelTol`ラベルによって変わらない。この条件で、1e-3/1e-4はVBM側の速度条件として比較した。

座標は、corpusのVBM直接呼び出し `BinaryMag(..., -x, y, ...)` に対応する検証済みのV2写像を使った。

```cpp
LensParams p{time, y, rho, 1.0 / q, s, true};
```

入力referenceとの照合は、既存VBM `RelTol=1e-6` referenceに対して行った。speed/error図では非OK V2点を捨てず、非OKはstatus集計とハッチで残している。これは非OKの数値をproduction成功として扱わないためである。

## 結果

`R = t_VBM / t_V2` とし、`R > 1`がV2の方が速いことを表す。

| profile | VBM target | V2 p50 ms | V2 p95 ms | VBM p50 ms | VBM p95 ms | R p50（全有限値） | R p50（V2=OK） | V2 OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 1e-3 | 0.2691 | 0.7299 | 0.06137 | 0.1996 | 0.272 | 0.281 | 3366/3608 (93.3%) |
| uniform | 1e-4 | 0.2691 | 0.7299 | 0.1075 | 0.3515 | 0.473 | 0.491 | 3366/3608 (93.3%) |
| linear LD (`c=0.5`) | 1e-3 | 0.3784 | 1.3908 | 0.3145 | 3.2612 | 0.880 | 0.918 | 3366/3608 (93.3%) |
| linear LD (`c=0.5`) | 1e-4 | 0.3784 | 1.3908 | 1.3054 | 14.3062 | 3.566 | 3.756 | 3366/3608 (93.3%) |

従ってこの純kernel境界では、uniformは両targetともVBMが速く、linear LDは1e-3でほぼ同等、1e-4でV2が約3.6倍速い。これはD14/topology込みのwhole-epoch結論ではない。

V2のVBM `RelTol=1e-6` referenceに対する相対差は次の通りだった。

| profile | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| uniform | `3.01e-5` | `2.11e-4` | `1.34e-3` | `2.63e-3` |
| linear LD | `2.90e-5` | `2.03e-4` | `1.35e-3` | `2.50e-3` |

非OKはuniform/linearとも242/3608点で、statusは全て`GRADIENT_UNRELIABLE`だった。今回の図は値の差だけでなく、このstatus欠落もハッチで示している。

## 作成した図とraw

4条件の速度・相対差図は次に保存した。

- [uniform, VBM 1e-3 PNG](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/protocol_edges/figures/q_rho_uniform_1e-3_v2_current_vbm1e-6.png)
- [uniform, VBM 1e-4 PNG](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/protocol_edges/figures/q_rho_uniform_1e-4_v2_current_vbm1e-6.png)
- [linear LD, VBM 1e-3 PNG](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/protocol_edges/figures/q_rho_linear_1e-3_v2_current_vbm1e-6.png)
- [linear LD, VBM 1e-4 PNG](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/protocol_edges/figures/q_rho_linear_1e-4_v2_current_vbm1e-6.png)
- [protocol-edge replot report](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/protocol_edges/REPORT.md)
- [machine-readable summary](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/summary.json)
- [V2 raw output](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/v2_results.tsv)
- [joined V2/VBM table](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/joined_v2_vbm.tsv)
- [figure-generation script](../../evidence/holonomic/v2_vbm_pure_kernel_20260911/make_figures.py)

VBM speedは既存 `/tmp/lcbinint_pure_kernel_confirm_full_20260909/merged_final/results.json`（SHA-256 `7233f909e2539dd5331acc5d412efc94a04238f5c9f3dbc7c031c88a5ade6027`）を再利用した。新規計算はV2の7216行だけである。

## 再現コマンド

```bash
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 \
  -I/rogue1_8/nunota/lcbinint/src -march=native -funroll-loops \
  -ffp-contract=fast -fno-math-errno \
  evidence/holonomic/v2_vbm_pure_kernel_20260911/v2_pure_kernel_runner.cpp \
  -o /tmp/v2_pure_kernel_runner -lquadmath

/tmp/v2_pure_kernel_runner \
  evidence/holonomic/v2_vbm_pure_kernel_20260911/input_snapshot.tsv \
  /tmp/v2_pure_kernel_results.tsv
```

図の再生成に使ったscript、実行時の入力・runner・summaryは同じevidence directoryに保存している。V2 runnerのSHA-256は `8c25119082065a1f69e5465deefb101edd3c8fb5a650d011ce8918b6f7efbf05`、入力snapshotは `6646492658ed416e75cb8c6c821ee853f79cdf7761fc41441186aa67b81fa05b`、V2 rawは `5f4c1da396414428496a7c24d0754bdc8a58b2db060e6390d7d62006b2c0aaaa` である。

この結果からは純kernelでの条件別速度を確認できる。D14/topologyを含むproduction whole epochやcold/warmの比較は、この図の対象外である。
