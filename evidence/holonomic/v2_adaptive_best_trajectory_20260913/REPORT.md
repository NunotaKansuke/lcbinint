# V2 adaptive best configuration trajectory benchmark（2026-09-13）

`dev/holonomic @ 08c03bd`に対し、検証済みのD14 `LOCAL_PAIRS` と
`ACTIVE_PRESEARCH (tol=1e-12, patience=2)`を既定ONにした作業treeを測定した。
値のみ、`Tol=1e-16`、`RelTol=1e-3 / 1e-4`、1,804 trajectory × 4 epoch ×
uniform/linearの14,432 rowsである。各rowは3回の中央値。`LensParams`生成はtimer外、
full-cold/full-warmではD14/topologyをtimer内に含む。

## Whole-epoch timing

値はp50 / p90 / p95 / p99 / max [ms]。

| profile | RelTol | lane | timing [ms] |
|---|---:|---|---:|
| uniform | 1e-3 | full-cold | 0.5205 / 0.8324 / 0.9849 / 3.4160 / 56.7497 |
| uniform | 1e-3 | full-warm | 0.4629 / 0.7774 / 0.9216 / 1.3526 / 56.9383 |
| uniform | 1e-3 | radial-only | 0.1327 / 0.2374 / 0.2909 / 0.4144 / 0.7195 |
| uniform | 1e-4 | full-cold | 0.5647 / 0.9177 / 1.1230 / 3.4446 / 57.1708 |
| uniform | 1e-4 | full-warm | 0.5272 / 0.8855 / 1.0686 / 2.2642 / 56.9943 |
| uniform | 1e-4 | radial-only | 0.1717 / 0.3182 / 0.3972 / 0.6746 / 2.3079 |
| linear | 1e-3 | full-cold | 0.5780 / 0.9161 / 1.0727 / 3.5536 / 57.6022 |
| linear | 1e-3 | full-warm | 0.5295 / 0.8565 / 1.0228 / 1.5413 / 56.9970 |
| linear | 1e-3 | radial-only | 0.1791 / 0.3465 / 0.4055 / 0.5446 / 0.9238 |
| linear | 1e-4 | full-cold | 0.6325 / 1.0382 / 1.2843 / 3.5748 / 57.3830 |
| linear | 1e-4 | full-warm | 0.6030 / 1.0166 / 1.2168 / 2.2423 / 57.0594 |
| linear | 1e-4 | radial-only | 0.2223 / 0.4687 / 0.5787 / 0.9048 / 1.7670 |

旧trajectory artifactに対し、linear RelTol=1e-3 full-warm p50は
0.6707→0.5295 ms、uniformは0.6014→0.4629 ms。RelTol=1e-4はlinear
0.7486→0.6030 ms、uniform 0.6568→0.5272 msとなった。

## VBM comparison and accuracy

既存VBM selected timingとのrow-wise比 `R=t_VBM/t_V2` の中央値は、full-warmで
uniform 1e-3/1e-4が0.146/0.226、linear 1e-3/1e-4が0.659/2.465。
したがってLDあり・1e-4ではV2が中央値で約2.46倍速い一方、1e-3ではVBMが速い。
uniformではVBMが大幅に速い。これは図上のbin分布も併せて判断する。

VBM RelTol=1e-6保存referenceに対する相対差は全lane共通で、p50
`4.22e-7`、p90 `4.01e-6`、p95 `6.66e-6`、p99 `1.41e-5`、最大
`2.71e-4`。全lane 14,432/14,432 `Converged`、status `OK`。
以前のsafe-candidate rawとのmu最大差0、status/convergence mismatch 0。

## Figure contract

従来と同じ2×3 layout、bin、color boundsを使用した。上段はVBM timing / V2
whole-epoch timing、下段はVBM 1e-6 referenceに対するcell p95相対差。
full-cold/full-warm、uniform/linear、RelTol=1e-3/1e-4の8図をPNG/PDFで保存した。

`full-warm`はL1 topology reuseを使わず、各trajectory初回3,608行がcold build、
後続10,824行がcertified previous D14 rootsをseedにしたL2再計算である。
