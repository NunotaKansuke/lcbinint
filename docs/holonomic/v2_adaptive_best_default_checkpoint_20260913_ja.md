# V2 adaptive best configuration default checkpoint（2026-09-13）

D14のsafe candidateであるpair-local mixed precisionとroot-wise active presearchを
既定ONにした。`HOLO_D14_LOCAL_PAIRS=0`、`HOLO_D14_ACTIVE_PRESEARCH=0`で旧経路へ
個別に戻せる。active tolerance `1e-12`、patience `2`、後段D14Real/qf residual、
completeness、topology certificateは変更していない。

14,432-row trajectoryで以前の明示的safe-candidate runとmu最大差0、status mismatch 0、
convergence mismatch 0。全CTest 19/19 pass。full-cold/full-warm/radial-onlyの
RelTol=1e-3/1e-4結果と2×3図は
`evidence/holonomic/v2_adaptive_best_trajectory_20260913/REPORT.md`に記録した。

代表的なfull-warm p50はuniformで0.4629/0.5272 ms、linearで
0.5295/0.6030 ms（RelTol=1e-3/1e-4）。全14,432行がConverged/OK。
