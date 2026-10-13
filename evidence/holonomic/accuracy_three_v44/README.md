# Three stored-reference discrepancies, v44

No solver changes. Diagnosis on experiment/seeded-boundary-events,
HEAD f3547723b79828d9c501182746028e99f7bad116 plus preserved v43 dirty source,
host rogue1, one thread per run. Reference is the existing VBM RelTol=1e-6
column. v43 output is the baseline. Current VBMicrolensing Python package 5.5.

Saved-reference discrepancies: case10 LD .0182422%, case10 uniform .0270597%,
case115 LD .0217776%; requested relative tolerance .01%. All high magnification.

Holonomic rtol 1e-4 -> 1e-7 changes only 5.02e-9, 1.00e-7, 7.32e-9 relative,
respectively. The 1e-8 case115 attempt correctly reports InnerAccuracyLimited;
do not call that converged. Unseeded classify_cells + epoch_adaptive at 1e-6
returns the same values (cold.log). This excludes a seeded handoff or new LD
product rule as the cause, but is not a third independent integration oracle.

VBM at rtol 1e-4..1e-8 is nonmonotonic for case10. Even the 1e-6 fresh values
differ from saved references. case115 appears stable at 1e-6..1e-8, but violates
exact binary-lens reflection symmetry: for the same real-axis lenses, y and -y
must give the same magnification. At rtol=1e-8, a1=.5, a2=0, linear profile,
VBM gives 1581.206716616568 versus 1580.8659820714684. That .02155% asymmetry is
comparable to the original discrepancy. Reflected VBM is within .000265% of
holonomic 1580.86179857, comfortably inside the .01% requested budget.

Conclusion: these rows demonstrate unreliable reference values, not evidence
of a new solver regression. No large-error or failure behavior is observed.
Self-convergence and alternate topology discovery support the solver values,
but do not prove absolute correctness or coverage. Keep the historical audit
rows visible; no tolerance relaxation or production promotion.

Reproduction from this checkout:

    bash evidence/holonomic/accuracy_three_v44/build.sh
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-seeded-events/probe_accuracy_v44 evidence/holonomic/accuracy_three_v44/input.tsv evidence/holonomic/accuracy_three_v44/holonomic.tsv 1
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/nunota/.miniconda3/envs/myenv/bin/python evidence/holonomic/accuracy_three_v44/vbm_probe.py

The symmetry call uses the last input row with BinaryMagDark(s,q,-time,+/-y,rho,
1e-12), fresh VBM objects, RelTol=1e-8, a1=.5, a2=0, LDlinear.
For cold_probe.cpp compile with -std=c++17 -O3 -march=native
-fext-numeric-literals -Isrc -lquadmath and TMPDIR=$PWD/build-seeded-events.
