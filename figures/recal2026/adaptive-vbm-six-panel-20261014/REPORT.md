# Ordinary adaptive kernel vs saved VBM: six-panel comparison

The four six-panel figures compare the ordinary production build at relative targets 1e-3 and 1e-4, starting from nbin16, against the saved direct VBM kernel timings and independent RelTol=1e-6 values. Each panel is regenerated from the 7,216-row input and 14,432 native results in this directory.

The current side uses one relative-tolerance request per timed evaluation. Its initial grid and relative normalization are handled in that request. A separate nbin16 scale call is not timed or added. Center image roots are computed before timing; caustic and seed geometry is built in an untimed warm call; three measured native calls provide each median. VBM times are retained from the previously measured warmed direct BinaryMag/BinaryMagDark calls and were not retimed here. The current C++ timer excludes Python and model setup; any binding-call overhead in the saved VBM times is retained.

The lower panels show actual relative difference from the saved VBM RelTol=1e-6 magnification, summarized as cell p95. They are not a mathematical error certificate. Rejected, inaccurate, and exceptional rows remain in joined.tsv and summary counts.

| profile | target | finite / rows | estimated accepted | actual pass | false accepts | median VBM / adaptive | p95 relative error | median adaptive ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|uniform|0.001|3456/3608|3259|3345|81|0.197|0.0007527|0.2891|
|uniform|0.0001|3456/3608|3241|3270|155|0.2784|0.0001054|0.3571|
|linear|0.001|3448/3608|3256|3366|54|0.6261|0.000581|0.3209|
|linear|0.0001|3448/3608|3241|3302|118|2.365|9.119e-05|0.4117|

The input cohort is a q-rho coverage corpus rather than an iid speed sample. The error estimator remains an empirical estimate, not a general mathematical guarantee.

Reproduce the native build with `cmake -S . -B build-adaptive-production -DCMAKE_BUILD_TYPE=Release -DLCBININT_BOUNDARY_SUPPORT_EXPERIMENTS=OFF -DLCBININT_BUILD_TESTS=OFF -DLCBININT_BUILD_PYTHON=ON -DLCBININT_ENABLE_JAX_FFI=OFF -DCMAKE_PREFIX_PATH=/home/nunota/.miniconda3/envs/myenv`, then `cmake --build build-adaptive-production -j2` and `python tests/diagnostics/recal2026/make_adaptive_vbm_six_panel_20261014.py`. The script resumes complete 32-input shards, reruns incomplete shards, and renders from the committed inputs and native measurements.

The input q-rho design covers 144/144 cells with at least 8 input rows per cell. The rho axis now keeps its requested 3e-5 lower bound. The previous edge logic floored the observed minimum (~3.14e-5) to 1e-5, adding an unintended low-rho row and shifting the bins.
After excluding non-finite measurements, 142/144 q-rho cells meet the 8-row display threshold. The remaining gray q-rho cells are: q=[0.0001,0.000215), rho=[3e-05,7.15e-05): 4/8 finite rows; q=[0.000215,0.000464), rho=[3e-05,7.15e-05): 4/8 finite rows. Gray means masked for insufficient finite data, never zero.
The A_fs projections are not a fully sampled independent grid because A_fs is derived from the epoch geometry; the speed maps support 65/144 A_fs-rho cells and 77/144 A_fs-q cells.
q-rho cells whose p95 cross-code difference exceeds the requested target: uniform 0.001: 22/142 cells; uniform 0.0001: 49/142 cells; linear 0.001: 15/142 cells; linear 0.0001: 42/142 cells. These cell counts include both accepted and rejected rows; they are distinct from the individual false-accept counts in the table.
