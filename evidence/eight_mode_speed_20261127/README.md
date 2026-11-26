# Algebraic-boundary public API timing

Run `bench.py BUILD OUTPUT [CASE_OFFSET] [--vary-lens]` from the parent lcbinint checkout, using the Python interpreter that built the extension. The script verifies the imported extension path. Binary inputs use center-of-mass coordinates and the reciprocal mass ratio of the VBM fixture. Triple inputs use VBM coordinates.

The comparison used the experiment worktree at 7038d6c2e7, GCC 11 Release, and one pinned Xeon Gold 6530 CPU with OMP and OpenBLAS thread counts set to one. Python extension SHA-256 before and after the edits:

```
b505651e2cb211bf140c958949724d670eb34d1b2140f56ed1e3883075263b6b
1880947a8cdf55d245182c9d02f6ce69027e7c3e9972c36ed441ef0d093b84eb
```

Each call evaluates two epochs, absolute tolerance 0, relative tolerance 1e-4. Each trajectory has four calls with different impact parameters. The trajectory workload combines case offsets 0 and 1. The second workload uses offset 0 and also perturbs separation and mass ratio between calls. Both include the first call. Timings cover the public magnification call, including the requested derivative and result construction. Curve construction and serializing the results are outside the timing boundary.

In `timing.tsv`, times are the median of three total calling-thread CPU times divided by the number of epochs. Columns count the integrated epochs, accepted value statuses, and usable gradient flags in each workload. Raw paired runs are in the parent checkout under `build-accuracy-20261127/experiment-fast/{old,final}-*jsonl`. Every repeat is compared on magnification, all derivatives, status, gradient validity, value convergence, seed support, coverage, discovery and integration route.

Independent value comparisons in `vbm.tsv` use the first two epochs per geometry and profile in cohort 0, 64 distinct epoch/profile inputs, each checked on value-only and gradient-enabled routes. VBM used relative tolerance 1e-6 and absolute tolerance 1e-8, BinaryMag2 or BinaryMagDark for binary and Multipoly TripleLightCurve for triple. The maximum fractional differences were 9.50e-6 and 3.97e-5 respectively. An earlier tighter VBM run in one process aborted in the allocator. The recorded results are from separate successful invocations per geometry.

All selected binary epochs returned COVERAGE_UNCERTIFIED and converged value candidates before and after the optimization. The triple rows returned OK. An independent value comparison on these sampled points cannot certify binary image coverage. Targeted regression checks passed on the rebuilt extension (26 binary/triple public API, gradient and event cases, 13 continuous caustic enclosure cases). The triple interval, surface and events CTest targets also passed.
