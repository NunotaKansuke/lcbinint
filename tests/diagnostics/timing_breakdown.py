"""Timing breakdown for the planetary_large_source_ld case."""
import time
import math
import numpy as np
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve

case = Case("planetary_large_source_ld", 1.0, 1e-3, -0.01, 0.5, 0.01, -0.5, 0.5)
times = np.linspace(-0.5, 0.5, 241)

def timed_run(opts, n=5):
    t0 = time.perf_counter()
    for _ in range(n):
        r = lc_curve(case, times, opts)
    ms_per_pt = (time.perf_counter() - t0) / n * 1000 / len(times)
    return r, ms_per_pt

opts_fixed = {}
for bins in [50, 100, 200]:
    opts_fixed[bins] = lcbinint.Options(source_bins=bins)
opts_ad = lcbinint.Options(source_bins=50, adaptive_source_bins=1,
                            max_source_bins=200, reltol=1e-4)

r50,  ms50  = timed_run(opts_fixed[50])
r100, ms100 = timed_run(opts_fixed[100])
r200, ms200 = timed_run(opts_fixed[200])
rad,  ms_ad = timed_run(opts_ad)

lvl = np.array(rad.finite_source_refinement_levels)
n_ref = np.count_nonzero(lvl)
n_lv1 = np.sum(lvl == 1)
n_lv2 = np.sum(lvl == 2)

print(f"fixed@50 : {ms50:.4f} ms/pt")
print(f"fixed@100: {ms100:.4f} ms/pt  ratio_vs_50={ms100/ms50:.2f}x")
print(f"fixed@200: {ms200:.4f} ms/pt  ratio_vs_50={ms200/ms50:.2f}x")
print(f"adaptive : {ms_ad:.4f} ms/pt  ratio_vs_50={ms_ad/ms50:.2f}x")
print()
print(f"refined points: total={n_ref}  lv1_only={n_lv1}  lv2={n_lv2}")
print()

# Expected cost from model (linear-bins model and quadratic-bins model)
non_ref = len(times) - n_ref
r50_ms = ms50  # cost of one 50-bin evaluation

# linear model: cost(bins) proportional to bins
for model, c100_over_c50, c200_over_c50 in [("linear",2,4), ("quadratic",4,16)]:
    expected_ms = (
        (non_ref * 1.0 + n_lv1 * (1 + c100_over_c50) + n_lv2 * (1 + c100_over_c50 + c200_over_c50))
        / len(times)
    ) * r50_ms
    print(f"Expected ({model} bins scale): {expected_ms:.4f} ms/pt  (ratio={expected_ms/r50_ms:.2f}x)")

# Measured overhead above expected
ovhd_linear = ms_ad - (expected_ms if model=="linear" else 0.0)
print()
print("Conclusion:")
print(f"  overhead per refined point (quadratic model) = {(ms_ad - (non_ref/len(times))*r50_ms) * len(times) / max(n_ref,1):.3f} ms")
c = lambda bins: r50_ms * (bins / 50.0) ** 2
total_3lv = n_lv2 * (c(50) + c(100) + c(200)) + n_lv1 * (c(50) + c(100)) + non_ref * c(50)
print(f"  quadratic model total: {total_3lv/len(times):.4f} ms/pt")
