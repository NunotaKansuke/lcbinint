import math
import numpy as np

import lcbinint
import VBMicrolensing

from adaptive_source_bins_sweep import Case, lc_curve

case = Case("resonant_low_q", 1.0, 0.1, 0.1, 0.0, 3e-3, -0.25, 0.25)
times = np.linspace(-0.25, 0.25, 241)

vbb = VBMicrolensing.VBMicrolensing()
vbb.Tol = 1e-3
vbb.RelTol = 0.0
ref = np.array(vbb.BinaryLightCurve(
    [math.log(1.0), math.log(0.1), 0.1, 0.0, math.log(0.003), 0.0, 0.0],
    times.tolist()
)[0])


def run_fixed(bins):
    opts = lcbinint.Options(source_bins=bins)
    r = lc_curve(case, times, opts)
    return np.array(r.magnifications), np.array(r.finite_source_error_estimates)


mag50, _ = run_fixed(50)
mag100, _ = run_fixed(100)
mag200, est200 = run_fixed(200)

opts_a = lcbinint.Options(
    source_bins=50, adaptive_source_bins=1, max_source_bins=200,
    reltol=1e-4,)
ra = lc_curve(case, times, opts_a)
a_mag = np.array(ra.magnifications)
a_conv = np.array(ra.finite_source_converged)
a_lvl = np.array(ra.finite_source_refinement_levels)

target = 1e-4 * np.maximum(np.abs(a_mag), 1.0)
uncov = ~a_conv
adj_50_100 = np.abs(mag100 - mag50)
adj_100_200 = np.abs(mag200 - mag100)
adj_50_200 = np.abs(mag200 - mag50)
min_hist = np.minimum(adj_100_200, adj_50_200)

print(f"unconverged={np.sum(uncov)}")
print()
print(f"{'i':>4} {'t':>7} {'rel':>8} {'tgt':>9} {'adj50-100':>10} {'adj100-200':>11} {'min_hist':>9} {'min/tgt':>8} {'est200':>9} {'e200/tgt':>8}")
for i in np.where(uncov & (a_lvl == 2))[0][:20]:
    tv = target[i]
    print(f"{i:4d} {times[i]:7.4f} {abs(mag200[i]/ref[i]-1):8.2e} {tv:9.2e} "
          f"{adj_50_100[i]:10.2e} {adj_100_200[i]:11.2e} "
          f"{min_hist[i]:9.2e} {min_hist[i]/tv:8.3f} {est200[i]:9.2e} {est200[i]/tv:8.3f}")
