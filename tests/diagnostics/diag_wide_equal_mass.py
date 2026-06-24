import math
import numpy as np

import lcbinint
import VBMicrolensing

from adaptive_source_bins_sweep import Case, lc_curve

case = Case("wide_equal_mass", 1.5, 1.0, -0.2, 0.0, 3e-3, -0.4, 0.4)
times = np.linspace(-0.4, 0.4, 241)

vbb = VBMicrolensing.VBMicrolensing()
vbb.Tol = 1e-3
vbb.RelTol = 0.0
ref = np.array(vbb.BinaryLightCurve(
    [math.log(1.5), math.log(1.0), -0.2, 0.0, math.log(0.003), 0.0, 0.0],
    times.tolist()
)[0])


def run_fixed(bins):
    opts = lcbinint.Options(source_bins=bins)
    r = lc_curve(case, times, opts)
    return np.array(r.magnifications), np.array(r.finite_source_error_estimates)


mag50, _ = run_fixed(50)
mag100, _ = run_fixed(100)
mag200, est200 = run_fixed(200)
mag400, est400 = run_fixed(400)

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
adj_200_400 = np.abs(mag400 - mag200)

rel200 = np.abs(mag200 / ref - 1)
rel400 = np.abs(mag400 / ref - 1)
accepted_bad = a_conv & (np.abs(a_mag / ref - 1) > 1e-4 * 1.05)

print(f"unconverged={np.sum(uncov)}, accepted_bad={np.sum(accepted_bad)}")
print()

print("Unconverged points at level 2 (worst 15 by min_hist/target):")
print(f"{'i':>4} {'t':>7} {'rel200':>8} {'tgt':>9} {'min_hist':>9} {'min/tgt':>8} {'est200':>9} {'e200/tgt':>9} {'rel400':>8} {'e200>tgt':>9}")
uncov_lv2 = uncov & (a_lvl == 2)
sort_idx = np.argsort(-min_hist[uncov_lv2])
for i in np.where(uncov_lv2)[0][sort_idx[:15]]:
    tv = target[i]
    print(f"{i:4d} {times[i]:7.4f} {rel200[i]:8.2e} {tv:9.2e} "
          f"{min_hist[i]:9.2e} {min_hist[i]/tv:8.3f} {est200[i]:9.2e} {est200[i]/tv:8.3f} "
          f"{rel400[i]:8.2e} {'YES' if rel200[i] > 1e-4 else 'no':>9}")

print()
print("Summary: min/tgt distribution for unconverged at lv2")
mv = min_hist[uncov_lv2] / target[uncov_lv2]
for thresh in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    n = np.sum(mv < thresh)
    print(f"  min/tgt < {thresh:.1f}: {n}/{np.sum(uncov_lv2)} points")
