"""Investigate 10% peak error for wide caustic finite source case."""
import numpy as np
import time
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve, vbm_curve

# Wide caustic finite source case from user report
CASE = Case(
    name="wide caustic finite source",
    separation=0.95,
    mass_ratio=0.01,
    u0=-0.01,
    alpha=0.5,
    rho=1.0e-2,
    t_min=-0.8,
    t_max=0.8,
    n_times=400,
)

times = np.linspace(CASE.t_min, CASE.t_max, CASE.n_times)

# Run both implementations
opts_lc = lcbinint.Options(source_bins=50, adaptive_source_bins=1,
                            max_source_bins=200, reltol=1e-4)
t0 = time.perf_counter()
result_lc = lc_curve(CASE, times, opts_lc)
t_lc = time.perf_counter() - t0

t0 = time.perf_counter()
result_vbm = vbm_curve(CASE, times)
t_vbm = time.perf_counter() - t0

mag_lc = np.array(result_lc.magnifications)
mag_vbm = result_vbm

# Find the peak region
peak_idx = np.argmax(mag_vbm)
peak_mag = mag_vbm[peak_idx]
peak_time = times[peak_idx]

print(f"Case: {CASE.name}")
print(f"Peak at t={peak_time:.4f}, mag={peak_mag:.4f}")
print(f"Timing: lcbinint={t_lc:.3f}s, vbm={t_vbm:.3f}s")
print()

# Analyze error in peak region (±20 points around peak)
win = 20
peak_region = slice(max(0, peak_idx - win), min(len(times), peak_idx + win + 1))
t_peak = times[peak_region]
mag_peak_lc = mag_lc[peak_region]
mag_peak_vbm = mag_vbm[peak_region]

rel_err = np.abs(mag_peak_lc - mag_peak_vbm) / mag_peak_vbm
max_err_idx = np.argmax(rel_err)

print("Peak region analysis (±20 points):")
print(f"  Time range: [{t_peak[0]:.4f}, {t_peak[-1]:.4f}]")
print(f"  Max relative error: {rel_err[max_err_idx]:.4%} at t={t_peak[max_err_idx]:.4f}")
print(f"    lcbinint={mag_peak_lc[max_err_idx]:.6f}, vbm={mag_peak_vbm[max_err_idx]:.6f}")
print()

# Refinement levels
ref_levels = np.array(result_lc.finite_source_refinement_levels)
n_ref = np.count_nonzero(ref_levels)
print(f"Refinement: {n_ref} points refined (lv1={np.sum(ref_levels==1)}, lv2={np.sum(ref_levels==2)})")
print()

# Check detailed error profile
print("Error profile across full time range:")
rel_err_all = np.abs(mag_lc - mag_vbm) / mag_vbm
top_5_err_idx = np.argsort(-rel_err_all)[:5]
for idx in top_5_err_idx:
    print(f"  t={times[idx]:7.4f}: lc={mag_lc[idx]:.6f}, vbm={mag_vbm[idx]:.6f}, "
          f"rel_err={rel_err_all[idx]:.4%}, ref_lv={ref_levels[idx]}")
