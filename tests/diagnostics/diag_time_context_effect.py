"""Test if single-point vs time-series context matters."""
import numpy as np
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve, vbbl_curve

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

# Get all times and magnifications from VBBL
times_full = np.linspace(CASE.t_min, CASE.t_max, CASE.n_times)
mags_vbbl_full = vbbl_curve(CASE, times_full)

# Find the problematic time point
idx_problem = np.argmin(np.abs(times_full - 0.006))
t_problem = times_full[idx_problem]

print(f"Problem point: index {idx_problem}, t={t_problem:.6f}")
print()

# Test 1: single point in isolation
times_single = np.array([t_problem])
mags_vbbl_single = vbbl_curve(CASE, times_single)
mag_vbbl_single = mags_vbbl_single[0]

opts_200 = lcbinint.Options(source_bins=200, vbbl_compatible=1)
result_single = lc_curve(CASE, times_single, opts_200)
mag_lc_single = result_single.magnifications[0]

print("Test 1: Single point in isolation")
print(f"  VBBL={mag_vbbl_single:.6f}")
print(f"  lcbinint@200={mag_lc_single:.6f}")
print(f"  Error={abs(mag_lc_single - mag_vbbl_single) / mag_vbbl_single * 100:.2f}%")
print()

# Test 2: narrow time window around the point
times_narrow = np.linspace(t_problem - 0.01, t_problem + 0.01, 21)
mags_vbbl_narrow = vbbl_curve(CASE, times_narrow)
result_narrow = lc_curve(CASE, times_narrow, opts_200)
mags_lc_narrow = result_narrow.magnifications
idx_center = np.argmin(np.abs(times_narrow - t_problem))

print("Test 2: Narrow window (±0.01)")
print(f"  At center (t={times_narrow[idx_center]:.6f}):")
print(f"    VBBL={mags_vbbl_narrow[idx_center]:.6f}")
print(f"    lcbinint@200={mags_lc_narrow[idx_center]:.6f}")
print(f"    Error={abs(mags_lc_narrow[idx_center] - mags_vbbl_narrow[idx_center]) / mags_vbbl_narrow[idx_center] * 100:.2f}%")
print()

# Test 3: Full 400-point range (original problem)
result_full = lc_curve(CASE, times_full, opts_200)
mags_lc_full = result_full.magnifications
mag_lc_full = mags_lc_full[idx_problem]

print("Test 3: Full 400-point time series")
print(f"  At t={times_full[idx_problem]:.6f}:")
print(f"    VBBL={mags_vbbl_full[idx_problem]:.6f}")
print(f"    lcbinint@200={mag_lc_full:.6f}")
print(f"    Error={abs(mag_lc_full - mags_vbbl_full[idx_problem]) / mags_vbbl_full[idx_problem] * 100:.2f}%")
print()

print("Key question: Is there a difference between the single-point and full-series results?")
print(f"  Single-point error: {abs(mag_lc_single - mag_vbbl_single) / mag_vbbl_single * 100:.2f}%")
print(f"  Narrow-window error: {abs(mags_lc_narrow[idx_center] - mags_vbbl_narrow[idx_center]) / mags_vbbl_narrow[idx_center] * 100:.2f}%")
print(f"  Full-series error: {abs(mag_lc_full - mags_vbbl_full[idx_problem]) / mags_vbbl_full[idx_problem] * 100:.2f}%")
