"""Detailed analysis of the peak error point (t=0.006)."""
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

times = np.linspace(CASE.t_min, CASE.t_max, CASE.n_times)

# Focus on times around the peak
idx_0_006 = np.argmin(np.abs(times - 0.006))
print(f"Investigating point at index {idx_0_006}, t={times[idx_0_006]:.6f}")
print()

# Run with fixed bins at different levels
for bins in [50, 100, 200]:
    opts = lcbinint.Options(source_bins=bins, vbbl_compatible=1)
    result = lc_curve(CASE, times, opts)
    mag = result.magnifications[idx_0_006]
    print(f"Fixed @{bins:3d} bins: {mag:.6f}")

# Run adaptive
opts_ad = lcbinint.Options(source_bins=50, adaptive_source_bins=1,
                            max_source_bins=200, reltol=1e-4, vbbl_compatible=1)
result_ad = lc_curve(CASE, times, opts_ad)
mag_ad = result_ad.magnifications[idx_0_006]
ref_lv = result_ad.finite_source_refinement_levels[idx_0_006]

print(f"Adaptive:    {mag_ad:.6f} (refinement_level={ref_lv})")
print()

# VBBL
mag_vbbl = vbbl_curve(CASE, times)[idx_0_006]
print(f"VBBL:        {mag_vbbl:.6f} (reference)")
print()

# Analyze error by bin count
results = {}
for bins in [50, 100, 200]:
    opts = lcbinint.Options(source_bins=bins, vbbl_compatible=1)
    result = lc_curve(CASE, times, opts)
    results[bins] = result.magnifications[idx_0_006]

print("Analysis:")
print(f"  50-bin  error: {abs(results[50] - mag_vbbl) / mag_vbbl * 100:.2f}%")
print(f"  100-bin error: {abs(results[100] - mag_vbbl) / mag_vbbl * 100:.2f}%")
print(f"  200-bin error: {abs(results[200] - mag_vbbl) / mag_vbbl * 100:.2f}%")
print()

# The adaptive chose level 1 (100 bins) but error is still >10%
# Why didn't it refine to level 2?
print("Why didn't adaptive refine to level 2?")
print(f"  Adaptive stopping at level 1: {mag_ad:.6f}")
print(f"  Level 1 (100 bins): {results[100]:.6f}")
print(f"  Level 2 (200 bins): {results[200]:.6f}")
print(f"  Improvement from lv1→lv2: {abs(results[100] - results[200]):.6f}")
print(f"  Relative improvement: {abs(results[100] - results[200]) / abs(results[100]) * 100:.3f}%")
