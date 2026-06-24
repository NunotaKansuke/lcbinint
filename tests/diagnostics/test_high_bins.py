"""Test with very high bin counts."""
import numpy as np
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve, vbm_curve

CASE = Case(
    name="wide caustic",
    separation=0.95,
    mass_ratio=0.01,
    u0=-0.01,
    alpha=0.5,
    rho=0.01,
    t_min=-0.8,
    t_max=0.8,
    n_times=1,
)

times = np.array([0.006])
mag_vbm = vbm_curve(CASE, times)[0]

print(f"VBM reference: {mag_vbm:.6f}")
print()

for bins in [50, 100, 150, 200, 250, 300, 350, 400]:
    opts = lcbinint.Options(source_bins=bins)
    result = lc_curve(CASE, times, opts)
    mag = result.magnifications[0]
    error = abs(mag - mag_vbm) / mag_vbm * 100
    print(f"bins={bins:3d}: mag={mag:.6f}, error={error:6.2f}%")
