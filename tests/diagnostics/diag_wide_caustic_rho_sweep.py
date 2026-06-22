"""Test error vs rho for wide caustic geometry."""
import numpy as np
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve, vbbl_curve

# Test at t=0.006 (the problematic point) with varying rho
rhos = [0.001, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]
times = np.array([0.006])

print("Wide caustic error analysis at t=0.006 vs rho:")
print()

for rho in rhos:
    case = Case(
        name=f"wide_rho_{rho:.3f}",
        separation=0.95,
        mass_ratio=0.01,
        u0=-0.01,
        alpha=0.5,
        rho=rho,
        t_min=-0.8,
        t_max=0.8,
        n_times=1,
    )

    # VBBL
    mag_vbbl = vbbl_curve(case, times)[0]

    # lcbinint fixed@200
    opts_200 = lcbinint.Options(source_bins=200, vbbl_compatible=1)
    result_200 = lc_curve(case, times, opts_200)
    mag_200 = result_200.magnifications[0]

    error = abs(mag_200 - mag_vbbl) / mag_vbbl * 100

    print(f"rho={rho:.4f}: lc_200={mag_200:.6f}, VBBL={mag_vbbl:.6f}, error={error:.2f}%")
