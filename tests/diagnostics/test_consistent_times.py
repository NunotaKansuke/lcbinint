"""Test at consistent time points across different grid sizes."""
import numpy as np
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve, vbbl_curve

# First, find times that actually exist in the 400-point grid
case_400 = Case(
    name="wide caustic",
    separation=0.95,
    mass_ratio=0.01,
    u0=-0.01,
    alpha=0.5,
    rho=0.01,
    t_min=-0.8,
    t_max=0.8,
    n_times=400,
)
times_400 = np.linspace(-0.8, 0.8, 400)

# Test a few times from the 400-point grid
test_times = [times_400[50], times_400[100], times_400[200], times_400[201], times_400[250], times_400[300]]

print("Testing at specific times with different grid sizes:")
print()

opts_200 = lcbinint.Options(source_bins=200, vbbl_compatible=1)

for test_time in test_times[:3]:  # Just first 3 to keep it manageable
    print(f"\nAt t={test_time:.6f}:")

    # Test with different grid sizes
    for n_points in [400, 500, 1000]:
        case = Case(
            name="wide caustic",
            separation=0.95,
            mass_ratio=0.01,
            u0=-0.01,
            alpha=0.5,
            rho=0.01,
            t_min=-0.8,
            t_max=0.8,
            n_times=n_points,
        )

        times = np.linspace(-0.8, 0.8, n_points)
        idx = np.argmin(np.abs(times - test_time))

        mag_vbbl = vbbl_curve(case, times)[idx]
        result = lc_curve(case, times, opts_200)
        mag_lc = result.magnifications[idx]

        error = abs(mag_lc - mag_vbbl) / mag_vbbl * 100
        print(f"  {n_points:4d} points: lc={mag_lc:.4f}, vbbl={mag_vbbl:.4f}, error={error:5.2f}%")
