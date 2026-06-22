"""Test if grid size affects the result at t=0.006."""
import numpy as np
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve, vbbl_curve

def test_grid(t_min, t_max, n_times, label):
    case = Case(
        name="wide caustic",
        separation=0.95,
        mass_ratio=0.01,
        u0=-0.01,
        alpha=0.5,
        rho=0.01,
        t_min=t_min,
        t_max=t_max,
        n_times=n_times,
    )

    times = np.linspace(t_min, t_max, n_times)
    idx = np.argmin(np.abs(times - 0.006))

    mag_vbbl = vbbl_curve(case, times)[idx]

    opts_200 = lcbinint.Options(source_bins=200, vbbl_compatible=1)
    result = lc_curve(case, times, opts_200)
    mag_lc = result.magnifications[idx]

    error = abs(mag_lc - mag_vbbl) / mag_vbbl * 100
    print(f"{label:40s}: lc={mag_lc:.6f}, vbbl={mag_vbbl:.6f}, error={error:6.2f}%")

print("Testing different grid sizes around t=0.006:")
print()

test_grid(-0.01, 0.01, 3, "3 points [-0.01, 0.006, 0.01]")
test_grid(-0.01, 0.01, 21, "21 points")
test_grid(-0.1, 0.1, 201, "201 points")
test_grid(-0.8, 0.8, 400, "400 points (original)")

print()
print("Key question: Does the grid size or range affect the magnification at t=0.006?")
