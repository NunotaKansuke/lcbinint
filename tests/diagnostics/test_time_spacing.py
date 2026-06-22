"""Test if time spacing affects the result."""
import numpy as np
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve, vbbl_curve

def test_grid(n_times, label):
    case = Case(
        name="wide caustic",
        separation=0.95,
        mass_ratio=0.01,
        u0=-0.01,
        alpha=0.5,
        rho=0.01,
        t_min=-0.8,
        t_max=0.8,
        n_times=n_times,
    )

    times = np.linspace(-0.8, 0.8, n_times)
    idx = np.argmin(np.abs(times - 0.006))
    t_actual = times[idx]
    dt = (0.8 - (-0.8)) / (n_times - 1)

    mag_vbbl = vbbl_curve(case, times)[idx]

    opts_200 = lcbinint.Options(source_bins=200, vbbl_compatible=1)
    result = lc_curve(case, times, opts_200)
    mag_lc = result.magnifications[idx]

    error = abs(mag_lc - mag_vbbl) / mag_vbbl * 100
    print(f"{label:50s}: dt={dt:.5f}, lc={mag_lc:.6f}, error={error:6.2f}%")

print("Testing time spacing effect (all over [-0.8, 0.8]):")
print()

test_grid(50, "50 points")
test_grid(100, "100 points")
test_grid(200, "200 points")
test_grid(300, "300 points")
test_grid(400, "400 points (original)")
test_grid(500, "500 points")
test_grid(800, "800 points")
