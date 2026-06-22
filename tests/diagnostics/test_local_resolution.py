"""Test t=0.006015 with different local resolutions."""
import numpy as np
import lcbinint
from adaptive_source_bins_sweep import Case, lc_curve, vbbl_curve

# The problematic time
t_problem = 0.006015

# Test with different local grid sizes around t=0.006015
for center_width in [0.01, 0.02, 0.05, 0.1, 0.2, 1.6]:
    for n_points in [10, 20, 50, 100]:
        case = Case(
            name="wide caustic",
            separation=0.95,
            mass_ratio=0.01,
            u0=-0.01,
            alpha=0.5,
            rho=0.01,
            t_min=t_problem - center_width/2,
            t_max=t_problem + center_width/2,
            n_times=n_points,
        )

        times = np.linspace(t_problem - center_width/2, t_problem + center_width/2, n_points)
        idx = np.argmin(np.abs(times - t_problem))

        mag_vbbl = vbbl_curve(case, times)[idx]

        opts_200 = lcbinint.Options(source_bins=200, vbbl_compatible=1)
        result = lc_curve(case, times, opts_200)
        mag_lc = result.magnifications[idx]

        error = abs(mag_lc - mag_vbbl) / mag_vbbl * 100

        if error > 1.0:  # Only print large errors
            print(f"width={center_width:.3f}, n={n_points:3d}: error={error:6.2f}%")

print()
print("Testing the full [-0.8, 0.8] range at different grid sizes:")
for n_points in [100, 200, 300, 400, 500, 600]:
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
    idx = np.argmin(np.abs(times - t_problem))

    mag_vbbl = vbbl_curve(case, times)[idx]
    opts_200 = lcbinint.Options(source_bins=200, vbbl_compatible=1)
    result = lc_curve(case, times, opts_200)
    mag_lc = result.magnifications[idx]

    error = abs(mag_lc - mag_vbbl) / mag_vbbl * 100
    t_actual = times[idx]
    print(f"n={n_points:3d}, t_actual={t_actual:.6f}: error={error:6.2f}%")
