import sys, math, importlib.util
import numpy as np

sys.path.insert(0, "tests/diagnostics")
spec = importlib.util.spec_from_file_location(
    "adaptive_source_bins_sweep", "tests/diagnostics/adaptive_source_bins_sweep.py"
)
sweep = importlib.util.module_from_spec(spec)
sys.modules["adaptive_source_bins_sweep"] = sweep
spec.loader.exec_module(sweep)

import lcbinint
import VBMicrolensing

case = sweep.Case("planetary_large_source", 1.0, 1e-3, -0.01, 0.0, 0.01, -0.5, 0.5)
times = np.linspace(-0.5, 0.5, 61)

vbb = VBMicrolensing.VBMicrolensing()
vbb.Tol = 1e-3
vbb.RelTol = 0.0
ref = np.array(
    vbb.BinaryLightCurve(
        [math.log(1.0), math.log(1e-3), -0.01, 0.0, math.log(0.01), 0.0, 0.0],
        times.tolist(),
    )[0]
)


def run(source_bins, max_bins, adaptive, reltol=1e-4):
    opts = lcbinint.Options(
        source_bins=source_bins,
        adaptive_source_bins=(1 if adaptive else 0),
        max_source_bins=max_bins,
        reltol=(reltol if adaptive else 0.0),    )
    r = sweep.lc_curve(case, times, opts)
    return (
        np.array(r.magnifications),
        np.array(r.finite_source_error_estimates),
        np.array(r.finite_source_converged),
        np.array(r.finite_source_refinement_levels),
    )


mag50, est50, _, _ = run(50, 50, False)
mag100f, _, _, _ = run(100, 100, False)
mag100a, est100a, conv100a, lvl100a = run(50, 100, True, 1e-4)
target = 1e-4 * np.maximum(np.abs(mag100a), 1.0)

accepted_bad = conv100a & (np.abs(mag100a - ref) > target * 1.05)
print(f"max_bins=100: accepted_bad={np.sum(accepted_bad)}, unconverged={np.sum(~conv100a)}")
for i in np.where(accepted_bad)[0]:
    adj = abs(mag100f[i] - mag50[i])
    print(
        f"  i={i} t={times[i]:.4f} lvl={lvl100a[i]} est={est100a[i]:.3e} "
        f"target={target[i]:.3e} adj={adj:.3e} adj/target={adj/target[i]:.4f} "
        f"rel={abs(mag100a[i] / ref[i] - 1):.3e}"
    )

print(f"\nLevel-1 accepts ({np.sum(conv100a & (lvl100a == 1))} total, first 8):")
for i in np.where(conv100a & (lvl100a == 1))[0][:8]:
    adj = abs(mag100f[i] - mag50[i])
    print(
        f"  i={i} t={times[i]:.4f} est={est100a[i]:.3e} target={target[i]:.3e} "
        f"adj={adj:.3e} adj/target={adj/target[i]:.4f} rel={abs(mag100a[i] / ref[i] - 1):.3e}"
    )
