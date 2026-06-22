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
import VBBinaryLensing

case = sweep.Case("planetary_large_source", 1.0, 1e-3, -0.01, 0.0, 0.01, -0.5, 0.5)
times = np.linspace(-0.5, 0.5, 61)

vbb = VBBinaryLensing.VBBinaryLensing()
vbb.Tol = 1e-3
vbb.RelTol = 0.0
ref = np.array(
    vbb.BinaryLightCurve(
        [math.log(1.0), math.log(1e-3), -0.01, 0.0, math.log(0.01), 0.0, 0.0],
        times.tolist(),
    )[0]
)


def run(source_bins, max_bins, adaptive=False, reltol=0.0):
    opts = lcbinint.Options(
        source_bins=source_bins,
        adaptive_source_bins=(1 if adaptive else 0),
        max_source_bins=max_bins,
        reltol=reltol,
        vbbl_compatible=1,
    )
    r = sweep.lc_curve(case, times, opts)
    return (
        np.array(r.magnifications),
        np.array(r.finite_source_error_estimates),
        np.array(r.finite_source_converged),
        np.array(r.finite_source_refinement_levels),
    )


mag50f, est50f, _, _ = run(50, 50)
mag100f, est100f, _, _ = run(100, 100)
mag200f, est200f, _, _ = run(200, 200)

target50 = 1e-4 * np.maximum(np.abs(mag50f), 1.0)
target100 = 1e-4 * np.maximum(np.abs(mag100f), 1.0)

# Focus on i=17 (bad accept) and nearby good ones
print("Fixed-bin diagnostics for key time points:")
print(f"{'i':>3} {'t':>8} {'est50/T50':>10} {'est100/T100':>11} {'adj50→100':>11} {'diag100/adj':>11} {'rel100':>8}")
for i in [14, 15, 16, 17, 18, 19, 27, 31, 34]:
    adj = abs(mag100f[i] - mag50f[i])
    ratio_diag_adj = est100f[i] / max(adj, 1e-20)
    print(f"{i:3d} {times[i]:8.4f} {est50f[i]/target50[i]:10.3f} {est100f[i]/target100[i]:11.3f} "
          f"{adj:11.3e} {ratio_diag_adj:11.2f} {abs(mag100f[i]/ref[i]-1):8.3e}")

print()
print("Mag comparison (bad accept i=17):")
i = 17
print(f"  ref  = {ref[i]:.6f}")
print(f"  mag50  = {mag50f[i]:.6f}  err={abs(mag50f[i]-ref[i]):.3e}")
print(f"  mag100 = {mag100f[i]:.6f}  err={abs(mag100f[i]-ref[i]):.3e}")
print(f"  mag200 = {mag200f[i]:.6f}  err={abs(mag200f[i]-ref[i]):.3e}")
print(f"  target = {target100[i]:.3e}")
print(f"  est50  = {est50f[i]:.3e} (fixed)")
print(f"  est100 = {est100f[i]:.3e} (fixed)")
print(f"  adj(50→100) = {abs(mag100f[i]-mag50f[i]):.3e}")
print(f"  adj/target = {abs(mag100f[i]-mag50f[i])/target100[i]:.4f}")
