"""Compare public binary LOM calls on identical trajectories in two builds."""
import json
from pathlib import Path
import sys
import time
import numpy as np

build = Path(sys.argv[1]).resolve()
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in str(f)]
sys.path.insert(0, str(build))
import jax
jax.config.update("jax_enable_x64", True)
import lcbinint as lc
assert Path(lc._native.__file__).is_relative_to(build)

for orbit in ("circular", "kepler"):
    for limb in (0., .5):
        curve = lc.LightCurve(orbital_motion=orbit, t_ref=.1,
            options=lc.Options(coordinates="vbm", tol=1e-6, reltol=1e-6),
            limb_darkening=lc.LimbDarkening.linear(limb))
        for q in (.001, .1, 1., 10., 1000.):
            params = dict(t0=0., u0=.03, tE=1., alpha=-.08, rho=.003,
                          q=q, s=1., g1=.2, g2=.03, g3=.04)
            if orbit == "kepler":
                params.update(lom_szs=.7, lom_ar=1.3)
            epochs = np.r_[np.linspace(-.35,.65,21)[::-1],.1,.1]
            for grad in (False, True):
                if q == .001:
                    curve(epochs, params, with_gradient=grad)  # warm the trajectory transform
                start = time.perf_counter_ns()
                r = curve(epochs, params, with_gradient=grad)
                ms = (time.perf_counter_ns() - start) / 1e6
                row = dict(orbit=orbit, limb=limb, q=q, gradient=grad, ms=ms)
                for key in ("magnifications", "gradients", "gradient_valid", "status",
                            "support_required", "value_converged", "support_certified"):
                    row[key] = np.asarray(getattr(r,key)).tolist()
                print(json.dumps(row))
