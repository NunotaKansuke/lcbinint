"""Tutorial 6: triple-lens light curve, trajectory, and caustics."""

import numpy as np
import lcbinint

from common import render_tutorial


times = np.linspace(-0.15, 0.25, 301)
params = dict(
    t0=0.0, tE=1.0, u0=0.01, alpha=0.5, s=1.0, q=1e-3,
    q2=1e-4, sep2=0.5, ang=1.2, rho=0.001,
)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
model = lcbinint.Model(lens="triple", source="single")
curve = lcbinint.LightCurve(options=options, model=model)
render_tutorial(
    "06-triple-lens", "triple lens", curve, times, params,
    geometry_limits=(-0.35, 0.35, -0.25, 0.25),
)
