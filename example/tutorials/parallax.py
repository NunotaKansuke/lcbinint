"""Tutorial 2: annual and terrestrial parallax."""

import numpy as np
import lcbinint

from common import render_tutorial


t_ref = 2459000.0
times = t_ref + np.linspace(-20.0, 20.0, 301)
params = dict(
    t0=t_ref, tE=30.0, u0=0.02, alpha=0.5, s=0.95, q=0.01, rho=0.005,
    piEN=0.15, piEE=-0.08,
)
baseline = lcbinint.LightCurve(options=lcbinint.Options(caustic_bins=600))
model = lcbinint.Model(
    parallax=True, terrestrial=True,
    sky=lcbinint.obs.SkyCoord(270.0, -30.0), t_ref=t_ref,
)
ground = lcbinint.LightCurve(
    model=model, options=lcbinint.Options(caustic_bins=600),
    site=lcbinint.obs.Site("ground", -29.0, 70.7),
)
render_tutorial(
    "02-parallax", "annual + terrestrial", ground, times, params,
    comparisons=(("geocentric baseline", baseline),),
)
