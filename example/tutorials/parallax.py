"""Tutorial 2: annual and terrestrial parallax."""

import numpy as np
import lcbinint

from common import render_tutorial


t_ref = 7500.0
times = np.linspace(7470.0, 7530.0, 300)
params = dict(
    t0=t_ref, tE=30.0, u0=0.0, alpha=1.0, s=0.9, q=0.1, rho=0.01,
    piEN=0.3, piEE=-0.2,
)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
baseline_model = lcbinint.Model(lens="binary", source="single")
baseline = lcbinint.LightCurve(options=options, model=baseline_model)
model = lcbinint.Model(
    parallax=True, terrestrial=True,
    sky=lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2"), t_ref=t_ref,
)
ground = lcbinint.LightCurve(
    model=model, options=options,
    site=lcbinint.obs.Site("ground", -29.0, 70.7),
)
render_tutorial(
    "02-parallax", "annual + terrestrial", ground, times, params,
    comparisons=(("geocentric baseline", baseline),),
)
