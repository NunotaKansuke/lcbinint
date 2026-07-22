"""Tutorial 4: circular xallarap (source orbital motion)."""

import numpy as np
import lcbinint

from common import render_tutorial


times = np.linspace(-15.0, 15.0, 301)
params = dict(
    t0=0.0, tE=20.0, u0=0.3, alpha=0.5, s=1.0, q=0.1, rho=0.003,
    xi_1=0.25, xi_2=-0.1, period_xa=35.0, inc_xa=1.1,
)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
plain_model = lcbinint.Model(lens="binary", source="single")
plain = lcbinint.LightCurve(options=options, model=plain_model)
xallarap_model = lcbinint.Model(lens="binary", source="single", xallarap="circular_elements")
xallarap = lcbinint.LightCurve(options=options, model=xallarap_model)
render_tutorial(
    "04-xallarap", "circular xallarap", xallarap, times, params,
    comparisons=(("no xallarap", plain),),
)
