"""Tutorial 4: circular xallarap (source orbital motion)."""

import numpy as np
import lcbinint

from common import render_tutorial


times = np.linspace(-15.0, 15.0, 301)
params = dict(
    t0=0.0, tE=20.0, u0=0.3, alpha=0.5, s=1.0, q=0.1, rho=0.003,
    xi_1=0.25, xi_2=-0.1, period_xa=35.0, inc_xa=1.1,
)
plain = lcbinint.LightCurve(options=lcbinint.Options(caustic_bins=600))
xallarap = lcbinint.LightCurve(
    model=lcbinint.Model(xallarap="circular_elements"),
    options=lcbinint.Options(caustic_bins=600),
)
render_tutorial(
    "04-xallarap", "circular xallarap", xallarap, times, params,
    comparisons=(("no xallarap", plain),),
)
