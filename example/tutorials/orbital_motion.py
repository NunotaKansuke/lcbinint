"""Tutorial 3: circular lens orbital motion."""

import numpy as np
import lcbinint

from common import render_tutorial


t_ref = 0.0
times = np.linspace(-15.0, 15.0, 301)
params = dict(
    t0=0.0, tE=20.0, u0=0.08, alpha=0.5, s=0.97, q=10.0**-1.5, rho=0.003,
    g1=0.004, g2=0.011, g3=0.006,
)
static = lcbinint.LightCurve(options=lcbinint.Options(caustic_bins=600))
orbit = lcbinint.LightCurve(
    model=lcbinint.Model(orbital_motion="circular", t_ref=t_ref),
    options=lcbinint.Options(caustic_bins=600),
)
render_tutorial(
    "03-orbital-motion", "circular lens orbit", orbit, times, params,
    comparisons=(("static lens", static),), caustic_time=t_ref,
)
