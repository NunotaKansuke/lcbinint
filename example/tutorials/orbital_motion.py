"""Tutorial 3: circular lens orbital motion."""

import numpy as np
import lcbinint

from common import render_tutorial


t_ref = 7500.0
times = np.linspace(7470.0, 7530.0, 300)
params = dict(
    t0=7500.0, tE=30.0, u0=0.0, alpha=1.0, s=0.9, q=0.1, rho=0.01,
    g1=0.011, g2=-0.005, g3=0.005,
)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
static_model = lcbinint.Model(lens="binary", source="single")
static = lcbinint.LightCurve(options=options, model=static_model)
orbit_model = lcbinint.Model(lens="binary", source="single", orbital_motion="circular", t_ref=t_ref)
orbit = lcbinint.LightCurve(options=options, model=orbit_model)
render_tutorial(
    "03-orbital-motion", "circular lens orbit", orbit, times, params,
    comparisons=(("static lens", static),), caustic_epochs=(7480.0, 7500.0, 7520.0),
)
