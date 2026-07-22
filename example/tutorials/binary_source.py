"""Tutorial 5: binary source with coupled circular xallarap."""

import numpy as np
import lcbinint

from common import render_tutorial


times = np.linspace(-15.0, 15.0, 301)
params = dict(
    t0=0.0, tE=20.0, u0=0.3, alpha=0.5, s=1.0, q=0.1, rho=0.003,
    xi_1=0.25, xi_2=-0.1, period_xa=35.0, inc_xa=1.1,
    q_source=0.5, q_mass=2.0,
)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
single_model = lcbinint.Model(lens="binary", source="single", xallarap="circular_elements")
single = lcbinint.LightCurve(options=options, model=single_model)
binary_model = lcbinint.Model(lens="binary", source="binary", xallarap="circular_elements")
binary = lcbinint.LightCurve(options=options, model=binary_model)
render_tutorial(
    "05-binary-source", "binary source", binary, times, params,
    comparisons=(("primary source only", single),),
)
