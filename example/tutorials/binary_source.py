"""Tutorial 5: binary source with coupled circular xallarap."""

import numpy as np
import lcbinint

from common import render_tutorial


times = np.linspace(7550.4 - 37.3, 7550.4 + 37.3, 300)
params = dict(
    # VBM BinSourceExtLightCurveXallarap example, expressed with lcbinint
    # names. q_source is VBM's flux ratio FR.
    t0=7550.4, tE=37.3, u0=0.1, alpha=0.0, s=1.0, q=1.0, rho=0.004,
    t0_2=7555.8, u0_2=0.05, q_source=0.4,
    piEN=0.03, piEE=-0.02, xi_1=0.1, xi_2=0.05,
    w1=0.021, w2=-0.02, w3=0.03,
)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
single_model = lcbinint.Model(lens="binary", source="single", xallarap="circular_velocity")
single = lcbinint.LightCurve(options=options, model=single_model)
binary_model = lcbinint.Model(lens="binary", source="binary", xallarap="circular_velocity")
binary = lcbinint.LightCurve(options=options, model=binary_model)
render_tutorial(
    "05-binary-source", "binary source", binary, times, params,
    comparisons=(("primary source only", single),),
)
