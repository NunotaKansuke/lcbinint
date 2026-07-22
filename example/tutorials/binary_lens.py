"""Tutorial 1: binary-lens light curve, trajectory, and caustics."""

import numpy as np
import lcbinint

from common import render_tutorial


times = np.linspace(7470.0, 7530.0, 300)
params = dict(t0=7500.0, tE=30.0, u0=0.0, alpha=1.0, s=0.9, q=0.1, rho=0.01)
options = lcbinint.Options(coordinates="vbm", nbin="auto", caustic_bins=600)
model = lcbinint.Model(lens="binary", source="single")
curve = lcbinint.LightCurve(options=options, model=model)
render_tutorial("01-binary-lens", "binary lens", curve, times, params)
