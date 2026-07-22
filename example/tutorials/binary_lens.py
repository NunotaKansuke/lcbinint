"""Tutorial 1: binary-lens light curve, trajectory, and caustics."""

import numpy as np
import lcbinint

from common import render_tutorial


times = np.linspace(-1.0, 1.0, 301)
params = dict(t0=0.0, tE=1.0, u0=0.02, alpha=0.5, s=0.95, q=0.01, rho=0.005)
curve = lcbinint.LightCurve(options=lcbinint.Options(caustic_bins=600))
render_tutorial("01-binary-lens", "binary lens", curve, times, params)
