[Previous: Critical curves and caustics](CriticalCurvesAndCaustics.md) · [Documentation home](readme.md) · [Next: Accuracy control](AccuracyControl.md)

# Limb darkening

> VBMicrolensing correspondence: [LimbDarkening.md](https://github.com/valboz/VBMicrolensing/blob/main/docs/python/LimbDarkening.md). The linear coefficient `0.51` and square-root coefficients `0.51, 0.3` are retained.

Limb darkening is a property of a light-curve calculation, so configure it when
creating `LightCurve`. It is not a per-call argument. Create one evaluator for
each source profile (or passband), then call each evaluator with the same times
and lens parameters.

## Configure `LightCurve`

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

params = {
    "s": 0.8, "q": 0.1, "u0": 0.01, "alpha": 0.0,
    "rho": 0.01, "tE": 1.0, "t0": 0.0,
}
t = np.linspace(-0.04, 0.04, 161)
options = lcbinint.Options(tol=1e-3, reltol=1e-3)

uniform_curve = lcbinint.LightCurve(
    options=options,
    limb_darkening=lcbinint.LimbDarkening.none(),
)
linear_curve = lcbinint.LightCurve(
    options=options,
    limb_darkening=lcbinint.LimbDarkening.linear(0.51),
)
square_root_curve = lcbinint.LightCurve(
    options=options,
    limb_darkening=lcbinint.LimbDarkening.square_root(0.51, 0.3),
)

uniform = uniform_curve(t, params)
linear = linear_curve(t, params)
square_root = square_root_curve(t, params)
```

`none()` is a uniform source. `linear(u)` uses the linear coefficient `u`.
`square_root(c, d)` uses the native two-coefficient profile
`I(μ) = 1 - c(1-μ) - d(1-√μ)`.

Plot the three light curves from those configured evaluators:

```python
plt.figure(figsize=(5.5, 3.2))
plt.plot(t, uniform, label="uniform")
plt.plot(t, linear, label="linear: 0.51")
plt.plot(t, square_root, label="square root: 0.51, 0.3")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
plt.show()
```

![Limb-darkening comparison](figures/LimbDarkening_comparison.png)

Quadratic, logarithmic, user-defined, and simultaneous multi-band profiles are
not native profiles in the current `lcbinint` Python API. The historical
`LimbDarkening.quadratic(c, d)` compatibility alias maps to the same two
coefficients as `square_root(c, d)`; use the explicit name in new code.

[Previous: Critical curves and caustics](CriticalCurvesAndCaustics.md) · [Documentation home](readme.md) · [Next: Accuracy control](AccuracyControl.md)
