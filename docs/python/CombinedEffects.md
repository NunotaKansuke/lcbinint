[Previous: Binary source + xallarap](BinarySourceXallarap.md) · [Documentation home](readme.md)

# Combining higher-order effects

For a single source, parallax, lens orbital motion, and xallarap are
composable. Numerical controls remain on `Options`.

| Effect | `Model` switch | Parameters used here |
| --- | --- | --- |
| Annual parallax | `parallax=True` | `piEN`, `piEE`, `sky`, `t_ref` |
| Lens orbital motion | `orbital_motion="circular"` | `g1`, `g2`, `g3` |
| Xallarap | `xallarap="circular_velocity"` | `xi_1`, `xi_2`, `w1`, `w2`, `w3` |

```python
import numpy as np
import lcbinint

sky = lcbinint.obs.SkyCoord("17:59:02.3", "-29:04:15.2")
curve = lcbinint.LightCurve(
    model=lcbinint.Model(
        parallax=True,
        orbital_motion="circular",
        xallarap="circular_velocity",
        sky=sky,
        t_ref=7500.0,
    )
)
parameters = {
    "s": 0.9, "q": 0.1, "t0": 7500.0, "u0": 0.1,
    "tE": 30.0, "alpha": 1.0, "rho": 0.01,
    "piEN": 0.3, "piEE": -0.2,
    "g1": 0.011, "g2": -0.005, "g3": 0.005,
    "xi_1": 0.04, "xi_2": -0.02,
    "w1": 0.01, "w2": 0.02, "w3": 0.015,
}
times = np.linspace(7470.0, 7530.0, 300)
magnification = curve(times, parameters)
```

Binary sources can also be combined with xallarap. They always use independent
`rho1`, `rho2`, and `flux_ratio`; `source_mass_ratio` is used only to distribute
the dynamical xallarap orbit about its centre of mass. See
[Binary source + xallarap](BinarySourceXallarap.md) for the two
velocity-coordinate choices and the integrated API.

[Previous: Binary source + xallarap](BinarySourceXallarap.md) · [Documentation home](readme.md)
