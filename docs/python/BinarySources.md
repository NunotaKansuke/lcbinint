[Previous: Orbital motion](OrbitalMotion.md) · [Documentation home](readme.md) · [Next: Xallarap](Xallarap.md)

# Binary sources

`source="binary"` describes two luminous source stars magnified by the same
lens. It changes the source trajectory and flux model only; it is independent
of the lens multiplicity.

The observed magnification is
`A(t) = (A1(t) + flux_ratio * A2(t)) / (1 + flux_ratio)`, with
`flux_ratio = F2 / F1`.

| Parameter | Meaning |
| --- | --- |
| `t0`, `u0`, `rho1` | Closest-approach epoch, impact parameter, and normalized radius of source 1. |
| `t0_2`, `u0_2`, `rho2` | Corresponding parameters for source 2. |
| `tE`, `alpha` | Shared Einstein timescale and trajectory angle. |
| `flux_ratio` | Source flux ratio, `F2 / F1`. |

Both `rho1` and `rho2` are required. Set either to zero for a point source.

```python
import numpy as np
import matplotlib.pyplot as plt
import lcbinint

options = lcbinint.Options(tol=1e-3, reltol=1e-3)
binary_curve = lcbinint.LightCurve(source="binary", options=options)
single_source_curve = lcbinint.LightCurve(options=options)
parameters = {
    "s": 0.9, "q": 0.1, "alpha": 1.0, "tE": 30.0,
    "t0": 7500.0, "u0": 0.10, "rho1": 0.004,
    "t0_2": 7501.2, "u0_2": -0.06, "rho2": 0.002,
    "flux_ratio": 0.4,
}
times = np.linspace(7470.0, 7530.0, 300)

source1_params = dict(parameters, rho=parameters["rho1"])
source2_params = dict(
    parameters,
    t0=parameters["t0_2"], u0=parameters["u0_2"], rho=parameters["rho2"],
)
for key in ("rho1", "t0_2", "u0_2", "rho2", "flux_ratio"):
    source1_params.pop(key)
    source2_params.pop(key)

source1_magnification = single_source_curve(times, source1_params)
source2_magnification = single_source_curve(times, source2_params)
binary_magnification = binary_curve(times, parameters)
```

```python
plt.figure(figsize=(4.8, 3.0))
plt.plot(times, source1_magnification, color="#0173B2", alpha=0.45, lw=1.0, label="source 1")
plt.plot(times, source2_magnification, color="#029E73", alpha=0.45, lw=1.0, label="source 2")
plt.plot(times, binary_magnification, color="black", lw=1.5, label="total")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.legend(loc="upper left", fontsize=8)
plt.show()
```

![Static binary-source light curve](figures/BinarySource_static_lightcurve.png)

```python
source1_trajectory = single_source_curve.source_trajectory(times, source1_params)
source2_trajectory = single_source_curve.source_trajectory(times, source2_params)
caustics = single_source_curve.caustics(source1_params)

plt.figure(figsize=(2.8, 2.8))
for x, y in zip(caustics.x, caustics.y):
    plt.plot(-np.asarray(x), -np.asarray(y), color="#6C6C6C", lw=1.1)
plt.plot(-np.asarray(source1_trajectory.x), -np.asarray(source1_trajectory.y), color="#0173B2", label="source 1")
plt.plot(-np.asarray(source2_trajectory.x), -np.asarray(source2_trajectory.y), color="#029E73", label="source 2")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.legend(fontsize=7)
plt.show()
```

![Static binary-source trajectories and caustics](figures/BinarySource_static_geometry.png)

For source orbital motion alone, see [Xallarap](Xallarap.md). For a physical
binary source with xallarap, see [Binary source + xallarap](BinarySourceXallarap.md).

[Previous: Orbital motion](OrbitalMotion.md) · [Documentation home](readme.md) · [Next: Xallarap](Xallarap.md)
